import torch
from torch import nn
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


def get_shapes(cfg):
    if cfg.dataset.tpu_graphs.custom:
        num_config_feat = 18 if cfg.dataset.tpu_graphs.tpu_task == 'layout' else 24
        if cfg.dataset.tpu_graphs.normalize:
            num_feat = 123
        else:
            num_feat = 179
        num_ops = 125
    else:
        num_config_feat = 18 if cfg.dataset.tpu_graphs.tpu_task == 'layout' else 24
        if cfg.dataset.tpu_graphs.normalize:
            num_feat = 112
        else:
            num_feat = 140
        num_ops = 120
    return num_config_feat, num_feat, num_ops


@register_node_encoder('TPUBslnNode')
class BslnTPUGraphEncoder(torch.nn.Module):

    def __init__(self, dim_in, init_factor: float = 100, *args, **kwargs):
        super().__init__()
        num_config_feat, num_feat, num_ops = get_shapes(cfg)
        dim_config_feat = num_config_feat
        dim_linear_map = num_feat + dim_config_feat + dim_in

        self.emb = nn.Embedding(num_ops, dim_in, max_norm=True)
        self.op_weights = nn.Parameter(
            torch.full((1, 1), init_factor), requires_grad=True)
        self.config_weights = nn.Parameter(
            torch.full((1, 1, dim_config_feat), init_factor),
            requires_grad=True)
        self.linear_map = nn.Linear(dim_linear_map, dim_in, bias=True)
        # self.linear_config_weights = nn.Linear(18, dim_in, bias=True)

    def forward(self, batch):
        op_emb = self.emb(batch.op_code.long())
        x = torch.cat((batch.op_feats, op_emb * self.op_weights), dim=-1)
        if 'config_feats_full' in batch:  # If there are node-level configs
            x = x[..., None, :].broadcast_to(
                *batch.config_feats_full.shape[:-1], x.shape[-1])
            x = torch.cat((x, batch.config_feats_full * self.config_weights),
                          dim=-1)
        else:
            x = x[..., None, :].broadcast_to(
                *x.shape[:-1], batch.config_feats.shape[-2], x.shape[-1])
            config_feats = batch.config_feats[batch.batch]
            x = torch.cat((x, config_feats * self.config_weights), dim=-1)
        x = self.linear_map(x)
        batch.x = x
        return batch


@register_node_encoder('TPUNode')
class TPUGraphEncoder(torch.nn.Module):

    def __init__(self, dim_in, init_factor: float = 100., posenc_dim: int = -1,
                 edge_dim: int = -1, max_edge_order: int = 500):
        super().__init__()

        num_config_feat, num_feat, num_ops = get_shapes(cfg)

        dim_config_feat = num_config_feat

        self.op_weights = nn.Parameter(torch.full((1, 1), init_factor),
                                       requires_grad=True)
        self.config_weights = nn.Parameter(
            torch.full((1, 1, dim_config_feat), init_factor),
            requires_grad=True)
        # self.linear_config_weights = nn.Linear(18, dim_in, bias=True)

        self.emb = nn.Embedding(num_ops, dim_in, max_norm=True)
        self.op_feats_lin = nn.Linear(num_feat, dim_in)
        # torch.nn.init.zeros_(self.op_feats_lin.weight)
        self.config_feats_lin = nn.Linear(dim_config_feat, dim_in)

        self.posenc_lin = None
        if posenc_dim > 0:
            self.posenc_lin = nn.Linear(posenc_dim, dim_in)

        self.edge_dim = edge_dim
        self.max_edge_order = max_edge_order
        self.encode_edges = edge_dim > 0
        if self.encode_edges:
            assert edge_dim % 2 == 0
            self.setup_sinusoidal(edge_dim, max_edge_order)
            self.edge_in_lin = nn.Linear(edge_dim + 1, edge_dim)
            self.edge_out_lin = nn.Linear(edge_dim + 1, edge_dim)

    def setup_sinusoidal(self, dim=64, max_seq_len=100):
        freq = max_seq_len ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq = 1. / freq
        position = torch.arange(0, max_seq_len, dtype=torch.float32)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        self.sinusoidal = nn.Parameter(
            torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1),
            requires_grad=False)

    def forward(self, batch):
        op_emb = self.op_weights * self.emb(batch.op_code.long())
        x = self.op_feats_lin(batch.op_feats.to(torch.float32)) + op_emb
        if self.posenc_lin is not None:
            posenc = batch.laplacian_eigenvector_plain_posenc
            if torch.is_complex(posenc):
                posenc = torch.concatenate((posenc.real, posenc.imag), dim=-1)
            x = x + self.posenc_lin(posenc)
        if 'config_feats_full' in batch:  # If there are node-level configs
            config_feats = self.config_feats_lin(
                self.config_weights * batch.config_feats.to(torch.float32))
            x = x[..., None, :].broadcast_to(
                *batch.config_feats_full.shape[:-1], x.shape[-1]).clone()
            x[batch.config_idx] += config_feats
        else:
            x = x[..., None, :].broadcast_to(
                *x.shape[:-1], batch.config_feats.shape[-2], x.shape[-1])
            config_feats = self.config_feats_lin(
                self.config_weights * batch.config_feats.to(torch.float32)
            )[batch.batch]
            x = x + config_feats
        batch.x = x

        if self.encode_edges:
            edge_attr_forward = batch.edge_attr
            edge_attr_forward[(batch.edge_index[0] != batch.edge_index[1])
                              | (edge_attr_forward >= 0)] += 1
            _, edge_attr_forward = add_remaining_self_loops(
                batch.edge_index, edge_attr_forward, 0, batch.num_nodes)

            edge_attr_forward = torch.concatenate(
                ((edge_attr_forward < 0).to(torch.float32)[:, None],
                 self.sinusoidal[edge_attr_forward.to(torch.int64)]), dim=-1)
            batch.edge_attr_forward = self.edge_in_lin(edge_attr_forward)

            edge_attr_backward = batch.tuple_idx[batch.edge_index[0]]
            edge_attr_backward[(batch.edge_index[0] != batch.edge_index[1])
                               | (edge_attr_backward >= 0)] += 1
            _, edge_attr_backward = add_remaining_self_loops(
                batch.edge_index, edge_attr_backward, 0, batch.num_nodes)

            edge_attr_backward = torch.concatenate(
                ((edge_attr_backward < 0).to(torch.float32)[:, None],
                 self.sinusoidal[edge_attr_backward.to(torch.int64)]), dim=-1)
            batch.edge_attr_backward = self.edge_out_lin(edge_attr_backward)
        return batch
