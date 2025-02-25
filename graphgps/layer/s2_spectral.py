"""Spectral layer for S2GNN."""

from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.graphgym.models.layer import new_layer_config
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (
    LayerConfig, GeneralMultiLayer)
from torch_geometric.graphgym.register import register_head
from torch_geometric.utils import scatter

from graphgps.layer.s2_filter_encoder import FilterEncoder, Window


class FeatureBatchSpectralLayer(nn.Module):
    """Spectral layer that allows for two batch dimensions: 1) multiple graphs
    and 2) multiple input features for identical graph structure."""

    def __init__(self, layer_config: LayerConfig, is_first: bool = True,
                 with_node_residual: bool = True, overwrite_x: bool = True,
                 eps=1e-3, **kwargs):
        super().__init__()
        self.with_node_residual = with_node_residual
        self.overwrite_x = overwrite_x

        if self.with_node_residual:
            assert layer_config.dim_in == layer_config.dim_out, 'Due to residual'
        dim_in, dim_out = layer_config.dim_in, layer_config.dim_out

        self.filter = FilterEncoder(
            dim_in, cfg.gnn.spectral.filter_encoder,
            cfg.gnn.spectral.num_heads_filter_encoder, is_first)

        self.spec_readout_residual = cfg.gnn.spectral.readout_residual

        if cfg.gnn.spectral.real_imag_x_merge == 'node_type':  # only implemented for TPUGraphs
            self.n_opcode = 125 if cfg.dataset.tpu_graphs.custom else 120
            self.factor_real = CheckpointedEmbedding(
                self.n_opcode, dim_out, max_norm=True)
            if cfg.posenc_MagLapPE.q > 0:
                self.factor_imag = CheckpointedEmbedding(
                    self.n_opcode, dim_out, max_norm=True)
        else:
            self.factor_split = nn.Parameter(
                torch.ones(dim_out, 2), requires_grad=True)

        self.feature_transform = SpecFeatureTransformLayer(dim_in)

        self.learnable_norm = None
        if cfg.gnn.spectral.learnable_norm:
            init_ = torch.full((dim_in,), cfg.gnn.spectral.learnable_norm_init,
                               dtype=torch.get_default_dtype())
            self.learnable_norm = nn.Parameter(init_, requires_grad=True)
            self.learnable_norm_bias = nn.Parameter(
                torch.zeros(dim_in), requires_grad=True)
        self.eps = eps

        dropout = layer_config.dropout
        if cfg.gnn.spectral.dropout >= 0:
            dropout = cfg.gnn.spectral.dropout
        self.dropout = nn.Dropout(dropout)

        self.filter_variant = cfg.gnn.spectral.filter_variant
        self.silu_stack, self.proj_out = None, None
        if self.filter_variant in ['silu_mix', 'silu', 'lin']:
            if self.filter_variant == 'silu':
                act = SiLU
            elif self.filter_variant == 'lin':
                act = nn.Identity
            else:
                act = partial(SiLUMixAndBias, dim=dim_out)
            self.silu_stack = nn.Sequential(*[
                nn.Sequential(MaybeComplexLinear(dim_in, dim_out, bias=False), act())
                for _ in range(max(layer_config.num_layers, 1))])
        elif dim_in != dim_out:
            self.proj_out = nn.Linear(dim_in, dim_out)

        self.window = None
        if cfg.gnn.spectral.window:
            self.window = Window(cfg.gnn.spectral.window,
                                 cfg.gnn.spectral.frequency_cutoff)

    def forward(self, batch):
        k = batch.laplacian_eigenvalue_plain.shape[-1]

        # Shape [b1, k, d] or [b1, k, 1, d]
        filter_ = self.filter(batch)

        # Shape [n, b2, d]
        x_shape = batch.x.shape
        _, *x_feat_shape = x_shape

        # Construct eigenvectors of shape [n, b1 * k]
        if 'batched_eigvec' not in batch:
            eigvec, eigvec_h = get_batched_eigenvectors(batch)
            batch.batched_eigvec, batch.batched_eigvec_h = eigvec, eigvec_h
        else:
            eigvec, eigvec_h = batch.batched_eigvec, batch.batched_eigvec_h

        x = self.feature_transform(batch)
        x = x.view(batch.num_nodes, -1)
        # Shapes: [b1 * k, n] @ [n, b2 * d] -> [b1 * k, b2 * d]
        x_hat = eigvec_h @ x
        x_hat = x_hat.view(batch.num_graphs, k, *x_feat_shape)

        # Shapes: either [b1, k, 1, d] x [b1, k, b2, d] -> [b1, k, b2, d]
        # or [b1, k, d] x [b1, k, d] -> [b1, k, d]
        y_hat = filter_ * x_hat

        if self.learnable_norm is not None:
            norm_factor = 0.5 * torch.tanh(self.learnable_norm) + 0.5
            bias = nn.functional.softplus(self.learnable_norm_bias)
            y_norm = torch.norm(y_hat, dim=1, keepdim=True)
            y_hat_normed = y_hat / (y_norm + bias + self.eps)

            if self.learnable_norm == 'feat_norm':
                x_norm = scatter(x * x.conj(), batch.batch, 0,
                                 batch.num_graphs, reduce='sum').sqrt()
                y_hat_normed = y_hat_normed * x_norm.unsqueeze(1)

            y_hat = (1 - norm_factor) * y_hat + norm_factor * y_hat_normed

        if self.silu_stack is not None:
            y_hat = self.silu_stack(y_hat)
        elif self.proj_out is not None:
            y_hat = self.proj_out(y_hat)

        if self.window is not None:
            window = self.window(batch)
            n_unsqueezes = y_hat.dim() - window.dim()
            y_hat = window[(...,) + (None,) * n_unsqueezes] * y_hat

        if self.spec_readout_residual:
            y_hat_sq = y_hat * y_hat.conj()
            if torch.is_complex(y_hat_sq):
                y_hat_sq = y_hat_sq.real
            y_hat_sq = y_hat_sq.sum(1)
            # y_hat_sq = (filter_ * x_hat_sq).sum(-3)
            if 'residual_y_hat_sq' in batch:
                batch.residual_y_hat_sq.append(y_hat_sq)
            else:
                batch.residual_y_hat_sq = [y_hat_sq]

        # Shapes: [n, b1 * k] @ [b1 * k, b2 * d] -> [n, b2 * d]
        y = (eigvec @ y_hat.view(batch.num_graphs * k, -1)).view(
            x_shape[0], y_hat.shape[-1])

        if torch.is_complex(y):
            if (hasattr(self, 'factor_split') and
                    isinstance(self.factor_split, nn.Parameter)):
                y_split = torch.stack([y.real, y.imag], -1)
                y = torch.sum(self.factor_split * y_split, -1)
            else:
                y = (self.factor_real(batch.op_code.long())[:, None] * y.real +
                     self.factor_imag(batch.op_code.long())[:, None] * y.imag)

        y = self.dropout(y)

        if self.with_node_residual:
            y = batch.x + y  # Residual connection

        if self.overwrite_x:
            batch.x = y
            return batch
        else:
            return y


@register_head('s2gnn_graph')
class S2GNNGraphHead(nn.Module):
    """
    GNN prediction head for graph prediction tasks.
    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out, is_first, scaling_init=0.001):
        super().__init__()

        self.config_node_readout = cfg.dataset.tpu_graphs.config_node_readout

        self.spec_readout = None
        if cfg.posenc_MagLapPE.enable:
            self.spec_readout = cfg.gnn.spectral.readout
            self.spec_readout_residual = cfg.gnn.spectral.readout_residual

        self.filter = None
        if self.spec_readout is not None:
            self.filter = FilterEncoder(
                dim_in, cfg.gnn.spectral.filter_encoder,
                cfg.gnn.spectral.num_heads_filter_encoder, is_first)

            self.split_real = self.split_imag = None
            self.emb_real = self.emb_imag = None
            self.layer_real = self.layer_imag = None
            if cfg.posenc_MagLapPE.q > 0:
                if cfg.gnn.spectral.feature_transform == 'glu':
                    bn = 0
                    if '_' in cfg.gnn.spectral.feature_transform:
                        bn = cfg.gnn.spectral.feature_transform.split(
                            '_')
                        bn = float(bn[-1])
                    self.layer_real = GLULayer(
                        dim_in, dim_in, simplified=True, bottle_neck=bn)
                    self.layer_imag = GLULayer(
                        dim_in, dim_in, simplified=True, bottle_neck=bn)
                elif cfg.gnn.spectral.feature_transform == 'node_type':
                    self.n_opcode = 125 if cfg.dataset.tpu_graphs.custom else 120
                    self.emb_real = CheckpointedEmbedding(
                        self.n_opcode, dim_in, max_norm=True)
                    self.emb_imag = CheckpointedEmbedding(
                        self.n_opcode, dim_in, max_norm=True)
                elif cfg.gnn.spectral.feature_transform is not None:
                    self.split_real = nn.Parameter(
                        torch.ones(1, 1, dim_in), requires_grad=True)
                    self.split_imag = nn.Parameter(
                        torch.ones(1, 1, dim_in), requires_grad=True)
            elif cfg.gnn.spectral.feature_transform == 'glu':
                bn = 0
                if '_' in cfg.gnn.spectral.feature_transform:
                    bn = cfg.gnn.spectral.feature_transform.split('_')
                    bn = float(bn[-1])
                self.layer_real = GLULayer(
                    dim_in, dim_in, simplified=True, bottle_neck=bn)
            elif cfg.gnn.spectral.feature_transform == 'node_type':
                self.n_opcode = 125 if cfg.dataset.tpu_graphs.custom else 120
                self.emb_real = CheckpointedEmbedding(
                    self.n_opcode, dim_in, max_norm=True)

            self.window = None
            if cfg.gnn.spectral.window:
                self.window = Window(cfg.gnn.spectral.window,
                                     cfg.posenc_MagLapPE.which == 'BE')

        dim_agg = dim_in
        dim_spat = dim_in
        if self.config_node_readout:
            dim_agg += dim_in
            dim_spat = dim_agg

        if self.spec_readout is not None:
            dim_agg += dim_in
            if self.spec_readout_residual:
                n_spec = (cfg.gnn.layers_mp - len(cfg.gnn.spectral.layer_skip))
                dim_agg += n_spec * dim_in
            if cfg.gnn.layer_type == 'combined':
                dim_agg += dim_in

        self.dropout = nn.Dropout(cfg.gnn.dropout)
        self.layer_post_mp = MLPMultiBatch(
            dim_agg, dim_out, cfg.gnn.layers_post_mp,
            has_batchnorm=False, dropout=0., final_act=False)

        self.spec_norm = None
        if (cfg.gnn.spectral.readout_sepnorm and cfg.posenc_MagLapPE.enable
                and (cfg.gnn.batchnorm_post_mp or cfg.gnn.layernorm_post_mp)):
            if cfg.gnn.batchnorm_post_mp:
                self.spec_norm = LastDimBatchNorm1d(
                    dim_agg - dim_spat, eps=cfg.bn.eps, momentum=cfg.bn.mom)
            elif cfg.gnn.layernorm_post_mp:
                self.spec_norm = LayerNorm([dim_agg - dim_spat])
            dim_agg = dim_spat

        self.norm = None
        if cfg.gnn.batchnorm_post_mp:
            self.norm = LastDimBatchNorm1d(
                dim_agg, eps=cfg.bn.eps, momentum=cfg.bn.mom)
        elif cfg.gnn.layernorm_post_mp:
            self.norm = LayerNorm([dim_agg])
        else:
            self.scaling_bias = nn.Parameter(torch.Tensor([scaling_init]),
                                             requires_grad=True)

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def _sum_over_nodes(self, batch, x, subset_idx=None):
        if subset_idx is not None:
            x = x[subset_idx]

        if batch.num_graphs > 1:
            batch_idx = batch.batch
            num_nodes = batch.num_nodes
            if subset_idx is not None:
                batch_idx = batch_idx[subset_idx]
                num_nodes = batch_idx.shape[0]

            x_shape = x.shape
            flat_x = x.view(x_shape[0], -1)
            agg_edge_index = torch.stack(
                (batch_idx, torch.arange(num_nodes, device=batch.x.device)))
            agg_edge_values = torch.ones_like(agg_edge_index[0],
                                              dtype=torch.float32)
            agg = torch.sparse_coo_tensor(agg_edge_index, agg_edge_values,
                                          (batch.num_graphs, num_nodes))

            graph_emb = agg @ flat_x
            graph_emb = graph_emb.view(agg.shape[0], *x_shape[1:])
        else:   # The above code led to issues for batch size 1 in the backward
            graph_emb = x.sum(0, keepdims=True)
        return graph_emb

    def _spectral_filter(self, batch):
        filter_ = self.filter(batch)

        if self.window is not None:
            window = self.window(batch)
            n_unsqueezes = filter_.dim() - window.dim()
            filter_ = window[(...,) + (None,) * n_unsqueezes] * filter_

        x_shape = batch.x.shape
        _, *x_feat_shape = x_shape
        # Construct eigenvectors of shape [n, b1 * k]
        if 'batched_eigvec' not in batch:
            eigvec, eigvec_h = get_batched_eigenvectors(batch)
            batch.batched_eigvec, batch.batched_eigvec_h = eigvec, eigvec_h
        else:
            eigvec, eigvec_h = batch.batched_eigvec, batch.batched_eigvec_h

        x = batch.x
        if torch.is_complex(eigvec):
            if self.layer_real is not None:
                x = self.layer_real(x) + 1j * self.layer_imag(x)
            elif self.emb_real is not None:
                weight_real = self.emb_real(batch.op_code.long())[:, None]
                weight_imag = self.emb_imag(batch.op_code.long())[:, None]
                x = weight_real * x + 1j * weight_imag * x
            elif self.split_real is not None:
                x = self.split_real * x + 1j * self.split_imag * x
            else:
                x = x.to(torch.complex64)
        else:
            if self.layer_real is not None:
                x = self.layer_real(x)
            elif self.emb_real is not None:
                weight_real = self.emb_real(batch.op_code.long())[:, None]
                x = weight_real * x
        flat_x = x.view(batch.num_nodes, -1)
        # Shapes: either [b1 * k, n] @ [n, b2 * d] -> [b1 * k, b2 * d]
        # or [b1 * k, n] @ [n, d] -> [b1 * k, d]
        x_hat = eigvec_h @ flat_x
        # Shapes: [b1, k, ...]
        x_hat = x_hat.view(batch.num_graphs, -1, *x_feat_shape)

        y_hat = filter_ * x_hat

        y_hat_sq = y_hat * y_hat.conj()
        if torch.is_complex(y_hat_sq):
            y_hat_sq = y_hat_sq.real

        # Shapes_ [b1, k, ...] -> [b1, ...]
        y_hat_sq = y_hat_sq.sum(1)

        return y_hat_sq

    def forward(self, batch, eps=1e-9):
        graph_emb = self._sum_over_nodes(batch, batch.x)

        if self.config_node_readout:
            assert 'config_feats_full' in batch
            node_conf_emb = self._sum_over_nodes(
                batch, batch.x, batch.config_idx)
            graph_emb = torch.concatenate((graph_emb, node_conf_emb), axis=-1)

        if self.norm is None:
            graph_emb = self.scaling_bias * graph_emb

        if self.filter is not None:
            y_hat_sq = self._spectral_filter(batch)

            if self.spec_readout_residual:
                y_hat_sq = torch.concatenate(
                    (*batch.residual_y_hat_sq, y_hat_sq), axis=-1)

            # Reverse quadratic scaling
            y_hat = (((y_hat_sq).relu() + eps).sqrt()
                     - ((-y_hat_sq).relu() + eps).sqrt())

            if self.spec_norm is not None:
                y_hat = self.spec_norm(y_hat)
                if self.norm is not None:
                    graph_emb = self.norm(graph_emb)
                else:
                    graph_emb = self.scaling_bias * graph_emb

            graph_emb = torch.concatenate((graph_emb, y_hat), dim=-1)

        graph_emb = self.dropout(graph_emb)

        if self.spec_norm is None:
            if self.norm is not None:
                graph_emb = self.norm(graph_emb)
            else:
                graph_emb = self.scaling_bias * graph_emb

        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


def get_batched_eigenvectors(batch):
    if batch.num_graphs == 1:
        return (batch.laplacian_eigenvector_plain,
                batch.laplacian_eigenvector_plain.transpose(0, 1).conj())
    # Construct eigenvectors of shape [n, b1 * k] with block sparse structure
    # for the b1 contained graphs
    k = batch.laplacian_eigenvalue_plain.shape[-1]
    node_idx = torch.arange(batch.num_nodes, device=batch.x.device)
    node_idx = node_idx[:, None].broadcast_to((batch.num_nodes, k))
    dim_idx = torch.arange(k, device=batch.x.device)
    dim_idx = dim_idx[None, :] + (k * batch.batch[:, None])
    eigvec = torch.sparse_coo_tensor(
        torch.stack((node_idx.flatten(), dim_idx.flatten())),
        batch.laplacian_eigenvector_plain.flatten(),
        (batch.num_nodes, k * batch.num_graphs))
    eigvec_h = eigvec.transpose(0, 1).conj()
    return eigvec, eigvec_h


class SpecFeatureTransformLayer(nn.Module):
    """Transforms features before applying spectral layer."""

    def __init__(self, dim):
        super().__init__()
        self.split_real = self.split_imag = None
        self.emb_real = self.emb_imag = None
        self.layer_real = self.layer_imag = None
        if (cfg.gnn.spectral.feature_transform is not None
                and cfg.posenc_MagLapPE.q > 0):
            if cfg.gnn.spectral.feature_transform.startswith('glu'):
                bn = 0
                if '_' in cfg.gnn.spectral.feature_transform:
                    bn = float(
                        cfg.gnn.spectral.feature_transform.split('_')[-1])
                self.layer_real = GLULayer(
                    dim, dim, simplified=True, bottle_neck=bn)
                self.layer_imag = GLULayer(
                    dim, dim, simplified=True, bottle_neck=bn)
            elif cfg.gnn.spectral.feature_transform == 'node_type':
                self.n_opcode = 125 if cfg.dataset.tpu_graphs.custom else 120  # TPUGraphs
                self.emb_real = CheckpointedEmbedding(
                    self.n_opcode, dim, max_norm=True)
                self.emb_imag = CheckpointedEmbedding(
                    self.n_opcode, dim, max_norm=True)
            elif cfg.gnn.spectral.feature_transform is not None:
                self.split_real = nn.Parameter(
                    torch.ones(1, 1, dim), requires_grad=True)
                self.split_imag = nn.Parameter(
                    torch.ones(1, 1, dim), requires_grad=True)
        elif cfg.gnn.spectral.feature_transform is not None:
            if cfg.gnn.spectral.feature_transform.startswith('glu'):
                bn = 0
                if '_' in cfg.gnn.spectral.feature_transform:
                    bn = float(
                        cfg.gnn.spectral.feature_transform.split('_')[-1])
                self.layer_real = GLULayer(
                    dim, dim, simplified=True, bottle_neck=bn)
            elif cfg.gnn.spectral.feature_transform == 'node_type':
                self.n_opcode = 125 if cfg.dataset.tpu_graphs.custom else 120  # TPUGraphs
                self.emb_real = CheckpointedEmbedding(
                    self.n_opcode, dim, max_norm=True)

    def forward(self, batch):
        x = batch.x
        if cfg.posenc_MagLapPE.q > 0:
            if self.layer_real is not None:
                x = self.layer_real(x) + 1j * self.layer_imag(x)
            elif self.emb_real is not None:
                weight_real = self.emb_real(batch.op_code.long())[:, None]
                weight_imag = self.emb_imag(batch.op_code.long())[:, None]
                x = weight_real * x + 1j * weight_imag * x
            elif self.split_real is not None:
                x = self.split_real * x + 1j * self.split_imag * x
            else:
                x = x.to(torch.complex64)
        else:
            if self.layer_real is not None:
                x = self.layer_real(x)
            elif self.emb_real is not None:
                weight_real = self.emb_real(batch.op_code.long())[:, None]
                x = weight_real * x
        return x


class SiLU(nn.Module):
    """SiLU activation that applies sigmoid to the absolute value of input"""

    def __init__(self, spec_dim=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spec_dim = spec_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(input, dim=self.spec_dim, keepdim=True)
        return input * torch.sigmoid(norm)


class SiLUMixAndBias(nn.Module):
    """SiLU activation that applies sigmoid to the absolute value of input"""

    def __init__(self, dim, spec_dim=1, bias=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spec_dim = spec_dim
        self.lin = nn.Linear(dim, dim, bias)

    def forward(self, input: torch.Tensor, eps=1e-9) -> torch.Tensor:
        norm = torch.linalg.norm(input, dim=self.spec_dim, keepdim=True)
        return input * torch.sigmoid(self.lin(norm))


class MaybeComplexLinear(nn.Linear):
    """Normal linear on real or identical linear mapping on imag if complex."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input) -> torch.Tensor:
        if torch.is_complex(input):
            out = (super().forward(input.real)
                   + 1j * super().forward(input.imag))
        else:
            out = super().forward(input)
        return out


class GLULayer(nn.Module):
    """Like Llama"""

    def __init__(self, dim_in, dim_out, p_drop=0., bottle_neck=0.,
                 simplified=False, checkpoint=False):
        super().__init__()
        self.bottle_neck = bottle_neck
        if self.bottle_neck > 0:
            dim_bottle = int(bottle_neck * dim_in)
            self.lin0 = nn.Linear(dim_in, dim_bottle, bias=False)
            self.lin1 = nn.Linear(dim_bottle, dim_out)
        else:
            self.lin1 = nn.Linear(dim_in, dim_out)
        self.simplified = simplified
        if not simplified:
            self.lin2 = nn.Linear(dim_in, dim_out)
            self.lin3 = nn.Linear(dim_out, dim_out)
        self.dropout = None
        if p_drop > 0:
            self.dropout = nn.Dropout(p_drop)
        self.checkpoint = checkpoint

    def propagate(self, input):
        input_fact = self.lin0(input) if self.bottle_neck else input
        if self.simplified:
            return F.silu(self.lin1(input_fact)) * input
        return self.lin3(F.silu(self.lin1(input_fact)) * self.lin2(input))

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            x = input
        else:
            x = input.x

        if self.dropout is not None:
            x = self.dropout(x)

        if self.checkpoint:
            x = checkpoint(self.propagate, x)
        else:
            x = self.propagate(x)

        if isinstance(input, torch.Tensor):
            out = x
        else:
            out = input
            out.x = x

        return out


class CheckpointedEmbedding(nn.Module):
    """Embedding with additional checkpointing. Does not support `padding idx´"""

    def __init__(self, num_embeddings: int, embedding_dim: int, *args,
                 device=None, dtype=None, **kwargs) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()
        self.args = args
        self.kwargs = kwargs

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, input):
        embedding = partial(nn.functional.embedding, **self.kwargs)
        out = checkpoint(embedding, input, self.weight, *self.args)
        return out


class LastDimBatchNorm1d(nn.BatchNorm1d):
    """Apply batch norm to last dimension."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        input_shape = input.shape
        input = input.reshape(-1, input_shape[-1])
        output = super().forward(input)
        output = output.reshape(*input_shape[:-1], output.shape[-1])
        return output


class LayerNorm(nn.LayerNorm):
    """LayerNorm either applied to tensor input or `x` attribute of input."""

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = super().forward(batch)
        else:
            batch.x = super().forward(batch.x)
        return batch


class MLPMultiBatch(GeneralMultiLayer):
    """MLP either applied to tensor input or `x` attribute of input."""

    def __init__(self, dim_in, dim_out, num_layers,
                 has_act=True, has_bias=True, name='linear', **kwargs):
        layer_config = new_layer_config(
            dim_in, dim_out, num_layers, has_act=has_act,
            has_bias=has_bias, cfg=cfg)
        for key, value in kwargs.items():
            setattr(layer_config, key, value)
        super().__init__(name, layer_config)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            x_shape = batch.shape
            batch = batch.reshape(-1, x_shape[-1])
            batch = super().forward(batch)
            batch = batch.reshape(*x_shape[:-1], batch.shape[-1])
        else:
            x_shape = batch.x.shape
            batch.x = batch.x.reshape(-1, x_shape[-1])
            batch = super().forward(batch)
            batch.x = batch.x.reshape(*x_shape[:-1], batch.x.shape[-1])
        return batch
