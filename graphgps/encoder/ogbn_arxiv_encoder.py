import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder

from graphgps.encoder.linear_node_encoder import LinearNodeEncoder


@register_node_encoder('OGBNArxivNode')
class OGBNArxivNodeEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_rate = cfg.dataset.ogbn_arxiv.mask_rate
        self.use_labels = cfg.dataset.ogbn_arxiv.use_labels

    def forward(self, batch):
        device = batch.train_mask.device
        if self.mask_rate is not None and self.use_labels:
            if self.training:
                mask = torch.rand(
                    batch.train_mask.shape, device=device) < self.mask_rate
                train_labels_mask = batch.train_mask * mask
                train_pred_mask = batch.train_mask * ~mask
            else:
                train_pred_mask = train_labels_mask = batch.train_mask
            batch.x = self._add_labels(batch.x, batch.y, train_labels_mask)
        elif self.training and self.mask_rate is not None:
            mask = torch.rand(
                batch.train_mask.shape, device=device) < self.mask_rate
            train_pred_mask = batch.train_mask * mask
        else:
            train_pred_mask = batch.train_mask
        batch.train_pred_mask = train_pred_mask
        return batch

    def _add_labels(self, x, y, idx):
        onehot = torch.zeros([x.shape[0], cfg.share.dim_out]).to(x.device)
        onehot[idx, y[idx, 0]] = 1
        return torch.cat([x, onehot], dim=-1)


@register_node_encoder('OGBNArxivLinearNode')
class OGBNArxivLinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__()
        self.ogbn_arxiv_encoder = OGBNArxivNodeEncoder()
        self.linear_node_encoder = LinearNodeEncoder(emb_dim)

    def forward(self, batch):
        batch = self.ogbn_arxiv_encoder(batch)
        return self.linear_node_encoder(batch)
