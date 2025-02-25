import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_head


@register_head('transductive_node_dummy')
class GNNTransductiveNodeDummyHead(nn.Module):
    """
    GNN prediction head for transductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, *args, **kwargs):
        if cfg.dataset.task != 'node':
            raise ValueError("'transductive_node' head can only be used for"
                             f"tasks of type 'node', not {cfg.dataset.task}")
        super().__init__()

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        pred, label = self._apply_index(batch)
        mask = self._get_mask(batch)
        return pred[mask], label[mask]

    def _get_mask(self, batch):
        if batch.split == 'train' and hasattr(batch, 'train_pred_mask'):
            return getattr(batch, 'train_pred_mask')
        else:
            return getattr(batch, f'{batch.split}_mask')
