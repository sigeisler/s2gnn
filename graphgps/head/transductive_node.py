import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head


@register_head('transductive_node')
class GNNTransductiveNodeHead(nn.Module):
    """
    GNN prediction head for transductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out, *args, **kwargs):
        if cfg.dataset.task != 'node':
            raise ValueError("'transductive_node' head can only be used for"
                             f"tasks of type 'node', not {cfg.dataset.task}")
        super(GNNTransductiveNodeHead, self).__init__()
        dropout = cfg.gnn.dropout
        L = cfg.gnn.layers_post_mp

        if L == 0:
            self.mlp = nn.Identity()
            return

        layers = []
        for _ in range(L - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_in, dim_in, bias=True))
            layers.append(register.act_dict[cfg.gnn.act]())

        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim_in, dim_out, bias=True))
        self.mlp = nn.Sequential(*layers)

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch.x = self.mlp(batch.x)
        pred, label = self._apply_index(batch)
        mask = self._get_mask(batch)
        return pred[mask], label[mask]

    def _get_mask(self, batch):
        if batch.split == 'train' and hasattr(batch, 'train_pred_mask'):
            return getattr(batch, 'train_pred_mask')
        else:
            return getattr(batch, f'{batch.split}_mask')
