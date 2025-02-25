import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head


@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out, *args, **kwargs):
        super(GNNInductiveNodeHead, self).__init__()
        dropout = cfg.gnn.dropout
        L = cfg.gnn.layers_post_mp

        layers = []
        for _ in range(L-1):
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
        return pred, label
