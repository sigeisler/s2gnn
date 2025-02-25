import math

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F
from torch_sparse import coalesce

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import LayerConfig


class ChebNetIILayer(nn.Module):

    def __init__(self, layer_config: LayerConfig, K=10, **kwargs):
        super().__init__()
        dim = layer_config.dim_in

        self.linear = nn.Linear(dim, dim, bias=layer_config.has_bias)
        self.act = nn.Sequential(
            register.act_dict[layer_config.act](),
            nn.Dropout(layer_config.dropout),
        )
        self.model = ChebnetII_prop(K)

    def forward(self, batch):
        batch.x = self.model(self.linear(batch.x), batch.edge_index)
        batch.x = self.act(batch.x)

        return batch


def cheby(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for ii in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


class ChebnetII_prop(MessagePassing):
    def __init__(self, K, exact_norm=False, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.exact_norm = exact_norm
        self.temp = nn.Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

        # for j in range(self.K + 1):
        #     x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
        #     self.temp.data[j] = x_j**2

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * \
                cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        num_nodes = x.size(self.node_dim)

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization='sym',
            dtype=x.dtype, num_nodes=num_nodes)

        if self.exact_norm:
            with torch.no_grad():
                adj = torch.sparse_coo_tensor(
                    edge_index1, norm1, (num_nodes, num_nodes))
                max_eigenvalue = torch.lobpcg(adj, k=1)[0]
            # Then the eigenvalue is always in the range [0, 2]
            norm1 = 2 / max_eigenvalue * norm1

        # L_tilde=2/lambda_ax * L - I
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0,
            num_nodes=num_nodes)
        edge_index_tilde, norm_tilde = coalesce(
            edge_index_tilde, norm_tilde, num_nodes, num_nodes)

        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x,
                              norm=norm_tilde, size=None)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1,
                                  norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(
            self.__class__.__name__, self.K, self.temp)
