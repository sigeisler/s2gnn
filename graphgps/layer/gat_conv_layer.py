from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor, set_diag, matmul
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

class GATConv(MessagePassing):
    """PyG implementation of GATConvDGL."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 1,
        feat_dropout: int = 0.0,
        attn_dropout: int = 0.0,
        negative_slope=0.2,
        norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._norm = norm
        
        self.fc = nn.Linear(
            self._in_channels, out_channels * num_heads, bias=False)
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_channels)))
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_channels)))
        self.feat_dropout = nn.Dropout(feat_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, x: Tensor, edge_index: Tensor, size=None):

        num_nodes = x.shape[0]
        adj = set_diag(
            SparseTensor.from_edge_index(edge_index,
                                         sparse_sizes=(num_nodes, num_nodes)))
        h_src = h_dst = self.feat_dropout(x)
        feat_src, feat_dst = h_src, h_dst
        feat_src = feat_dst = self.fc(h_src).view(
            -1, self._num_heads, self._out_channels)

        if self._norm:
            degs = adj.sum(0).clamp(min=1)  # out-degrees
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

        src, dst, _ = adj.coo()
        e = el[src] + er[dst]
        e = self.leaky_relu(e)
        alpha = softmax(e, dst)
        alpha = self.attn_dropout(alpha)

        rsts = []
        for i in range(self._num_heads):
            alpha_head_i = SparseTensor(
                row=src, col=dst, value=alpha[:, i, 0],
                sparse_sizes=(num_nodes, num_nodes)).t()
            rsts.append(self.propagate(
                adj.t(), x=feat_src[:, i, :], alpha=alpha_head_i, size=size))
        rst = torch.stack(rsts, dim=1)

        if self._norm:
            degs = adj.sum(1).clamp(min=1)  # in-degrees
            norm = torch.pow(degs, 0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        return rst

    def message_and_aggregate(
            self,
            adj_t: SparseTensor,
            x: Tensor,
            alpha: SparseTensor):
        return matmul(alpha, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self._in_channels,
                                             self._out_channels,
                                             self._num_heads)
