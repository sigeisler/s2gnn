import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.graphgym.models.layer import MLP, new_layer_config


@register_node_encoder('MagLapPE')
class MagLapPENodeEncoder(torch.nn.Module):

    def __init__(self, dim_in: int, expand_x: bool = False):
        super().__init__()

        assert not expand_x

        if cfg.posenc_MagLapPE.dim_pe:
            dim_in = cfg.posenc_MagLapPE.dim_pe

        posenc_dim = cfg.posenc_MagLapPE.max_freqs
        if cfg.posenc_MagLapPE.q > 0:
            posenc_dim *= 2

        self.posenc_lin = nn.Linear(posenc_dim, dim_in)

    def forward(self, batch):
        # Convenience for hyperparam search
        if not hasattr(batch, 'laplacian_eigenvector_plain_posenc'):
            return batch
        posenc = batch.laplacian_eigenvector_plain_posenc
        if torch.is_complex(posenc):
            posenc = torch.concatenate((posenc.real, posenc.imag), dim=-1)

        posenc = + self.posenc_lin(posenc)

        if cfg.posenc_MagLapPE.dim_pe:
            # Concatenate final PEs to input embedding
            batch.x = torch.cat((batch.x, posenc), -1)
        else:
            batch.x = batch.x + posenc

        return batch
