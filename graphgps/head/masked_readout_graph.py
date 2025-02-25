from torch_geometric.graphgym.register import register_head

from graphgps.head.mlp_graph import MLPGraphHead


@register_head('masked_readout_graph')
class MaskedReadoutGraphHead(MLPGraphHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_fun = lambda x, *args, **kwargs: x

    def forward(self, batch):
        batch.x = batch.x[batch.mask]
        return super().forward(batch)
