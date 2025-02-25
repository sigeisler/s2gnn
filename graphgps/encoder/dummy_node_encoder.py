import torch
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('DummyNode')
class DummyNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.encoder = torch.nn.Embedding(num_embeddings=1,
                                          embedding_dim=emb_dim)

    def forward(self, batch):
        device = next(self.encoder.parameters()).device
        dummy_x = torch.zeros(batch.num_nodes, dtype=int, device=device)
        batch.x = self.encoder(dummy_x)
        return batch
