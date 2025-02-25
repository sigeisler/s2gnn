import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('cross_entropy')
def cross_entropy(pred, true):
    if cfg.model.loss_fun == 'cross_entropy':
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
        true = true.float()
        return bce_loss(pred, true), torch.sigmoid(pred)
