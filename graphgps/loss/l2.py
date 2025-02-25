import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('l2_losses')
def l1_losses(pred, true):
    if cfg.model.loss_fun == 'l2':
        mse_loss = nn.MSELoss()
        loss = mse_loss(pred, true.type(pred.dtype))
        return loss, pred
