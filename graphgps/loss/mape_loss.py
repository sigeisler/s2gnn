import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
from torch.nn import functional as F


@register_loss('mape')
def mape_loss_batch(pred, true, weight=None, epsilon=1e-5,
                    true_divisor=20_000_000_000):
    if cfg.model.loss_fun == 'mape':
        # pred: (batch_size, num_preds)
        # true: (batch_size, num_preds)
        true = true / true_divisor
        abs_diff = torch.abs(pred - true)
        abs_per_error = abs_diff / torch.clamp(torch.abs(true), min=epsilon)

        loss = torch.mean(abs_per_error)

        return loss, pred
