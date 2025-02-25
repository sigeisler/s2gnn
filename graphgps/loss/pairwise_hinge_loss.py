import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
from torch.nn import functional as F


@register_loss('pairwise_hinge')
def pairwise_hinge_loss_batch(pred, true, weight=None):
    # pred: (batch_size, num_preds)
    # true: (batch_size, num_preds)
    if cfg.model.loss_fun == 'pairwise_hinge':
        batch_size = pred.shape[0]
        num_preds = pred.shape[1]
        if num_preds >= 1024 and not pred.requires_grad:
            return torch.tensor(0, device=pred.device), pred
        i_idx = torch.arange(num_preds).repeat(num_preds)
        j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
        pairwise_true = (true[:, i_idx] > true[:, j_idx]).float()
        element_wise_loss = F.relu(0.1 - (pred[:, i_idx] - pred[:, j_idx])) * pairwise_true
        if weight is not None:
            if isinstance(weight, list):
                weight = torch.tensor(weight, device=pred.device)
            element_wise_loss = element_wise_loss * weight[:, None]
        # loss = torch.sum(torch.nn.functional.relu(0.1 - (pred[:,i_idx] - pred[:,j_idx])) * pairwise_true.float()) / batch_size
        loss = torch.sum(element_wise_loss) / pairwise_true.sum()
        return loss, pred
