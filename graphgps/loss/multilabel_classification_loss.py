import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('multilabel_cross_entropy')
def multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss.
    """
    if cfg.dataset.task_type == 'classification_multilabel':
        if cfg.model.loss_fun != 'cross_entropy':
            raise ValueError("Only 'cross_entropy' loss_fun supported with "
                             "'classification_multilabel' task_type.")
        bce_loss = nn.BCEWithLogitsLoss()
        is_labeled = true == true  # Filter our nans.
        return bce_loss(pred[is_labeled], true[is_labeled].float()), pred
    

@register_loss('cross_entropy_ogbn-arxiv')
def cross_entropy_ogbn_arxiv(pred, true):
    if cfg.model.loss_fun == 'cross_entropy_ogbn-arxiv':
        epsilon = 1 - math.log(2)
        pred = F.log_softmax(pred, dim=-1)
        y = F.cross_entropy(pred, true, reduction='none')
        y = torch.log(epsilon + y) - math.log(epsilon)
        return torch.mean(y), pred
