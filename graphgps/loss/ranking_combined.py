from torch_geometric.graphgym.register import register_loss

from torch_geometric.graphgym.config import cfg
from graphgps.loss.pairwise_hinge_loss import pairwise_hinge_loss_batch
from graphgps.loss.list_mle import list_mle


@register_loss('ranking_combined')
def ranking_combined(pred, true, divisor=30.):
    if cfg.model.loss_fun == 'ranking_combined':
        list_mle_loss, pred = list_mle(pred, true)
        list_mle_loss = list_mle_loss / cfg.model.list_mle_divisor
        pairwise_hinge_loss, _ = pairwise_hinge_loss_batch(pred, true)
        return list_mle_loss + pairwise_hinge_loss, pred
