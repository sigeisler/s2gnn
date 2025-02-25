import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('list_mle')
def list_mle(y_pred, y_true, eps=1e-8):
    """
    Slightly modified from https://github.com/allegro/allRank/blob/c88475661cb72db292d13283fdbc4f2ae6498ee4/allrank/models/losses/listMLE.py#L7

    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :return: loss value, a torch.Tensor
    """
    if cfg.model.loss_fun == 'list_mle':
        # pred: (batch_size, num_preds)
        # true: (batch_size, num_preds)
        # shuffle for randomised tie resolution (same order for batch)
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]),
                            dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + eps) + max_pred_values

        observation_loss = observation_loss - preds_sorted_by_true

        loss = torch.mean(observation_loss)

        return loss, y_pred
