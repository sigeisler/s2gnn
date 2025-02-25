import glob
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Union

import torch

from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg

MODEL_STATE = 'model_state'
OPTIMIZER_STATE = 'optimizer_state'
SCHEDULER_STATE = 'scheduler_state'
DATA_STATE = 'data_state'


def load_ckpt(
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    return_data: bool = False,
    epoch: int = -1,
) -> int:
    r"""Loads the model checkpoint at a given epoch."""
    epoch = get_ckpt_epoch(epoch)
    path = get_ckpt_path(epoch)

    if not osp.exists(path):
        return 0

    ckpt = torch.load(path)
    if model is not None and MODEL_STATE is ckpt:
        model.load_state_dict(ckpt[MODEL_STATE])
    if optimizer is not None and OPTIMIZER_STATE in ckpt:
        optimizer.load_state_dict(ckpt[OPTIMIZER_STATE])
    if scheduler is not None and SCHEDULER_STATE in ckpt:
        scheduler.load_state_dict(ckpt[SCHEDULER_STATE])
    if return_data:
        data = ckpt[DATA_STATE] if DATA_STATE in ckpt else None
        return epoch + 1, data

    return epoch + 1


def save_ckpt(
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    data: Optional[Dict[str, Batch]] = None,
):
    r"""Saves the model checkpoint at a given epoch."""
    ckpt: Dict[str, Any] = {}
    if model is not None:
        ckpt[MODEL_STATE] = model.state_dict()
    if optimizer is not None:
        ckpt[OPTIMIZER_STATE] = optimizer.state_dict()
    if scheduler is not None:
        ckpt[SCHEDULER_STATE] = scheduler.state_dict()
    if data is not None:
        ckpt[DATA_STATE] = {key: data_split.to_dict()
                            for key, data_split in data.items()}

    os.makedirs(get_ckpt_dir(), exist_ok=True)
    torch.save(ckpt, get_ckpt_path(get_ckpt_epoch(epoch)))


def remove_ckpt(epoch: int = -1):
    r"""Removes the model checkpoint at a given epoch."""
    os.remove(get_ckpt_path(get_ckpt_epoch(epoch)))


def clean_ckpt():
    r"""Removes all but the last model checkpoint."""
    for epoch in get_ckpt_epochs()[:-1]:
        os.remove(get_ckpt_path(epoch))


###############################################################################


def get_ckpt_dir() -> str:
    return osp.join(cfg.run_dir, 'ckpt')


def get_ckpt_path(epoch: Union[int, str]) -> str:
    return osp.join(get_ckpt_dir(), f'{epoch}.ckpt')


def get_ckpt_epochs() -> List[int]:
    paths = glob.glob(get_ckpt_path('*'))
    return sorted([int(osp.basename(path).split('.')[0]) for path in paths])


def get_ckpt_epoch(epoch: int) -> int:
    if epoch < 0:
        epochs = get_ckpt_epochs()
        epoch = epochs[epoch] if len(epochs) > 0 else 0
    return epoch
