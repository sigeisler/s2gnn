from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False
    cfg.train.ckpt_data_splits = []
    cfg.train.ckpt_data_attrs = ['y', 'pred', 'batch']

    cfg.train.mode = 'standard'
    cfg.device = 'cuda:0'

    # Train and val config for sampling configurations in tpu graphs
    cfg.train.num_sample_configs = 16
    cfg.train.scale_num_sample_configs = True
    cfg.val.num_sample_configs = 1_000
    cfg.val.num_sample_batch = 100
    cfg.model.list_mle_divisor = 250
