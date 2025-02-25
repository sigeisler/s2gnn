from torch_geometric.graphgym.register import register_config

from graphgps.config.posenc_config import CN


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our S^2GNN network model.
    """
    cfg.gnn.layer_type = 'lin_gnn'
    cfg.gnn.make_undirected = True
    cfg.gnn.use_edge_attr = False
    cfg.gnn.adj_norm = 'dir'
    cfg.gnn.dir_aggr = 'cat'

    # Use residual connections between the spatial GNN layers.
    cfg.gnn.residual = True

    cfg.gnn.batchnorm_post_mp = False
    cfg.gnn.layernorm_post_mp = False

    # Next two options apply to encoded features (before the first GNN layer)
    cfg.gnn.node_dropout = 0.

    # Allows skipping a message passing layer (e.g., only apply spectral conv)
    cfg.gnn.layer_skip = []

    # GAT specific configurations to reproduce results of prior work
    cfg.gnn.gatconv = CN()
    cfg.gnn.gatconv.pre_dropout = 0.1
    cfg.gnn.gatconv.num_heads = 3
    cfg.gnn.gatconv.negative_slope = 0.2
    cfg.gnn.gatconv.attn_dropout = 0.05
    cfg.gnn.gatconv.feat_dropout = 0.75
    cfg.gnn.gatconv.norm = True
    cfg.gnn.gatconv.backend = 'PyG'
    cfg.gnn.gatconv.with_linear = True

    cfg.gnn.spectral = CN()
    
    # Allows skipping a spectral layer (e.g., only apply spatial conv)
    cfg.gnn.spectral.layer_skip = [-1]

    # Max eigenvalue until which the spectral filter is constructed
    cfg.gnn.spectral.frequency_cutoff = None

    # The windowing function to smoothen filter at frequency cutoff
    cfg.gnn.spectral.window = None

    # The parametrization of the spectral filter
    cfg.gnn.spectral.filter_encoder = 'basis'  # either basis, attn, mlp
    # If > 0 construct only `num_heads_filter_encoder` spectral filters
    cfg.gnn.spectral.num_heads_filter_encoder = -1  # Number of spectral filters
    cfg.gnn.spectral.mlp_layers_filter_encoder = 2  # For filter_encoder=='mlp'
    cfg.gnn.spectral.basis_bottleneck = 0.25  # For filter_encoder=='basis'
    cfg.gnn.spectral.basis_num_gaussians = 50  # For filter_encoder=='basis'
    cfg.gnn.spectral.basis_init_type = 'default'  # For filter_encoder=='basis'

    # Non-linearities etc in spectral domain (see "Neural Network for the Spectral Domain")
    cfg.gnn.spectral.filter_variant = 'naive'
    cfg.gnn.spectral.filter_layers = 1
    cfg.gnn.spectral.filter_value_trans = None  # or acos, nacos

    # If positive, use different dropout rate for spectral conv
    cfg.gnn.spectral.dropout = -1.

    # Spectral readout for graph level tasks (ses "Computational Remarks")
    cfg.gnn.spectral.readout = None
    cfg.gnn.spectral.readout_sepnorm = False
    cfg.gnn.spectral.readout_residual = False

    # Regular feature transformations
    cfg.gnn.spectral.feature_transform = None
    cfg.gnn.spectral.real_imag_x_merge = None

    # Use residual connections around spectral conv
    cfg.gnn.spectral.residual = True

    # Learnable norm see ("Spectral Normalization")
    cfg.gnn.spectral.learnable_norm = False
    cfg.gnn.spectral.learnable_norm_init = 0
    
    cfg.gnn.spectral.eigv_scale = -1  # To mimic feature scaling of TPUGraphs

    # Options to merge with message passing, e.g., to implement layers of form `spectral() + spatial()``
    # Mostly used for ablation/variation studies
    cfg.gnn.spectral.combine_with_spatial = None
    cfg.gnn.spectral.combine_with_spatial_norm = True
