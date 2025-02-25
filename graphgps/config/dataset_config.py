from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # Config for TPUGraphs
    cfg.dataset.tpu_graphs = CN()
    # layout or tile
    cfg.dataset.tpu_graphs.tpu_task = 'layout'
    # nlp or xla
    cfg.dataset.tpu_graphs.source = ['nlp']
    # random or default
    cfg.dataset.tpu_graphs.search = ['random']
    # -1 or int > 0
    cfg.dataset.tpu_graphs.subsample = 500
    # Factor to apply to op embedding and configuration features
    cfg.dataset.tpu_graphs.encoder_factor = 100.
    # TPUGraph: drop last node (high degree "aggregation" node)
    cfg.dataset.tpu_graphs.drop_last_node_above_deg = -1
    # Filter high deg sinks/sources of min degree `drop_last_node_above_deg`
    cfg.dataset.tpu_graphs.drop_high_deg_sinks = False
    cfg.dataset.tpu_graphs.drop_high_deg_sources = False
    # If true the custom dataformat is used
    cfg.dataset.tpu_graphs.custom = False
    # Normalize features
    cfg.dataset.tpu_graphs.normalize = False
    # Additionally only readout the config nodes
    cfg.dataset.tpu_graphs.config_node_readout = False
    # If true, the validation graphs are also used for training
    cfg.dataset.tpu_graphs.include_valid_in_train = False

    # Config for synthetic source distance prediction task
    cfg.dataset.source_dist = CN()
    cfg.dataset.source_dist.n_graphs = (50000, 2500, 2500)
    cfg.dataset.source_dist.train_n_nodes = (500, 1000)
    cfg.dataset.source_dist.valid_n_nodes = (1000, 1100)
    cfg.dataset.source_dist.test_n_nodes = (1100, 1200)
    cfg.dataset.source_dist.p_add_edges_from_tree = 0

    # Config for custom cluster task
    cfg.dataset.custom_cluster = CN()
    cfg.dataset.custom_cluster.n_graphs = (10000, 1000, 1000)
    cfg.dataset.custom_cluster.n_clusters = 6
    cfg.dataset.custom_cluster.size_min = 5
    cfg.dataset.custom_cluster.size_max = 35
    cfg.dataset.custom_cluster.graph_type = "gmm"
    cfg.dataset.custom_cluster.random_p = 0.55
    cfg.dataset.custom_cluster.random_q = 0.25
    cfg.dataset.custom_cluster.gmm_dim = 2
    cfg.dataset.custom_cluster.gmm_range_clusters = 10
    cfg.dataset.custom_cluster.gmm_std_clusters = 2
    cfg.dataset.custom_cluster.gmm_edges_min = 1
    cfg.dataset.custom_cluster.gmm_edges_max = 10
    cfg.dataset.custom_cluster.gmm_cluster_from_posterior = True

    # Config for oversquashing task
    cfg.dataset.over_squashing = CN()
    cfg.dataset.over_squashing.gen_mode = 'full'
    cfg.dataset.over_squashing.n_graphs = (5_000, 500, 5_000)
    cfg.dataset.over_squashing.n_classes = 5
    cfg.dataset.over_squashing.topology = 'ring_lollipop'
    cfg.dataset.over_squashing.train_n_nodes = (4, 50)
    cfg.dataset.over_squashing.valid_n_nodes = (4, 50)
    cfg.dataset.over_squashing.test_n_nodes = (52, 100)

    # Config for associative recall task
    cfg.dataset.associative_recall = CN()
    cfg.dataset.associative_recall.n_graphs = (25_000, 500, 500)
    cfg.dataset.associative_recall.num_vocab = 30
    cfg.dataset.associative_recall.num_keys = 1
    cfg.dataset.associative_recall.train_n_nodes = (20, 1_000)
    cfg.dataset.associative_recall.valid_n_nodes = (20, 1_000)
    cfg.dataset.associative_recall.test_n_nodes = (1_000, 1_200)
    cfg.dataset.associative_recall.precalc_eigdec_k = 10

    # Config for ogbn-arxiv task
    cfg.dataset.ogbn_arxiv = CN()
    cfg.dataset.ogbn_arxiv.mask_rate = 0.5
    cfg.dataset.ogbn_arxiv.use_labels = True

    # Config for arxiv-year task
    cfg.dataset.arxiv_year = CN()
    cfg.dataset.arxiv_year.num_split = 0
    cfg.dataset.arxiv_year.with_ogbn_arxiv_labels = False
