import os
from typing import Callable
import torch

import numpy as np
import torch_geometric.graphgym.register as register
import torch_geometric.transforms as T
from torch_geometric.data import download_url
from torch_geometric.datasets import (
    PPI,
    Amazon,
    Coauthor,
    KarateClub,
    MNISTSuperpixels,
    Planetoid,
    QM7b,
    TUDataset,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.transform import (
    create_link_label,
    neg_sampling_transform,
)
from torch_geometric.loader import (
    ClusterLoader,
    DataLoader,
    DynamicBatchSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    NeighborSampler,
    RandomNodeLoader,
)
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import (
    index_to_mask,
    negative_sampling,
    to_undirected,
)
from graphgps.transform.precalc_eigvec import AddMagneticLaplacianEigenvectorPlain
from graphgps.utils import get_mask, even_quantile_labels

index2mask = index_to_mask  # TODO Backward compatibility


def planetoid_dataset(name: str) -> Callable:
    return lambda root: Planetoid(root, name)


# register.register_dataset('Cora', planetoid_dataset('Cora'))
# register.register_dataset('CiteSeer', planetoid_dataset('CiteSeer'))
# register.register_dataset('PubMed', planetoid_dataset('PubMed'))
# register.register_dataset('PPI', PPI)


def load_pyg(name, dataset_dir):
    """
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        name (str): dataset name
        dataset_dir (str): data directory

    Returns: PyG dataset object

    """
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset = TUDataset(dataset_dir, name, transform=T.Constant())
        else:
            dataset = TUDataset(dataset_dir, name[3:])
    elif name == 'Karate':
        dataset = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset = Coauthor(dataset_dir, name='CS')
        else:
            dataset = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset = Amazon(dataset_dir, name='Computers')
        else:
            dataset = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset


def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


def load_ogb(name, dataset_dir):
    r"""

    Load OGB dataset objects.


    Args:
        name (str): dataset name
        dataset_dir (str): data directory

    Returns: PyG dataset object

    """
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset
    from ogb.nodeproppred import PygNodePropPredDataset

    if name[:4] == 'ogbn':

        # Optionally precompute eigendecomposition
        if cfg.posenc_MagLapPE.enable and cfg.posenc_MagLapPE.precompute:
            pre_transform = AddMagneticLaplacianEigenvectorPlain(
                cfg.posenc_MagLapPE.max_freqs,
                q=cfg.posenc_MagLapPE.q,
                drop_trailing_repeated=cfg.posenc_MagLapPE.drop_trailing_repeated,
                which=cfg.posenc_MagLapPE.which.upper(),
                normalization=cfg.posenc_MagLapPE.laplacian_norm,
                scc=cfg.posenc_MagLapPE.largest_connected_component,
                positional_encoding=cfg.posenc_MagLapPE.positional_encoding,
                sparse=cfg.posenc_MagLapPE.sparse,
                **cfg.posenc_MagLapPE.kwargs)
            print("Precomputing/loading precomputed MagLapPE...")
        else:
            pre_transform = None

        dataset = PygNodePropPredDataset(name=name, root=dataset_dir,
                                         pre_transform=pre_transform)
        split_names = ['train_mask', 'val_mask', 'test_mask']
        splits = dataset.get_idx_split()
        for i, key in enumerate(splits.keys()):
            mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_names[i], mask, len(mask))
        edge_index = to_undirected(dataset.data.edge_index)
        set_dataset_attr(dataset, 'edge_index', edge_index,
                         edge_index.shape[1])

    elif name[:4] == 'ogbg':
        dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for i, key in enumerate(splits.keys()):
            id = splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif name[:4] == "ogbl":
        dataset = PygLinkPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_edge_split()
        id = splits['train']['edge'].T
        if cfg.dataset.resample_negative:
            set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
            dataset.transform = neg_sampling_transform
        else:
            id_neg = negative_sampling(edge_index=id,
                                       num_nodes=dataset.data.num_nodes,
                                       num_neg_samples=id.shape[1])
            id_all = torch.cat([id, id_neg], dim=-1)
            label = create_link_label(id, id_neg)
            set_dataset_attr(dataset, 'train_edge_index', id_all,
                             id_all.shape[1])
            set_dataset_attr(dataset, 'train_edge_label', label, len(label))

        id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'val_edge_label', label, len(label))

        id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'test_edge_label', label, len(label))

    else:
        raise ValueError('OGB dataset: {} non-exist')
    return dataset


def load_arxiv_year(dataset_dir):

    from ogb.nodeproppred import PygNodePropPredDataset

    # Optionally precompute eigendecomposition
    if cfg.posenc_MagLapPE.enable and cfg.posenc_MagLapPE.precompute:
        pre_transform = AddMagneticLaplacianEigenvectorPlain(
            cfg.posenc_MagLapPE.max_freqs,
            q=cfg.posenc_MagLapPE.q,
            drop_trailing_repeated=cfg.posenc_MagLapPE.drop_trailing_repeated,
            which=cfg.posenc_MagLapPE.which.upper(),
            normalization=cfg.posenc_MagLapPE.laplacian_norm,
            scc=cfg.posenc_MagLapPE.largest_connected_component,
            positional_encoding=cfg.posenc_MagLapPE.positional_encoding,
            sparse=cfg.posenc_MagLapPE.sparse,
            **cfg.posenc_MagLapPE.kwargs)
        print("Precomputing/loading precomputed MagLapPE...")
    else:
        pre_transform = None

    # Load ogbn-arxiv dataset
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     root=dataset_dir,
                                     pre_transform=pre_transform)

    # Create new year label
    x, y = dataset.data.x, dataset.data.y
    num_nodes = dataset.data.num_nodes
    year = even_quantile_labels(
        dataset.data.node_year.flatten().numpy(), nclasses=5, verbose=False)
    set_dataset_attr(dataset, 'y', torch.as_tensor(year)[:, None], num_nodes)

    # Optionally append original one-hot class labels to features
    if cfg.dataset.arxiv_year.with_ogbn_arxiv_labels:
        num_classes = torch.unique(y).shape[0]
        onehot = torch.zeros([x.shape[0], num_classes]).to(x.device)
        idx = range(num_nodes)
        onehot[idx, y[idx, 0]] = 1
        x = torch.cat([x, onehot], dim=-1)
    set_dataset_attr(dataset, 'x', x, num_nodes)

    # Get dataset splits
    github_url = f"https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
    split_file_name = f"arxiv-year-splits.npy"
    local_dir = os.path.join(dataset_dir, 'ogbn_arxiv', 'raw')
    download_url(os.path.join(github_url, split_file_name),
                 local_dir, log=False)
    splits = np.load(os.path.join(local_dir, split_file_name),
                     allow_pickle=True)
    split_number = cfg.dataset.arxiv_year.num_split
    split_idx = splits[split_number % len(splits)]
    set_dataset_attr(dataset, 'train_mask',
                     get_mask(split_idx["train"], num_nodes), num_nodes)
    set_dataset_attr(dataset, 'val_mask',
                     get_mask(split_idx["valid"], num_nodes), num_nodes)
    set_dataset_attr(dataset, 'test_mask',
                     get_mask(split_idx["test"], num_nodes), num_nodes)

    return dataset


def load_dataset():
    r"""

    Load dataset objects.

    Returns: PyG dataset object

    """
    format = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        dataset = func(format, name, dataset_dir)
        if dataset is not None:
            return dataset
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    # Load from OGB formatted data
    elif format == 'OGB':
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(format))
    return dataset


def set_dataset_info(dataset):
    r"""
    Set global dataset information

    Args:
        dataset: PyG dataset object

    """

    # get dim_in and dim_out
    try:
        cfg.share.dim_in = dataset.data.x.shape[1]
    except Exception:
        cfg.share.dim_in = 1
    try:
        if cfg.dataset.task_type == 'classification':
            cfg.share.dim_out = torch.unique(dataset.data.y).shape[0]
        else:
            cfg.share.dim_out = dataset.data.y.shape[1]
    except Exception:
        cfg.share.dim_out = 1

    if (('arxiv' in cfg.dataset.name) and
        (cfg.dataset.ogbn_arxiv.mask_rate is not None) and
        cfg.dataset.ogbn_arxiv.use_labels):
        cfg.share.dim_in = cfg.share.dim_in + cfg.share.dim_out

    # count number of dataset splits
    cfg.share.num_splits = 1
    for key in dataset.data.keys:
        if 'val' in key:
            cfg.share.num_splits += 1
            break
    for key in dataset.data.keys:
        if 'test' in key:
            cfg.share.num_splits += 1
            break


def create_dataset():
    r"""
    Create dataset object

    Returns: PyG dataset object

    """
    dataset = load_dataset()
    set_dataset_info(dataset)

    return dataset


def get_loader(dataset, sampler, batch_size, shuffle=True):
    pw = cfg.num_workers > 0
    if sampler == "full_batch_dyn":
        sampler_ = DynamicBatchSampler(
            dataset, shuffle=shuffle, max_num=batch_size)
        loader_train = DataLoader(dataset, batch_sampler=sampler_,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True, persistent_workers=pw)
    elif sampler == "full_batch" or len(dataset) > 1:
        loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=cfg.num_workers,
                                  pin_memory=True, persistent_workers=pw)
    elif sampler == "neighbor":
        loader_train = NeighborSampler(
            dataset[0], sizes=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=True)
    elif sampler == "random_node":
        loader_train = RandomNodeLoader(dataset[0],
                                        num_parts=cfg.train.train_parts,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True, persistent_workers=pw)
    elif sampler == "saint_rw":
        loader_train = \
            GraphSAINTRandomWalkSampler(dataset[0],
                                        batch_size=batch_size,
                                        walk_length=cfg.train.walk_length,
                                        num_steps=cfg.train.iter_per_epoch,
                                        sample_coverage=0,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True,
                                        persistent_workers=pw)
    elif sampler == "saint_node":
        loader_train = \
            GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=pw)
    elif sampler == "saint_edge":
        loader_train = \
            GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=pw)
    elif sampler == "cluster":
        loader_train = \
            ClusterLoader(dataset[0],
                          num_parts=cfg.train.train_parts,
                          save_dir="{}/{}".format(cfg.dataset.dir,
                                                  cfg.dataset.name.replace(
                                                      "-", "_")),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers,
                          pin_memory=True,
                          persistent_workers=pw)

    else:
        raise NotImplementedError("%s sampler is not implemented!" % sampler)
    return loader_train


def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True)
        ]

    if cfg.dataset.name == 'TPUGraphs':
        loaders[-1].collate_fn = dataset.collate_fn_train

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))
        if cfg.dataset.name == 'TPUGraphs':
            loaders[-1].collate_fn = dataset.collate_fn_eval

    return loaders
