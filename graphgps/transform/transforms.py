import logging

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data


def clip_last_node(data, degree):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    degree_last_node = (data.edge_index[0] == N - 1).sum()

    if degree_last_node < degree:
        return data

    if hasattr(data, 'edge_attr'):
        edge_attr = data.edge_attr
    else:
        edge_attr = None
    edge_index, edge_attr = subgraph(list(range(N - 1)),
                                     data.edge_index, edge_attr)
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x[:N - 1]
        data.num_nodes = N - 1
    else:
        data.num_nodes = N - 1
    if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
        data.node_is_attributed = data.node_is_attributed[:N - 1]
        data.node_dfs_order = data.node_dfs_order[:N - 1]
        data.node_depth = data.node_depth[:N - 1]
    if hasattr(data, 'op_code'):
        data.op_code = data.op_code[:N - 1]
    if hasattr(data, 'op_feats'):
        data.op_feats = data.op_feats[:N - 1]
    data.edge_index = edge_index
    if hasattr(data, 'edge_attr'):
        data.edge_attr = edge_attr
    return data


def clip_high_deg_sinks_and_sources(data, degree, drop_sinks, drop_sources):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.

    node_in, deg_in = torch.unique(data.edge_index[0, :], return_counts=True)
    node_out, deg_out = torch.unique(data.edge_index[1, :], return_counts=True)

    deg_in_full = torch.zeros(N, dtype=int)
    deg_in_full[node_in] = deg_in

    deg_out_full = torch.zeros(N, dtype=int)
    deg_out_full[node_out] = deg_out

    mask = ~((drop_sinks & (deg_in_full > degree) & (deg_out_full == 0))
             | (drop_sources & (deg_out_full > degree) & (deg_in_full == 0)))

    data.edge_attr = torch.tensor(data.edge_attr)

    if not mask.all():
        return data

    if hasattr(data, 'edge_attr'):
        edge_attr = data.edge_attr
    else:
        edge_attr = None
    edge_index, edge_attr = subgraph(mask, data.edge_index, edge_attr,
                                     relabel_nodes=True)
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x[mask]
    data.num_nodes = mask.sum()
    if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
        data.node_is_attributed = data.node_is_attributed[mask]
        data.node_dfs_order = data.node_dfs_order[mask]
        data.node_depth = data.node_depth[mask]
    if hasattr(data, 'op_code'):
        data.op_code = data.op_code[mask]
    if hasattr(data, 'op_feats'):
        data.op_feats = data.op_feats[mask]
    data.edge_index = edge_index
    if hasattr(data, 'edge_attr'):
        data.edge_attr = edge_attr
    if hasattr(data, ''):
        new_idx = mask.astype(int).cumsum() - 1
        data.config_idx = new_idx[data.config_idx]
    return data
