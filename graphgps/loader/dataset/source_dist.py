import os.path as osp

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm


class SourceDistanceDataset(InMemoryDataset):
    def __init__(self,
                 n_graphs=(50000, 2500, 2500),
                 train_n_nodes=(500, 1000),
                 valid_n_nodes=(500, 1000),
                 test_n_nodes=(500, 1000),
                 p_add_edges_from_tree=0,
                 pre_transform=None,
                 seed=242,
                 root='datasets'):
        self.original_root = root
        dataset_name =  f'source-dist_' + \
                        f'{train_n_nodes[0]}to{train_n_nodes[1] - 1}_' + \
                        f'{valid_n_nodes[0]}to{valid_n_nodes[1] - 1}_' + \
                        f'{test_n_nodes[0]}to{test_n_nodes[1] - 1}' + \
                        f'-{p_add_edges_from_tree}'
        self.folder = osp.join(root, 'source-dist', dataset_name)
        self.n_graphs = dict(zip(['train', 'valid', 'test'], n_graphs))
        self.train_n_nodes = train_n_nodes
        self.valid_n_nodes = valid_n_nodes
        self.test_n_nodes = test_n_nodes
        self.p_add_edges_from_tree = p_add_edges_from_tree
        self.random = np.random.RandomState(seed)

        super().__init__(self.folder, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        data_list = []
        for split, n_graphs in self.n_graphs.items():
            print(f'Generating {n_graphs} random DAGs for {split} split...')
            for _ in tqdm(range(n_graphs)):
                data = self.generate_graph(split)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        n_train, n_valid, n_test = self.n_graphs.values()
        return {
            'train': torch.arange(n_train),
            'valid': torch.arange(n_valid) + n_train,
            'test': torch.arange(n_test) + n_train + n_valid,
        }

    def _generate_nx_graph(self, split):
        n_nodes = self.random.choice(range(*getattr(self, f'{split}_n_nodes')))
        G = nx.random_tree(n_nodes, self.random)
        dist_sort = nx.single_source_shortest_path_length(G, 0)
        edges_dir = list(map(
            lambda x: x if dist_sort[x[0]] < dist_sort[x[1]] else x[::-1],
            G.edges(),
        ))
        G_dir = nx.DiGraph()
        G_dir.add_nodes_from(G.nodes())
        G_dir.add_edges_from(edges_dir)
        if self.p_add_edges_from_tree > 0:
            for _ in range(self.p_add_edges_from_tree * n_nodes // 100):
                i, j = self.random.choice(G_dir.nodes(), size=2)
                s, t = (i, j) if dist_sort[i] < dist_sort[j] else (j, i)
                G_dir.add_edge(s, t)
        adj = nx.adjacency_matrix(G_dir)
        dist = np.array(list(dict(sorted(
            nx.single_source_shortest_path_length(G_dir, 0).items())).values()))
        shuffle = self.random.permutation(n_nodes)
        return adj[:, shuffle][shuffle, :], dist[shuffle]

    def generate_graph(self, split):
        adj, dist = self._generate_nx_graph(split)
        dist = torch.tensor(dist)
        edge_index = from_scipy_sparse_matrix(adj)[0]
        x = (dist == 0).type(torch.float32)[:, None]
        return Data(edge_index=edge_index, x=x, y=dist, num_nodes=dist.shape[0])
   

if __name__ == '__main__':
    dataset = SourceDistanceDataset()
