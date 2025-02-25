import os.path as osp

import networkx as nx
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm


class CustomClusterDataset(InMemoryDataset):
    def __init__(self,
                 n_graphs=(10000, 1000, 1000),
                 n_clusters=6,
                 size_min=100,
                 size_max=200,
                 graph_type="gmm",
                 random_p=0.025,
                 random_q=0.001,
                 gmm_dim=2,
                 gmm_range_clusters=10,
                 gmm_std_clusters=2,
                 gmm_edges_min=1,
                 gmm_edges_max=10,
                 gmm_cluster_from_posterior=True,
                 gmm_include_pos=False,
                 pre_transform=None,
                 seed=242,
                 root='datasets'):
        """
        Params:
            type: 'sbm' or 'gmm'
        """
        self.original_root = root
        dataset_name =  f'custom-cluster_' + \
                        f'{graph_type}_' + \
                        f'{n_clusters}_' + \
                        f'{size_min}to{size_max}_'
        if graph_type == 'sbm':
            dataset_name += f'p{int(random_p * 100)}_q{int(random_q * 100)}'
        elif graph_type == 'gmm':
            dataset_name += f'dim{gmm_dim}_' + \
                            f'rc{gmm_range_clusters}_' + \
                            f'stdc{gmm_std_clusters}_' + \
                            f'e{gmm_edges_min}to{gmm_edges_max}_'
            if gmm_cluster_from_posterior:
                dataset_name += 'post'
            else:
                dataset_name += 'sample'
        self.folder = osp.join(root, 'custom-cluster', dataset_name)
        self.n_graphs = dict(zip(['train', 'valid', 'test'], n_graphs))
        self.n_clusters = n_clusters
        self.size_min, self.size_max = size_min, size_max
        self.graph_type = graph_type
        self.random_p, self.random_q = random_p, random_q
        self.gmm_dim = gmm_dim
        self.gmm_range_clusters = gmm_range_clusters
        self.gmm_std_clusters = gmm_std_clusters
        self.gmm_edges_min, self.gmm_edges_max = gmm_edges_min, gmm_edges_max
        self.gmm_cluster_from_posterior = gmm_cluster_from_posterior
        self.gmm_include_pos = gmm_include_pos
        self.random = np.random.RandomState(seed)

        super().__init__(self.folder, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        data_list = []
        for split, n_graphs in self.n_graphs.items():
            print(f'Generating {n_graphs} random graphs for {split} split...')
            for _ in tqdm(range(n_graphs)):
                data = self.generate_graph()
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

    def generate_graph(self):
        if self.graph_type == 'sbm':
            W, c = self.unbalanced_block_model()
            W, c, _ = self.shuffle(W, c)
        elif self.graph_type == 'gmm':
            while True:
                W, c, pos = self.gmm_model()
                if nx.is_connected(nx.from_numpy_array(W)):
                    break
                W, c, pos, _ = self.shuffle(W, c, pos)
        else:
            raise ValueError(f"Type {self.graph_type} not supported!")
        u = self.create_signal(c)
        edge_index = dense_to_sparse(torch.tensor(W))[0]
        x = (one_hot(torch.tensor(u), num_classes=self.n_clusters+1)
             .type(torch.float32))
        data = Data(edge_index=edge_index,
                    x=x,
                    y=torch.tensor(c).type(torch.int64))
        if self.graph_type == 'gmm' and self.gmm_include_pos:
            data.pos = pos
        return data

    def gmm_model(self):
        cluster_centers = self.random.uniform(
            low=-self.gmm_range_clusters,
            high=self.gmm_range_clusters,
            size=(self.n_clusters, self.gmm_dim))
        cluster_sizes = self.random.randint(
            low=self.size_min, high=self.size_max, size=self.n_clusters)
        c = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
        offsets = self.random.normal(
            scale=self.gmm_std_clusters, size=(len(c), self.gmm_dim))
        nodes = cluster_centers[c, :] + offsets
        dist = np.linalg.norm(nodes[None, :, :] - nodes[:, None, :], axis=-1)
        W = np.zeros_like(dist)
        sorted_indices = np.argsort(dist, axis=-1)
        rows, _ = np.indices(dist.shape)
        for row in range(W.shape[0]):
            k = self.random.randint(
                low=self.gmm_edges_min, high=self.gmm_edges_max)
            W[rows[row, :k], sorted_indices[row, 1:k+1]] = 1
        W = (W + W.T != 0).astype(int)
        if self.gmm_cluster_from_posterior:
            dist_cluster_centers = np.linalg.norm(
                nodes[:, None, :] - cluster_centers[None, :, :], axis=-1)
            c = np.argmin(dist_cluster_centers, axis=-1)
        return W, c, nodes
    
    def unbalanced_block_model(self):  
        c = []
        for r in range(self.n_clusters):
            if self.size_max == self.size_min:
                clust_size_r = self.size_max
            else:
                clust_size_r = self.random.randint(
                    self.size_min, self.size_max, size=1)[0]
            val_r = np.repeat(r, clust_size_r, axis=0)
            c.append(val_r)
        c = np.concatenate(c)
        W = self.block_model(c)  
        return W, c

    def block_model(self, c):
        n = len(c)
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if c[i] == c[j]:
                    prob = self.random_p
                else:
                    prob= self.random_q
                if self.random.binomial(1, prob) == 1:
                    W[i, j] = 1
                    W[j, i] = 1
        return W

    def create_signal(self, c):
        u = np.zeros(c.shape[0], dtype=int)
        for r in range(self.n_clusters):
            cluster = np.where(c == r)[0]
            s = cluster[self.random.randint(cluster.shape[0])]
            u[s] = r + 1
        return u

    
    def shuffle(self, W, c, pos=None):
        idx = self.random.permutation(W.shape[0])
        W_new = W[idx, :]
        W_new = W_new[:, idx]
        c_new = c[idx]
        if pos is None:
            return W_new, c_new, idx
        else:
            pos_new = pos[idx]
            return W_new, c_new, pos_new, idx
   

if __name__ == '__main__':
    dataset = CustomClusterDataset()