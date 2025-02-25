import math
import os.path as osp
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from graphgps.transform.precalc_eigvec import AddMagneticLaplacianEigenvectorPlain


# Start adapted from https://github.com/HazyResearch/safari/blob/02220c69d247e5473616cd053a443ad99fd2559b/src/dataloaders/synthetics.py
class Vocab:
    """Custom vocab."""

    def __init__(self, vocab_size: int, special_vocabs: Dict):
        # Special tokens hold copy_prefix and noop/pad token etc
        assert "copy_prefix" in special_vocabs
        self.special_vocabs = special_vocabs
        vocab = [v for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(
            list(set(vocab + list(self.special_vocabs.values()))))
        self.v2id = {v: i for i, v in enumerate(self.vocab)}
        self.vocab_size = len(vocab)

    def get_next_vocab(self, token: str):
        """Gets next token excluding special_vocabs."""
        id = (self.get_id(token) + 1) % self.vocab_size
        while self.get_vocab(id) in self.special_vocabs:
            id = (id + 1) % self.vocab_size
        return self.get_vocab(id)

    @property
    def copy_prefix(self):
        return self.special_vocabs["copy_prefix"]

    @property
    def noop(self):
        return self.special_vocabs["noop"]

    @property
    def special_tokens(self):
        return set(self.special_vocabs.values())

    def get_id(self, token: str):
        return self.v2id[token]

    def get_vocab(self, id: int):
        return self.vocab[id]

    def __len__(self):
        return len(self.vocab)


def generate_assoc_recall(
    vocab: Vocab,
    input_seq_len: int,
    num_keys: int,
    rng: np.random.Generator,
    allow_dot: bool = False,
):
    """Generate sequence where the input has a sequence of key value pairs
    and the copy prefix at the end, and then a key value pair is inserted
    after the copy prefix."""
    non_special_vocab_size = len(vocab.non_special_vocab)
    keys = vocab.non_special_vocab[non_special_vocab_size // 2:]
    values = vocab.non_special_vocab[:non_special_vocab_size // 2]
    keys_multi = [[key] for key in keys]
    for i in range(num_keys - 1):
        keys_multi = [key + [key2] for key in keys_multi for key2 in keys]
    kv_map = {
        tuple(k): rng.choice(values) for k in keys_multi
    }

    key_present = {}
    vocab_seq = []
    for _ in range(input_seq_len // (num_keys + 1)):
        k = list(kv_map.keys())[rng.choice(len(kv_map.keys()))]
        v = kv_map[k]
        vocab_seq += list(k) + [v]
        key_present[k] = True
        # vocab_seq.append(v)

    k = list(kv_map.keys())[rng.choice(len(kv_map.keys()))]
    if not allow_dot:
        while k not in key_present:
            k = list(key_present.keys())[rng.choice(len(key_present.keys()))]
    to_copy = [vocab.copy_prefix] + \
        list(k) + [kv_map[k] if k in key_present else vocab.noop]
    vocab_seq = vocab_seq + to_copy
    return vocab_seq
# End


class AssociativeRecallDataset(InMemoryDataset):
    def __init__(self,
                 n_graphs=(5_000, 500, 500),
                 train_n_nodes=(10, 20),
                 valid_n_nodes=(10, 20),
                 test_n_nodes=(20, 30),
                 num_vocab: int = 10,
                 num_keys: int = 1,
                 seed: int = 242,
                 precalc_eigdec_k: int = -1,
                 root='datasets'):
        self.original_root = root
        dataset_name = (
            f'associative-recall_{num_vocab}_{num_keys}_' +
            f'samples_{"_".join(map(str, n_graphs))}_' +
            f'nodes_{train_n_nodes[0]}to{train_n_nodes[1] - 1}_' +
            f'{valid_n_nodes[0]}to{valid_n_nodes[1] - 1}_' +
            f'{test_n_nodes[0]}to{test_n_nodes[1] - 1}')
        self.folder = osp.join(root, 'associative-recall', dataset_name)
        self.n_graphs = dict(zip(['train', 'valid', 'test'], n_graphs))
        self.train_n_nodes = train_n_nodes
        self.valid_n_nodes = valid_n_nodes
        self.test_n_nodes = test_n_nodes

        self.num_vocab = num_vocab
        self.num_keys = num_keys
        self.precalc_eigdec_k = precalc_eigdec_k

        self.special_vocabs = {
            "copy_prefix": num_vocab,
            "noop": num_vocab + 1
        }
        self.random = np.random.RandomState(seed)
        self.vocab = Vocab(self.num_vocab, self.special_vocabs)

        super().__init__(self.folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        data_list = []
        for split, n_graphs in self.n_graphs.items():
            print(f'Generating {n_graphs} graphs for {split} split...')
            for _ in tqdm(range(n_graphs)):
                n_nodes = self.random.choice(
                    range(*getattr(self, f'{split}_n_nodes')))
                data = self.generate_graph(n_nodes)
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

    def generate_graph(self, n_nodes):
        sequence = generate_assoc_recall(
            self.vocab, n_nodes, self.num_keys, self.random, allow_dot=False)

        y = torch.tensor([sequence[-1]])
        x = torch.tensor(sequence)
        x[-1] = self.special_vocabs['noop']
        x = torch.nn.functional.one_hot(x, num_classes=self.num_vocab + 2)
        x = x.float()

        edge_index = torch.vstack((
            torch.arange(len(sequence) - 1),
            torch.arange(1, len(sequence))))

        mask = torch.zeros_like(x[:, 0], dtype=bool)
        mask[-1] = 1  # Source node is the node of interest

        # Return a Torch Geometric Data object containing all graph information
        data = dict(x=x, edge_index=edge_index, mask=mask, y=y)

        # Return or add eigenvectors with approximate form
        if self.precalc_eigdec_k <= 0:
            return Data(**data)

        # To obtain sufficiently small q without many digits
        q = 1 / 10 ** math.ceil(math.log10(self.test_n_nodes[-1]))
        transform = AddMagneticLaplacianEigenvectorPlain(
            k=self.precalc_eigdec_k, q=q, positional_encoding=True)
        eigvec_config = transform._repr_dict()

        num_nodes = len(sequence)
        node_idx = torch.arange(num_nodes)[:, None]
        node_idx = node_idx.to(torch.float64)
        freq_idx = torch.arange(self.precalc_eigdec_k)[None, :]
        freq_idx = freq_idx.to(torch.float64)

        eig_vecs = torch.cos((node_idx + 1 / 2) * freq_idx * np.pi / num_nodes)

        deg_sqrt = torch.ones(eig_vecs.shape[0])
        deg_sqrt[1:-1] = np.sqrt(2)
        eig_vecs = deg_sqrt[:, None] * eig_vecs

        eig_vecs /= torch.linalg.norm(eig_vecs, axis=0, keepdims=True)
        eig_vecs = eig_vecs * torch.exp(-1j * np.pi * 2 * q * node_idx)
        eig_vals = (2 * (1 - torch.cos(np.pi * freq_idx / num_nodes)))
        eigenvalue_max = (
            2 * (1 - np.cos(np.pi * (num_nodes - 1) / num_nodes)))
        eig_vals /= 1 / 2 * eigenvalue_max

        laplacian_config = (
            'AddMagneticLaplacianEigenvectorPlain(' +
            ', '.join([f'{k}={v}' for k, v in eigvec_config.items()]) +
            ')')

        edge_index_und = to_undirected(edge_index, reduce='mean')
        posenc = transform._calc_positional_encoding(
            edge_index_und, eig_vals[0], eig_vecs,
            num_nodes, sigma=np.clip(eig_vals[0, 1], a_min=None, a_max=1e-7))

        eig_vals = eig_vals.to(torch.float32)
        eig_vecs = eig_vecs.to(torch.complex64)
        posenc = posenc.to(torch.complex64)

        data['laplacian_eigenvector_plain'] = eig_vecs
        data['laplacian_eigenvalue_plain'] = eig_vals
        data['laplacian_eigenvalue_plain_mask'] = torch.ones_like(eig_vals)
        data['laplacian_eigenvector_plain_posenc'] = posenc
        data['laplacian_config'] = laplacian_config

        return Data(**data)


if __name__ == '__main__':
    dataset = AssociativeRecallDataset()
