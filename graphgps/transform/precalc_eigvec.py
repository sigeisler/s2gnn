from typing import Optional, Tuple

import numpy as np
import scipy
import scipy.sparse as sp

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    get_laplacian,
    scatter,
    is_undirected,
    to_scipy_sparse_matrix,
    to_undirected
)


@functional_transform('add_magnetic_laplacian_eigenvector_plain')
class AddMagneticLaplacianEigenvectorPlain(BaseTransform):
    r"""Adds the (naive) Magnetic Laplacian eigenvector positional encoding from `"Transformers Meet Directed Graphs" <https://arxiv.org/abs/2302.00049>` to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number eigenvectors to consider.
        attr_eigvec_name (str, optional): The attribute name of the data object
            to add positional encodings to. (default: 
            :obj:`"laplacian_eigenvector_plain"`)
        attr_eigval_name (str, optional): The attribute name of the data object
            to add eigenvalues to. (default: 
            :obj:`"laplacian_eigenvalue_plain"`)
        attr_repr_name (str, optional): The attribute name of the data object
            to add the configuration to (`repr` of this class). Can be used,
            e.g., for storing eigenvectors along with their config.
        positional_encoding (bool, optional): If set to :obj:`True`, the
            positional encoding from `"Spatio-Spectral Graph Neural Networks"
            <https://arxiv.org/abs/2405.19121>` is added to data object with 
            attribute name :obj:`"<attr_eigvec_name>_posenc"`.
        normalization (str, optional): Normalization of Laplacian. See
            :meth:`torch_geometric.utils.get_laplacian`.
        q (float, optional): The magnetic "potential". (default: :obj:`1e-4`)
        dtype (torch.dtype, optional): The desired data type of returned tensor
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        which (str, optional): Which eigenvalues to compute. See 
            :meth:`scipy.sparse.linalg.eigsh`.
        sparse (bool, optional): If set to :obj:`True`, use 
            :meth:`scipy.sparse.linalg.eigsh` and otherwise 
            :meth:`scipy.linalg.eigh`. (default: :obj:`True`) 
        n_failover (int, optional): Number of failovers for eigendecomposition.
        **kwargs (optional): Additional arguments for 
            :meth:`scipy.sparse.linalg.eigsh` etc.
    """

    def __init__(
        self,
        k: int,
        attr_eigvec_name: Optional[str] = 'laplacian_eigenvector_plain',
        attr_eigval_name: Optional[str] = 'laplacian_eigenvalue_plain',
        attr_repr_name: Optional[str] = 'laplacian_config',
        positional_encoding: bool = False,
        normalization: str = 'sym',
        q: float = 1e-4,
        drop_trailing_repeated: bool = False,
        scc: bool = False,
        which: str = 'LM',
        sparse: bool = True,
        dtype=torch.float32,
        n_failover: int = 10,
        **kwargs,
    ):
        self.k = k
        self.attr_eigvec_name = attr_eigvec_name
        self.attr_eigval_name = attr_eigval_name
        self.attr_repr_name = attr_repr_name
        self.normalization = normalization if normalization != 'none' else None
        self.which = which
        self.q = q
        self.drop_trailing_repeated = drop_trailing_repeated
        self.scc = scc
        self.positional_encoding = positional_encoding
        self.sparse = sparse
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype = torch.float32
            elif dtype == 'float64':
                dtype = torch.float64
        self.dtype = dtype
        self.n_failover = n_failover
        self.kwargs = kwargs

    def _calc_magnetic_laplacian(
            self, data: Data, edge_index_und: torch.Tensor,
            edge_weight_und: torch.Tensor):
        num_nodes = data.num_nodes

        normalization = self.normalization
        if normalization == 'rw' and self.q == 0 and self.sparse:
            # Calculate eigenvectors for (symmetric) I - D^{-1/2} A D^{-1/2}
            # and subsequently then rescale eigenvectors by degree
            normalization = 'sym'
        lap_edge_index, lap_edge_weight = get_laplacian(
            edge_index_und,
            edge_weight_und,
            normalization=normalization,
            num_nodes=num_nodes,
            dtype=self.dtype
        )
        # Make sure that the laplacian edge weights are aligned with theta
        lap_edge_index, lap_edge_weight = coalesce(
            lap_edge_index, lap_edge_weight, data.num_nodes)

        if self.q != 0:
            edge_index = torch.concatenate(
                (data.edge_index, torch.flip(data.edge_index, [0,])), dim=-1)
            edge_weight = None
            if edge_weight is None:
                edge_weight = self.q * torch.ones_like(data.edge_index[0],
                                                       dtype=torch.float32)
            edge_weight = torch.concatenate(
                (edge_weight, -edge_weight), dim=-1)
            edge_index, edge_weight = coalesce(
                edge_index, edge_weight, num_nodes, reduce='add')
            theta = np.exp(2j * np.pi * edge_weight)
            edge_index, theta = add_self_loops(
                edge_index, theta, fill_value=1., num_nodes=num_nodes)
            # Make sure that theta is aligned with the laplacian edge weights
            edge_index, theta = coalesce(
                edge_index, theta, data.num_nodes)
            assert (edge_index == lap_edge_index).all()
            lap_edge_weight = lap_edge_weight * theta

        L = to_scipy_sparse_matrix(lap_edge_index, lap_edge_weight, num_nodes)
        return L

    def _calc_positional_encoding(
            self, edge_index_und: torch.Tensor, eig_vals: np.ndarray,
            eig_vecs: np.ndarray, num_nodes: int, sigma=1e-7) -> torch.Tensor:
        row, col = edge_index_und
        prod_edge = eig_vecs[row] * np.conjugate(eig_vecs[col])

        filter_ = scipy.special.softmax(
            -(eig_vals[None, :] - eig_vals[:, None]) ** 2 / sigma, axis=-1)

        single_frequency_convs = np.einsum('mc,cd->md', prod_edge, filter_)

        posenc = scatter(
            torch.tensor(single_frequency_convs), torch.tensor(row), dim=0,
            dim_size=num_nodes, reduce='sum')

        return posenc

    def _calc_eig(self, L: scipy.sparse.coo_matrix, which: str, k: int,
                  deg: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if self.sparse:
            if self.normalization == 'rw' and self.q != 0:
                eig = sp.linalg.eigs
            else:
                eig = sp.linalg.eigsh

            v0 = None
            if self.q == 0:
                v0 = np.sqrt(deg)
                v0 = v0 / np.linalg.norm(v0)

            eig_vals, eig_vecs = eig(
                L,
                k=k,
                which=which,
                return_eigenvectors=True,
                v0=v0,
                **kwargs,
            )

            if self.normalization == 'rw' and self.q == 0:
                eig_vecs /= np.sqrt(deg)[:, None]
                eig_vecs /= np.linalg.norm(eig_vecs, axis=0, keepdims=True)
        else:
            assert self.normalization != 'rw' or self.q == 0
            assert which == 'SM' or which == 'SA'
            eig_vals, eig_vecs = scipy.linalg.eigh(L.toarray(), **kwargs)
            eig_vals = eig_vals[:k]
            eig_vecs = eig_vecs[:, :k]
        return eig_vals, eig_vecs

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes

        edge_index_und = data.edge_index
        edge_weight_und = torch.ones(data.num_edges, dtype=self.dtype) \
            if data.edge_weight is None else data.edge_weight.to(self.dtype)
        if not is_undirected(data.edge_index, data.edge_weight):
            edge_index_und, edge_weight_und = to_undirected(
                edge_index_und, edge_weight_und, reduce='mean')

        deg = scatter(edge_weight_und, edge_index_und[0], 0,
                      dim_size=num_nodes, reduce='sum')
        L = self._calc_magnetic_laplacian(
            data, edge_index_und, edge_weight_und)

        scc_subset = None
        if self.scc:
            num_components, component = sp.csgraph.connected_components(
                np.abs(L), connection='weak')

            if num_components > 1:
                _, count = np.unique(component, return_counts=True)
                scc_subset = np.in1d(component, count.argmax())
                print(f'Drop {num_nodes - scc_subset.sum()} nodes for '
                      'eigenvector calculation')
                L = L.tocsr()[:, scc_subset][scc_subset, :]
                deg = deg[scc_subset]

        k = self.k
        if self.drop_trailing_repeated:
            assert self.which == 'LM' or self.which == 'SA'
            k += 1
        mask = np.ones(self.k, dtype=bool)

        which = self.which
        kwargs = self.kwargs.copy()
        # Heuristic that worked better on TPU Graphs
        if (self.normalization is None
                and 'sigma' not in self.kwargs and self.which == 'SA'):
            which = 'LM'
            kwargs['sigma'] = 0

        if k < L.shape[0] - 1:
            eig_vals = None
            for _ in range(self.n_failover):  # Fail over for new random init
                try:
                    eig_vals, eig_vecs = self._calc_eig(
                        L, k=k, which=which, deg=deg.numpy(), **kwargs)
                except:  # noqa E722
                    print('Eigval calc failed')

                if eig_vals is not None:
                    break
            else:
                raise ValueError('Eigval calc failed repeatedly')
        else:
            eig_vals, eig_vecs = scipy.linalg.eigh(L.toarray())

        eig_vals_order = eig_vals.argsort()[:min(k, len(eig_vals))]
        eig_vals = eig_vals[eig_vals_order]
        eig_vecs = eig_vecs[:, eig_vals_order]

        if self.drop_trailing_repeated and k <= L.shape[0]:
            eig_val_bounds = torch.tensor([eig_vals.max()])
            print(f'Max eigenvalue is {eig_val_bounds.item()} '
                    f'for graph with {num_nodes} nodes')
            while (np.allclose(eig_vals[k - 2], eig_vals[k - 1])
                    and k < L.shape[0] and k > 2):
                k -= 1
            if k <= self.k:
                print(f'Drop {self.k - k + 1} trailing '
                        'repeated eigenvalues')
            eig_vals[k - 1:] = 0
            eig_vecs[:, k - 1:] = 0
            mask[k - 1:] = False

            eig_vals = eig_vals[:self.k]
            mask = mask[:self.k]
            eig_vecs = eig_vecs[:, :self.k]

        if scc_subset is not None:
            eig_vecs_complete = np.zeros((num_nodes, self.k),
                                         dtype=eig_vecs.dtype)
            eig_vecs_complete[scc_subset, :eig_vecs.shape[-1]] = eig_vecs
            eig_vecs = eig_vecs_complete

        if self.k > L.shape[0]:
            eig_vals_ = np.zeros(self.k, dtype=eig_vals.dtype)
            eig_vals_[:eig_vals.shape[0]] = eig_vals
            eig_vals = eig_vals_
            eig_vecs_ = np.zeros((num_nodes, self.k), dtype=eig_vecs.dtype)
            eig_vecs_[:, :eig_vecs.shape[1]] = eig_vecs
            eig_vecs = eig_vecs_
            mask_ = np.zeros(self.k, dtype=bool)
            mask_[:mask.shape[0]] = mask
            mask_[L.shape[0]:] = False
            mask = mask_

        if self.positional_encoding:
            posenc = self._calc_positional_encoding(
                edge_index_und, eig_vals, eig_vecs, data.num_nodes)
            posenc[:, ~mask] = 0.
            data[self.attr_eigvec_name + '_posenc'] = posenc

        eig_vals = torch.from_numpy(eig_vals)
        eig_vecs = torch.from_numpy(eig_vecs)

        data[self.attr_eigval_name] = eig_vals[None, :]
        data[self.attr_eigval_name + '_mask'] = torch.from_numpy(mask)
        data[self.attr_eigvec_name] = eig_vecs

        if self.drop_trailing_repeated and k < num_nodes:
            data[self.attr_eigval_name + '_bounds'] = eig_val_bounds
        else:
            # Set value well above the largest eigenvalue
            data[self.attr_eigval_name + '_bounds'] = (
                torch.tensor([eig_vals.max()]) + 1)

        data[self.attr_repr_name] = self.__repr__()

        return data

    def _repr_dict(self):
        return dict(attr_eigvec_name=self.attr_eigvec_name,
                    attr_eigval_name=self.attr_eigvec_name,
                    normalization=self.normalization,
                    which=self.which,
                    q=self.q,
                    drop_trailing_repeated=self.drop_trailing_repeated,
                    scc=self.scc,
                    positional_encoding=self.positional_encoding,
                    sparse=self.sparse)

    def __repr__(self):
        return ('AddMagneticLaplacianEigenvectorPlain(' +
                ', '.join([f'{k}={v}' for k, v in self._repr_dict().items()]) +
                ')')
