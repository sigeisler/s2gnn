from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
import torch
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN

from graphgps.layer.s2_spectral import FeatureBatchSpectralLayer


def balanced_binary_tree(n: int):
    n_orig = n
    if n % 2 == 0:
        n += 1
    A = np.zeros((n, n), dtype=np.float64)
    receivers = np.arange(1, n)
    senders = np.arange(n // 2).repeat(2)
    A[senders, receivers] = 1
    if n_orig % 2 == 0:
        A = A[:-1, :-1]
    return A


def magnetic_laplacian(A, q, symmetric=False):
    A_s = (A + A.T)
    A_s[A_s > 1] = 1
    D_s = np.diag(A_s.sum(axis=0))
    inv_D_s = np.diag(1 / np.sqrt(A_s.sum(axis=0)))
    if q != 0:
        rotation = np.exp(2j * np.pi * q * (A - A.T))
    else:
        rotation = np.ones_like(A)

    if symmetric:
        L = np.eye(A.shape[0]) - (inv_D_s.dot(A_s).dot(inv_D_s)) * rotation
    else:
        L = D_s - A_s * rotation
    return L


@dataclass
class Batch:
    x: torch.Tensor
    laplacian_eigenvalue_plain: torch.Tensor
    laplacian_eigenvector_plain: torch.Tensor
    num_nodes: int
    laplacian_eigenvalue_plain_mask: Optional[torch.Tensor] = None
    num_graphs: int = 1

    def __contains__(self, item):
        return item in ['x', 'laplacian_eigenvalue_plain', 'num_nodes',
                        'laplacian_eigenvector_plain', 'num_graphs']


class TestFeatureBatchSpectralLayer():

    @torch.no_grad()
    @pytest.mark.parametrize('filter_variant', ['naive', 'silu'])
    @pytest.mark.parametrize('factor_imag', [1., 10.])
    def test_phase_shift_invariance(self, filter_variant, factor_imag):
        n = 100
        b1 = 1
        k = 9
        d = 36

        cfg.gnn = CN()
        cfg.gnn.layers_mp = 1
        cfg.gnn.spectral.dropout = -1.
        cfg.posenc_MagLapPE = CN()
        cfg.gnn.spectral.filter_encoder = 'attn'
        cfg.gnn.spectral.readout_residual = True
        cfg.gnn.spectral.filter_variant = filter_variant
        cfg.gnn.spectral.eigv_scale = 10
        cfg.gnn.spectral.basis_init_type = None
        cfg.gnn.spectral.window = 'tukey'
        cfg.posenc_MagLapPE.which = 'SA'
        cfg.posenc_MagLapPE.q = 1e-5
        cfg.gnn.spectral.num_heads_filter_encoder = -1
        cfg.gnn.spectral.real_imag_x_merge = None
        cfg.gnn.spectral.feature_transform = None
        cfg.gnn.spectral.learnable_norm = False
        cfg.gnn.spectral.frequency_cutoff = None
        cfg.gnn.spectral.filter_value_trans = None

        layer_config = LayerConfig(dim_in=d,  dim_out=d)
        layer = FeatureBatchSpectralLayer(layer_config, is_first=True)
        layer.factor_split[:, 1] = factor_imag

        x = np.random.normal(size=(n, d))

        eigval = (np.arange(k) / k)[None, :]

        amp = np.random.uniform(low=0, high=1, size=(n, k))
        angle = np.random.uniform(low=0, high=2 * np.pi, size=(n, k))
        eigvec = amp * np.exp(1j * angle)

        batch = Batch(x=torch.from_numpy(x).to(torch.float32),
                      num_nodes=torch.tensor(n),
                      num_graphs=torch.tensor(b1),
                      laplacian_eigenvalue_plain=torch.from_numpy(
                          eigval).to(torch.float32),
                      laplacian_eigenvector_plain=torch.from_numpy(
                          eigvec).to(torch.complex64))
        batch.laplacian_eigenvalue_plain_mask = torch.ones_like(
            batch.laplacian_eigenvalue_plain, dtype=bool)

        output = layer(batch)

        for _ in range(5):
            rotations = np.random.uniform(size=(1, k), low=0, high=2 * np.pi)
            angle = angle + rotations
            eigvec = amp * np.exp(1j * angle)
            batch = deepcopy(batch)
            batch.laplacian_eigenvector_plain = torch.from_numpy(
                eigvec).to(torch.complex64)
            batch.x = torch.from_numpy(x).to(torch.float32)

            rotated_output = layer(batch)

            assert torch.allclose(output.x, rotated_output.x,
                                  rtol=5e-02, atol=1e-04)

            assert torch.allclose(output.residual_y_hat_sq[0],
                                  rotated_output.residual_y_hat_sq[0],
                                  rtol=1e-03, atol=1e-05)

    @pytest.mark.parametrize('q', [0, 2 * np.pi])
    @pytest.mark.parametrize('filter_variant', ['naive', 'silu'])
    @torch.no_grad()
    def test_binary_tree(self, q, filter_variant):
        b1 = 1
        b2 = 2
        k = 9
        d = 36
        
        ns = [63, 127]
        q = q / max(ns)

        cfg.gnn = CN()
        cfg.gnn.layers_mp = 1
        cfg.gnn.spectral.dropout = -1.
        cfg.posenc_MagLapPE = CN()
        cfg.gnn.spectral.filter_encoder = 'attn'
        cfg.gnn.spectral.readout_residual = True
        cfg.gnn.spectral.filter_variant = filter_variant
        cfg.gnn.spectral.basis_init_type = None
        cfg.gnn.spectral.eigv_scale = 10
        cfg.gnn.spectral.window = 'tukey'
        cfg.posenc_MagLapPE.which = 'SA'
        cfg.posenc_MagLapPE.q = q
        cfg.gnn.spectral.num_heads_filter_encoder = -1
        cfg.gnn.spectral.real_imag_x_merge = None
        cfg.gnn.spectral.feature_transform = None
        cfg.gnn.spectral.learnable_norm = False
        cfg.gnn.spectral.frequency_cutoff = None
        cfg.gnn.spectral.filter_value_trans = None

        layer_config = LayerConfig(dim_in=d,  dim_out=d)
        layer = FeatureBatchSpectralLayer(layer_config, is_first=True)

        eigvec_dtype = torch.float32 if q == 0 else torch.complex64

        for n in ns:
            x = np.random.normal(size=(n, b2, d))

            for k in [15, 25]:
                A = balanced_binary_tree(n)
                L = magnetic_laplacian(A, q)
                eigenvalues, eigenvectors = np.linalg.eigh(L)

                while np.allclose(eigenvalues[k - 1], eigenvalues[k]) and k < n:
                    k += 1
                print(n, k)

                eigenvalues = eigenvalues[:k][None, :]
                eigenvectors = eigenvectors[:, :k]

                batch = Batch(
                    x=torch.from_numpy(x).to(torch.float32),
                    num_nodes=torch.tensor(n),
                    num_graphs=torch.tensor(b1),
                    laplacian_eigenvalue_plain=torch.from_numpy(
                        eigenvalues).to(torch.float32),
                    laplacian_eigenvector_plain=torch.from_numpy(
                        eigenvectors).to(eigvec_dtype))
                batch.laplacian_eigenvalue_plain_mask = torch.ones_like(
                    batch.laplacian_eigenvalue_plain, dtype=bool)

                output = layer(batch)

                P = np.random.permutation(np.arange(n))
                A = np.array([A[v, u] for v in P for u in P]).reshape(n, n)

                L = magnetic_laplacian(A, q)
                eigenvalues_p, eigenvectors_p = np.linalg.eigh(L)

                eigenvalues_p = eigenvalues_p[:k][None, :]
                eigenvectors_p = eigenvectors_p[:, :k]

                batch = deepcopy(batch)
                batch.laplacian_eigenvalue_plain = torch.from_numpy(
                    eigenvalues_p).to(torch.float32)
                batch.laplacian_eigenvector_plain = torch.from_numpy(
                    eigenvectors_p).to(eigvec_dtype)
                batch.x = torch.from_numpy(x[P]).to(torch.float32)

                output_p = layer(batch)
                output_p_x = output_p.x[np.argsort(P)]

                assert torch.allclose(
                    output.x, output_p_x, rtol=1e-03, atol=1e-05)

                assert torch.allclose(
                    output.residual_y_hat_sq[0], output_p.residual_y_hat_sq[0],
                    rtol=1e-03, atol=1e-05)

if __name__ == '__main__':
    TestFeatureBatchSpectralLayer().test_phase_shift_invariance('naive', 2.)
    TestFeatureBatchSpectralLayer().test_binary_tree(2 * np.pi, 'naive')
    print('Tests w/o silu successful')
    TestFeatureBatchSpectralLayer().test_phase_shift_invariance('silu', 2.)
    TestFeatureBatchSpectralLayer().test_binary_tree(2 * np.pi, 'silu')
    print('Tests w/ silu successful')
    TestFeatureBatchSpectralLayer().test_phase_shift_invariance('silu_mix', 2.)
    TestFeatureBatchSpectralLayer().test_binary_tree(2 * np.pi, 'silu_mix')
    print('Tests w/ silu_mix successful')
