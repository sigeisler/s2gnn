"""Concerns function g(lambda) of a spectrally parametrized convolution."""

import math
from typing import Optional

import torch
from torch import nn
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.graphgym.models.layer import new_layer_config
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP


def tukey_window(eigenvalues: torch.Tensor,
                 alpha=0.5, wiggle: float = 0.025,
                 eigenvalue_bounds: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
    r"""Return a one-sided Tukey window, also known as a tapered cosine window.

    Derived from the scipy implementation.

    Parameters
    ----------
    eigenvalues : torch.Tensor, optional
        To calculate the window values. Shape [b, k].
    alpha : float or torch.Tensor, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region. If tensor, must be compatible 
        to shape [b, k].
    wiggle : float, optional
        overestimate the window width to give last eigenvalue at least some
        contribution.
    eigenvalue_bounds : torch.Tensor, optional
        if provided, the bounds will be used instead of the `wiggle`.
        Shape [b, 1].

    Returns
    -------
    w : torch.Tensor
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Tukey_window

    """
    assert 0 < alpha and alpha < 1

    if eigenvalue_bounds is None:
        M = eigenvalues.max(-1, keepdims=True).values * (1 + wiggle)
    else:
        M = eigenvalue_bounds

    eigenvalues = eigenvalues / M

    window = torch.cos(torch.pi * (eigenvalues - alpha) / (2 - 2 * alpha))
    window = window.clamp_min(0)

    window[eigenvalues <= alpha] = 1

    return window


def exponential_window(eigenvalues: torch.Tensor,
                       tau=1. / math.log(10),
                       eigenvalue_bounds: Optional[torch.Tensor] = None,
                       **kwargs) -> torch.Tensor:
    r"""Return an exponential (or Poisson) window.

    Derived from the scipy implementation.

    Parameters
    ----------
    eigenvalues : torch.Tensor, optional
        To calculate the window values. Shape [b, k].
    tau : float, optional
        Parameter defining the decay.  Default yields window factor 0.1 at max.
    eigenvalue_bounds : torch.Tensor, optional
        if provided, the bounds will be used instead of the `wiggle`.
        Shape [b, 1].

    Returns
    -------
    w : torch.Tensor
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The (positive, centered around 0) Exponential window is defined as

    .. math::  w(n) = e^{-n / n_max / \tau}

    References
    ----------
    .. [1] S. Gade and H. Herlufsen, "Windows to FFT analysis (Part I)",
           Technical Review 3, Bruel & Kjaer, 1987.

    """
    assert tau > 0

    if eigenvalue_bounds is None:
        M = eigenvalues.max(-1, keepdims=True).values
    else:
        M = eigenvalue_bounds

    eigenvalues = eigenvalues / M

    w = torch.exp(-torch.abs(eigenvalues) / tau)

    return w


class BasisFunctionsLayer(nn.Module):
    """Using Gaussian 'smearing' for g(lambda) where the eigenvalues are
    mapped to a sequence of equally spaced Gaussians."""

    def __init__(self,
                 n_filters: int,
                 start: float = 0.0,
                 stop: float = 2.0,
                 bottle_neck_factor: float = 0.1,
                 num_gaussians: int = 50,
                 init_type: str = 'default') -> None:
        super().__init__()
        self.init_type = init_type
        self.distance_expansion = GaussianSmearing(start, stop, num_gaussians)
        if bottle_neck_factor != 1:
            bottle_neck_d = int(bottle_neck_factor * n_filters)
            linears = [nn.Linear(num_gaussians, bottle_neck_d, bias=False),
                       nn.Linear(bottle_neck_d, n_filters)]
        else:
            linears = [nn.Linear(num_gaussians, n_filters)]
        self.linear = nn.Sequential(*linears)
        self.reset_parameters()

    def forward(self, eigvals: torch.Tensor) -> torch.Tensor:
        """Model filter via learnable weighting of a set of fixed Gaussians.

        Args:
            eigvals (torch.Tensor): Eigenvalues of shape b x k.

        Returns:
            torch.Tensor: filter of shape b x k x d.
        """
        basis_representation = self.distance_expansion(eigvals)
        filters = self.linear(basis_representation)
        filters = filters.reshape(eigvals.shape[:-1] + filters.shape[-1:])
        return filters

    def reset_parameters(self):
        if isinstance(self.linear, nn.Sequential):
            linear_layers = list(self.linear._modules.values())
        else:
            linear_layers = [self.linear]
        for idx, linear in enumerate(linear_layers):
            if self.init_type == 'orthogonal':
                nn.init.orthogonal_(linear.weight)
            elif self.init_type == 'zeros':
                nn.init.zeros_(linear.weight)
            elif self.init_type == 'virtual_node':
                assert len(linear_layers) <= 2
                fill_ = torch.zeros_like(linear.weight)
                if idx == 0:
                    fill_[0, 0] = 1
                    with torch.no_grad():
                        linear.weight.set_(fill_)
                else:
                    fill_[:, 0] = 1
                    with torch.no_grad():
                        linear.weight.set_(fill_)

            if linear.bias is not None:
                nn.init.zeros_(linear.bias)


class AttentionFilterEncoder(nn.Module):
    """Using attention for g(lambda)."""

    def __init__(self, dim: int, num_heads: int = 4, input_scale: float = -1.,
                 filter_zero_init: bool = False) -> None:
        super().__init__()

        self.cond_pre_linear = nn.Linear(3, dim)
        self.cond_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)
        init_type = 'default'
        if filter_zero_init:
            init_type = 'zeros'
        self.cond_post_linear = LinearCustomInit(dim, dim, init_type=init_type)
        self.input_scale = None
        if input_scale > 0:
            self.input_scale = nn.Parameter(torch.Tensor([input_scale]),
                                            requires_grad=True)

    def forward(self, eigenvalues, eigval_mask, num_nodes):
        # Shape [b1, k, 1]
        cond = eigenvalues
        eigenvalues_mean = eigenvalues.mean(-2)
        eigv_node_fraction = eigval_mask[..., 0].sum(1) / num_nodes
        cond = torch.cat([
            cond,
            eigenvalues_mean[..., None].expand(cond.shape),
            eigv_node_fraction[..., None, None].expand(cond.shape),
        ], -1)

        if self.input_scale is not None:
            cond = self.input_scale * cond

        cond = (self.cond_pre_linear(cond)).relu()

        # Duplicate times number heads and then flatten batch * heads
        not_eigval_mask = ~eigval_mask[..., None, :, :].broadcast_to(
            eigval_mask.shape[0], self.cond_attention.num_heads,
            *eigval_mask.shape[1:])
        not_eigval_mask = not_eigval_mask.view(-1, *not_eigval_mask.shape[2:])

        mask = (torch.zeros_like(not_eigval_mask)
                | not_eigval_mask[..., None, :, 0])
        cond_trans = self.cond_attention(cond, cond, cond, attn_mask=mask)[0]

        # cond_trans = (self.cond_trans_linear(cond_trans)).relu()
        cond = cond + self.layer_norm(cond_trans)

        filter_ = self.cond_post_linear(cond)
        return filter_, cond


class Window(nn.Module):
    """The windowing implementation that assures a smooth filter transition at
    truncation of spectrum."""

    def __init__(self, window_type: str = 'tukey', frequency_cutoff: Optional[float] = None) -> None:
        super().__init__()
        if cfg.gnn.spectral.window == 'tukey':
            self.window = tukey_window
        elif cfg.gnn.spectral.window == 'exp':
            self.window = exponential_window
        else:
            raise ValueError(f'Window {window_type} not supported')
        self.frequency_cutoff = frequency_cutoff

    def forward(self, batch):
        eigval, eigval_bounds = get_eigenvalues(batch)

        if self.frequency_cutoff:
            if eigval_bounds is not None:
                eigval_bounds = torch.clamp(
                    eigval_bounds, max=self.frequency_cutoff)
            else:
                eigval_bounds = self.frequency_cutoff

        if isinstance(eigval_bounds, torch.Tensor):
            eigval_bounds = eigval_bounds.view(batch.num_graphs, -1)

        window_kwargs = dict(eigenvalue_bounds=eigval_bounds)
        if self.frequency_cutoff:
            window_kwargs['wiggle'] = 0

        window = self.window(eigval, **window_kwargs)
        return window


class FilterEncoder(nn.Module):
    """The filter encoder that transforms eigenvalues to filter magnitudes."""

    def __init__(self, dim: int, encoder_type: str,
                 num_heads: int = -1, is_first: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.n_repeat = 1 if num_heads <= 0 else math.ceil(dim / num_heads)
        num_filters = dim if num_heads <= 0 else num_heads

        self.encoder_type = encoder_type
        if encoder_type == 'mlp':
            self.filter = MLP(new_layer_config(
                1, num_filters, cfg.gnn.spectral.mlp_layers_filter_encoder,
                has_act=True, has_bias=True, cfg=cfg))
        elif encoder_type == 'basis':
            stop = 2
            if cfg.gnn.spectral.frequency_cutoff:
                stop = cfg.gnn.spectral.frequency_cutoff
            self.filter = BasisFunctionsLayer(
                num_filters, stop=stop,
                bottle_neck_factor=cfg.gnn.spectral.basis_bottleneck,
                num_gaussians=cfg.gnn.spectral.basis_num_gaussians,
                init_type=cfg.gnn.spectral.basis_init_type)
        elif encoder_type == 'attn':
            zero_init = (cfg.gnn.spectral.basis_init_type == 'zeros')
            if is_first:
                self.filter = AttentionFilterEncoder(
                    num_filters, input_scale=cfg.gnn.spectral.eigv_scale,
                    filter_zero_init=zero_init)
            else:
                init_type = 'zeros' if zero_init else 'default'
                self.filter = LinearCustomInit(
                    num_filters, num_filters, init_type=init_type)
        else:
            raise ValueError(f'Filter encoder {encoder_type} unknown')

    def forward(self, batch) -> torch.Tensor:
        eigval = get_eigenvalues(batch)[0]
        eigval_mask = batch.laplacian_eigenvalue_plain_mask
        eigval = eigval.view(batch.num_graphs, -1, 1)
        eigval_mask = eigval_mask.view(batch.num_graphs, -1, 1)
        if self.encoder_type == 'attn' and 'cond' not in batch:
            num_nodes = batch.ptr.diff() if 'ptr' in batch else batch.num_nodes
            filter_, cond = self.filter(eigval, eigval_mask, num_nodes)
            batch.cond = cond
        elif self.encoder_type == 'attn' and 'cond' in batch:
            filter_ = self.filter(batch.cond)
        else:
            filter_ = self.filter(eigval)
        filter_ = filter_[..., None, :] * eigval_mask[..., None]

        # For headed spectral filter
        if self.n_repeat > 1:
            repeat_per_dim = [1] * (filter_.ndim - 1) + [self.n_repeat]
            filter_ = filter_.repeat(*repeat_per_dim)[..., :self.dim]

        # Allow for batching node-level signal
        if batch.x.dim() == 2:
            filter_ = filter_[:, :, 0, :]

        return filter_


def acos_transform(eigval: torch.Tensor) -> torch.Tensor:
    """Numerically more stable version of

    `1 / torch.pi * torch.acos(1 - torch.clamp(eigval, 0, 2))`

    where `eigval` can be rather close to zero.

    Args:
        eigval (torch.Tensor): eigenvalues

    Returns:
        torch.Tensor: transformed eigenvalues
    """
    eigval = torch.clamp(eigval, 0, 2)
    a = torch.sqrt(torch.clamp(2 * eigval - eigval ** 2, min=0))
    b = 1 - eigval
    return 1 / torch.pi * torch.atan2(a, b)


def get_eigenvalues(batch):
    """Either return eigenvalues plain or after a transformation."""
    eigval = batch.laplacian_eigenvalue_plain
    bounds = None
    if 'laplacian_eigenvalue_plain_bounds' in batch:
        bounds = batch.laplacian_eigenvalue_plain_bounds

    if cfg.gnn.spectral.filter_value_trans is not None:
        if 'acos' in cfg.gnn.spectral.filter_value_trans:
            eigval = acos_transform(eigval)
            if bounds is not None:
                bounds = acos_transform(bounds)

        if cfg.gnn.spectral.filter_value_trans == 'nacos':
            eigval = eigval * batch.ptr.diff()[:, None]
            if bounds is not None:
                bounds = bounds * batch.ptr.diff()

    return eigval, bounds


class LinearCustomInit(nn.Linear):
    """Linear layer that allows for a zero initialization of weights."""

    def __init__(self, *args, init_type='zeros', **kwargs) -> None:
        self.init_type = init_type
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        if self.init_type == 'zeros':
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            super().reset_parameters()

