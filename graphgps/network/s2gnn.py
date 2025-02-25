from functools import partial
import math
from typing import List

import torch
from torch import nn
from torch_geometric.graphgym.models.layer import new_layer_config
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.gnn import FeatureEncoder

from graphgps.layer.chebnet_conv_layer import ChebNetIILayer
from graphgps.layer.s2_message_passing import *  # noqa f403
from graphgps.layer.s2_spectral import *  # noqa f403


class BatchS2GNNGNNLayer(nn.Module):
    """Main and sole S^2GNN layer implementation for all tasks."""

    def __init__(self, layer_config: LayerConfig, spat_layer: nn.Module,
                 spec_layer: nn.Module, with_node_residual=True,
                 aggr_mode: str='cat', norm: bool=True):
        assert layer_config.dim_in == layer_config.dim_out
        super().__init__()
        self.with_node_residual = with_node_residual
        self.spat_layer = spat_layer
        self.spec_layer = spec_layer
        self.aggr_mode = aggr_mode
        self.norm = norm
        self.norm_factor = 1 + norm * \
            (with_node_residual + (aggr_mode == 'sum'))
        if aggr_mode == 'mamba_like':
            self.linear = nn.Linear(layer_config.dim_in, layer_config.dim_out)
            self.act = nn.SiLU()

    def forward(self, batch):
        if self.aggr_mode == 'mamba_like':
            z = self.act(self.linear(batch.x))
            batch.x = self.act(self.spat_layer(batch))
            y = self.spec_layer(batch)
            batch.x = y * z
            return batch
        else:
            spat_out = self.spat_layer(batch)
            spec_out = self.spec_layer(batch)
            # Aggregate spec/spat part
            if self.aggr_mode == 'sum':
                y = spec_out + spat_out
            elif self.aggr_mode == 'cat':
                y = torch.cat([spec_out, spat_out], dim=-1)
            else:
                raise ValueError(f'Unknown aggregation mode: {self.aggr_mode}')
            # Residual
            if self.with_node_residual:
                y = y + batch.x
            # Normalization
            batch.x = 1 / math.sqrt(self.norm_factor) * y
            return batch


@register_network('s2gnn')
class S2GNN(nn.Module):
    """Main and sole S^2GNN network implementation for all tasks.

    We customize the torch_geometric.graphgym.models.gnn.GNN to support
    specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        # Init dataloader
        if cfg.dataset.node_encoder_name in ['TPUNode', 'TPUBslnNode']:
            edge_dim = cfg.gnn.dim_inner // 2 if cfg.gnn.use_edge_attr else -1
            posenc_dim = -1
            if cfg.posenc_MagLapPE.positional_encoding:
                posenc_dim = cfg.posenc_MagLapPE.max_freqs
                if cfg.posenc_MagLapPE.q > 0:
                    posenc_dim *= 2
            encoder_fn = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.encoder = encoder_fn(
                dim_in, cfg.dataset.tpu_graphs.encoder_factor,
                posenc_dim, edge_dim)
        elif cfg.dataset.node_encoder_name == 'OGBNArxivNode':
            self.encoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]()
        else:
            self.encoder = FeatureEncoder(cfg.gnn.dim_inner)
            dim_in = self.encoder.dim_in

        # Optional MLP prior to message passing
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = MLPMultiBatch(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)

            dim_in = cfg.gnn.dim_inner

        if cfg.gnn.node_dropout:
            self.node_dropout = Dropout1d(cfg.gnn.node_dropout)

        # Init GNN layers
        spat_model = self.build_spatial_layer(cfg.gnn.layer_type,
                                           cfg.gnn.make_undirected,
                                           cfg.gnn.use_edge_attr,
                                           cfg.gnn.adj_norm,
                                           cfg.gnn.dir_aggr)
        spec_model = self.build_spectral_layer()
        layers = []

        spat_layer_skip = [i % cfg.gnn.layers_mp
                           for i in cfg.gnn.layer_skip
                           if i < cfg.gnn.layers_mp]
        spec_layer_skip = [i % cfg.gnn.layers_mp
                           for i in cfg.gnn.spectral.layer_skip
                           if i < cfg.gnn.layers_mp]

        combined_spec_spat_layer = None
        if cfg.gnn.spectral.combine_with_spatial:
            assert spat_model is not None and spec_model is not None

            def combined_spec_spat_layer(layer_cfg, is_first, is_last):  # noqa F811
                with_node_residual = (
                    cfg.gnn.spectral.combine_with_spatial == 'mamba_like')
                spat_layer = spat_model(
                    layer_cfg, is_first=is_first, is_last=is_last,
                    overwrite_x=False, with_node_residual=with_node_residual)
                spec_layer = spec_model(
                    layer_cfg, is_first=is_first, overwrite_x=False,
                    with_node_residual=with_node_residual)
                return BatchS2GNNGNNLayer(
                    layer_cfg, spat_layer, spec_layer,
                    with_node_residual=cfg.gnn.residual,
                    aggr_mode=cfg.gnn.spectral.combine_with_spatial,
                    norm=cfg.gnn.spectral.combine_with_spatial_norm)

        for i in range(cfg.gnn.layers_mp):
            is_first, is_last = i == 0, i == cfg.gnn.layers_mp - 1
            # Potentially different input dim in first mp step
            dim_in_ = dim_in if is_first else cfg.gnn.dim_inner
            # Potentially different output dim in last mp step
            dim_out_ = cfg.gnn.dim_inner
            if is_last and (cfg.gnn.layers_post_mp == 0):
                dim_out_ = dim_out
            # Potentially different output dim for combined w/ concat
            if (combined_spec_spat_layer is not None and
                (i not in spat_layer_skip) and (i not in spec_layer_skip) and
                cfg.gnn.spectral.combine_with_spatial == 'cat'):
                dim_out_ = cfg.gnn.dim_inner // 2

            layer_cfg = new_layer_config(
                dim_in_, dim_out_, cfg.gnn.spectral.filter_layers,
                has_act=True, has_bias=True, cfg=cfg)

            if (combined_spec_spat_layer is not None and
                    (i not in spat_layer_skip) and (i not in spec_layer_skip)):
                layers.append(combined_spec_spat_layer(
                    layer_cfg, is_first, is_last))
            else:
                if spat_model and i not in spat_layer_skip:
                    layers.append(spat_model(
                        layer_cfg, is_first=is_first, is_last=is_last,
                        with_node_residual=cfg.gnn.residual))
                if spec_model and i not in spec_layer_skip:
                    layers.append(spec_model(
                        layer_cfg, is_first=is_first,
                        with_node_residual=cfg.gnn.spectral.residual))

        self.gnn_layers = nn.Sequential(*layers)

        # Init output head
        GNNHead = register.head_dict[cfg.gnn.head]
        is_first = (
            cfg.gnn.layers_mp <= 0 or
            (cfg.gnn.layer_type == 'lin_gnn' and cfg.gnn.layers_mp == 1))
        self.post_mp = GNNHead(cfg.gnn.dim_inner, dim_out, is_first)

    def build_spatial_layer(self, model_type, make_undirected, use_edge_attr,
                            adj_norm, dir_aggr):
        """Prototype/constructor for spatial layer (message passing)."""
        if model_type == 'none' or model_type is None:
            return None
        elif model_type == 'lin_gnn':
            return partial(FeatureBatchGNNLayer,
                           make_undirected=make_undirected,
                           use_edge_attr=use_edge_attr,
                           normalize=adj_norm,
                           dir_aggr=dir_aggr)
        elif model_type == 'gcnconv':
            # TODO: consider dropping since lin_gnn is more general
            return partial(GCNConvGNNLayer,
                           make_undirected=make_undirected,
                           use_edge_attr=use_edge_attr,
                           normalize=adj_norm)
        elif model_type == 'gatconv':
            return GATConvGNNLayer
        elif model_type == 'gatedgcnconv':
            return GatedGCNConvGNNLayer
        elif model_type.startswith('chebconv'):
            kwargs = {}
            if '-' in model_type:
                kwargs['K'] = int(model_type.split('-')[-1])
            return partial(ChebNetIILayer, **kwargs)
        else:
            return GNNLayer

    def build_spectral_layer(self):
        """Prototype/constructor for spectral layer."""
        if not cfg.posenc_MagLapPE.enable:
            return None
        else:
            return FeatureBatchSpectralLayer

    def forward(self, batch):
        # For obg products, we need to handle the subsampling of nodes
        if (cfg.train.sampler == 'random_node'
                and hasattr(batch, 'laplacian_eigenvector_plain')):
            # Normalize eigenvectors
            batch.laplacian_eigenvector_plain /= torch.clamp_min(
                batch.laplacian_eigenvector_plain.norm(dim=0, keepdim=True),
                1e-2)
        # Set heavily used constant num_graphs if not available
        if not hasattr(batch, 'num_graphs'):
            batch.num_graphs = 1

        for module in self.children():
            batch = module(batch)
        return batch

    def last_layer_keys(self) -> List[str]:
        """Returns parameter keys of last layer."""
        proto = 'post_mp.layer_post_mp.Layer_'
        matches = [int(n.replace(proto, '').split('.')[0])
                   for n, _ in self.named_parameters() if n.startswith(proto)]
        layer_idx = max(matches)
        return [f'model.{n}' for n, _ in self.named_parameters()
                if n.startswith(f'{proto}{layer_idx}')]

    def spatial_keys(self) -> List[str]:
        """Returns parameter keys for spatial layers."""
        if cfg.gnn.layer_type == 'combined':
            matches = [f'model.{n}' for n, _ in self.named_parameters()
                       if '.gnn_layers' in n and '.spec' not in n]
        else:
            proto = 'gnn_layers.'
            matches = [
                f'model.{n}' for n, _
                in self.named_parameters() if n.startswith(proto) and
                int(n.replace(proto, '').split('.')[0]) % 2 == 0]
        return matches

    def spec_keys(self) -> List[str]:
        """Returns parameter keys for spectral layers."""
        if cfg.gnn.layer_type == 'combined':
            matches = [f'model.{n}' for n, _ in self.named_parameters()
                       if '.gnn_layers' in n and '.spec' in n]
        else:
            proto = 'gnn_layers.'
            matches = [
                f'model.{n}' for n, _
                in self.named_parameters() if n.startswith(proto) and
                int(n.replace(proto, '').split('.')[0]) % 2 == 1]
        return matches
