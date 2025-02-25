from typing import Optional, Callable, List
import copy
import re
import os
import glob
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import (Batch, InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip)
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.transforms import BaseTransform, Compose
from torch_sparse import SparseTensor
from tqdm import tqdm


class TPUGraphs(InMemoryDataset):

    def __init__(self, root: str, thres: int = 1000,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 task: str = 'layout',
                 subsample: int = 500,
                 source: str = 'nlp',  # 'nlp' or 'xla'
                 search: str = 'random',  # 'random' or 'default'
                 custom: bool = False,
                 normalize: bool = True,
                 num_sample_configs_train: int = 100,
                 scale_num_cfgs_train: bool = False,
                 num_sample_configs_eval: int = 1_000,
                 num_sample_batch_eval: int = 100,
                 include_valid_in_train: bool = False,
                 subsample_seed: int = 42):
        assert task in ('layout', 'tile')
        assert source in ('nlp', 'xla')
        assert search in ('random', 'default')
        assert task != 'tile' or (source == 'xla' and subsample < 0)
        self.thres = thres
        self.task = task
        self.source = source
        self.search = search
        self.custom = custom
        self.normalize = normalize
        self.subsample = subsample
        self.subsample_seed = subsample_seed
        self.num_sample_configs_train = num_sample_configs_train
        self.scale_num_cfgs_train = scale_num_cfgs_train
        self.num_sample_configs_eval = num_sample_configs_eval
        self.num_sample_batch_eval = num_sample_batch_eval
        self.include_valid_in_train = include_valid_in_train

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if self.normalize:
            norm_transform = TPUGraphsNormalize(self.custom)
            self.add_transform(norm_transform)
        else:
            op_feats_mean = torch.mean(self.data.op_feats, dim=0, keepdim=True)
            op_feats_std = torch.std(self.data.op_feats, dim=0, keepdim=True)
            op_feats_std[op_feats_std < 1e-6] = 1
            self.data.op_feats = (
                self.data.op_feats - op_feats_mean) / op_feats_std

    def add_transform(self, new_transform: BaseTransform):
        if self.transform is None:
            self.transform = new_transform
        elif isinstance(self.transform, Compose):
            self.transform = Compose(
                self.transform.transforms + [new_transform])
        else:
            self.transform = Compose([self.transform, new_transform])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,
                        'raw_custom' if self.custom else 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.task == 'tile':
            return [f'npz/tile/{self.source}']
        return [f'npz/layout/{self.source}/{self.search}']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_segment_{}.pt'.format(self.thres), 'split_dict_segment_{}.pt'.format(self.thres)]

    @property
    def processed_dir(self) -> str:
        prefix = osp.join(self.root,
                          'processed_custom' if self.custom else 'processed',
                          self.task,
                          self.source)
        if self.task == 'tile':
            return osp.join(prefix, str(self.subsample))
        else:
            return osp.join(prefix, self.search, str(self.subsample))

    def process(self):
        data_list = []
        split_names = ['train', 'valid', 'test']
        split_dict = {'train': [], 'valid': [], 'test': []}
        graphs_cnt = 0
        parts_cnt = 0

        rand_state = np.random.get_state()
        np.random.seed(self.subsample_seed)

        for raw_path in self.raw_paths:
            for split_name in split_names:
                filenames = glob.glob(osp.join(os.path.join(raw_path, split_name), '*.npz'))
                graph_id_prefix = f'{self.task}:{self.source}'
                if self.task == 'layout':
                    graph_id_prefix = f'{graph_id_prefix}:{self.search}'
                for filename in tqdm(filenames, desc=f'Processing {split_name} {raw_path}'):
                    split_dict[split_name].append(graphs_cnt)
                    graph_id = os.path.splitext(os.path.basename(filename))[0]
                    graph_id = f'{graph_id_prefix}:{graph_id}'
                    np_file = dict(np.load(filename))
                    if "edge_index" not in np_file:
                      print('error in', filename)
                    edge_index = torch.tensor(np_file["edge_index"].T)
                    op = torch.tensor(np_file["node_feat"])
                    op_code = torch.tensor(np_file["node_opcode"])

                    input_num_configs = np_file["config_runtime"].shape[0]
                    if "node_config_feat" in np_file:
                        if (self.subsample >= 0
                              and split_name != 'test'
                              and self.subsample < input_num_configs):
                            n_best = int(0.01 * self.subsample)
                            n_worst = int(0.01 * self.subsample)
                            if split_name == 'valid':
                                n_best = 1
                                n_worst = 1
                            n_random = self.subsample - n_best - n_worst
                            n_rand_choose_from = input_num_configs - n_best - n_worst
                            argsort_config_runtimes = np.argsort(np_file["config_runtime"])
                            lin_weight = (
                                (input_num_configs - np.arange(n_rand_choose_from)) 
                                / n_rand_choose_from).astype(np.float32) 
                            # Skew sampling towards good runtimes.
                            # Sample wrt GumbalSoftmax([NumConfs, NumConfs-1, ..., 1])
                            select_idx = np.argsort(
                                lin_weight - np.log(-np.log(np.random.uniform(0, 1, [n_rand_choose_from]))))[-n_random:]
                            select_idx = np.concatenate((np.arange(n_best),
                                                        select_idx + n_best,
                                                        -np.arange(1, n_worst + 1)))

                            select_idx = argsort_config_runtimes[select_idx]
                            # num_configs = config_samples
                            runtime = np_file["config_runtime"][select_idx]
                            config_feats = np_file["node_config_feat"][select_idx]
                        else:
                            runtime = np_file["config_runtime"]
                            config_feats = np_file["node_config_feat"]

                        num_config = torch.tensor(config_feats.shape[0])
                        num_config_idx = torch.tensor(config_feats.shape[1])
                        config_feats = torch.tensor(config_feats)
                        config_feats = config_feats.view(-1, config_feats.shape[-1])
                        config_idx = torch.tensor(np_file["node_config_ids"])
                        num_nodes = torch.tensor(np_file["node_feat"].shape[0])
                        num_parts = num_nodes // self.thres + 1
                        interval = num_nodes // num_parts
                        partptr = torch.arange(0, num_nodes, interval+1)
                        if partptr[-1] != num_nodes:
                            partptr = torch.cat([partptr, torch.tensor([num_nodes])])
                        edge_attr = None
                        if 'edge_attr' in np_file:
                            edge_attr = torch.tensor(np_file['edge_attr'])
                        data = Data(edge_index=edge_index, op_feats=op, op_code=op_code, config_feats=config_feats, config_idx=config_idx,
                                    num_config=num_config, num_config_idx=num_config_idx, y=runtime, num_nodes=num_nodes, partptr=partptr, partition_idx=parts_cnt, graph_id=graph_id, edge_attr=edge_attr)
                    else:
                        runtime = np_file["config_runtime"]
                        config_feats = np_file["config_feat"]
                        config_runtime_normalizers = np_file["config_runtime_normalizers"]

                        num_config = torch.tensor(config_feats.shape[0])
                        config_feats = torch.tensor(config_feats)
                        num_nodes = torch.tensor(np_file["node_feat"].shape[0])
                        num_parts = num_nodes // self.thres + 1
                        interval = num_nodes // num_parts
                        partptr = torch.arange(0, num_nodes, interval+1)
                        if partptr[-1] != num_nodes:
                            partptr = torch.cat([partptr, torch.tensor([num_nodes])])
                        data = Data(edge_index=edge_index, op_feats=op, op_code=op_code, config_feats=config_feats,
                                    num_config=num_config, config_runtime_normalizers=config_runtime_normalizers, y=runtime, num_nodes=num_nodes, partptr=partptr, partition_idx = parts_cnt, graph_id=graph_id, edge_attr=None)

                    data_list.append(data)
                    graphs_cnt += 1
                    parts_cnt += num_parts * num_config
            torch.save(self.collate(data_list), self.processed_paths[0])
            torch.save(split_dict, self.processed_paths[1])
        np.random.set_state(rand_state)

    def get_idx_split(self):
        split_dict = torch.load(self.processed_paths[1])
        if self.include_valid_in_train:
            split_dict['train'].extend(split_dict['valid'])
        return split_dict

    def collate_fn_train(self, data):
        batch = Batch.from_data_list(data)

        num_sample_configs = self.num_sample_configs_train
        if self.scale_num_cfgs_train:
            num_sample_configs_orig = num_sample_configs
            factor = np.sqrt((batch.num_graphs * 40_000) / batch.num_nodes)
            # factor = (batch.num_graphs * 40_000) / batch.num_nodes
            num_sample_configs = int(
                (num_sample_configs * factor).round().item())

        batch_list = batch.to_data_list()
        processed_batch_list = []
        for i, g in enumerate(batch_list):
            sample_idx = torch.randint(
                0, g.num_config.item(), (num_sample_configs,))
            # if g.num_config.item() >= num_sample_configs:
            #     sample_idx = torch.randperm(
            #         g.num_config.item())[:num_sample_configs]
            # else:
            #     sample_idx = torch.randint(
            #         0, g.num_config.item(), (num_sample_configs,))
            g.y = torch.tensor(g.y[sample_idx])[None, :]
            if 'num_config_idx' in g:
                g.config_feats = g.config_feats.view(
                    g.num_config, g.num_config_idx, -1)[sample_idx, ...]
                g.config_feats = g.config_feats.transpose(0, 1)
                g.config_feats_full = torch.zeros(
                    (g.num_nodes, num_sample_configs, g.config_feats.shape[-1]),
                    device=g.config_feats.device)
                g.config_feats_full[g.config_idx, ...] += g.config_feats
                g.config_idx += batch.ptr[i]
            else:
                if num_sample_configs > g.y.shape[-1]:
                    continue
                g.config_feats = g.config_feats[None, sample_idx, ...]
            # g.adj = SparseTensor(row=g.edge_index[0], col=g.edge_index[1], sparse_sizes=(g.num_nodes, g.num_nodes))
            processed_batch_list.append(g)
        batch = Batch.from_data_list(processed_batch_list)
        #batch.y = torch.stack(batch.y)
        return batch

    def collate_fn_eval(self, data):
        num_sample_configs = self.num_sample_configs_eval
        num_sample_batch = self.num_sample_batch_eval

        subbatches = []
        for batch in data:
            num_sample_configs_ = num_sample_configs
            # def preprocess_batch_val(batch, num_sample_configs=1_000, num_sample_batch=100):
            if num_sample_configs_ > batch.num_config.item():
                sample_idx_all = torch.randperm(batch.num_config.item())
            else:
                if num_sample_configs_ < 0:
                    num_sample_configs_ = batch.num_config.item()
                sample_idx_all = torch.arange(num_sample_configs_)
            num_sample_configs_ = min(num_sample_configs_,
                                      batch.num_config.item())
            processed_batch_list = []
            for i in range(int(np.ceil(num_sample_configs_ / num_sample_batch))):
                g = batch.clone()
                last_id = min((i + 1) * num_sample_batch, g.num_config.item())
                batch_size = last_id - i * num_sample_batch
                sample_idx = sample_idx_all[i * num_sample_batch:last_id]
                y = torch.tensor(g.y[sample_idx])
                if y.ndim == 0:  # Handle that sample_idx is a single element
                    y = y.unsqueeze(0)
                g.y = y.unsqueeze(0)
                if 'num_config_idx' in g:
                    g.config_feats = g.config_feats.view(g.num_config, g.num_config_idx, -1)[sample_idx, ...]
                    g.config_feats = g.config_feats.transpose(0, 1)
                    g.config_feats_full = torch.zeros((g.num_nodes, batch_size, g.config_feats.shape[-1]), device=g.config_feats.device)
                    g.config_feats_full[g.config_idx, ...] += g.config_feats
                else:
                    g.config_feats = g.config_feats[None, sample_idx, ...]
                # g.adj = SparseTensor(row=g.edge_index[0], col=g.edge_index[1], sparse_sizes=(g.num_nodes, g.num_nodes))
                g.batch = torch.zeros((g.num_nodes,), device=g.edge_index.device, dtype=torch.int64)
                g.num_graphs = 1
                processed_batch_list.append(g)
            subbatches.append(processed_batch_list)
        return subbatches


class TPUGraphsNormalize(BaseTransform):

    def __init__(self, custom: bool = False) -> None:
        super().__init__()

        self.custom = custom

        self.min_op_feats = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.max_op_feats = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 184320000.0, 372736.0, 372736.0, 372736.0, 2048.0, 256.0, 184320000.0, 2949120000.0, 322.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 35.0, 4.0, 4.0, 4.0, 5.0, 4.0, 5.0, 512.0, 512.0, 128.0, 3.0, 1.0, 0.0, 1024.0, 2097152.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0, 8.0, 16.0, 79.0, 127.0, 4.0, 0.0, 0.0, 0.0, 127.0, 16.0, 4.0, 4.0, 4.0, 1.0, 0.0, 0.0, 8.0, 18.0, 4.0, 4.0, 2.0, 1.0, 1.0, 0.0, 8.0, 16.0, 4.0, 4.0, 3.0, 1.0, 1.0, 0.0, 8.0, 16.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 5.0, 4.0, 4.0, 3.0, 2.0, 3.0, 0.0, 4.0, 4.0, 1.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3840.0, 3840.0, 1024.0, 119646.0, 119646.0, 1024.0, 1.0, 1.0, 5.0, 1.0, 12375.0, 196608.0, 196614.0, 805306368.0, 12.0, 2048.0, 33784.0, 25887744.0, 1.0, 64.0, 448.0, 1.0, 3072.0, 1023.0, 33784.0, 25887744.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3502.0, 0.0, 0.0, 4.0, 3.0, 4.0, 3.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 5.0, 4.0, 5.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0])

        self.is_constant = (self.max_op_feats - self.min_op_feats) == 0
        self.is_constant[146] = True

        self.is_custom = torch.zeros_like(self.min_op_feats, dtype=bool)
        self.is_custom[30:38] = True
        self.is_custom[141:145] = True
        self.is_custom[146:169] = True
        self.is_custom[175:] = True

        self.use_log_scale = torch.zeros_like(self.min_op_feats, dtype=bool)
        self.use_log_scale[21:30] = True
        self.use_log_scale[38:145] = True
        self.use_log_scale[147:178] = True
        self.use_log_scale[self.min_op_feats < 0] = False

        self.tuple_index_dim = 146

        self.max_op_feats_scaled = self.max_op_feats.clone()
        log_mask = self.use_log_scale & ~self.is_constant
        self.max_op_feats_scaled[log_mask] = self.max_op_feats_scaled[
            log_mask].log() + 1

        if not self.custom:
            self.default_fields_idx = torch.where(~self.is_custom)[0]
            self.is_constant = self.is_constant[self.default_fields_idx]
            self.use_log_scale = self.use_log_scale[self.default_fields_idx]
            self.min_op_feats = self.min_op_feats[self.default_fields_idx]
            self.max_op_feats = self.max_op_feats[self.default_fields_idx]
            self.max_op_feats_scaled = self.max_op_feats_scaled[
                self.default_fields_idx]

    def __call__(self, data: Data) -> Data:
        if self.custom:
            data.tuple_idx = data.op_feats[:, 146]

        is_not_constant_idx = torch.where(~self.is_constant)[0]
        op_feats = data.op_feats

        op_feats = op_feats[:, is_not_constant_idx]

        log_mask = ((op_feats >= 1)
                    & self.use_log_scale[None, is_not_constant_idx])
        op_feats[log_mask] = op_feats[log_mask].log() + 1

        op_feats /= self.max_op_feats_scaled[~self.is_constant]

        data.op_feats = op_feats.to(torch.float32)

        return data


if __name__ == '__main__':
    dataset = TPUGraphs(root='datasets/TPUGraphs')
