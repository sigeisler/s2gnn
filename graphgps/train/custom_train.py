import logging
import time
import os
from typing import List

import numpy as np
import torch

from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train, loss_dict
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_sparse import SparseTensor
import torch.nn.functional as F

from graphgps.checkpoint import load_ckpt, save_ckpt, clean_ckpt, get_ckpt_dir
from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name


def train_epoch(logger, loader, model, avg_model,
                optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    # with torch.autograd.detect_anomaly():
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        if cfg.dataset.name == 'source-dist':
            # Get predictions to a reasonable range
            num_nodes_per_graph = batch.ptr.diff()[batch.batch][:, None]
            pred = torch.tanh(pred / num_nodes_per_graph) * num_nodes_per_graph
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if avg_model is not None:
                avg_model.update_parameters(model)

        data = None
        if batch.split in cfg.train.ckpt_data_splits:
            # For storing the raw data
            data = batch.cpu()
        logger.update_stats(true=_true,
                            pred=_pred,
                            data=data,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()


def maybe_pad(elements: List[torch.Tensor], dim: int = 1):
    shapes = [el.shape[dim] for el in elements]
    max_dims = max(shapes)
    if max_dims == min(shapes):
        return
    for idx, el in enumerate(elements):
        if el.shape[dim] == max_dims:
            continue
        pad = (2 * el.ndim) * [0]
        pad[2 * dim] = max_dims - el.shape[dim]
        pad = tuple(reversed(pad))
        fill_value = el.max() + 1
        elements[idx] = torch.nn.functional.pad(
            el, pad, "constant", fill_value)


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    sample_ids = []
    for batch in loader:
        if cfg.dataset.name == 'TPUGraphs':
            preds = []
            trues = []
            if isinstance(batch, list):
                for e in batch:
                    sample_ids.append(e[0].graph_id)
            else:
                sample_ids.append(batch.graph_id)
            if split == 'train':
                batch = [[batch]]
            for i in range(len(batch)):
                subbatches = batch[i]
                sub_preds = []
                sub_trues = []
                for subbatch in subbatches:
                    subbatch.split = split
                    subbatch = subbatch.to(torch.device(cfg.device))
                    pred, true = model(subbatch)
                    subbatch.to('cpu', non_blocking=True)
                    sub_preds.append(pred)
                    sub_trues.append(true)
                sub_pred = torch.concatenate(sub_preds, dim=-2)
                sub_true = torch.concatenate(sub_trues, dim=-1)
                preds.append(sub_pred)
                trues.append(sub_true)
            maybe_pad(preds)
            pred = torch.concatenate(preds, dim=0).to(torch.device(cfg.device))
            maybe_pad(trues)
            true = torch.concatenate(trues, dim=0).to(torch.device(cfg.device))
            extra_stats = {}
        elif cfg.gnn.head == 'inductive_edge':
            batch.to(torch.device(cfg.device))
            pred, true, extra_stats = model(batch)
        else:
            batch.to(torch.device(cfg.device))
            batch.split = split
            out = model(batch)
            if len(out) == 2:
                pred, true = out
                extra_stats = {}
            else:
                pred, true, extra_stats = out

        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)

        data = None
        if (not isinstance(batch, list)
                and batch.split in cfg.train.ckpt_data_splits):
            # For storing the raw data
            data = batch.cpu()
        logger.update_stats(true=_true,
                            pred=_pred,
                            data=data,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
    return sample_ids


def build_ckpt_data(data, loggers):
    data_save = {}
    for logger, batch_list in zip(loggers, data):
        if len(batch_list) > 0:
            for b in batch_list:
                for attr in set(b._store.keys()) - set(cfg.train.ckpt_data_attrs):
                    delattr(b, attr)
            batch = Batch.from_data_list(batch_list)
            data_save[logger.name] = batch.detach().cpu()
    return data_save


@torch.no_grad()
def submission_csv(preds, test_loader, output_dir=None, suffix=None):
    tpu_task = cfg.dataset.tpu_graphs.tpu_task
    source = '-'.join(cfg.dataset.tpu_graphs.source)
    search = '-'.join(cfg.dataset.tpu_graphs.search)
    output_csv_filename = f'inference_{tpu_task}_{source}_{search}'
    if suffix is not None:
        output_csv_filename = f'{output_csv_filename}_{suffix}'
    output_csv_filename = f'{output_csv_filename}.csv'

    print('Generating submission csv')
    if output_dir is None:
        output_dir = get_ckpt_dir()
        os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, output_csv_filename)

    # prefix = f'{cfg.dataset.tpu_graphs.tpu_task}:{source}'
    # if cfg.dataset.tpu_graphs.tpu_task == 'layout':
    #     prefix = f'{prefix}:{search}'
    with open(output_csv, 'w+') as fout:
        fout.write('ID,TopConfigs\n')
        for batch, pred in zip(test_loader, preds):
            for graph, all_scores in zip(batch, pred):
                graph_id = graph[0].graph_id
                ranks = all_scores.argsort().numpy()
                if cfg.dataset.tpu_graphs.tpu_task == 'tile':
                    ranks = ranks[:5]
                ranks = ';'.join(ranks.astype(str).tolist())
                fout.write(f'{graph_id},{ranks}\n')

    print('Wrote', output_csv)


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        tags = cfg.wandb.tags.split(',') if cfg.wandb.tags else None
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name, tags=tags, save_code=True)
        run.config.update(cfg_to_dict(cfg))
        wandb.watch(model, log_freq=1_000)

    avg_model = None
    if cfg.optim.model_averaging is not None:
        assert not cfg.gnn.batchnorm, 'No BatchNorm with averaging'
        assert not cfg.gnn.batchnorm_post_mp, 'No BatchNorm with averaging'
        if cfg.optim.model_averaging == 'ema':
            def avg_fn(avg_parameter, curr_parameter, *args, decay=0.999):
                return decay * avg_parameter + (1 - decay) * curr_parameter
        else:
            avg_fn = None
        avg_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=avg_fn)

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start_time = time.perf_counter()
        pred, data = [], []
        train_epoch(loggers[0], loaders[0], model,
                    avg_model if cur_epoch >= cfg.optim.model_averaging_start else None,
                    optimizer, scheduler, cfg.optim.batch_accumulation)
        pred.append(loggers[0]._pred)
        data.append(loggers[0]._data)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if (cfg.optim.model_averaging
                and cur_epoch >= cfg.optim.model_averaging_start):
            # Swap model with averaged model for validation etc
            model_ = model
            model = avg_model

        if is_eval_epoch(cur_epoch) or cur_epoch == start_epoch:
            for i in range(1, num_splits):
                sample_ids = eval_epoch(loggers[i], loaders[i], model,
                                        split=split_names[i - 1])
                splits = None
                if split_names[i - 1] == 'val' and cfg.dataset.name == 'TPUGraphs':
                    splits = {}
                    for idx, sample_id in enumerate(sample_ids):
                        split_id = '_' + '_'.join(sample_id.split(':')[:-1])
                        if split_id in splits:
                            splits[split_id].append(idx)
                        else:
                            splits[split_id] = [idx]
                pred.append(loggers[i]._pred)
                data.append(loggers[i]._data)
                perf[i].append(loggers[i].write_epoch(cur_epoch, splits))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch,
                      build_ckpt_data(data, loggers))

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            key_loss = 'loss'
            m = 'loss'
            if cfg.dataset.name == 'TPUGraphs':
                if cfg.dataset.tpu_graphs.tpu_task == 'layout':
                    key_loss = 'kendal_tau'
                else:
                    key_loss = 'one_minus_slowdown'
                val_all = np.array([vp[key_loss] for vp in val_perf])
                if np.isnan(val_all).all():
                    best_epoch = -1
                else:
                    best_epoch = np.nanargmax(val_all)
            else:
                best_epoch = np.nanargmin(
                    np.array([vp[key_loss] for vp in val_perf]))
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
            if m in perf[0][best_epoch]:
                best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
            else:
                # Note: For some datasets it is too expensive to compute
                # the main metric on the training set.
                best_train = f"train_{m}: {0:.4f}"
            best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
            best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

            if cfg.wandb.use:
                bstats = {"best/epoch": best_epoch}
                for i, s in enumerate(['train', 'val', 'test']):
                    bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                    if m in perf[i][best_epoch]:
                        bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                        run.summary[f"best_{s}_perf"] = \
                            perf[i][best_epoch][m]
                    for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                        if x in perf[i][best_epoch]:
                            bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                run.log(bstats, step=cur_epoch)
                run.summary["full_epoch_time_avg"] = np.mean(
                    full_epoch_times)
                run.summary["full_epoch_time_sum"] = np.sum(
                    full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch,
                          build_ckpt_data(data, loggers))
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()

            if cfg.dataset.name == 'TPUGraphs':
                # Generate submission csv
                submission_csv(pred[-1], loaders[-1], suffix=cur_epoch)
                if best_epoch == cur_epoch:
                    submission_csv(pred[-1], loaders[-1])
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train {key_loss}: {perf[0][best_epoch][key_loss]:.4f} {best_train}\t"
                f"val {key_loss}: {perf[1][best_epoch][key_loss]:.4f} {best_val}\t"
                f"test {key_loss}: {perf[2][best_epoch][key_loss]:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")

            if (cfg.optim.stop_patience and
                    cur_epoch - best_epoch > cfg.optim.stop_patience):
                logging.info(f"Patience stop in epoch {best_epoch}")
                break

        if (cfg.optim.model_averaging
                and cur_epoch >= cfg.optim.model_averaging_start):
            model = model_  # Swap back

    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(
        f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()

    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None, csv_path=None, suffix=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test'][-num_splits:]
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        print(f'Split: {split_names[i]}')
        sample_ids = eval_epoch(loggers[i], loaders[i], model,
                                split=split_names[i])
        splits = None
        if split_names[i] == 'val':
            splits = {}
            for idx, sample_id in enumerate(sample_ids):
                split_id = '_' + '_'.join(sample_id.split(':')[:-1])
                if split_id in splits:
                    splits[split_id].append(idx)
                else:
                    splits[split_id] = [idx]
        submission_csv(loggers[i]._pred, loaders[-1], csv_path,
                       f'{split_names[i]}_{suffix}')
        perf[i].append(loggers[i].write_epoch(cur_epoch, splits))
        if split_names[i] == 'val':
            if csv_path is not None:
                with open(f'{csv_path}/stats.txt', 'a') as file:
                    file.write(f"{suffix},{perf[i][-1]['kendal_tau']}\n")

    best_epoch = 0
    best_train = best_val = best_test = ""

    key_best = 'loss'
    if cfg.dataset.name == 'TPUGraphs':
        if cfg.dataset.tpu_graphs.tpu_task == 'layout':
            key_best = 'kendal_tau'
        else:
            key_best = 'one_minus_slowdown'

    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    if num_splits >= 3:
        logging.info(
            f"> Inference | "
            f"train {key_best}: {perf[0][best_epoch][key_best]:.4f} {best_train}\t"
            f"val {key_best}: {perf[1][best_epoch][key_best]:.4f} {best_val}\t"
            f"test {key_best}: {perf[2][best_epoch][key_best]:.4f} {best_test}"
        )
        logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
        for logger in loggers:
            logger.close()


@register_train('PCQM4Mv2-inference')
def ogblsc_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on OGB-LSC PCQM4Mv2.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from ogb.lsc import PCQM4Mv2Evaluator
    evaluator = PCQM4Mv2Evaluator()

    num_splits = 3
    split_names = ['valid', 'test-dev', 'test-challenge']
    assert len(loaders) == num_splits, "Expecting 3 particular splits."

    # Check PCQM4Mv2 prediction targets.
    logging.info(f"0 ({split_names[0]}): {len(loaders[0].dataset)}")
    assert (all([not torch.isnan(d.y)[0] for d in loaders[0].dataset]))
    logging.info(f"1 ({split_names[1]}): {len(loaders[1].dataset)}")
    assert (all([torch.isnan(d.y)[0] for d in loaders[1].dataset]))
    logging.info(f"2 ({split_names[2]}): {len(loaders[2].dataset)}")
    assert (all([torch.isnan(d.y)[0] for d in loaders[2].dataset]))

    model.eval()
    for i in range(num_splits):
        all_true = []
        all_pred = []
        for batch in loaders[i]:
            batch.to(torch.device(cfg.device))
            pred, true = model(batch)
            all_true.append(true.detach().to('cpu', non_blocking=True))
            all_pred.append(pred.detach().to('cpu', non_blocking=True))
        all_true, all_pred = torch.cat(all_true), torch.cat(all_pred)

        if i == 0:
            input_dict = {'y_pred': all_pred.squeeze(),
                          'y_true': all_true.squeeze()}
            result_dict = evaluator.eval(input_dict)
            # Get MAE.
            logging.info(f"{split_names[i]}: MAE = {result_dict['mae']}")
        else:
            input_dict = {'y_pred': all_pred.squeeze()}
            evaluator.save_test_submission(input_dict=input_dict,
                                           dir_path=cfg.run_dir,
                                           mode=split_names[i])
