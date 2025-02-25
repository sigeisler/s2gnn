# Spatio-Spectral Graph Neural Networks (S²GNN)

This repository contains the accompanying code for `Spatio-Spectral Graph Neural Networks (S²GNN)`.

## Install

The codebase is developed based on [GraphGPS](https://github.com/rampasek/GraphGPS). You may install the environment following their instructions. We provide the full conda environment that we used in the `environment.yml` file.

## Implementation Notes

Many components of the original `GraphGPS` are not usable due to the additional batch dimension, required for in TPUGraphs. The S²GNN implementation is located in `graphgps/network/s2gnn.py`, and the spectral layer is in `graphgps/layer/s2_spectral.py`. The parametrization and windowing for the spectral filter is in `graphgps/layer/s2_filter_encoder.py`.

A minimal installation routine that should work for most environments is:
```bash
conda create -n <env_name> python=3.10 pip pytorch torchvision torchaudio pytorch-cuda=<cuda_version> -c pytorch -c nvidia
conda activate <env_name>
conda install pyg pytorch-scatter -c pyg

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install tensorboardX
pip install ogb
pip install wandb
```

## Datasets and Configurations

Except for TPUGraphs, all datasets will be generated or downloaded at the beginning of the experiment execution. For TPU Graphs we largely refer to the official instructions. The configurations for the experiments are located in the folder `configs`, where each dataset has its own subfolder that contain the actual experiment configurations.

## Reproducing Results

For all experiments (but TPUGraphs), the `main.py` is the entry point and the results, e.g., for `peptides-func` can be reproduced with:

```bash
python main.py \
  --cfg configs/peptides-func/peptides-func-s2gnn.yaml \
  out_dir tests/results/peptides-func-default \
  wandb.use False \
  seed 1  # we use 1..10
```

For the other experiments consider swapping out `configs/peptides-func/peptides-func-s2gnn.yaml`. For TPU graphs, you should use `main_tpugraphs.py` instead.
