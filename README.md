# CSCL
PyTorch implementation for[Cross-Level Semantic Coherence Learning for IncompleteMulti-View Clustering]

## Environment

python ==3.8

pytorch==2.4.1 (CUDA 11.8)

To run the code, you first need to create a Python 3.8 environment and install PyTorch 2.4.1 with CUDA 11.8 support.

## Requirements

Install the dependencies in requirements.txt.



## run

Modify the flag in the train to select the data set, and then run the python train.py command.



## Datasets

All datasets are stored in the data directory. Before running the code, please make sure that the datasets are placed in the corresponding subdirectories.

To run the model on a specific dataset, select the corresponding dataset flag in `train.py` and execute the training script.

Datasets Download

Download Link: https://pan.quark.cn/s/bf1092360ce9

Extraction Code: 2yfm

After downloading, please extract the datasets into the data directory before running the code.



## Ablation Study

In the ablation study, we use a Transformer Encoder to replace the Mamba module in the original model to investigate the impact of different sequence modeling architectures on model performance. The Transformer Encoder consists of a single encoder layer with 8 attention heads, and the feed-forward dimension is set to four times the latent dimension.
