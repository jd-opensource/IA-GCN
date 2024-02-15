# IA-GCN: Interactive Graph Convolutional Network for Recommendation

## Overview
This is our Tensorflow implementation for our CIKM 2023 short paper:  
>Zhang, Yinan, et al. "BI-GCN: Bilateral Interactive Graph Convolutional Network for Recommendation." Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 2023. (https://dl.acm.org/doi/abs/10.1145/3583780.3615232).

We also provide a long version on arxiv: IA-GCN: Interactive Graph Convolutional Network for Recommendation (https://arxiv.org/abs/2204.03827).


## Introduction 
In this work, we propose a novel graph attention model named Interactive GCN (IA-GCN), which introduces bilateral interactive guidance into each user-item pair for preference prediction. By this manner, we can obtain target-aware representations, i.e., the information of the target item/user is explicitly encoded in the user/item representation, for more precise matching. 

## Requirements
The required packages are as follows:
numpy (1.15.0)
tensorflow (1.12.0)

## Quick Start
cd electronics/l2_hyper_weights
python -u /export/App/training_platform/PinoModel/Light_GCN_ops.py --dataset=electronics --regs=[1e-4] --embed_size=64 --layer_size=[64,64] --lr=6.5e-05 --batch_size=1024 --epoch=1000

## Example to run 2-layer IA-GCN
