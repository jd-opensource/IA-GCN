# IA-GCN: Interactive Graph Convolutional Network for Recommendation

## Overview
This is our Tensorflow implementation for the paper "IA-GCN: Interactive Graph Convolutional Network for Recommendation".

## Introduction 
we propose a novel graph attention model named Interactive GCN (IA-GCN), which introduces bilateral interactive guidance into each user-item pair for preference prediction. By this manner, we can obtain target-aware representations,i.e., the information of the target item/user is explicitly encoded in the user/item representation, for more precise matching. 

test

## Requirements
numpy (1.15.0)
tensorflow (1.12.0)

## Quick Start
cd electronics/l2_hyper_weights
python -u /export/App/training_platform/PinoModel/Light_GCN_ops.py --dataset=electronics --regs=[1e-4] --embed_size=64 --layer_size=[64,64] --lr=6.5e-05 --batch_size=1024 --epoch=1000
