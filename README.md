# IA-GCN: Interactive Graph Convolutional Network for Recommendation

## Overview
This is our Tensorflow implementation for our CIKM 2023 short paper:  
>Zhang, Yinan, et al. "BI-GCN: Bilateral Interactive Graph Convolutional Network for Recommendation." Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 2023. (https://dl.acm.org/doi/abs/10.1145/3583780.3615232).

We also provide a long version on arxiv: IA-GCN: Interactive Graph Convolutional Network for Recommendation (https://arxiv.org/abs/2204.03827).


## Introduction 
In this work, we propose a novel graph attention model named Interactive GCN (IA-GCN), which introduces bilateral interactive guidance into each user-item pair for preference prediction. By this manner, we can obtain target-aware representations, i.e., the information of the target item/user is explicitly encoded in the user/item representation, for more precise matching. 

## Requirements
The required packages are as follows: 
* numpy (1.15.0) 
* tensorflow (1.12.0)

## Quick Start
```
cd electronics/l2_hyper_weights
python -u /export/App/training_platform/PinoModel/Light_GCN_ops.py --dataset=electronics --regs=[1e-4] --embed_size=64 --layer_size=[64,64] --lr=6.5e-05 --batch_size=1024 --epoch=1000
```

## Example to run 2-layer IA-GCN
* For data preprocessing, run make_pkl function located in electronics/l2_hyper_weights/utility/load_data.py to generate 'data_bin'. Note: the parameter layer_num equals 2 in this example but needs to be changed accordingly.
* for custom op compliation, run following commands to generate the 'tree_out_load_more.so' file, and put it in the main workspace. 
  ```
  cd ops/l2
  sh 1.build.sh
  ```
* To Train a model, run the following command
  ```
  python -u /export/App/training_platform/PinoModel/Light_GCN_ops.py --dataset=electronics --regs=[1e-4] --embed_size=64 --layer_size=[64,64] --lr=6.5e-05 --batch_size=1024 --epoch=1000
  ```
  
## Dataset
We use four open datasets: Amazon-Electronics, Gowalla, Yelp2018, Amazon-Book, which vary in domains, scale, and density. We closely follow the same data split strategy as existing GCN- based CF works [1, 2]
| Dataset | #Users | #Items | #Interactions | Density |
| :-----| :-----| :----- | :----- | :----- |
| Amazon-Electronics | 1435 | 1522 | 35931 | 0.01654 |
| Gowalla | 29528 | 40981 | 1027370 | 0.00084 |
| Yelp2018 | 31668 | 38048 | 1561406 | 0.00130 |
| Amazon-Book | 52643 | 91599 | 2984108 | 0.00062 |

## References
[1] He, Xiangnan, et al. "Lightgcn: Simplifying and powering graph convolution network for recommendation." Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 2020.

[2] Mao, Kelong, et al. "UltraGCN: ultra simplification of graph convolutional networks for recommendation." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.
