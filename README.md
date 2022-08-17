# SparseMask
Pytorch implementation of "Exploring Fine-Grained Sparsity in Convolutional Neural Networks for Efficient Inference", TPAMI 2022

[[Paper]](https://ieeexplore.ieee.org/document/9841044) 

## Motivation
- Connection between the sparsity in human brains and the sparsity in CNNs
<p align="center"> <img src="figs/brain.png" width="40%"> </p>

- Feature sparsity in CNNs 
<p align="center"> <img src="figs/sparsity.png" width="85%"> </p>


## Overview

- Sparse Mask Generation
<p align="center"> <img src="figs/SM_generation.png" width="70%"> </p>

- Sparse Mask Convolution
<p align="center"> <img src="figs/SM_conv.png" width="80%"> </p>


## Applications
### 1. Point Cloud Semgantic Segmentation [[code]](https://github.com/LongguangWang/SparseMask/tree/master/point_cloud_semantic_segmentation)

- Network Architecture
<p align="center"> <img src="figs/SMPointSeg.png" width="100%"> </p>

- Results
<p align="center"> <img src="figs/results_pc_1.png" width="100%"> </p>

<p align="center"> <img src="figs/results_pc_2.png" width="100%"> </p>

### 2. Singe Image Super-Resolution [[code]](https://github.com/LongguangWang/SMSR)

- Network Architecture
<p align="center"> <img src="figs/SMSR.png" width="75%"> </p>

- Results
<p align="center"> <img src="figs/results_sr_1.png" width="100%"> </p>

<p align="center"> <img src="figs/results_sr_2.png" width="100%"> </p>

### 3. Stereo Matching

- Network Architecture
<p align="center"> <img src="figs/SMStereo.png" width="75%"> </p>

- Results
<p align="center"> <img src="figs/results_stereo_1.png" width="100%"> </p>

<p align="center"> <img src="figs/results_stereo_2.png" width="100%"> </p>

## Citation
```
@Article{Wang2022Exploring,
  author  = {Longguang Wang and Yulan Guo and Xiaoyu Dong and Yingqian Wang and Xinyi Ying and Zaiping Lin and Wei An},
  title   = {Exploring Fine-Grained Sparsity in Convolutional Neural Networks for Efficient Inference},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2022},
}
```
