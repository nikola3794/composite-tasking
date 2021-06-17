# CompositeTasking: Understanding Images by Spatial Composition of Tasks

This repository implements ideas discussed in the CVPR2021 paper "N. Popovic, D. Pani Paudel, T. Probst, G. Sun, L. Van Gool - CompositeTasking: Understanding Images by Spatial Composition of Tasks" (https://arxiv.org/abs/2012.09030).

# Abstract


# Requirements
This project is implemented using Python and the PyTorch Deep Learning framework. Following libraries are used:

* pytorch 1.7.1+cu110
* torchvision 0.8.2+cu110
* pytorch-lightning 1.3.5
* torchmetrics 0.3.2
* numpy 1.19.2
* albumentations 1.0.0
* wandb 0.10.27
* pyyaml 5.3.1
* pillow 7.2.0
* opencv-python 4.5.2
* scipy 1.6.3.
* sckit-image 0.17.2
* matplotlib 3.3.2

The code has only been used and tested on the Linux OS. It should work on other OS as well.
The code has only been used and tested with a NVIDIA CUDA capable GPU. It should also worn with a CPU.

# Data set
The data set used in this project is the PASCAL-MT data set, which is an extension of PASCAL introduced in "K. K. Maninis et al. - Attentive Single-Tasking of Multiple Tasks". 
The data set contains 4998 training and 5105 validation images, as well as labels for the task of semantic segemntation, human body parts, surface normals, saliency and edges.
While constructing the data set, authors distilled labels for some of the tasks while others were used from PASCAL or PASCAL-Context.
For more details about the data set take a look at their paper or code: https://arxiv.org/abs/1904.08918 ; https://github.com/facebookresearch/astmt

The data set can be downloaded at the following link - https://data.vision.ee.ethz.ch/nipopovic/PASCAL_MT.zip. It contains some additional metadata and labels used in this work. Create a directory for the data set and unizip it inside it. The .zip also contains a readme.txt file with basic information about the dataset and what is contained where.



