# CompositeTasking: Understanding Images by Spatial Composition of Tasks

This repository implements ideas discussed in the CVPR2021 paper "Nikola Popovic, Danda Pani Paudel, Thomas Probst, Guolei Sun, Luc Van Gool - CompositeTasking: Understanding Images by Spatial Composition of Tasks" (https://arxiv.org/abs/2012.09030).

![CTN_illustration](https://github.com/nikola3794/composite-tasking/blob/main/images/composit_net_smaller.png)

# Abstract
We define the concept of CompositeTasking as the fusion of multiple, spatially distributed tasks, for various aspects of image understanding. 
Learning to perform spatially distributed tasks is motivated by the frequent availability of only sparse labels across tasks, and the desire for a compact multi-tasking network.
To facilitate CompositeTasking, we introduce a novel task conditioning model -- a single encoder-decoder network that performs multiple, spatially varying tasks at once.
The proposed network takes an image and a set of pixel-wise dense task requests as inputs, and performs the requested prediction task for each pixel. 
Moreover, we also learn the composition of tasks that needs to be performed according to some CompositeTasking rules, which includes the decision of where to apply which task.
It not only offers us a compact network for multi-tasking, but also allows for task-editing. 
Another strength of the proposed method is demonstrated by only having to supply sparse supervision per task. 
The obtained results are on par with our baselines that use dense supervision and a multi-headed multi-tasking design.

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
* scipy 1.6.3
* sckit-image 0.17.2
* matplotlib 3.3.2

The code has only been used and tested on the Linux OS. It should work on other OS as well.
Also, the code has only been used and tested with a NVIDIA CUDA capable GPU. It should also worn with a CPU.

# Data set
The data set used in this project is the PASCAL-MT data set, which is an extension of PASCAL introduced in "K. K. Maninis et al. - Attentive Single-Tasking of Multiple Tasks". 
The data set contains 4998 training and 5105 validation images, as well as labels for the task of semantic segemntation, human body parts, surface normals, saliency and edges.
While constructing the data set, authors distilled labels for some of the tasks while others were used from PASCAL or PASCAL-Context.
For more details about the data set take a look at their paper or code: https://arxiv.org/abs/1904.08918 ; https://github.com/facebookresearch/astmt

The data set can be downloaded at the following link - https://data.vision.ee.ethz.ch/nipopovic/PASCAL_MT.zip. It contains some additional metadata and labels used in this work. Create a directory for the data set and unizip it inside it. The .zip also contains a readme.txt file with basic information about the dataset and what is contained where.

# Code structure
The directory src contains the source code of this project and it is structured in the following way:
```
src
├── data_sets
│   ├── pascal_mt
│   ├── task_palette_gen
│   └── utils
├── experiment_configs
│   └── composite_tasking
├── systems
│   ├── composite_tasking
│   ├── multi_tasking
│   └── single_tasking
├── models
│   ├── original_implementation
│   └── blocks
├── losses
├── metrics
└── misc
```

* `/root/src/data_sets/` contains code related to data loading and processing.
    * `/root/src/data_sets/pascal_mt` contains code to load the PASCAL-MT data set. The core is inside the `data_set.py` script.
    * `/root/src/data_sets/task_palette_gen` contains the code to generate Task Palettes in various different regimes.
    * `/root/src/data_sets/utils` contains some helper files for data loading.
* `/root/src/experiment_configs` contains code related to configuring an experiment. The script `./experiment_config.py` is an abstract class which takes the experiment arguments, creates a local directory for saving model checkpoints, logging (local .txt file, tensorboard, wandb logging) and various experiment metadata snapshots. It also creates data loaders for the used data partitions, and creates a PyTorch Lightning system (contains model, optimizer, losses, metrics) and trainer (for running the training procedure). This abstract class is supposed to serve as a template which can be used in different Deep Learning projects. The user needs to implement some problem-specific functions like constructing data loaders and PyTorch Lightning systems.
    * `/root/src/experiment_configs/composite_tasking` inherites the abstract experiment configuring class `../experiment_config.py` and implements functions specific for the problem of CompositeTasking. It implements problem-specific data loaders, systems and saves problem-specific metadata to the current experiments log directory.
* `/root/src/systems` contains code related to constructing PyTorch Lightning systems. `./system.py` is an abstract

# Results
The predictions of the CompositeTasking Network which has been trained using the semantic R2 Task Palette rule can be seen in the following image:
![R2_rule_predictions](https://github.com/nikola3794/composite-tasking/blob/main/images/semantic_rule_pred.PNG)

The prediction of the of the CompositeTasking Network which has been trained using the semantic R2 Task Palette rule can be seen in the following image:
![Rnd_rule_predictions](https://github.com/nikola3794/composite-tasking/blob/main/images/random_rule_pred.png)

# Contact
Please feel free to rech out if there are any questions, suggestion or issues with the code. My e-mail is nipopovic@vision.ee.ethz.ch.

# Citation
If you use this code, please consider citing the following paper:
```bash
@inproceedings{Popovic21CompositeTasking,
      title = {CompositeTasking: Understanding Images by Spatial Composition of Tasks},
      author    = {Popovic, Nikola and
                   Paudel, ‪Danda Pani and
                   Probst, Thomas and
                   Sun, Guolei and
                   Van Gool, Luc},
      year = {2021},
      booktitle = {2021 {IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR} 2021}
}
```
