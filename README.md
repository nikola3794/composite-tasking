# CompositeTasking: Understanding Images by Spatial Composition of Tasks

This repository implements ideas discussed in the CVPR2021 paper ["Nikola Popovic, Danda Pani Paudel, Thomas Probst, Guolei Sun, Luc Van Gool - CompositeTasking: Understanding Images by Spatial Composition of Tasks"](https://arxiv.org/abs/2012.09030).

![CTN_illustration](https://github.com/nikola3794/composite-tasking/blob/main/images/composit_net_smaller.png)

# Abstract
We define the concept of CompositeTasking as the fusion of multiple, spatially distributed tasks, for various aspects of image understanding. 
Learning to perform spatially distributed tasks is motivated by the frequent availability of only sparse labels across tasks, and the desire for a compact multi-tasking network.
To facilitate CompositeTasking, we introduce a novel task conditioning model -- a single encoder-decoder network that performs multiple, spatially varying tasks at once.
The proposed network takes an image and a set of pixel-wise dense task requests as inputs, and performs the requested prediction task for each pixel. 
One strength of the proposed method is demonstrated by only having to supply sparse supervision per task.
The obtained results are on par with our baselines that use dense supervision and a multi-headed multi-tasking design.

# Requirements
This project is implemented using Python3 and the PyTorch Deep Learning framework. Following libraries are used:

* python 3.8.5
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

The code has only been used and tested on the Linux OS, but it should work on other OS as well.
Also, the code has only been used and tested with a NVIDIA CUDA capable GPU, but it should also work with a CPU.

# Data set
The data set used in this project is the PASCAL-MT data set, which is an extension of PASCAL for the purpose of Multi-Tasking introduced in "K. K. Maninis et al. - Attentive Single-Tasking of Multiple Tasks". 
The data set contains 4998 training and 5105 validation images, as well as labels for the task of semantic segemntation, human body parts, surface normals, saliency and edges.
While constructing the data set, authors distilled labels for some of the tasks while others were used from PASCAL or PASCAL-Context.
For more details about the data set take a look at their [paper](https://arxiv.org/abs/1904.08918) or [code](https://github.com/facebookresearch/astmt).

The data set can be downloaded at the [following link](https://data.vision.ee.ethz.ch/nipopovic/PASCAL_MT.zip). It contains some additional metadata and labels used in this work. The .zip also contains a readme.txt file with basic information about the dataset and what is contained where. Choose a directory where you want to store the data set and unzip it (unzipping will create the main data set directory with all the content inside it). The path to the root of the data set folder will need to be specified in the training and evaluation scripts.

# Code structure
The directory `/root/src/` contains the source code of this project and it is structured in the following way:
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
    * `/root/src/data_sets/pascal_mt/` contains code which loads the PASCAL-MT data set. The core code is in the `./data_set.py` script.
    * `/root/src/data_sets/task_palette_gen/` contains the code to generate Task Palettes in various different regimes.
    * `/root/src/data_sets/utils/` contains some helper code for data loading.

* `/root/src/experiment_configs/` contains code related to configuring an experiment. The script `./experiment_config.py` is an abstract class which loads the experiment configuration arguments, creates a local directory for saving model checkpoints and various experiment metadata and performs logging (local .txt files, tensorboard logging, wandb logging). This abstract class is supposed to serve as a template which can be used in different Deep Learning projects. It is left to the user to implement problem-specific functions such as data loaders for the used data partitions, a system (which should implement models, optimizers, losses, metrics and training/validation/test loops) and a trainer (responsible for running the training, valdiation and test loops of the system).
    * `/root/src/experiment_configs/composite_tasking/` inherites the abstract experiment configuring class `../experiment_config.py` and implements functions specific for the problem of CompositeTasking. It implements CompositeTasking-specific data loaders, systems, trainers and saves CompositeTasking-specific metadata to the current experiment's log directory.

* `/root/src/systems/` contains code related to constructing PyTorch Lightning systems. `./system.py` is an abstract class which constructs a system by taking the provided experiment arguments and constructs the model, optimizer, losses and metrics. It also implements the training, validation and test loops which are called in the PyTorch Lightning trainer. It is intended to be implemented as an abstract system which can be used for other DeepLearning projects (it actually contains a few CompositeTasking-specific functionallities, which can be removed for further re-use). Since different experiments, even inside the same problen, can have different models, optimizers, losses and metric updates, they should be implemented in the actual system which inherites this class.
    * `/root/src/systems/composite_tasking/` inherites `../system.py` and implements the system for running CompositeTasking experiments.
    * `/root/src/systems/multi_tasking/` inherites `../system.py` and implements the system for running Multi-tasking experiments.
    * `/root/src/systems/single_tasking/` inherites `../system.py` and implements the system for running Single-Tasking experiments.

* `/root/src/models/` contains code related to constructing model architectures. Scripts which construct the whole models can be found in this directory.
    * `/root/src/models/blocks/` contains various building-blocks of the defined architectures.
    * `/root/src/models/original_implementation/` contains original implementations of the models used in the CompositeTasking paper.

* `/root/src/losses/` contains code related to loss function calculations.

* `/root/src/metrics/` contains code related to metric calculations.

* `/root/src/misc/` contains some miscellaneous helper scripts.

The directories `/root/run_training/` and `/root/run_evaluation/` contain scripts which call the core code from `/root/src/` in order to run experiments and conduct evaluations. They will be commented later.

# Train models
Use the following script `/root/run_training/train.py` to train models. It uses a command line argparser for specifying the experiment configuration. One option is to provide the argument values to the script when calling it from the command line. Another option is to specify the argument values in a .yaml file and provide the path of the .yaml file when calling the training script (along with argument values not specified in the .yaml file). For more details on the descriptions and expected values for most of the input arguments, take a look at the following file `/root/run_training/configs/composite_tasking_paper/main_hyperparameters.yaml` and it's comments. The directory `/root/run_training/configs/composite_tasking_paper/` contains .yaml files specifying configurations of the most important experiments from the CompositeTasking paper.

An example of the training scripts call from the `/root/` directory of the project:
```
python3 -u run_training/train.py \
--config_file_path=PATH_TO_YAML_CONFIG_FILE \
--data_root_dir=PATH_TO_DATASET_ROOT_DIR \
--code_root_dir=PATH_TO_CODE_ROOT_DIR \
--exp_root_dir=PATH_TO_EXPERIMENT_ROOT_DIR
```
`PATH_TO_EXPERIMENT_ROOT_DIR` should be a path to a directory, where each new experiment is going to create a logging directory (model checkpoints, log files, current training metadata). `PATH_TO_CODE_ROOT_DIR` is specified so that a current snapshot of the code is saved in the experiment's logging directory.

The progress of the experiment should be shown on the print console. If specified, the progress is also going to be shown in the current experiment directory's .txt file, tensorboard directory, as well as in wandb online.

Currently, multi-GPU training does not work. There is an issue with the metric syncing.

# Evaluate models
Use the following script `/root/run_evaluation/evaluate.py` to evaluate models. It also uses a command line argparser for specifying the evaluation configuration, just like in the training script. The argument values can also be given through the command line, .yaml file, or mixed.
For more details on the descriptions and expected values for most of the input arguments, take a look at the following file `/root/run_evaluation/configs/composite_tasking_paper/main_hyperparameters.yaml` and it's comments. The directory `/root/run_evaluation/configs/composite_tasking_paper/` contains .yaml files specifying configurations of the most important evaluations from the CompositeTasking paper.

An example of the evaluation scripts call from the `/root/` directory of the project:
```
python3 -u run_evaluation/evaluate.py \
--config_file_path=PATH_TO_YAML_CONFIG_FILE \
--checkpoint_path=PATH_TO_CHECKPOINT_FILE \
--data_root_dir=PATH_TO_DATASET_ROOT_DIR \
```
`PATH_TO_CHECKPOINT_FILE` should be a path to the checkpoint file to be evaluated. The structure of the experiment's logging directory, where the checkpoint is file is saved, should not be changed. The evaluation script will extract the path of the experiment's logging directory and load some other metadata files saved inside it. The result of the evaluation is going to be displayed in the print console and saved in a new .txt file generated in the experiment's directory.

Currently, multi-GPU training does not work. There is an issue with the metric syncing.

For the proper evaluation of edge detecetion, which is reported in the CompositeTasking paper, the [seism repository](https://github.com/jponttuset/seism) needs to be used (evaluation code is written in MATLAB and it is required to have it installed). In order to do so, edge predictions need to be saved as .png images with the same size as the input image and provided to the seism evaluation protocol along with the labels contained in the dataset. This can be done by uncommenting the `save_edge_preds()` call in the `training_step()` method of `/root/src/systems/system.py`, when necessary. If there is trouble with running the seism repository, take a look at [this repository](https://github.com/nikola3794/edge-evaluation-PASCAL-MT-tmp), which is a quick and temporary solution with instructions on how to call the seism repository.

# Example results
The predictions of the CompositeTasking Network which has been trained using the semantic R2 Task Palette rule can be seen in the following image:
![R2_rule_predictions](https://github.com/nikola3794/composite-tasking/blob/main/images/semantic_rule_pred.PNG)

The prediction of the of the CompositeTasking Network which has been trained using the completely random Task Palette rule  Rrnd can be seen in the following image:
![Rnd_rule_predictions](https://github.com/nikola3794/composite-tasking/blob/main/images/random_rule_pred.png)

# Contact
Please feel free to reach out if there are any questions, suggestion or issues with the code. My e-mail is nipopovic@vision.ee.ethz.ch.

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
