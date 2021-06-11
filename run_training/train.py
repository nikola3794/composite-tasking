import sys
import os
if __name__ == "__main__":
    sys.path.append(os.getcwd())

import argparse
import yaml

import albumentations as albu

import numpy as np
import torch
from torchvision import transforms

import random

from src.misc.important_dirs import get_important_dirs

from src.experiment_configs.composite_tasking.experiment_config import CompositeTaskingExperimentConfig
from src.data_sets.pascal_mt.data_augmentations import get_tr_augmentation, get_tr_augmentation_normals


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def str_with_none(v):
        if v in ["None", "none", ""]:
            return None
        else:
            return v
    
    def int_with_none(v):
        if v in ["None", "none", ""]:
            return None
        else:
            return int(v)

    def float_with_none(v):
        if v in ["None", "none", ""]:
            return None
        else:
            return float(v)

    def str_lower_case(v):
        return v.lower()

    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    default_cfg_pth = '/home/nipopovic/Projects/composite_tasking/big_storage/code_shortcut/composite-tasking/run_training/configs/debug_config.yaml'
    parser.add_argument('--config_file_path', default=default_cfg_pth, type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    # Create the argument parser for parsing the experiment config
    parser = argparse.ArgumentParser(description='Experiment training')

    # Setup config
    # ------------
    parser.add_argument('--project_name', default="CompositeTasking", type=str)
    parser.add_argument('--which_system', default="composite_tasking_1", type=str)

    # Data set & task palette config
    # ------------------------------
    parser.add_argument('--img_size', type=int, nargs=2, default=[32, 32])
    parser.add_argument('--task_code_len', type=int, default=20)
    parser.add_argument('--palette_mode', type=str, default="rnd_4_mosaic")
    parser.add_argument('--task_list', type=str, nargs='+', default=['parts', 'seg', 'normals', 'saliency', 'edges'])
       
    # Model config    
    # ------------
    parser.add_argument('--which_model', default="composite_tasking_net_v1", type=str)

    parser.add_argument('--encoder_arch', default="resnet34", type=str)
    parser.add_argument('--encoder_pre_trained', default=True, type=str2bool)
    parser.add_argument('--decoder_arch', default="resnet34", type=str)

    parser.add_argument('--n_fc_z_map', default=6, type=int)
    parser.add_argument('--latent_w_dim', default=128, type=int)

    parser.add_argument('--which_cond', default="task_composition_v1", type=str)
    parser.add_argument('--cond_cfg_txt', default="cond_batch1x1", type=str)

    parser.add_argument('--skip_conv_ks', default=1, type=int)
    parser.add_argument('--dec_conv_ks', default=3, type=int)
    parser.add_argument('--net_output_ch', default=3, type=int)

    # Training config    
    # ---------------
    parser.add_argument('--optimizer', default="adam", type=str_lower_case)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--lr_div_encoder', default=100.0, type=float)    
    parser.add_argument('--lr_scheduler', default="step_lr", type=str_lower_case)
    parser.add_argument('--lr_scheduler_step', default=30, type=int)
    parser.add_argument('--lr_scheduler_patience', default=12, type=float)
    parser.add_argument('--lr_scheduler_factor', default=0.3, type=float)
    parser.add_argument('--pl_grad_clip_val', default=0.0, type=float)
    parser.add_argument('--wd', default=0.00001, type=float)
    parser.add_argument('--sgd_momentum', default=0.9, type=float)
    parser.add_argument('--sgd_nesterov_momentum', default=True, type=bool)

    parser.add_argument('--pl_max_epochs', default=100, type=int)
    parser.add_argument('--pl_min_epochs', default=100, type=int)
    parser.add_argument('--pl_early_stop_patience', default=-1, type=int)
    parser.add_argument('--b_size', default=2, type=int)
    parser.add_argument('--n_workers', default=0, type=int)

    parser.add_argument('--seg_l_w', type=float, default=3.0)
    parser.add_argument('--parts_l_w', type=float, default=4.0)
    parser.add_argument('--edges_l_w', type=float, default=50.0)
    parser.add_argument('--saliency_l_w', type=float, default=8.0)
    parser.add_argument('--normals_l_w', type=float, default=4.0)

    parser.add_argument('--seg_l_focal_gamma', type=float, default=2.0)

    parser.add_argument('--pl_benchmark', default=True, type=bool)
    parser.add_argument('--pl_precision', default=32, type=int)
    parser.add_argument('--pl_sync_batchnorm', default=True, type=bool)
    parser.add_argument('--pl_which_profiler', default="simple", type=str)

    parser.add_argument('--pl_progress_bar_refresh_rate', default=100, type=int)
    parser.add_argument('--pl_log_every_n_steps', default=100, type=int)
    parser.add_argument('--pl_flush_logs_every_n_steps', default=100, type=int)
    parser.add_argument('--pl_check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--pl_limit_train_batches', default=1.0, type=float)
    parser.add_argument('--pl_limit_val_batches', default=1.0, type=float)
    
    # Miscellaneous
    # -------------
    parser.add_argument('--data_root_dir', default="/home/nipopovic/scratch_second/workspace/cvpr2021/DATA/PASCAL_MT", type=str)
    parser.add_argument('--code_root_dir', default="path", type=str)
    parser.add_argument('--exp_root_dir', default="/srv/beegfs02/scratch/switchtasks/data/NIKOLA/Experiments", type=str)

    parser.add_argument('--debug', default=True, type=str2bool, metavar='T/F')
    parser.add_argument('--message', default="experiment message..", type=str)

    # Parse the arguments
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config_file_path:
        assert os.path.isfile(args_config.config_file_path)
        with open(args_config.config_file_path, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    args.img_size = tuple(args.img_size)
    
    args.message = args.message.replace("_", " ")

    return args, args_config.config_file_path

def main():
    # Parse arguments
    args, config_file_path = parse_args()

    # Make sure single-tasking regime has only 1 task specified
    if args.palette_mode == "single_task":
        assert len(args.task_list) == 1
    
    # Load the augmentations and pack appropriately
    tr_augmentation = {
        "regular": get_tr_augmentation(img_size=args.img_size),
        "normals": get_tr_augmentation_normals(img_size=args.img_size),
    }
    val_augmentation = None
    augmentation = {
        "train": tr_augmentation,
        "val": val_augmentation
    }

    # Input normalization
    input_transform = transforms.Compose(
        [transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
        ]
    )

    # Setup config
    # ------------
    setup_cfg = {
        "project_name": args.project_name,
        "which_system": args.which_system,
        "exp_root_dir": args.exp_root_dir,
        "code_root_dir": args.code_root_dir,
        "config_file_pth": config_file_path
    }

    # Data set & task palette config
    # ------------------------------
    data_set_cfg = {
        "data_root": args.data_root_dir,
        "img_size": args.img_size,
        "palette_mode": args.palette_mode,
        "task_code_len": args.task_code_len,
        "task_list": args.task_list,
    }
           
    # Model config    
    # ------------
    model_cfg = {
        "which_model": args.which_model,

        "encoder_arch": args.encoder_arch,
        "encoder_pre_trained": args.encoder_pre_trained,
        "decoder_arch": args.decoder_arch,

        "n_fc_z_map": args.n_fc_z_map,
        "latent_w_dim": args.latent_w_dim,

        "which_cond": args.which_cond,
        "cond_cfg_txt": args.cond_cfg_txt,

        "skip_conv_ks": args.skip_conv_ks,
        "dec_conv_ks": args.dec_conv_ks,
        "net_output_ch": args.net_output_ch,
    }

    # Training config    
    # ---------------
    training_cfg = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "lr_div_encoder": args.lr_div_encoder,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_step": args.lr_scheduler_step,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "pl_grad_clip_val": args.pl_grad_clip_val,
        "wd": args.wd,
        "sgd_momentum": args.sgd_momentum,
        "sgd_nesterov_momentum": args.sgd_nesterov_momentum,

        "pl_max_epochs": args.pl_max_epochs,
        "pl_min_epochs": args.pl_min_epochs,
        "pl_early_stop_patience": args.pl_early_stop_patience,
        "b_size": args.b_size,
        "n_workers": args.n_workers,

        "seg_l_w": args.seg_l_w,
        "parts_l_w": args.parts_l_w,
        "edges_l_w": args.edges_l_w,
        "saliency_l_w": args.saliency_l_w,
        "normals_l_w": args.normals_l_w,

        "seg_l_focal_gamma": args.seg_l_focal_gamma,

        "pl_benchmark": args.pl_benchmark,
        "pl_precision": args.pl_precision,
        "pl_sync_batchnorm": args.pl_sync_batchnorm,
        "pl_which_profiler": args.pl_which_profiler,

        "pl_progress_bar_refresh_rate": args.pl_progress_bar_refresh_rate,
        "pl_log_every_n_steps": args.pl_log_every_n_steps,
        "pl_flush_logs_every_n_steps": args.pl_flush_logs_every_n_steps,
        "pl_check_val_every_n_epoch": args.pl_check_val_every_n_epoch,
        "pl_limit_train_batches": args.pl_limit_train_batches,
        "pl_limit_val_batches": args.pl_limit_val_batches,
    }

    # Merge all different configs into one config dictionary
    cfg = {
        "setup_cfg": setup_cfg,
        "data_set_cfg": data_set_cfg,
        "model_cfg": model_cfg,
        "training_cfg": training_cfg,
    } 

    exp_config = CompositeTaskingExperimentConfig()
    exp_config.init_new_exp(
        cfg=cfg, 
        augmentation=augmentation, 
        input_transform=input_transform, 
        debug=args.debug
    )

    exp_config.train()
    
if __name__ == "__main__":
    main()