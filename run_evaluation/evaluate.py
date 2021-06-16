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
    default_cfg_pth = '/home/nipopovic/Projects/composite_tasking/big_storage/code_shortcut/composite-tasking/run_evaluation/configs/debug_config.yaml'
    parser.add_argument('--config_file_path', default=default_cfg_pth, type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    # Create the argument parser for parsing the experiment config
    parser = argparse.ArgumentParser(description='Experiment training')

    parser.add_argument('--checkpoint_path', default="CompositeTasking", type=str)

    # Setup config
    # ------------
    parser.add_argument('--which_system', default="composite_tasking_1", type=str)

    # Data set & task palette config
    # ------------------------------
    parser.add_argument('--img_size', type=int, nargs=2, default=[32, 32])
    # parser.add_argument('--task_code_len', type=int, default=20)
    parser.add_argument('--palette_mode', type=str, default="rnd_4_mosaic")
    parser.add_argument('--task_list', type=str, nargs='+', default=['parts', 'seg', 'normals', 'saliency', 'edges'])
       
    # Training config    
    # ---------------
    parser.add_argument('--b_size', default=2, type=int)
    parser.add_argument('--n_workers', default=0, type=int)

    parser.add_argument('--pl_precision', default=32, type=int)

    parser.add_argument('--pl_progress_bar_refresh_rate', default=100, type=int)
    parser.add_argument('--pl_log_every_n_steps', default=100, type=int)
    parser.add_argument('--pl_flush_logs_every_n_steps', default=100, type=int)
    parser.add_argument('--pl_logger_use', default=False, type=str2bool)
    
    # Miscellaneous
    # -------------
    parser.add_argument('--data_root_dir', default="SPECIFY PATH", type=str)
    parser.add_argument('--code_root_dir', default="SPECIFY PATH", type=str)
    # parser.add_argument('--exp_root_dir', default="SPECIFY PATH", type=str)

    parser.add_argument('--debug', default=True, type=str2bool, metavar='T/F')

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
        "which_system": args.which_system,
    }

    # Data set & task palette config
    # ------------------------------
    data_set_cfg = {
        "data_root": args.data_root_dir,
        "img_size": args.img_size,
        "palette_mode": args.palette_mode,
        "task_list": args.task_list,
    }
           
    # Model config    
    # ------------
    model_cfg = None

    # Training config    
    # ---------------
    training_cfg = {
        "b_size": args.b_size,
        "n_workers": args.n_workers,

        "pl_precision": args.pl_precision,

        "pl_progress_bar_refresh_rate": args.pl_progress_bar_refresh_rate,
        "pl_log_every_n_steps": args.pl_log_every_n_steps,
        "pl_flush_logs_every_n_steps": args.pl_flush_logs_every_n_steps,
        "pl_logger_use": args.pl_logger_use,
    }

    # Merge all different configs into one config dictionary
    cfg_overwrite = {
        "setup_cfg": setup_cfg,
        "data_set_cfg": data_set_cfg,
        "model_cfg": model_cfg,
        "training_cfg": training_cfg,
    } 


    exp_config = CompositeTaskingExperimentConfig()
    exp_config.load_existing_exp(
        checkpoint_path=args.checkpoint_path, 
        load_trainer_ckp=False,
        cfg_overwrite=cfg_overwrite, 
        augmentation=augmentation, 
        input_transform=input_transform, 
        debug=True
    )


    exp_config.evaluate()
    

if __name__ == "__main__":
    main()