import sys
import os

import torch
import numpy as np

import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .exp_dir_management import ExpDirManagement


class ExperimentConfig:

    def __init__(self):
        pass

    def _init_new_exp_generic(self, cfg, augmentation, input_transform, debug=False):
        # Store input arguments
        self.cfg = cfg
        self.augmentation = augmentation
        self.input_transform = input_transform
        self.debug = debug

        # Check the validity of the input arguments
        self._check_input_arguments_validity_new_exp()

        # Create the experiment main directory and save code
        self.exp_main_dir = self._create_main_exp_dir()

        # If an experiment directory is created, modify the pythons print
        # function so it writes both on the console and to a log file in the dir
        self.print_to_log_file_also()

        # Save the configurations in the experiment directory
        self._save_config()

        # Print the description of the new experiment
        self.print_new_experiment_description()

        # Create data set partitions
        self.data_sets = self._create_data_sets()

        # Create dataloaders for each partition
        self.data_loaders = self._create_data_loaders()

        # Create the system
        # (containing the models, losses, metrics, optimizers, lr schedulers, ...)
        self.system = self._create_system()

        # Create trainer to train the system
        self.trainer = self._create_trainer()
    
    def _create_data_sets(self):
        raise NotImplementedError

    def _create_data_loaders(self):
        raise NotImplementedError

    def _create_system(self):
        raise NotImplementedError   

    def evaluate(self):
        raise NotImplementedError

    def _check_input_arguments_validity_new_exp(self):
            """
            Check are the initialization input arguments valid
            """
            # Check config file validity
            if not isinstance(self.cfg, dict):
                raise AssertionError  
                
            if not isinstance(self.cfg["setup_cfg"], dict):
                raise ValueError   
            if not isinstance(self.cfg["data_set_cfg"], dict):
                raise ValueError
            if not isinstance(self.cfg["model_cfg"], dict):
                raise ValueError
            if not isinstance(self.cfg["training_cfg"], dict):
                raise ValueError  

            if not isinstance(self.cfg["setup_cfg"]["project_name"], str):
                raise ValueError

            if not isinstance(self.cfg["setup_cfg"]["which_system"], str):
                raise ValueError

            if self.cfg["setup_cfg"]["exp_root_dir"] is not None:
                if not isinstance(self.cfg["setup_cfg"]["exp_root_dir"], str):
                    raise ValueError
            if self.cfg["setup_cfg"]["code_root_dir"] is not None:
                if not isinstance(self.cfg["setup_cfg"]["code_root_dir"], str):
                    raise ValueError  
        
    def train(self):
        # Trains the model in the train split 
        # and validates on the validation split
        self.trainer.fit(
            model=self.system,
            train_dataloader=self.data_loaders["train"],
            val_dataloaders=self.data_loaders["val"],
        )  

    def _create_trainer(self):
        n_available_gpus = torch.cuda.device_count()
        # ------------------------------------> TODO <------------------------------------
        #                        Unused parameters which PL offeres
        # ------------------------------------> TODO <------------------------------------
        # TODO Check out arguments:
        # TODO accumulate_grad_batches, amp_backend, amp_level, auto_scale_batch_size
        # TODO auto_lr_find, auto_select_gpus, deterministic, default_root_dir
        # TODO num_tpu_cores, overfit_batches, process_position, test_percent_check
        # TODO track_grad_norm, train_percent_check, val_percent_check, truncated_bptt_steps
        # TODO weights_save_path, weights_summary, replace_sampler_ddp, prepare_data_per_node
        trainer_arguments = {
            # Should speed up when the input size doesnt change
            "benchmark": self.cfg["training_cfg"]["pl_benchmark"], 
            # Number of GPUs to train on or which GPUs to train on
            "gpus": 0 if n_available_gpus == 0 else -1,
            # Number of GPU nodes for distributed training
            "num_nodes": 1, # ----------> TODO Parametrize when needed <-------------
            # Which distributed backend to use (multi-gpu) 
            # One needs to be careful with dp and ddp2 because they devide the
            # current batch into subsets, while the ddp and ddp_spawn devide
            # the whole dataset into subsets and each process gets the data subset
            "accelerator": None if n_available_gpus <= 1 else "ddp", 
            # Plugins allow you to connect arbitrary backends, precision libraries, SLURM, etc…
            # TODO 
            "plugins": None,
            # Runs n batch of train, test and val to find any bugs
            "fast_dev_run": False, 
            # Sanity check runs n validation batches before starting the training routine.
            "num_sanity_val_steps": 0,
            # 0 means no clipping 
            "gradient_clip_val": 0 if "pl_grad_clip_val" not in self.cfg["training_cfg"] else self.cfg["training_cfg"]["pl_grad_clip_val"], 
            # None, 'all' and 'min_max' options,
            "log_gpu_memory": "all", 
            # How often to refresh progress bar (in steps)
            "progress_bar_refresh_rate": self.cfg["training_cfg"]["pl_progress_bar_refresh_rate"], 
            # How often to add logging rows (does not write to disk)
            "log_every_n_steps": self.cfg["training_cfg"]["pl_log_every_n_steps"], 
            # Writes logs to disk every n steps
            "flush_logs_every_n_steps": self.cfg["training_cfg"]["pl_flush_logs_every_n_steps"], 
            # Force training for at least these many epochs
            "min_epochs": self.cfg["training_cfg"]["pl_min_epochs"], 
            # Stop after x epochs
            "max_epochs": self.cfg["training_cfg"]["pl_max_epochs"], 
            # Force training for at least these number of steps
            "min_steps": None, 
            # Stop training after this number of steps
            "max_steps": None,
            # Full precision (32) or half precision (16) 
            "precision": self.cfg["training_cfg"]["pl_precision"], 
            # Reload the dataloader every epoch
            "reload_dataloaders_every_epoch": False,
            # 
            "resume_from_checkpoint": None, # --------> TODO <----------
            # Run val loop after every epoch
            "check_val_every_n_epoch": self.cfg["training_cfg"]["pl_check_val_every_n_epoch"], 
            # How often within one training epoch to check the validation set
            "val_check_interval": 1.0, 
            # How much of training dataset to use each epoch
            "limit_train_batches": self.cfg["training_cfg"]["pl_limit_train_batches"], 
            # How much of val dataset to use each epoch
            "limit_val_batches": self.cfg["training_cfg"]["pl_limit_val_batches"], 
            # How much of test dataset use each epoch
            "limit_test_batches": 1.0, 
            # Enable synchronization between batchnorm layers across all GPUs
            "sync_batchnorm": self.cfg["training_cfg"]["pl_sync_batchnorm"], 
        }
        # Logger
        trainer_arguments["logger"] = self._create_pl_logger()
        # Profiler
        trainer_arguments["profiler"] = self._create_pl_profiler()

        # Override some arguments while in debug mode:
        if self.debug:
            # before start of training
            trainer_arguments["num_sanity_val_steps"] = 0
            # length of epochs
            trainer_arguments["fast_dev_run"] = False # TODO This will end everything after first "epoch"
            trainer_arguments["limit_train_batches"] = 0.01
            trainer_arguments["limit_val_batches"] = 0.01
            trainer_arguments["limit_test_batches"] = 0.01
            # length of training
            trainer_arguments["min_epochs"] = 3
            trainer_arguments["max_epochs"] = 3
            # logging frequency
            trainer_arguments["progress_bar_refresh_rate"] = 1
            trainer_arguments["log_every_n_steps"] = 1
            trainer_arguments["flush_logs_every_n_steps"] = 1
            # miscellaneous
            trainer_arguments["reload_dataloaders_every_epoch"] = False
            trainer_arguments["checkpoint_callback"] = False

        callbacks = [] 

        # Making model checkpoints
        checkpoint_callback = self._create_pl_model_checkpoint_callback()
        if checkpoint_callback:
            callbacks.append(checkpoint_callback)

        # Early stopping
        early_stop_callback = self._create_pl_early_stopping_callback()
        if early_stop_callback:
            callbacks.append(early_stop_callback)
        
        # Defined callbacks
        trainer_arguments["callbacks"] = callbacks

        return pl.Trainer(**trainer_arguments) 

    def _create_pl_logger(self):
        if "exp_main_dir" in self.cfg["setup_cfg"]:
            logger = []

            # Tensorboard logger
            tb_logger = TensorBoardLogger(
                save_dir=self.cfg["setup_cfg"]["exp_main_dir"],
                name="tensorboard",
                version="",
                # TODO For this to work, define self.example_input_array attribute in model
                log_graph=False, 
            )
            logger.append(tb_logger)

            # Wandb logger
            wandb_name = f"{os.sep}".join(os.path.normpath(self.cfg["setup_cfg"]["exp_main_dir"]).split(os.sep)[-2:])
            wandb_name = f"{self.cfg['setup_cfg']['which_system']}{os.sep}{wandb_name}"
            wandb_logger = WandbLogger(
                entity="nikola3794",
                project=self.cfg["setup_cfg"]["project_name"],
                name=wandb_name,
                save_dir=self.cfg["setup_cfg"]["exp_main_dir"],
                id=None,

            )
            # Log gradients in wandb
            wandb_logger.watch(
                self.system, 
                log='gradients', 
                log_freq=self.cfg["training_cfg"]["pl_log_every_n_steps"],
            )
            logger.append(wandb_logger)
        else:
            logger = False
        return logger
        
    def _create_pl_profiler(self):
        # Only if an experiment direcotyr exists
        if "exp_main_dir" in self.cfg["setup_cfg"]:
            prof_out_file = os.path.join(self.cfg["setup_cfg"]["exp_main_dir"], "runtime_profiling.txt")
        else:
            return None

        if self.cfg["training_cfg"]["pl_which_profiler"].lower() == "simple":
            return SimpleProfiler(
                output_filename=prof_out_file,
                extended=True
            )
        elif self.cfg["train_cfg"]["pl_which_profiler"].lower() == "advanced":
            return AdvancedProfiler(
                output_filename=prof_out_file,
                line_count_restriction = 1.0
            )
        elif self.cfg["train_cfg"]["pl_which_profiler"].lower() in ["none", ""]:
            return None
        else:
            raise NotImplementedError 
    
    def _create_pl_model_checkpoint_callback(self):
        if "exp_main_dir" in self.cfg["setup_cfg"]:
            return ModelCheckpoint(
                dirpath=self.cfg["setup_cfg"]["exp_main_dir"],
                filename=None,
                # Quantity to monitor
                monitor=None,
                verbose=True,
                # (Redundant when save_top_k is None) Always saves the model at the end of the epoch
                #save_last=True, 
                # The best k models (quantity monitored) will be saved
                save_top_k=None, 
                mode='min',
                # Save either full model (F) or just weights (T)
                save_weights_only=False, 
                #  Interval (number of epochs) between checkpoints.
                period=1, 
            )
        else:
            return None

    def _create_pl_early_stopping_callback(self):
        if self.cfg["training_cfg"]["pl_early_stop_patience"] > 0:
            return EarlyStopping(
                patience=self.cfg["training_cfg"]["pl_early_stop_patience"], 
                # Quantity to be monitored (must be logged with self.log)
                monitor='val/loss/loss_total', 
                mode='min',
                # Minimum change in the monitored quantity to qualify as an improvement
                min_delta = 0.0, 
                # Whether to crash the training if monitor is not found in the validation metrics
                strict=True, 
                verbose=True,
            )
        else:
            return None

    def _create_main_exp_dir(self):
        # The configuration should be specified
        if not self.cfg:
            raise ValueError
        
        # No directory creating while in debug mode
        if self.debug:
            return None

        # The specified root should exist
        if not os.path.isdir(self.cfg["setup_cfg"]["exp_root_dir"]):
            raise NotADirectoryError
            
        # Create the main experiment directory. 
        # Everything will be logged here.
        exp_main_dir = ExpDirManagement.create_exp_main_dir(
            exp_root_dir=self.cfg["setup_cfg"]["exp_root_dir"], 
            short_description="exp"
            )

        # Save current code snapshot in the experiment directory
        ExpDirManagement.save_code_to_zip(
            code_root_dir=self.cfg["setup_cfg"]["code_root_dir"], 
            save_dir=exp_main_dir, 
            ignore_hidden=True
            )
        
        # Save the experiment main directory to the configuration
        self.cfg["setup_cfg"]["exp_main_dir"] = exp_main_dir
        
        return exp_main_dir
        
    def _save_config(self):
        # Do not save configuration in debug mode
        if self.debug:
            return

        # Save the experiment configuration to a .json
        ExpDirManagement.save_dict_to_json(
            dictionary=self.cfg, 
            save_path=self.exp_main_dir, 
            file_name="exp_cfg.json") 
    
    def print_to_log_file_also(self):
        """
        If an experiment directory is created, modify the pythons print
        function so it writes both on the console and to a log file in the dir
        Otherwise, just write in the console
        """
        class Logger(object):
            def __init__(self, exp_main_dir):
                self.terminal = sys.stdout
                self.exp_main_dir = exp_main_dir

            def write(self, message):
                self.terminal.write(message)
                if self.exp_main_dir is not None:
                    with open(os.path.join(self.exp_main_dir, "print_log.txt"), "a", encoding = 'utf-8') as self.log:            
                        self.log.write(message)

            def flush(self):
                #this flush method is needed for python 3 compatibility.
                #this handles the flush command by doing nothing.
                #you might want to specify some extra behavior here.
                pass
        sys.stdout = Logger(exp_main_dir=self.exp_main_dir)  

    # TODO Adapt to more nesting in the configuration dict (make recursive)
    def print_new_experiment_description(self):
        """
        Print the description of the new experiment in the console and the log files.
        """
        cfg_msg = ""
        cfg_msg += "[Experiment configuration]\n"
        for config_key in self.cfg.keys():
            if not isinstance(self.cfg[config_key], dict):
                cfg_msg += f"{config_key}: {self.cfg[config_key]}\n"
            else:
                cfg_msg += f"\n{config_key}\n"
                cfg_msg += f"------------------\n"
                for config_key_2 in self.cfg[config_key].keys():
                    cfg_msg += f"    {config_key_2}: {self.cfg[config_key][config_key_2]}\n"
        cfg_msg += "---------------------------------------------------------------------------------\n"
        cfg_msg += "Data augmentation: \n{:s}\n".format(str(self.augmentation))
        cfg_msg += "Input image transformations: \n{:s}\n".format(str(self.input_transform))
        cfg_msg += "---------------------------------------------------------------------------------\n"
        cfg_msg += "---------------------------------------------------------------------------------\n\n"
        
        print("---------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------")
        print(f"              STARTING NEW EXPERIMENT : {self.exp_main_dir}")
        print(cfg_msg)
        print("---------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------")
        
        return cfg_msg

    def _which_device(self):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu" 
