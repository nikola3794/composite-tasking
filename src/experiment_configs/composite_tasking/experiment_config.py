import os
import copy
import json
import shutil

import torch

from ..experiment_config import ExperimentConfig

from ...data_sets.pascal_mt.data_set import PascalMT

from ...systems.composite_tasking.system import CompositeTaskingSystem
from ...systems.multi_tasking.system import MultiTaskingSystem
from ...systems.single_tasking.system import SingleTaskingSystem


class CompositeTaskingExperimentConfig(ExperimentConfig):

    def __init__(self):
        pass

    def init_new_exp(self, cfg, augmentation, input_transform, debug=False):
        # Initialize the general experiment class
        self._init_new_exp(
            cfg=cfg, 
            augmentation=augmentation, 
            input_transform=input_transform, 
            debug=debug
        )
        
        # Bookkeep informationr related to task id's and task codes
        self.task_to_id_dict=self.data_sets["train"].get_task_to_id_dict()
        self.colourmaps=self.data_sets["train"].get_colourmaps()
        self.task_z_code_dict=self.data_sets["train"].get_task_z_code_dict()

        # Save task_to_id_dict if there is an experiment folder
        self._save_task_info_to_json()

    def load_existing_exp(self, checkpoint_path, load_trainer_ckp, cfg_overwrite, augmentation, input_transform, debug=False):
        # Initialize the general experiment class
        self._load_existing_exp(
            checkpoint_path=checkpoint_path,
            load_trainer_ckp=load_trainer_ckp,
            cfg_overwrite=cfg_overwrite, 
            augmentation=augmentation, 
            input_transform=input_transform, 
            debug=debug
        )
    
    def _load_system(self, checkpoint_path):
        # Load informationr related to task id's and task codes
        self.task_z_code_dict, self.task_to_id_dict, self.colourmaps = self._load_task_info_from_json()

        # Get the appropriate system
        system, _ = self._get_system_constructor()
        # Load the system from a checkpoint
        system = system.load_from_checkpoint(
            checkpoint_path,
            task_z_code_dict = self.task_z_code_dict,
            task_to_id_dict = self.task_to_id_dict,
            colourmaps = self.colourmaps,
            used_seg_cls_ids=self.data_sets["train"].used_seg_cls_ids,
            used_parts_cls_ids=self.data_sets["train"].used_parts_cls_ids,
        )
        
        # TODO <--------------------------------
        # The system will automatically load all __init__ arguments from the checpoints saved with save_hyperparameters()
        # Since these contain some parameters  that do not necessarily describe the model
        # (like data_set parameters), forecefully overwrite them with the provided ones.
        # TODO <--------------------------------
        # TODO Is this necessary?
        system_cfg_tmp = self.cfg.copy()
        system_cfg_tmp["model_cfg"] = system.cfg["model_cfg"].copy()
        system.cfg = system_cfg_tmp

        # TODO <--------------------------------
        # When constructing the system only the cfg dictinonary is saved as a torch lightning
        # hyperparameter. When loading the system, other arguments need to be passed, and torch lightning
        # automatically considers them as hyperparameters and fails later because of that.
        # Here I have hardcoded to remove all unwanted arguments from the hyperparameter dictionary
        # TODO <--------------------------------
        keys = ["task_to_id_dict", "colourmaps", "task_z_code_dict", 
                 "used_seg_cls_ids", "used_parts_cls_ids"]
        for k in keys:
            if k in system._hparams_initial:
                del system._hparams_initial[k]
            if k in system._hparams:
                del system._hparams[k]

        return system

    def _create_data_sets(self):
        data_sets = {}
        for which in ["train", "val"]:
            data_sets[which] = self.create_data_set(which_split=which)
        # PASCAL_MT only has a train and validation split, 
        # so the validation split is used as a test split
        data_sets["test"] = self.create_data_set(which_split="val")
        return data_sets

    def create_data_set(self, which_split):
        """
        Creates the specified partition of the PASCAL-MT dataset.
        """
        # Load augmentations
        if which_split == "train":
            augmentation = self.augmentation["train"]
        else:
            augmentation = self.augmentation["val"]

        # Specify how frequent to re-generate the Task Palette
        if self.cfg["data_set_cfg"]["palette_mode"] in ["equal_4_mosaic", "rnd_4_mosaic", "rnd_all_mosaic"]:
            change_palette_after = self.cfg["training_cfg"]["b_size"]
        else:
            change_palette_after = None
        # Task palette config
        task_palette_cfg = {
            "change_map_after": change_palette_after,
            "task_list": self.cfg["data_set_cfg"]["task_list"]
            }

        # Data set config
        data_set_cfg = {
            "palette_mode": self.cfg["data_set_cfg"]["palette_mode"],
            "data_root": self.cfg["data_set_cfg"]["data_root"],
            "task_code_len": self.cfg["data_set_cfg"]["task_code_len"],
            "which_split": which_split,
            "img_size": self.cfg["data_set_cfg"]["img_size"],
            "task_palette_cfg": task_palette_cfg
            }

        # Create the specified data set partition
        data_set = PascalMT(
            cfg=data_set_cfg, 
            augmentation=augmentation, 
            input_transform=self.input_transform
        )

        print(f"Created {which_split} dataset containing {data_set.__len__()} data points.")

        return data_set
    
    def _create_data_loaders(self):
        data_loaders = {}
        for which in self.data_sets:
            data_loaders[which] = self.create_data_loader(which_split=which)
        return data_loaders

    def create_data_loader(self, which_split):
        shuffle = True if which_split == "train" else False

        data_loader = torch.utils.data.DataLoader(
            self.data_sets[which_split], 
            batch_size=self.cfg["training_cfg"]["b_size"], 
            shuffle=shuffle,
            pin_memory=True, 
            num_workers=self.cfg["training_cfg"]["n_workers"],
            )
        
        return data_loader

    def _get_system_constructor(self):
        system_arg_dict = {
            "cfg": self.cfg,
            "task_to_id_dict": self.data_sets["train"].get_task_to_id_dict(),
            "colourmaps": self.data_sets["train"].get_colourmaps(), 
            "task_z_code_dict": self.data_sets["train"].get_task_z_code_dict()
        }
        if self.cfg["data_set_cfg"]["palette_mode"] in ["semantic_rule_R2", "semantic_rule_R3"]:
            system_arg_dict["used_seg_cls_ids"] = self.data_sets["val"].used_seg_cls_ids
            system_arg_dict["used_parts_cls_ids"] = self.data_sets["val"].used_parts_cls_ids

        if self.cfg["setup_cfg"]["which_system"] == "composite_tasking":
            return CompositeTaskingSystem, system_arg_dict
        elif self.cfg["setup_cfg"]["which_system"] == "multi_tasking":
            return MultiTaskingSystem, system_arg_dict
        elif self.cfg["setup_cfg"]["which_system"] == "single_tasking":
            # Check that a single task data-set has been defined
            assert len(self.cfg["data_set_cfg"]["task_list"]) == 1
            return SingleTaskingSystem, system_arg_dict
        elif self.cfg["setup_cfg"]["which_system"] == "single_tasking_parallel":
            # TODO
            raise NotImplementedError
            # return SingleTaskingSystemParallel, system_arg_dict
        else:
            raise NotImplementedError

    def _save_task_info_to_json(self):        
        # Save the task code dictionary
        # Only save if an experiment directory exists
        if self.exp_main_dir:

            # Save the config yaml file if specified
            save_path = os.path.join(self.exp_main_dir, "config_file.yaml")
            if not os.path.isfile(save_path):
                shutil.copyfile(
                    src=self.cfg["setup_cfg"]["config_file_pth"],
                    dst=save_path,
                )
                with open(save_path, "w") as fh:
                    json.dump(self.task_to_id_dict, fh)

            #Save the task z code dictionary
            save_path = os.path.join(self.exp_main_dir, "task_z_code_dict.json")
            if not os.path.isfile(save_path):
                # self.task_z_code_dict is a torch Parameter class
                # convert back to regular dictionary for saving
                task_z_codes_tmp = {}
                for k in self.task_z_code_dict:
                    task_z_codes_tmp[k] = self.task_z_code_dict[k].tolist()
                with open(save_path, "w") as fh:
                    json.dump(task_z_codes_tmp, fh)

            # Save the ID to task name mapping
            save_path = os.path.join(self.exp_main_dir, "task_to_id_dict.json")
            if not os.path.isfile(save_path):
                with open(save_path, "w") as fh:
                    json.dump(self.task_to_id_dict, fh)

            # Save the coulorumap mapping
            save_path = os.path.join(self.exp_main_dir, "colourmaps.json")
            if not os.path.isfile(save_path):
                # self.colourmaps is a torch Parameter class
                # convert back to regular dictionary for saving
                colourmaps_tmp = {}
                for k in self.colourmaps:
                    colourmaps_tmp[k] = self.colourmaps[k].tolist()
                with open(save_path, "w") as fh:
                    json.dump(colourmaps_tmp, fh)
        
    def _load_task_info_from_json(self):

        # Load the task z code dictionary
        load_path = os.path.join(self.exp_main_dir, "task_z_code_dict.json")
        assert os.path.isfile(load_path)
        # self.task_z_code_dict is a torch Parameter class
        # convert back to regular dictionary for saving
        with open(load_path, "r") as fh:
            task_z_code_dict = json.load(fh)
        # Convert to torch.nn.ParameterDict where every entry is a torch.nn.parameter.Parameter
        for k in task_z_code_dict:
            task_z_code_dict[k] = torch.nn.parameter.Parameter(torch.tensor(task_z_code_dict[k], dtype=torch.float32), requires_grad=False)
        task_z_code_dict = torch.nn.ParameterDict(task_z_code_dict)
        
        # Load the ID to task name mapping
        load_path = os.path.join(self.exp_main_dir, "task_to_id_dict.json")
        assert os.path.isfile(load_path)
        with open(load_path, "r") as fh:
            task_to_id_dict = json.load(fh)

        # Load the coulorumap mapping
        load_path = os.path.join(self.exp_main_dir, "colourmaps.json")
        assert os.path.isfile(load_path)
        with open(load_path, "r") as fh:
            colourmaps = json.load(fh)

        # Convert to torch.nn.ParameterDict where every entry is a torch.nn.parameter.Parameter
        for k in colourmaps:
            colourmaps[k] = torch.tensor(colourmaps[k], dtype=torch.float32, requires_grad=False)

        return task_z_code_dict, task_to_id_dict, colourmaps

