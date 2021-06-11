import os
import copy
import json

import torch

from ..experiment_config import ExperimentConfig

from ...data_sets.pascal_mt.data_set import PascalMT

from ...systems.composite_tasking.system import CompositeTaskingSystem
from ...systems.multi_tasking.system import MultiTaskingSystem
from ...systems.single_tasking.system import SingleTaskingSystem


class CompositeTaskingExperimentConfig(ExperimentConfig):

    def __init__(self):
        pass
    
    # TODO Implement experiment loading

    def init_new_exp(self, cfg, augmentation, input_transform, debug=False):
        # Initialize the general experiment class
        self._init_new_exp_generic(
            cfg=cfg, 
            augmentation=augmentation, 
            input_transform=input_transform, 
            debug=debug)
        
        # Bookkeep informationr elated to task id's and codes
        self.task_to_id_dict=self.data_sets["train"].get_task_to_id_dict(), 
        self.colourmaps=self.data_sets["train"].get_colourmaps(), 
        self.task_z_code_dict=self.data_sets["train"].get_task_z_code_dict()

        # Save task_to_id_dict if there is an experiment folder
        self._save_task_info_to_json()

    def _create_data_sets(self):
        data_sets = {}
        for which in ["train", "val"]:
            data_sets[which] = self.create_data_set(which_split=which)
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

    def evaluate(self):
        self.trainer.test(
            model=self.system,
            ckpt_path=None, # None for last, 'best' for best, or providde a path
            test_dataloaders=self.data_loaders["train"],
            verbose=True
        )
    
    def _create_system(self):
        if self.cfg["setup_cfg"]["which_system"] == "composite_tasking":
            return CompositeTaskingSystem(
                cfg=self.cfg, 
                task_to_id_dict=self.data_sets["train"].get_task_to_id_dict(), 
                colourmaps=self.data_sets["train"].get_colourmaps(), 
                task_z_code_dict=self.data_sets["train"].get_task_z_code_dict()
            )
        elif self.cfg["setup_cfg"]["which_system"] == "multi_tasking":
            return MultiTaskingSystem(
                cfg=self.cfg, 
                task_to_id_dict=self.data_sets["train"].get_task_to_id_dict(), 
                colourmaps=self.data_sets["train"].get_colourmaps(), 
                task_z_code_dict=self.data_sets["train"].get_task_z_code_dict()
            )
        elif self.cfg["setup_cfg"]["which_system"] == "single_tasking":
            # Check that a single task data-set has been defined
            assert len(self.cfg["data_set_cfg"]["task_list"]) == 1
            return SingleTaskingSystem(
                cfg=self.cfg, 
                task_to_id_dict=self.data_sets["train"].get_task_to_id_dict(), 
                colourmaps=self.data_sets["train"].get_colourmaps(), 
                task_z_code_dict=self.data_sets["train"].get_task_z_code_dict()
            )
        else:
            raise NotImplementedError

    def _save_task_info_to_json(self):        
        # Save the task code dictionary
        # Only save if an experiment directory exists
        if "exp_main_dir" in self.cfg["setup_cfg"]:

            #Save the task z code dictionary
            save_path = os.path.join(self.cfg["setup_cfg"]["exp_main_dir"], "task_z_code_dict.json")
            if not os.path.isfile(save_path):
                with open(save_path, "w") as fh:
                    json.dump(self.task_z_code_dict, fh)

            # Save the ID to task name mapping
            save_path = os.path.join(self.cfg["setup_cfg"]["exp_main_dir"], "task_to_id_dict.json")
            if not os.path.isfile(save_path):
                with open(save_path, "w") as fh:
                    json.dump(self.task_to_id_dict, fh)

            # Save the coulorumap mapping
            save_path = os.path.join(self.cfg["setup_cfg"]["exp_main_dir"], "colourmaps.json")
            if not os.path.isfile(save_path):
                with open(save_path, "w") as fh:
                    json.dump(self.colourmaps, fh)
        
    # def _load_config_for_ongoing(self):
    #     with open(os.path.join(self.load_dir, f"exp_cfg.json"), "r") as fh:
    #         cfg = json.load(fh)
    #     return cfg
