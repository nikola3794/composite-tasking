import time

import numpy as np
import torch

from .task_palette_gen_abstract import TaskPaletteGenAbstract


class TaskPaletteGenMosaics(TaskPaletteGenAbstract):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # Helpers
        self.cnt = None

        # To avoid different data loader workers ahving the same random seed 
        # and thus randomly sampling the same tasks regions for every worker
        np.random.seed(int(time.time())) 

    def _check_input_validity(self):
        assert isinstance(self.cfg, dict)
        
        if isinstance(self.cfg["img_size"], list):
            self.cfg["img_size"] = tuple(self.cfg["img_size"])
        assert isinstance(self.cfg["img_size"], tuple)
        assert len(self.cfg["img_size"]) == 2

        assert isinstance(self.cfg["palette_type"], str)
        assert self.cfg["palette_type"] in ["equal_4_mosaic", "rnd_4_mosaic", "rnd_all_mosaic"]

        assert isinstance(self.cfg["task_list"], list)
        if self.cfg["palette_type"] == "equal_4_mosaic":
            assert len(self.cfg["task_list"]) == 4

        if self.cfg["palette_type"] in ["equal_4_mosaic"]:
            assert self.cfg["change_map_after"] is None
        elif self.cfg["palette_type"] in ["rnd_4_mosaic", "rnd_all_mosaic"]:
            assert isinstance(self.cfg["change_map_after"], int)
        else:
            raise NotImplementedError

        assert isinstance(self.cfg["task_to_id"], dict)

    def _generate_task_id_map(self, **kwargs):
        """
        Retrieve a task id map according to the specified configurations. 
        Change the current saved one if the coutner reaches the self.cfg["change_map_after"] parameter
        """
        # When method is called for the first time. Generate a task map like specified
        if self.cnt is None:
            self._generate_mosaic_task_id_map()
            self.cnt = 0

        # If self.cfg["change_map_after"] parameter is given, re-generate the task map when necessary
        if self.cfg["change_map_after"] is not None:
            assert self.cfg["change_map_after"] >= self.cnt
            if self.cfg["change_map_after"] == self.cnt:
                self._generate_mosaic_task_id_map()  
                self.cnt = 0

        # Increment the counter
        self.cnt += 1
    
    def _generate_mosaic_task_id_map(self):
        """
        Generate a task id map according to the specified configurations.
        """
        if self.cfg["palette_type"] == "equal_4_mosaic":
            self._generate_equal_4mosaic_id_map()
        elif self.cfg["palette_type"] == "rnd_4_mosaic":
            self._generate_rnd_4mosaic_id_map()
        elif self.cfg["palette_type"] == "rnd_all_mosaic":
            self._generate_rnd_all_mosaic_id_map()
        else:
            raise NotImplementedError

    def _generate_equal_4mosaic_id_map(self):
        """
        Divide the target task id map into 4 equal size rectangular regions in the image (divison point in image center).
        """

        # Choose the center as the division point
        h_center = int(self.cfg["img_size"][0]/2)
        w_center = int(self.cfg["img_size"][1]/2)

        # Save the list of unique tasks used in the map
        self.curr_unique_task_ids = [self.cfg["task_to_id"][self.cfg["task_list"][i]] for i in [0, 1, 2, 3]]

        # Create the task map
        task_id_map = self._generate_4mosaic_id_map(
            h_center=h_center, 
            w_center=w_center, 
            curr_unique_task_ids=self.curr_task_id_map
        )

        #Save the task map in the class instance state so that it can be retrieved later
        self.curr_task_id_map = task_id_map

    def _generate_rnd_4mosaic_id_map(self):
        """
        Divide the target tasks id map randomly into 4 rectangular regions in the image (with a random divison point).
        """

        # Randomly choose a division point
        # Torch used for random generation, 
        # because only torch has a different random seed in each DataLoader worker process
        h_center = torch.randint(int(self.cfg["img_size"][0]/4), int(3*self.cfg["img_size"][0]/4), (1,))[0].item()
        w_center = torch.randint(int(self.cfg["img_size"][0]/4), int(3*self.cfg["img_size"][0]/4), (1,))[0].item()

        # Shuffle the order of tasks and select first 4
        # Torch used for random generation, 
        # because only torch has a different random seed in each DataLoader worker process
        rnd_perm = torch.randperm(len(self.cfg["task_list"]))[:4]

        # Save the list of unique tasks used in the map
        self.curr_unique_task_ids = [self.cfg["task_to_id"][self.cfg["task_list"][i]] for i in rnd_perm]

        # Create the task map
        task_id_map = self._generate_4mosaic_id_map(
            h_center=h_center, 
            w_center=w_center, 
            curr_unique_task_ids=self.curr_task_id_map
        )

        #Save the task map in the class instance state so that it can be retrieved later
        self.curr_task_id_map = task_id_map
    
    def _generate_4mosaic_id_map(self, h_center, w_center, curr_unique_task_ids):
        # Create the task map. Instead of string task names use defined numerical task ids.
        # If the number of tasks is less than 4 repeat some
        task_id_map = -1 * torch.ones(self.cfg["img_size"], dtype=torch.int8)
        task_id_map[:h_center, :w_center] = self.curr_unique_task_ids[0]
        task_id_map[:h_center, w_center:] = self.curr_unique_task_ids[1]
        if len(self.curr_unique_task_ids) == 2:
            task_id_map[h_center:, :w_center] = self.curr_unique_task_ids[0]
            task_id_map[h_center:, w_center:] = self.curr_unique_task_ids[1]
        else:
            task_id_map[h_center:, :w_center] = self.curr_unique_task_ids[2]
            if len(self.curr_unique_task_ids) == 3:
                task_id_map[h_center:, w_center:] = self.curr_unique_task_ids[0]
            else:
                task_id_map[h_center:, w_center:] = self.curr_unique_task_ids[3]
        return task_id_map

    def _generate_rnd_all_mosaic_id_map(self):
        """
        Divide the target tasks id map randomly into 4 rectangular regions in the image (with a random divison point).
        """
        # Save the task map in the class instance state so that it can be retrieved later
        self.curr_task_id_map = torch.randint(low=0, high=len(self.cfg["task_list"]), size=self.cfg["img_size"], dtype=torch.int8)
        # Save the list of unique tasks used in the map
        self.curr_unique_task_ids = torch.unique(self.curr_task_id_map).tolist()
