import time

import numpy as np
import torch

from .task_palette_gen_abstract import TaskPaletteGenAbstract


class TaskPaletteGenSingleTask(TaskPaletteGenAbstract):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # Generate the task palette of the signle task rule
        # This stayed fixed, and it is always the same task
        # TODO Add an options to change the single task later with a method?
        self._generate_single_task_id_map() 

    def _check_input_validity(self):
        assert isinstance(self.cfg, dict)
        
        if isinstance(self.cfg["img_size"], list):
            self.cfg["img_size"] = tuple(self.cfg["img_size"])
        assert isinstance(self.cfg["img_size"], tuple)
        assert len(self.cfg["img_size"]) == 2

        assert isinstance(self.cfg["task_list"], list)
        assert len(self.cfg["task_list"]) == 1

        assert isinstance(self.cfg["task_to_id"], dict)

    def _generate_task_id_map(self, **kwargs):
        """
        No need to generate. It is already generated in the beginning.
        """
        return

    def _generate_single_task_id_map(self):
        """
        Task map consisting on one task id at every spatial location.
        """
        # Task id
        self.curr_unique_task_ids = [self.cfg["task_to_id"][self.cfg["task_list"][0]]]
        # Save the task map in the class instance state so that it can be retrieved later
        self.curr_task_id_map = torch.ones(self.cfg["img_size"], dtype=torch.int8) * self.curr_unique_task_ids[0]