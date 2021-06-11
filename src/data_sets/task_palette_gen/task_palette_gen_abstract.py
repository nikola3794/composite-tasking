import time

import numpy as np
import torch


class TaskPaletteGenAbstract:

    def __init__(self, cfg):
        self.cfg = cfg
        self._check_input_validity()

        # Initialzie task palette state
        self.curr_task_id_map = None
        self.curr_unique_task_ids = None

    def _check_input_validity(self):
        raise NotImplementedError

    def get_task_list(self):
        """
        List of all tasks used for the current task palette.
        """
        return self.cfg["task_list"]

    def get_task_id_map(self, **kwargs):
        """
        Retrieve a task id map according to the specified configurations. 
        """
        self._generate_task_id_map(**kwargs)
        return self.curr_task_id_map, self.curr_unique_task_ids
    
    def _generate_task_id_map(self, **kwargs):
        """
        Generate the task id map for the current iteration
        """
        raise NotImplementedError