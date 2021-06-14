import time

import numpy as np
import torch
from torchvision import transforms

from .task_palette_gen_abstract import TaskPaletteGenAbstract

from ..utils.helpers import label_to_classes

from ..pascal_mt.metadata import *


class TaskPaletteGenSemanticMap(TaskPaletteGenAbstract):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # Which semantic rule is being used?
        if self.cfg["palette_type"] == "semantic_rule_R2":
            self.semantic_rule = VOC_GROUPS_2
        elif self.cfg["palette_type"] == "semantic_rule_R3":
            self.semantic_rule = VOC_GROUPS_3
        else:
            raise NotImplementedError

    def _check_input_validity(self):
        assert isinstance(self.cfg, dict)
        
        if isinstance(self.cfg["img_size"], list):
            self.cfg["img_size"] = tuple(self.cfg["img_size"])
        assert isinstance(self.cfg["img_size"], tuple)
        assert len(self.cfg["img_size"]) == 2

        assert isinstance(self.cfg["palette_type"], str)
        assert self.cfg["palette_type"] in ["semantic_rule_R2", "semantic_rule_R3"]

        assert isinstance(self.cfg["task_to_id"], dict)

        assert "colourmap_seg" in self.cfg
        assert torch.is_tensor(self.cfg["colourmap_seg"])

    def _generate_task_id_map(self, seg_label, **kwargs):
        """
        Generate the task id map for the current iteration
        """
        
        # Load the segmentation label to determin which other labels to load and where to put them
        seg_label_tesnor = transforms.ToTensor()(seg_label)
        seg_target_label = label_to_classes( # [0,1] --> [0,255]
            label=seg_label_tesnor.unsqueeze(0)*255.0, 
            cls_centroids=self.cfg["colourmap_seg"]
        ) 
        curr_class_ids = torch.unique(seg_target_label).tolist()

        # List of task labels that need to be loaded
        unique_task_ids = []
        for task in TASK_LIST:
            intersect = [cls_name for cls_name in self.semantic_rule[task] if VOC_CLS_NAME_TO_ID[cls_name] in curr_class_ids]
            if intersect:
                unique_task_ids.append(TASK_TO_ID[task])

        self.curr_unique_task_ids = unique_task_ids

        # Construct the task palette with respect to the VOC segmentation class rules
        task_id_map = torch.zeros((self.cfg["img_size"][0], self.cfg["img_size"][1]), dtype=torch.float32)
        for class_id in curr_class_ids:
            # Find the task that corresponds to this class
            class_name = VOC_CLS_ID_TO_NAME[class_id]
            corresponding_task = None
            for task_id in unique_task_ids:
                task = ID_TO_TASK[task_id]
                if class_name in self.semantic_rule[task]:
                    assert corresponding_task is None
                    corresponding_task = task
            
            # Skip if no corresponding task
            if corresponding_task is None:
                raise AssertionError
            else:
                corresponding_task_id = TASK_TO_ID[corresponding_task]

            # Construct a task ID map
            task_id_map = torch.where(seg_target_label.squeeze(0) == class_id, corresponding_task_id*torch.ones_like(task_id_map), task_id_map)

        self.curr_task_id_map = task_id_map