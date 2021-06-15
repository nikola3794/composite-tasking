# Data download link:
# https://data.vision.ee.ethz.ch/nipopovic/PASCAL_MT.zip

import os

import time
import json

import torch
import torchvision
from torchvision import transforms

import numpy as np

from PIL import Image
import cv2
import scipy.io as sio
from skimage.morphology import thin
import albumentations as albu

from .metadata import *

from ..task_palette_gen.task_palette_gen_mosaics import TaskPaletteGenMosaics
from ..task_palette_gen.task_palette_gen_single_task import TaskPaletteGenSingleTask
from ..task_palette_gen.task_palette_gen_semantic_map import TaskPaletteGenSemanticMap

from ..utils.helpers import label_to_classes


class PascalMT(torch.utils.data.Dataset):

    def __init__(self, cfg, augmentation, input_transform):
        self.cfg = cfg
        self.augmentation = augmentation
        self.input_transform = input_transform
        
        # Check validity of input arguments of this method
        self._check_input_validity()

        # Load the path of the data split directory 
        # (Every split is a .txt file with a list of its image names)
        splits_dir = os.path.join(self.cfg["data_root"], DATA_SPLIT_PATH)

        # Load the requested image paths (depends on which split requested)
        images_dir = os.path.join(self.cfg["data_root"], IMG_PATH)
        split_file = os.path.join(splits_dir, self.cfg["which_split"].rstrip('\n') + '.txt')
        with open(os.path.join(split_file), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.img_paths = [os.path.join(images_dir, x + ".jpg") for x in file_names]

        if self.cfg["palette_mode"] != "no_tasks":
            # Load the requested label paths (depends on which split is requested)
            label_dirs = {
                "seg": os.path.join(self.cfg["data_root"], SEG_PATH),
                "parts": os.path.join(self.cfg["data_root"], PARTS_PATH),
                "edges": os.path.join(self.cfg["data_root"], EDGES_PATH),
                "normals": os.path.join(self.cfg["data_root"], NORMALS_PATH),
                "saliency": os.path.join(self.cfg["data_root"], SALIENCY_PATH)
                }
            self.label_paths = {}
            for task in label_dirs.keys():
                extension = ".mat" if task == "edges" else ".png"
                self.label_paths[task] = [os.path.join(label_dirs[task], x + extension) for x in file_names]
                assert (len(self.img_paths) == len(self.label_paths[task]))

            # Create task encoding for conditioning
            self.task_z_code_dict = self._create_task_codes()

            # Create colourmaps for segmentation and parts
            # Different colors are used to visualize different calsses
            self.colourmaps = self._load_colourmaps()

        self.task_palette_generator = None
        self.requires_pre_load_seg = False
        if isinstance(self.cfg["task_palette_cfg"], dict):
            # This is needed for every task palette generator
            self.cfg["task_palette_cfg"]["img_size"] = self.cfg["img_size"]
            self.cfg["task_palette_cfg"]["task_to_id"] = TASK_TO_ID
        if self.cfg["palette_mode"] in ["equal_4_mosaic", "rnd_4_mosaic", "rnd_all_mosaic"]:
            # Create a task map generator
            self.cfg["task_palette_cfg"]["palette_type"] = self.cfg["palette_mode"]
            self.task_palette_generator = TaskPaletteGenMosaics(cfg=self.cfg["task_palette_cfg"])    
        if self.cfg["palette_mode"] in ["semantic_rule_R2", "semantic_rule_R3"]:
            # Create a task map generator
            self.cfg["task_palette_cfg"]["palette_type"] = self.cfg["palette_mode"]
            self.cfg["task_palette_cfg"]["colourmap_seg"] = self.colourmaps["seg"]
            self.requires_pre_load_seg = True
            self.task_palette_generator = TaskPaletteGenSemanticMap(cfg=self.cfg["task_palette_cfg"])       
        elif self.cfg["palette_mode"] in ["single_task"]:
            # Create a task map generator
            self.task_palette_generator = TaskPaletteGenSingleTask(cfg=self.cfg["task_palette_cfg"])
        
        # Load the overlap between the PASCAL and NYU classes to only use normals distilled at those locations
        # Also creates a list of paths to labels from the pascal-context dataset
        self.normals_valid_classes, self.context = self._normals_loading_info(file_names=file_names)

        # If a semantic rule is selected, returns information about the classes
        # which are still used in certain tasks...because some might get completely discarded
        self.used_seg_cls_ids, self.used_parts_cls_ids = self._semantic_rule_info()

    def _check_input_validity(self):
        """
        Check validity of input arguments upon class instance creation.
        """
        assert isinstance(self.cfg, dict)

        assert isinstance(self.cfg["data_root"], str)
        assert os.path.isdir(self.cfg["data_root"])

        assert self.cfg["which_split"] in ["train", "val", "trainval"]

        if isinstance(self.cfg["img_size"], list):
            self.cfg["img_size"] = tuple(self.cfg["img_size"])
        assert isinstance(self.cfg["img_size"], tuple)
        assert len(self.cfg["img_size"]) == 2

        assert isinstance(self.cfg["palette_mode"], str)
        assert isinstance(self.cfg["task_palette_cfg"], dict)
        
        assert isinstance(self.cfg["task_code_len"], int)

        if self.augmentation is not None:
            assert ("regular" in self.augmentation) and ("normals" in self.augmentation)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if self.task_palette_generator is not None:
            return self._getitem_task_palette(index)
        if self.cfg["palette_mode"] == "no_tasks":
            return self._getitem_just_img(index)
        elif self.cfg["palette_mode"] == "all_tasks":
            return self._getitem_all_tasks(index)
        else:
            raise NotImplementedError

    def _getitem_task_palette(self, index, load_all_existing_labels=False):
        # Pre-load segmentation label if necessary
        kwargs = {}
        if self.requires_pre_load_seg:
            kwargs["seg_label"] = self._load_label(index=index, task="seg")

        # Load the Task Palette and id of used tasks
        task_palette, unique_task_ids = self.task_palette_generator.get_task_id_map(**kwargs)

        # Get the specified datapoint
        return self._getitem_regular(
            index=index, 
            task_palette=task_palette, 
            unique_task_ids=unique_task_ids,
            seg_label=kwargs["seg_label"] if self.requires_pre_load_seg else None,
            load_all_existing_labels=load_all_existing_labels,
        )

    def _getitem_regular(self, index, task_palette, unique_task_ids, seg_label=None, load_all_existing_labels=False):
        # Which labels to load and return
        if load_all_existing_labels:
            task_ids_to_load = []
            for task in self.get_task_list():
                task_ids_to_load.append(TASK_TO_ID[task])
        else:
            task_ids_to_load = unique_task_ids

        # Load the image and its original size before it was downscaled
        image, orig_cv2_size = self._load_image(index=index)

        # Load label, and a dictionary needed for data augmentation
        loaded_labels, additional_augment_tasks = self._load_all_needed_labels(
            index=index, 
            unique_task_ids=task_ids_to_load,
            seg_label=seg_label
        )
        
        # Apply appropriate augmentation to the label mask for every task
        image, loaded_labels = self._apply_data_augmentation(
            image=image,
            loaded_labels=loaded_labels, 
            unique_task_ids=task_ids_to_load, 
            additional_augment_tasks=additional_augment_tasks
        )

        # Convert the input image into a tesor and apply transformations if specified
        image = transforms.ToTensor()(image)
        if self.input_transform is not None:
            image = self.input_transform(image)

        # Convert all task label maps to tensors
        for task_id in task_ids_to_load:
            task = ID_TO_TASK[task_id]
            loaded_labels[task] = transforms.ToTensor()(loaded_labels[task])

        # Mark images that do not contain a human in them, so they dont get evaluated in the parts metric
        valid_parts_img = self._which_images_are_relevant_for_parts(
            loaded_labels=loaded_labels, 
            unique_task_ids=task_ids_to_load
        )

        # Make a zero-one vector where the ones tell which tasks have been requested for this data point
        used_tasks = self._which_tasks_were_used(unique_task_ids=task_ids_to_load)

        # Construt the data point dictionary
        data_point = {
            "image": image,
            "used_tasks": used_tasks,
            "img_name": os.path.basename(self.img_paths[index]),
            "orig_cv2_size": orig_cv2_size,
            "valid_parts_img": valid_parts_img,
        }
        
        # TODO <--------------------------------------------------
        # If all labels are to be also returned in the normal mode
        # it needs to be made sure that each label gets some filler (null) values
        # for batch elements where that task is not lodded at all.
        # Otherwise the default collate function will break.
        # TODO <--------------------------------------------------
        if load_all_existing_labels:
            data_point["loaded_labels"] = loaded_labels

        if task_palette is not None:
            # Construct the fina label acording to the task palette
            data_point["label"] = self._construct_final_label(
                loaded_labels=loaded_labels, 
                task_palette=task_palette, 
                unique_task_ids=unique_task_ids
            )
            data_point["task_palette"] = task_palette

        return data_point 

    def _getitem_all_tasks(self, index):
        # Return a datapoint with all labels loaded 
        # in a standard multi-task fashion
        return self._getitem_regular(
            index=index, 
            task_palette=None, 
            unique_task_ids=None, 
            load_all_existing_labels=True,
            seg_label=None,
        )

    def _getitem_just_img(self, index):
        # Load the image and its original size before it was downscaled
        image, orig_cv2_size = self._load_image(index=index)
        
        # Apply appropriate augmentation to the label mask for every task
        image, _ = self._apply_data_augmentation(image=image)

        # Convert the input image into a tesor and apply transformations if specified
        image = transforms.ToTensor()(image)
        if self.input_transform is not None:
            image = self.input_transform(image)

        # Construt the data point dictionary
        data_point = {
            "image": image,
            "orig_cv2_size": orig_cv2_size,
            "img_name": os.path.basename(self.img_paths[index]),
        }

        return data_point

    def _load_image(self, index):
        # Load the image and its original size
        image = cv2.imread(self.img_paths[index])
        orig_cv2_size = torch.tensor(image.shape, dtype=torch.int32)
        # Change to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image
        image = cv2.resize(image, (self.cfg["img_size"][1], self.cfg["img_size"][0]),interpolation=cv2.INTER_CUBIC)

        return image, orig_cv2_size
    
    def _load_all_needed_labels(self, index, unique_task_ids, seg_label=None):
        """
        Load labels for all requested tasks
        """
        loaded_labels = {}
        additional_augment_tasks = {}
        for task_id in unique_task_ids:
            task = ID_TO_TASK[task_id]
            # Signalize that it is a mask type label so that 
            # the augmentation fucntion knows what to do with the label
            additional_augment_tasks[task] = "mask"

            if task == "seg" and seg_label is not None:
                # If the segmentation label was already loaded  (i.e. to construct the semantic rule),
                # use it so that it is not loaded twice.
                label = seg_label
            else:
                # Load the current label
                label = self._load_label(index=index, task=task)

            loaded_labels[task] = label 

        return loaded_labels, additional_augment_tasks

    def _apply_data_augmentation(self, image, loaded_labels=None, unique_task_ids=None, additional_augment_tasks=None):
        """
        Apply appropriate augmentation to the label mask for every task
        """
        if self.augmentation is None:
            return image, loaded_labels

        # If normals task is present, affine transformations shouldnt be used in augmentation
        if TASK_TO_ID["normals"] in unique_task_ids:
            augmentation = self.augmentation["normals"]
        else:
            augmentation = self.augmentation["regular"]
        augmentation = albu.Compose(augmentation, additional_targets=additional_augment_tasks)

        # Pack the data for the augmeanttion function
        aug_data = {"image": image}
        if loaded_labels is not None:
            for task in additional_augment_tasks:
                aug_data[task] = loaded_labels[task]

        # Apply data augmentation
        sample = augmentation(**aug_data)

        image = sample["image"]
        # Unpack augmented data
        for s in sample:
            if s == "image":
                continue
            else:
                loaded_labels[s] = sample[s]

        return image, loaded_labels

    def _construct_final_label(self, loaded_labels, task_palette, unique_task_ids):
        # Construct the fina label that combines all label maps according to the task id map
        label_final = torch.zeros((3, self.cfg["img_size"][0], self.cfg["img_size"][1]), dtype=torch.float32)
        for task_id in unique_task_ids:
            task = ID_TO_TASK[task_id]
            label_final = torch.where(task_palette==task_id, loaded_labels[task], label_final)
        label_final = label_final.detach()

        return label_final

    def _which_images_are_relevant_for_parts(self, loaded_labels, unique_task_ids):
        # Signalize images that do not contain a human in them, so they dont get evaluated in the parts metric
        if TASK_TO_ID["parts"] in unique_task_ids:
            # Background class is [0, 0, 0] ... all other classes contain at least one non-zero element in the 3d vector
            if (len(torch.unique(loaded_labels["parts"])) == 1) and (torch.unique(loaded_labels["parts"])[0].item() == 0):
                valid_parts_img = torch.tensor(0, dtype=torch.int8)
            else:
                valid_parts_img = torch.tensor(1, dtype=torch.int8)
        else:
            valid_parts_img = torch.tensor(-1, dtype=torch.int8)
        
        return valid_parts_img

    def _which_tasks_were_used(self, unique_task_ids):
        # Make a zero-one vector where the ones tell which tasks have been requested for this data point
        used_tasks = torch.zeros(len(TASK_TO_ID), dtype=torch.int8)
        for task_id in unique_task_ids:
            used_tasks[task_id] += 1 

        return used_tasks    

    def get_idx_of_img_name(self, img_name):
        """
        Get the datapoint index of an image with the specified name
        """
        for idx in range(self.__len__()):
            if os.path.basename(self.img_paths[idx]) == img_name:
                return idx
        return None

    def _load_label(self, index, task):
        """
        Load the label for the specified task/datapoint.
        """
        if task == "edges":
            label = self._load_edge(index)
        elif task == "normals":
            label = self._load_normals(index)
        elif task == "saliency":
            label = self._load_saliency(index)
        elif task in ["seg", "parts"]:
            label = self._load_seg_parts(index, task=task)
        else:
            raise NotImplementedError

        return label
    
    def _load_edge(self, index):
        # Load semantical regions
        _tmp = sio.loadmat(self.label_paths["edges"][index])
        # Resize label
        _tmp = cv2.resize(_tmp['LabelMap'], (self.cfg["img_size"][1], self.cfg["img_size"][0]), interpolation=cv2.INTER_NEAREST)
        # Detect edges of semantical regions
        _edge = cv2.Laplacian(_tmp, cv2.CV_64F)
        _edge = (255 * thin(np.abs(_edge) > 0)).astype(np.uint8)
        # Repeat the same edge label across all 3 channels
        _edge = np.repeat(np.expand_dims(_edge, axis=2), repeats=3, axis=2)
        return _edge

    def _load_saliency(self, index):
        # Load saliency label
        label = cv2.imread(self.label_paths["saliency"][index]) 
        # Convert to RGB
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        #label = (255 * (label > 0.5*255.0)).astype(np.uint8)

        # Resize the label
        label = cv2.resize(label, (self.cfg["img_size"][1], self.cfg["img_size"][0]), interpolation=cv2.INTER_NEAREST)

        return label

    def _load_normals(self, index):
        # Load the normals label
        label = cv2.imread(self.label_paths["normals"][index]) 
        # Change to RGB
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Return to original domain of [-1, 1] (aved as [0,255] image)
        label = 2.0 * label / 255.0 - 1.0

        # Put zeros on locations of non-PASCAL classes
        context_labels = sio.loadmat(self.context[index])
        context_labels = context_labels['LabelMap']
        _normals = np.zeros(label.shape, dtype=np.float32)
        for x in np.unique(context_labels):
            if x in self.normals_valid_classes:
                _normals[context_labels == x, :] = label[context_labels == x, :]
        label = _normals

        # Resize label
        label = cv2.resize(label, (self.cfg["img_size"][1], self.cfg["img_size"][0]), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        
        return label

    def _load_seg_parts(self, index, task):
        # Load label
        label = cv2.imread(self.label_paths[task][index]) 
        # Convert to RGB
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        # Resize label
        label = cv2.resize(label, (self.cfg["img_size"][1], self.cfg["img_size"][0]), interpolation=cv2.INTER_NEAREST)
        
        return label

    def get_task_list(self):
        return self.cfg["task_palette_cfg"]["task_list"]
    
    def get_img_size(self):
        return self.cfg["img_size"]
    
    # TODO <--------------------------------------------------
    # This is currently hardcoded for 5 tasks
    # TODO <--------------------------------------------------
    def _create_task_codes(self):
        """
        Generates vector codes for all tasks in the Pascal VOC dataset.
        This codes are going to be used as inputs to tell the network 
        which tasks needs to be solved where.
        """
        task_z_code_dict = {}
        c_len = self.cfg["task_code_len"]
        for task in TASK_LIST:
            # Input code for the corresponding task
            z = torch.zeros([5*c_len], dtype=torch.float32)
            z = torch.nn.parameter.Parameter(z, requires_grad=False)
            if task == 'seg':
                z[:1*c_len] = 1
                z[1*c_len:] = 0
            elif task == 'parts':
                z[:1*c_len] = 0
                z[1*c_len:2*c_len] = 1
                z[2*c_len:] = 0
            elif task == 'edges':
                z[:2*c_len] = 0
                z[2*c_len:3*c_len] = 1
                z[3*c_len:] = 0
            elif task == 'normals':
                z[:3*c_len] = 0
                z[3*c_len:4*c_len] = 1
                z[4*c_len:] = 0
            elif task== 'saliency':
                z[:4*c_len] = 0
                z[4*c_len:5*c_len] = 1
            if task in self.get_task_list():
                task_z_code_dict[task] = z
        return torch.nn.ParameterDict(task_z_code_dict)
    
    # def get_task_codes(self, keys):
    #     # Choose the type of returned dictionary keys
    #     if not keys in ["id", "name"]:
    #         raise NotImplementedError

    #     task_z_code_dict = self.task_z_code_dict.copy()
    #     new_task_z_code_dict = {}
    #     for k in task_z_code_dict:
    #         if keys == "id":
    #             new_k = TASK_TO_ID[k]
    #         elif keys == "name":
    #             new_k = k
    #         else:
    #             raise NotImplementedError
    #         new_task_z_code_dict[new_k] = task_z_code_dict[k]  
    #     return new_task_z_code_dict
    
    def _load_colourmaps(self):
        colourmap = {}
        # Load class colors for segmentation task
        cmap = np.loadtxt(os.path.join(self.cfg["data_root"], COLOURMAPS_PATH, 'colourmap_seg21.txt') ,delimiter='\t')
        colourmap['seg'] = np.zeros_like(cmap)
        colourmap['seg'][:,0] = cmap[:,2]
        colourmap['seg'][:,1] = cmap[:,1]
        colourmap['seg'][:,2] = cmap[:,0]
        colourmap['seg'] = torch.from_numpy(colourmap['seg']).type(torch.float32)

        # Load class colors for parts task
        cmap = np.loadtxt(os.path.join(self.cfg["data_root"], COLOURMAPS_PATH, 'colourmap_parts7.txt') ,delimiter='\t')
        colourmap['parts'] = np.zeros_like(cmap)
        colourmap['parts'][:,0] = cmap[:,2]
        colourmap['parts'][:,1] = cmap[:,1]
        colourmap['parts'][:,2] = cmap[:,0]
        colourmap['parts'] = torch.from_numpy(colourmap['parts']).type(torch.float32)

        return colourmap

    def _semantic_rule_info(self):
        # Makes a list of all seg/parts classes that are being used in the selected mode (selected rule)
        if self.cfg["palette_mode"] == "semantic_rule_R3":
            used_seg_cls_ids = []
            used_parts_cls_ids = [1, 2, 3, 4, 5, 6]
        elif self.cfg["palette_mode"] == "semantic_rule_R2":
            used_seg_cls_ids = []
            for cls_i in VOC_GROUPS_2["seg"]:
                used_seg_cls_ids.append(VOC_CLS_NAME_TO_ID[cls_i])
            used_parts_cls_ids = [1, 2, 3, 4, 5, 6]
        else:
            used_seg_cls_ids = None
            used_parts_cls_ids = None

        return used_seg_cls_ids, used_parts_cls_ids

    def _normals_loading_info(self, file_names):
        # Load the overlap between the PASCAL and NYU classes, 
        # because losses/metrics for surface normals will be only evaluated on those classes.
        # Labels were constructed with a network trained on the NYU dataset.
        with open(os.path.join(self.cfg["data_root"], 'json/nyu_classes.json')) as f:
            cls_nyu = json.load(f)
        with open(os.path.join(self.cfg["data_root"], 'json/context_classes.json')) as f:
            cls_context = json.load(f)
        # Find common classes between the two datasets to use for normals
        normals_valid_classes = []
        for cl_nyu in cls_nyu:
            if cl_nyu in cls_context and cl_nyu != 'unknown':
                normals_valid_classes.append(cls_context[cl_nyu])
        # Custom additions due to incompatibilities
        normals_valid_classes.append(cls_context['tvmonitor'])

        # Context dir: Used for Normals loading
        context_dir = os.path.join(self.cfg["data_root"], 'pascal-context', 'trainval')
        context = [os.path.join(context_dir, x + ".mat") for x in file_names]

        return normals_valid_classes, context

    def get_colourmaps(self):
        return self.colourmaps

    def get_id_to_task_dict(self):
        return ID_TO_TASK

    def get_task_to_id_dict(self):
        return TASK_TO_ID

    def get_task_z_code_dict(self):
        return self.task_z_code_dict
