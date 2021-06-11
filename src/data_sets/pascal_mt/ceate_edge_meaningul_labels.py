import sys
import os
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from data_sets.pascal_mt.data_set import PascalMT
import scipy.io as sio
import cv2
import numpy as np

import matplotlib.pyplot as plt

NUM_PASCAL_CONTEXT_CLS = 459

def main():
    # TODO <----------------------------------------------------
    # Specify the path where to save helper files which are 
    # going to serve as labels for evaluating the edge estimation 
    # results of sematical rules R2 and R3
    # TODO <----------------------------------------------------
    new_edge_path_root = "save_path"

    task_map_cfg = {
        "img_size": (256, 256),
        "task_list": ["edges"],
        "change_map_after": None,
        }

    data_set_cfg = {
        "mode": "single_task",
        "data_root": "/srv/beegfs02/scratch/switchtasks/data/NIKOLA/DATA/PASCAL_MT",
        "task_code_len": 20,
        "which_split": "val",
        "img_size": (256, 256),
        "task_map_cfg": task_map_cfg,
        }

    data_set = PascalMT(cfg=data_set_cfg, augmentations=None, input_transform=None)

    for idx in range(data_set.__len__()):
        # Load detailed semantic regions which are used to construct edge labels
        curr_edge_path = data_set.label_paths["edges"][idx]
        _tmp = sio.loadmat(curr_edge_path)
        label_map = _tmp['LabelMap'].copy()
        all_curr_labels = np.unique(label_map)

        # Load the semantic segmentaion labels from PASCAL-MT
        sem_seg = cv2.imread(data_set.label_paths["seg"][idx]) \
        # Every class has a 3d embedding
        # The background class is the only one which sums all those dimensions to 0
        sem_seg = np.sum(sem_seg, 2)

        # Take the detailed semantic regions and choose a class id
        # which is not present and put it everywhere where PASCAL-MT segmentation
        # has a label other than the background
        # In that was, if you construct edge labels from the new file,
        # there will only be edges in the background, since the edges
        # are actually the borders of the detailed semantic regions
        new_label = 24
        assert new_label not in all_curr_labels
        label_map[sem_seg!=0] = new_label 
        
        # Save the new detailed semantic regions.
        # When we construct edge labels from this new file, we will only have edges
        # where the PASCAL-MT segemntation task has a label background, which is according
        # to the task request rules R2 and R2
        new_edge_path = os.path.join(new_edge_path_root, os.path.basename(curr_edge_path))
        _tmp['LabelMap'] = label_map
        sio.savemat(new_edge_path, _tmp, do_compression=True)

if __name__ == "__main__":
    main()