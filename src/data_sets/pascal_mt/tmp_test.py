import sys
import os
if __name__ == "__main__":
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from data_sets.pascal_mt.data_set import * 

from torch.utils.data import DataLoader


# The purpose of this script is to test whether the data set
# class is implemented correctly.
# This script creates an instance of the data set class
# and loads data points according to the specifications.

if __name__ == "__main__":
    IMG_SIZE = (256, 256)

    task_palette_cfg = {
        "change_map_after": 1,
        "task_list": ["seg", "parts", "edges", "saliency", "normals"]
        }

    data_set_cfg = {
        "palette_mode": "semantic_rule_R2",
        "data_root": "/home/nipopovic/scratch_second/workspace/cvpr2021/DATA/PASCAL_MT",
        "task_code_len": 20,
        "which_split": "val",
        "img_size": IMG_SIZE,
        "task_palette_cfg": task_palette_cfg
        }

    tr_augmentations = {
        "regular": get_tr_augmentation(img_size=IMG_SIZE),
        "normals": get_tr_augmentation_normals(img_size=IMG_SIZE),
    }

    augmentations = tr_augmentations
    augmentations = None

    trfinput = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trfinput = None

    data_set = PascalMT(cfg=data_set_cfg, augmentations=augmentations, input_transform=trfinput)

    data_loader = DataLoader(data_set, batch_size=2, shuffle=False, num_workers=0)

    for i, sample in enumerate(data_loader):
        # TODO Fix sample indicies and fix unique task ids...and display color iof images/labels
        for j in range(sample["image"].shape[0]):
            label = sample["label"][j, :, :, :]
            plt.imshow(transforms.ToPILImage()(sample["image"][j, :, :, :]))
            plt.show()
            plt.imshow(transforms.ToPILImage()(label))
            plt.show()