import torch
import numpy as np

from data_sets.VOC.data_set import VOCDataSet
from data_sets.VOC.metadata import ID_TO_TASK, TASK_TO_ID
from utils.get_dirs import get_dirs


# This script is used to compute some staitstics of the data set

if __name__ == "__main__":
    data_set_cfg = {
        "mode": "all_labels",
        "data_root": get_dirs(which_cluster="aws")["data_root_dir"],
        "task_code_len": 20,
        "which_split": "trainval",
        "img_size": (512, 512),
        "mask_normals": True,
        }

    data_set = VOCDataSet(cfg=data_set_cfg, augmentations=None, input_transform=None)

    edges_perc = []
    saliency_perc = []
    normals_perc = []
    seg_perc = []
    parts_perc = []
    parts_count = []
    for idx in range(data_set.__len__()):
        if idx % 20 == 0:
            print(idx)
            
        sample = data_set.__getitem__(idx)
        seg = sample["all_labels"]["seg"]
        parts = sample["all_labels"]["parts"]
        normals = sample["all_labels"]["normals"]
        saliency = sample["all_labels"]["saliency"]
        edges = sample["all_labels"]["edges"]

        seg_perc.append(torch.mean(1.0*(torch.sum(seg, 0) != 0)).item())
        parts_perc.append(torch.mean(1.0*(torch.sum(parts, 0) != 0)).item())
        normals_perc.append(torch.mean(1.0*(torch.sum(normals, 0) != 0)).item())
        saliency_perc.append(torch.mean(1.0*(torch.sum(saliency, 0) != 0)).item())
        edges_perc.append(torch.mean(1.0*(torch.sum(edges, 0) != 0)).item())
        
    
    print(f"Segmentation content %: {np.mean(seg_perc) * 100.0}")
    print(f"Segmentation content # images: {np.sum(1*(np.array(seg_perc)>0))}/{data_set.__len__()}")
    print(" ")

    print(f"Parts content %: {np.mean(parts_perc) * 100.0}")
    print(f"Parts content # images: {np.sum(1*(np.array(parts_perc)>0))}/{data_set.__len__()}")
    print(" ")

    print(f"Normals content %: {np.mean(normals_perc) * 100.0}")
    print(f"Normals content # images: {np.sum(1*(np.array(normals_perc)>0))}/{data_set.__len__()}")
    print(" ")

    print(f"Edges content %: {np.mean(edges_perc) * 100.0}")
    print(f"Edges content # images: {np.sum(1*(np.array(edges_perc)>0))}/{data_set.__len__()}")
    print(" ")

    print(f"Saliency content %: {np.mean(saliency_perc) * 100.0}")
    print(f"Saliency content # images: {np.sum(1*(np.array(saliency_perc)>0))}/{data_set.__len__()}")
    print(" ")

