import torch
import numpy as np


def label_to_classes(label, cls_centroids):
    """
    Transforms the (N_batch, d, H, W) tensor with class labels 
    encoded as d-dimensional vecotrs in [0,255]^d into (N_batch, H, W) 
    tensor where class labels are encoded with their class id.
    """
    dev = label.device

    # Create a tensor [B,1,H,W] such that at every location it contains the
    # number of the target class (i.e. at (B=1,H=100,W=50) class 5 is present) 
    class_labels = -1.0 * torch.ones(
        (label.shape[0], label.shape[2], label.shape[3]), 
        dtype=torch.int, 
        device=dev
    )
    for i in range(0, cls_centroids.shape[0]):
        # Centroid of the current class
        b = cls_centroids[i, :].to(dev)
        # Distance of current class centroid to labels
        diff = label.permute(0,2,3,1) - b
        dis = torch.norm(diff, p=2, dim=3, keepdim=False)
        # All spatial locations with the centroid for class number i get the value i 
        class_labels[dis < 1e-4] = i

    return class_labels.type(torch.long)


def pred_to_classes(pred, cls_centroids):
    """
    Transforms the (N_batch, d, H, W) tensor with class predictions encoded 
    as d-dimensional vecotrs in [0,255]^d into (N_batch, H, W)  tensor 
    where class labels are encoded with their class id.
    Also output the d-dimensional segmentation class centroids for the predicted class ids.
    """
    dev = pred.device

    # l2 is a [B,N_C,H,W] tensor. For each [b,:,h,w] location in the tensor the distance of the networks prediction at that 
    # location is computed to the N_C centroids. The centroids represent the codes of each predicted class
    # (i.e. the network predicted at location (N=1, :, H=100,W=50) [2.0, 50.8, 4.1] for a 3-dimensional representation)
    # (i.e. centroid for class 5 is [0,	64,	128] ..... then l2[B=1,N_c=5,H=100,W=50] is L2([2.0, 50.8, 4.1], [0, 64, 128])) )
    l2 = torch.zeros((pred.shape[0], cls_centroids.shape[0], pred.shape[2], pred.shape[3]), device=dev)
    for i in range(0, cls_centroids.shape[0]):
        # Code of the current class
        b = cls_centroids[i, :].unsqueeze(0).unsqueeze(2).unsqueeze(3).to(dev)
        b = b.repeat(pred.shape[0], 1, pred.shape[2], pred.shape[3])
        l2[:,i,:,:] = torch.norm((pred-b), p=2, dim=1) 
    
    # The class is chosen as the closest class centroid with respect to the predicted position
    pred_labels = torch.argmin(l2, dim=1).type(torch.long).to(dev)
    
    # For every choosen class, select its centroid and pack into a tensor
    pred_centroids = cls_centroids[pred_labels].permute(0, 3, 1, 2).to(dev)

    return pred_labels, pred_centroids
