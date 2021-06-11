import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

import math


class CosineSImilarityLoss(nn.Module):
    """
    Computes the cosine similarity loss.
    The target will have value 0 at spatial locations which should not be taken into consideration.
    """
    def __init__(self, normalize_vec=True, zero_discard=True, reduction="mean"):
        super().__init__()
        
        self.normalize_vec = normalize_vec
        self.zero_discard = zero_discard
        self.reduction = reduction

        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target, select_mask=None):
        # select_mask is a mask that tells on which spatial locations
        # to compute the loss.
        # If it is None, take into consideration the whole prediction.
        if select_mask == None:
            select_mask = torch.ones_like(pred).to(pred.device) == 1

        # TODO For now asserts that every channel (dim=1) makes the same
        # TODO select choice in one spatial location i.e. (B,:,H,W) = same_val.
        # TODO This is hardcoded for 3 channel outputs
        assert torch.sum(torch.sum(select_mask,axis=1)==1) == 0
        assert torch.sum(torch.sum(select_mask,axis=1)==2) == 0

        # Reshape the tensor to shape (-1,).
        select_mask = torch.reshape(select_mask.all(dim=1), (-1,))

        if self.zero_discard:
            # Do not take into consideration the pred spatial locations which
            # have a target value of [0,0,0] for normals.
            # (for normals only compute the loss on the objects that come from
            # an intersection of PASCAL-context and NYU objects)
            target_zero_mask = (torch.abs(torch.norm(target, p=2, dim=1)) > 0)
            # Reshape the tensor to shape (-1,) This is to make sure 
            # that it is in the correct shape to easily select elements which
            # are going to be used to compute the loss (with the logical indexing mask)
            target_zero_mask = torch.reshape(target_zero_mask, (-1,))

            # Combine the target_zero_mask and the select_mask.
            # The loss is taken into account only where both masks are True.
            mask = target_zero_mask * select_mask
        else:
            mask = select_mask

        # Reshape the tensor to shape (-1, D) and select masked values.
        pred_sel = torch.reshape(pred.permute(0, 2, 3, 1), (-1, pred.shape[1]))[mask, :]
        # Reshape the tensor to shape (-1, D) and select masked values.
        target_sel = torch.reshape(target.permute(0, 2, 3, 1), (-1, target.shape[1]))[mask, :]

        # Normalize to unit vectors
        if self.normalize_vec:
            pred_sel = self.normal_ize(pred_sel, dim=1)
            target_sel = self.normal_ize(target_sel, dim=1)

        # Translate [-1, 1] --> [0, 2], and invert such that smaller value means better similarity.
        # Observation: when I tried the angular loss (applying arccosine and transforming to degrees) it was very unstabile
        loss = 1.0 - self.cos_sim(pred_sel, target_sel.detach())

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError
        
    @staticmethod
    def normal_ize(x, dim):
        norm = x.norm(p=2, dim=dim, keepdim=True).detach()
        return x / (norm + 1e-8)
