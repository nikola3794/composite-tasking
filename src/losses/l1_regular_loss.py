
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

import math


class L1Loss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

        self.L1_loss = nn.L1Loss(reduction=reduction)

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

        # Reshape the tensor to form (-1, D). 
        pred = torch.reshape(pred.permute(0, 2, 3, 1), (-1, pred.shape[1]))
        # Reshape the tensor to form (-1, D). 
        target = torch.reshape(target.permute(0, 2, 3, 1), (-1, target.shape[1]))

        # Calculate the L1 loss only on masked locations
        loss = self.L1_loss(pred[select_mask, :], target[select_mask, :].detach())
        
        return loss