import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

import math


class BalancedBCELoss(nn.Module):
    """
    Balanced Cross Entropy Loss.
    Optionally image regions can be specified where to compute the loss.
    This loss has been taken from: 
    https://github.com/facebookresearch/astmt/blob/master/fblib/layers/loss.py
    """

    def __init__(self, pos_weight=None, reduction="mean"):
        super().__init__()

        self.reduction = reduction
        self.pos_weight = pos_weight

        # In the current implementation, the weights for the positive 
        # and negative class sum up to 1
        if pos_weight is not None:
            assert self.pos_weight >= 0.0 and self.pos_weight <= 1.0

    def forward(self, pred, target, select_mask=None):
        output = pred
        label = target
        
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

        assert (output.size() == label.size())
        
        labels = torch.ge(label, 0.5).float()
        if self.pos_weight is None:
            # Make weights according to the count in the current batch.
            # The weights for positive and negative class sum up to 1.
            num_labels_pos = torch.sum(labels[select_mask] == 1) + 1.0
            num_labels_neg = torch.sum(labels[select_mask] == 0) + 1.0
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            # Use the provided weight of the positive class.
            # For the negative class, 1 - w will be used.
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        loss_pos_pix = loss_pos_pix[select_mask]
        loss_neg_pix = loss_neg_pix[select_mask]
        final_loss_pix = w * loss_pos_pix + (1 - w) * loss_neg_pix

        if self.reduction == "mean":
            return torch.mean(final_loss_pix)
        elif self.reduction == "sum":
            return torch.sum(final_loss_pix)
        elif self.reduction == "none":
            return final_loss_pix
        else:
            raise NotImplementedError