import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

import math

from ..data_sets.utils.helpers import label_to_classes


class FocalCEGridToSeg(nn.Module):

    def __init__(self, cls_centroids, focal_gamma=2.0, discard_empty=False, reduction="mean"):
        super().__init__()
        
        self.cls_centroids = cls_centroids
        self.focal_gamma = focal_gamma
        self.discard_empty = discard_empty 
        self.reduction = reduction

        self.NLL_loss = nn.NLLLoss(reduction="none")
        
        assert focal_gamma >= 0
    
    def forward(self, pred, target, select_mask=None, valid_imgs=None):
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
        
        # Some images only have the background, and thus are not interesting
        # data points for supervision. (It is used for parts images without humans)
        # TODO This is currently hardcoded to human body parts and them having 7 classes
        if (valid_imgs is not None) and (len(self.cls_centroids) == 7) and (self.discard_empty):
            select_mask = (select_mask * valid_imgs.unsqueeze(1).unsqueeze(2).unsqueeze(3)) == 1.0

        # Transform the network output to quasi logits
        quasi_logits = self.pred_to_quasi_logits(pred=pred) 

        # Transform the labels from class codes to class numbers (i.e. [0, 64, 128] --> 5)
        target_t = label_to_classes(
            label=torch.round(target*255.0), # [0,1] --> [0,255] 
            cls_centroids=self.cls_centroids
        ) 

        # Also reshape the tensor to shape (-1, N_C). This is to make sure that it 
        # is in the correct shape to easily select elements which are going to be used 
        # to compute the loss (with the logical indexing mask).
        quasi_logits = torch.reshape(
            quasi_logits.permute(0, 2, 3, 1), 
            (-1, quasi_logits.shape[1])
        )
        # Also reshape the tensor to shape (-1, ).
        target_t = torch.reshape(target_t, (-1, ))

        # Select relevant items
        quasi_logits = quasi_logits[select_mask, :]
        target_t = target_t[select_mask]

        # TODO Quick fix:
        # If there is no human in the batch, for parts none of the predicted pixels will be selected
        # This is a quick fix, in order not to destabilize things during backprop.
        if target_t.nelement() == 0:
            assert quasi_logits.nelement() == 0
            return torch.tensor(0.0, dtype=torch.float32).to(pred.device).detach()

        # Finally, calculate the loss
        if self.focal_gamma != 0:
            # Focal CE implementation
            pt = nn.Softmax(dim=-1)(quasi_logits)
            pt = pt.gather(1, target_t.unsqueeze(1).detach()).view(-1) + 1e-8 
            log_pt = pt.log()
            loss = -1 * torch.pow(torch.sub(1.0, pt), self.focal_gamma) * log_pt
        else:
            # Regular CE with torch implementation
            log_pt = nn.LogSoftmax(dim=-1)(quasi_logits)
            loss = self.NLL_loss(log_pt, target_t.detach()) 

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

    def pred_to_quasi_logits(self, pred):
        # !!!!! pred must be a tensor with values in [0, 1] !!!!!

        # [B,N_C,H,W] tensor. For each [b,:,h,w] location in the tensor the distance 
        # of the networks prediction at that location is computed to the N_C centroids.
        # The centroids represent the codes of each predicted class
        d1 = torch.zeros(
            (pred.shape[0], self.cls_centroids.shape[0], pred.shape[2], pred.shape[3]), 
            device=pred.device
        )
        for i in range(0, self.cls_centroids.shape[0]):
            # Centroid of the current class
            b = self.cls_centroids[i].unsqueeze(0).unsqueeze(2).unsqueeze(3).to(pred.device)
            # Since the pred is in [0,1] also transform the label centroid to [0,255] --> [0,1]
            b = (b / 255.0)
            # Repeat to be compatible with the prediction
            b = b.repeat(pred.shape[0], 1, pred.shape[2], pred.shape[3])

            # Calculate the distance of predictions to the current class centroid
            # TODO If the number of coding dimensions noticeably increases,
            # TODO this should be normalized with respect to the # of dimensions
            dist = (pred-b.detach())
            d1[:,i,:,:] = torch.norm(dist, p=2, dim=1)            

        # Then each element in the tensor is inversed so that the meaning can be 
        # interpreted the same as for logits (bigger value means a more certain prediction)
        quasi_logits = torch.ones_like((d1)).div(d1 + 1e-8)

        return quasi_logits
