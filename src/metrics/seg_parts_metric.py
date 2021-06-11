import numpy as np
import torch
from torchmetrics import Metric

from .helpers import precision_recall, jaccard

from ..data_sets.utils.helpers import label_to_classes, pred_to_classes
from ..data_sets.pascal_mt.metadata import ID_TO_TASK


class SegMetric(Metric):
    def __init__(self, cls_centroids, task, cls_ids_of_interest=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task = task
        self.n_classes = cls_centroids.size()[0]
        self.cls_centroids = cls_centroids
        self.cls_ids_of_interest = cls_ids_of_interest
       
        # Initialize metric bookkeeping
        # Everything defined with self.add_state should
        # be automatically restarted when self.compute is called
        self.add_state(
            "tp", 
            default=torch.tensor([0] * self.n_classes, requires_grad=False), 
            dist_reduce_fx="sum"
        )
        self.add_state(
            "fp", 
            default=torch.tensor([0] * self.n_classes, requires_grad=False), 
            dist_reduce_fx="sum"
            )
        self.add_state(
            "fn", 
            default=torch.tensor([0] * self.n_classes, requires_grad=False), 
            dist_reduce_fx="sum"
        )
            
        if not self.task in ["seg", "parts"]:
            raise NotImplementedError
    
    def update(self, target, pred, select_mask=None, valid_parts_img=None):
        # If no task map has been specified, that means every pixel corresponds to the task
        if select_mask is None:
            select_mask = torch.ones_like(pred).to(pred.device) == 1

        # TODO For now asserts that every channel (dim=1) makes the same
        # TODO select choice in one spatial location i.e. (B,:,H,W) = same_val.
        # TODO This is hardcoded for 3 channel outputs
        assert torch.sum(torch.sum(select_mask,axis=1)==1) == 0
        assert torch.sum(torch.sum(select_mask,axis=1)==2) == 0

        # Reshape task mask to lose the RGB dimension 
        select_mask = select_mask.all(dim=1)

        # For parts - discard data point when the whole image doesnt contain a human
        if (valid_parts_img is not None) and (self.task == "parts"):
            select_mask = (select_mask * valid_parts_img.unsqueeze(1).unsqueeze(2)) == 1.0

        # Transform predictions and labels from the coding in d-dimensional space to classes
        target_labels = label_to_classes(
            label=torch.round(target*255.0), # [0,1] --> [0,255]
            cls_centroids=self.cls_centroids
        ) 
        pred_labels, _ = pred_to_classes(
            pred=pred*255.0, # [0,1] --> [0,255]
            cls_centroids=self.cls_centroids
        )

        # Select only relevant pixels
        target_labels = target_labels[select_mask]
        pred_labels = pred_labels[select_mask]

        # TODO This was not used in the current dataset
        # Code for pixels not to be taken into account during metric calculation
        valid = (target_labels != 255)

        # Skip discarded images
        if target_labels.numel() == 0:
            assert pred_labels.numel() == 0.0
            return 

        # Accumulate tp, fp, fn for every class
        for i_part in range(0, self.n_classes):
            tmp_gt = (target_labels == i_part)
            tmp_pred = (pred_labels == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()
    
    def compute(self):
        metric = {}

        jac = [0] * self.n_classes
        for i_part in range(0, self.n_classes):
            # Blog with pictures that nicely explain why this formula equals IoU:
            # https://www.jeremyjordan.me/evaluating-image-segmentation-models/
            jac[i_part] = float(self.tp[i_part]) / \
                        max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        metric['jaccards_all_categs'] = jac
        metric['mIoU'] = np.mean(jac)

        # If requested, also calculate mIoU while only looking at specified classes 
        if self.cls_ids_of_interest is not None:
            mIoU_of_interest = []
            for i_part in range(0, self.n_classes):
                if i_part in self.cls_ids_of_interest:
                    mIoU_of_interest.append(jac[i_part])
            metric['mIoU_of_interest'] = np.mean(mIoU_of_interest)

        return metric

