import math

import torch
from torchmetrics import Metric


class NormalsMetric(Metric):

    def __init__(self, task, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.task = task

        # Initialize metric bookkeeping
        # Everything defined with self.add_state should
        # be automatically restarted when self.compute is called
        self.add_state("angular_error", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

        if not self.task in ["normals"]:
            raise NotImplementedError

    def update(self, target, pred, select_mask=None):
        # If no task map has been specified, that means every pixel corresponds to the task
        if select_mask is None:
            select_mask = torch.ones_like(pred).to(pred.device) == 1

        # TODO For now asserts that every channel (dim=1) makes the same
        # TODO select choice in one spatial location i.e. (B,:,H,W) = same_val.
        # TODO This is hardcoded for 3 channel outputs
        assert torch.sum(torch.sum(select_mask,axis=1)==1) == 0
        assert torch.sum(torch.sum(select_mask,axis=1)==2) == 0

        # Positions with zero norm vectors should not be taken into account 
        # while calculating the metric (for normals only compute the metric on 
        # the objects that come from an intersection of PASCAL-context and NYU objects)
        zero_norm_mask = (torch.abs(torch.norm(target, p=2, dim=1, keepdim=True)) > 0)
        mask = zero_norm_mask * select_mask

        # Reshape the mask to shape (-1,)
        mask = torch.reshape(mask.all(dim=1), (-1,))
        
        # Reshape the tensor to shape (-1, D) and select according to mask
        pred_sel = torch.reshape(pred.permute(0, 2, 3, 1), (-1, pred.shape[1]))[mask, :]
        target_sel = torch.reshape(target.permute(0, 2, 3, 1), (-1, target.shape[1]))[mask, :]
        
        # Normalize to unit vectors
        pred_sel = self.normal_ize(pred_sel, dim=1)
        target_sel = self.normal_ize(target_sel, dim=1)

        # TODO Generalization problems: feels like too much bookkeping for big data splits
        # TODO (This can alternatively be implemented as a sum that keeps track of
        # TODO how much elements (pixel positions) were summed so far)
        # TODO (Potential overflow could be a problem with this mentioned solution)
        # This is the angular error metric computed for surface normals
        angular_error_tmp = torch.sum((pred_sel * target_sel), dim=1)
        angular_error_tmp = torch.clamp(angular_error_tmp, min=-1, max=1)
        angular_error_tmp = torch.acos(angular_error_tmp) * (180.0/math.pi)
        self.angular_error = torch.cat([self.angular_error, angular_error_tmp])

    def compute(self):
        metric = {}

        metric['mean'] = torch.mean(self.angular_error).item()
        metric['median'] = torch.median(self.angular_error).item()
        metric['rmse'] = (torch.mean(self.angular_error ** 2) ** 0.5).item()
        metric['11.25'] = (torch.mean((self.angular_error < 11.25).type(torch.float32)) * 100).item()
        metric['22.5'] = (torch.mean((self.angular_error < 22.5).type(torch.float32)) * 100).item()
        metric['30'] = (torch.mean((self.angular_error < 30).type(torch.float32)) * 100).item()

        return metric

    @staticmethod
    def normal_ize(x, dim):
        norm = x.norm(p=2, dim=dim, keepdim=True).detach()
        return x / (norm + 1e-8)
