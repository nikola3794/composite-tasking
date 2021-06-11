import numpy as np
import torch
from torchmetrics import Metric

from .helpers import precision_recall, jaccard


class EdgesSaliencyMetric(Metric):

    def __init__(self, task, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task = task
        
        # Initialize metric bookkeeping
        # Everything defined with self.add_state should
        # be automatically restarted when self.compute is called
        self.add_state("prec", default=torch.tensor([[]], dtype=torch.long), dist_reduce_fx="cat")
        self.add_state("rec", default=torch.tensor([[]], dtype=torch.long), dist_reduce_fx="cat")
        if task == "edges":
            self.mask_thres = np.linspace(0, 1, 100)
        elif task == "saliency":
            self.mask_thres = np.linspace(0.2, 0.9, 15)
        self.add_state("all_jaccards", default=torch.tensor([[]], dtype=torch.long), dist_reduce_fx="cat")

        if not self.task in ["edges", "saliency"]:
            raise NotImplementedError

    def update(self, target, pred, select_mask=None):        
        # If no task map has been specified, then every pixel corresponds to the task
        if select_mask is None:
            select_mask = torch.ones_like(pred).to(pred.device) == 1        

        # TODO For now asserts that every channel (dim=1) makes the same
        # TODO select choice in one spatial location i.e. (B,:,H,W) = same_val.
        # TODO This is hardcoded for 3 channel outputs
        assert torch.sum(torch.sum(select_mask,axis=1)==1) == 0
        assert torch.sum(torch.sum(select_mask,axis=1)==2) == 0
        
        # Make sure that targets are 0 and 1
        target = 1.0 * (target > 0.5)

        prec_tmp = []
        rec_tmp = []
        if self.task == "saliency":
            all_jacc_tmp = []
        
        for j, thres in enumerate(self.mask_thres):
            # FOr every threshold
            gt_eval = (target > thres)
            pred_eval = (pred > thres)
            if self.task == "saliency":
                # Jaccard index calculation
                all_jacc_tmp.append(jaccard(gt_eval[select_mask], pred_eval[select_mask]).item())
            # Precision and recall calculation
            prec_rec_tmp = precision_recall(gt_eval[select_mask], pred_eval[select_mask])
            prec_tmp.append(prec_rec_tmp[0].item())
            rec_tmp.append(prec_rec_tmp[1].item())

        # Solves concatenation problems of first element
        if self.prec.numel() == 0:
            assert self.rec.numel() == 0
            self.prec = torch.tensor(prec_tmp).unsqueeze(0)
            self.rec = torch.tensor(rec_tmp).unsqueeze(0)
            if self.task == "saliency":
                assert self.all_jaccards.numel() == 0
                self.all_jaccards = torch.tensor(all_jacc_tmp).unsqueeze(0)

        self.prec = torch.cat([self.prec, torch.tensor(prec_tmp).unsqueeze(0)]) 
        self.rec = torch.cat([self.rec, torch.tensor(rec_tmp).unsqueeze(0)]) 
        if self.task == "saliency":
            self.all_jaccards = torch.cat([self.all_jaccards, torch.tensor(all_jacc_tmp).unsqueeze(0)])
    
    def compute(self):
        metric = {}

        if self.task == 'saliency':
            metric['mIoUs'] = torch.mean(self.all_jaccards, dim=0)
            metric['max_mIoU'] = torch.max(metric['mIoUs']).item()
            metric['mIoUs'] = metric['mIoUs'].tolist()

        metric['mPrec'] = torch.mean(self.prec, dim=0)
        metric['mRec'] = torch.mean(self.rec, dim=0)
        metric['F'] = 2 * metric['mPrec'] * metric['mRec'] / \
                        (metric['mPrec'] + metric['mRec'] + 1e-12)
        metric['maxF'] = torch.max(metric['F']).item()

        metric['mPrec'] = metric['mPrec'].tolist()
        metric['mRec'] = metric['mRec'].tolist()
        metric['F'] = metric['F'].tolist()

        return metric