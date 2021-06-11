import torch
import numpy as np


def precision_recall(gt, pred, select_mask=None):
    """
    Calcualte precision and recall.
    Calculate on positions which are specified with the select_mask.
    """
    if select_mask is None:
        select_mask = torch.ones(gt.shape, device=gt.device)

    gt = gt.type(torch.bool)
    pred = pred.type(torch.bool)
    select_mask = select_mask.type(torch.bool)

    tp = torch.sum(((pred & gt) & select_mask)).type(torch.float32)
    fn = torch.sum(((~pred & gt) & select_mask)).type(torch.float32)

    fp = torch.sum(((pred & ~gt) & select_mask)).type(torch.float32)

    prec = tp / (tp + fp)
    prec = torch.tensor(0.0, device=prec.device) if torch.isnan(prec) else prec
    rec = tp / (tp + fn)
    rec = torch.tensor(0.0, device=prec.device) if torch.isnan(rec) else rec

    return prec, rec

def jaccard(gt, pred, select_mask=None):
    """
    Calculate the Jaccard index.
    Calculate on positions which are specified with the select_mask.
    """
    assert(gt.shape == pred.shape)

    if select_mask is None:
        select_mask = torch.ones(gt.shape, device=gt.device)
    assert(select_mask.shape == gt.shape)

    gt = gt.type(torch.bool)
    pred = pred.type(torch.bool)
    select_mask = select_mask.type(torch.bool)

    if torch.isclose(torch.sum(gt & select_mask), torch.tensor(0).to(gt.device)) and torch.isclose(torch.sum(pred & select_mask), torch.tensor(0).to(gt.device)):
        # If there is not a gt present nor predicted on the whole tensor
        # (the part of the tensor that is valid for metric calculation to be exact - select_mask=1)
        return torch.tensor(1)
    else:
        return torch.sum(((gt & pred) & select_mask), dtype=torch.float32).type(torch.float32) / \
               torch.sum(((gt | pred) & select_mask), dtype=torch.float32).type(torch.float32)