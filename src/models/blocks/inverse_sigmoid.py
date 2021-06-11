import torch
import torch.nn as nn


class InvSigmoid(nn.Module):
    """
    Class for computing the inverse of a sigmoid
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        assert torch.sum((x < 0)*(x > 1)) == 0
        x = torch.where(x==0, x+1e-6, x)
        x = torch.where(x==1, x-1e-6, x)
        return torch.log(x/(1-x))