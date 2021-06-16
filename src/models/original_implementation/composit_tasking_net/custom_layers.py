import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
import math
from math import sqrt

import random


class InvSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        assert torch.sum((x < 0)*(x > 1)) == 0
        x = torch.where(x==0, x+1e-6, x)
        x = torch.where(x==1, x-1e-6, x)
        return torch.log(x/(1-x))

        
class Intermediate_Generator(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input task code vector z to an intermediate latent space w.
    '''
    def __init__(self, n_fc, input_dim, output_dim):
        super().__init__()
        # L2 normalization of the input code vector to be of unit length
        layers = [VectorL2Normalize()]

        # fc x (linear layer followed by leaky relu)
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.LeakyReLU(0.2))
        for _ in range(n_fc-1):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.LeakyReLU(0.2))

        # Combine everything into a torch Sequential module
        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        # Map the input task code vector z to an intermediate latent space w
        latent_w = self.mapping(latent_z)
        return latent_w

class VectorL2Normalize(nn.Module):
    """
    A module that normalizes the input vector to length 1 
    by dividing it wit its L2 norm.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        norm = x.norm(p=2, dim=0, keepdim=False).detach()
        x_normalized = x / norm
        return x_normalized
