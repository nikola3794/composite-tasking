import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchsummary import summary
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
    def __init__(self, n_fc, input_dim, output_dim, custom_w_init):
        super().__init__()
        # L2 normalization of the input code vector to be of unit length
        layers = [VectorL2Normalize()]

        # fc x (linear layer followed by leaky relu)
        layers.append(SLinear(input_dim, output_dim, custom_w_init))
        layers.append(nn.LeakyReLU(0.2))
        for _ in range(n_fc-1):
            layers.append(SLinear(output_dim, output_dim, custom_w_init))
            layers.append(nn.LeakyReLU(0.2))

        # Combine everything into a torch Sequential module
        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        # Map the input task code vector z to an intermediate latent space w
        latent_w = self.mapping(latent_z)
        return latent_w


class SLinear(nn.Module):
    """
    One linear NN layer with a manually implemented initialization.
    TODO The meanually implemented initialization looks like He initialization from the code
    """
    def __init__(self, dim_in, dim_out, custom_w_init):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)

        if custom_w_init:
            linear.bias.data.zero_()
            linear.weight.data.normal_()
            self.linear = quick_scale(linear)
        else:
            self.linear = linear

    def forward(self, x):
        return self.linear(x)


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
        # x_normalized = x * torch.rsqrt(torch.mean(x ** 2, dim=0, keepdim=True) + 1e-8)
        return x_normalized


# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module


class ScaleW:
    '''
    Manually implemented # TODO He?? TODO initialization
    '''

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)