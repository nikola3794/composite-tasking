import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
import math
from math import sqrt

import re

import random

class NoCondBlock(nn.Module):

    def __init__(self, cfg_txt, in_channels, out_channels, dim_latent_w, ks):
        super().__init__()

        # Convolutional layer
        pd = ks // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=pd)

        # Batch norm without spatial conditioning
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels, affine=True)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

    def forward(self, feature_map, latent_w_spat_pyramid, global_stride):

        result = self.conv1(feature_map)

        result = self.batch_norm(result)

        result = self.lrelu(result)

        return result

    def get_all_param_except_bn(self):
        return self.conv1.parameters()
    
    def get_bn_parametern(self):
        return self.batch_norm.parameters()
    

class SpatialAffineCondBlock(nn.Module):

    def __init__(self, cfg_txt, in_channels, out_channels, dim_latent_w, ks):
        super().__init__()

        # Convolutional layer
        pd = ks // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=pd)

        # Spatial conditioning
        self.spatial_cond = SpatialAffineCond(cfg_txt=cfg_txt, dim_latent_w=dim_latent_w, n_channels=out_channels)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

    def forward(self, feature_map, latent_w_spat_pyramid, global_stride):

        result = self.conv1(feature_map)

        result = self.spatial_cond(feature_map=result, latent_w_spat_pyramid=latent_w_spat_pyramid, global_stride=global_stride)

        result = self.lrelu(result)

        return result


class SpatialAffineCond(nn.Module):

    # --> SPADE implementation: https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py <--

    def __init__(self, cfg_txt, dim_latent_w, n_channels):
        super().__init__()

        # Parse the configuration text that says which parameter free norm to use
        # and what filter size to use for computing the affine transformation parameters
        assert cfg_txt.startswith('cond')
        parsed = re.search('cond_(\D+)(\d)x\d_hid(\d+)', cfg_txt)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2
        n_hidden = int(parsed.group(3))

        assert ks == 1 and pw == 0

        # Computes intermediate embedding space
        # TODO PArametrize so this can be repeated
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels=dim_latent_w, out_channels=n_hidden, kernel_size=ks, padding=pw, bias=True),
            nn.LeakyReLU()
        )

        # Choose the appropriate parameter free normalization
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(num_features=n_channels, affine=False, track_running_stats=True)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(num_features=n_channels, affine=False)
        else:
            raise ValueError(f'{param_free_norm_type} is not a recognized param-free norm type for the conditioning')
        
        # Layers that are going to compute the conditioning affine transformation parameters gamma and beta
        self.transform_gamma = torch.nn.Conv2d(in_channels=n_hidden, out_channels=n_channels, kernel_size=ks, stride=1, padding=pw, bias=True)
        self.transform_beta = torch.nn.Conv2d(in_channels=n_hidden, out_channels=n_channels, kernel_size=ks, stride=1, padding=pw, bias=True)
        # Bias initialization
        self.transform_beta.bias.data[:] = 0
        self.transform_gamma.bias.data[:] = 1
    
    def forward(self, feature_map, latent_w_spat_pyramid, global_stride):
        # Pass the latent code w map through a conv layer
        spat_w_code = self.conv_shared(latent_w_spat_pyramid[f"s{global_stride}"])

        # Compute affine conditioning tensors
        # (conditioned on the target task latent w map)
        spat_gamma = self.transform_gamma(spat_w_code)
        spat_beta = self.transform_beta(spat_w_code)

        res = self.param_free_norm(feature_map) * spat_gamma + spat_beta

        return res