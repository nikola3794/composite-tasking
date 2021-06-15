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

import re

import random

class NoCondBlock(nn.Module):

    def __init__(self, cfg_txt, in_channels, out_channels, dim_latent_w, ks, spade_cond_fn=False):
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

    def __init__(self, cfg_txt, in_channels, out_channels, dim_latent_w, ks, spade_cond_fn=False):
        super().__init__()

        # Convolutional layer
        pd = ks // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=pd)

        # Spatial conditioning
        self.spatial_cond = SpatialAffineCond(cfg_txt=cfg_txt, dim_latent_w=dim_latent_w, n_channels=out_channels, spade_cond_fn=spade_cond_fn)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

    def forward(self, feature_map, latent_w_spat_pyramid, global_stride):

        result = self.conv1(feature_map)

        result = self.spatial_cond(feature_map=result, latent_w_spat_pyramid=latent_w_spat_pyramid, global_stride=global_stride)

        result = self.lrelu(result)

        return result


class SpatialAffineCond(nn.Module):

    # --> SPADE implementation: https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py <--

    def __init__(self, cfg_txt, dim_latent_w, n_channels, spade_cond_fn=False):
        super().__init__()

        self.spade_cond_fn = spade_cond_fn

        # Parse the configuration text that says which parameter free norm to use
        # and what filter size to use for computing the affine transformation parameters
        assert cfg_txt.startswith('cond')
        parsed = re.search('cond_(\D+)(\d)x\d_hid(\d)', cfg_txt)
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
        if self.spade_cond_fn is False:
            # Old way
            self.transform_beta.bias.data[:] = 0
            self.transform_gamma.bias.data[:] = 1
        else:
            # Spade way
            self.transform_beta.bias.data[:] = 0
            self.transform_gamma.bias.data[:] = 0
    
    def forward(self, feature_map, latent_w_spat_pyramid, global_stride):
        # Pass the latent code w map through a conv layer
        spat_w_code = self.conv_shared(latent_w_spat_pyramid[f"s{global_stride}"])

        # Compute affine conditioning tensors
        # (conditioned on the target task latent w map)
        spat_gamma = self.transform_gamma(spat_w_code)
        spat_beta = self.transform_beta(spat_w_code)

        if self.spade_cond_fn is False:
            # Old way
            res = self.param_free_norm(feature_map) * spat_gamma + spat_beta
        else:
            # Spade way
            res = self.param_free_norm(feature_map) * (1.0 + spat_gamma) + spat_beta
        return res


# TODO <------------------------------------------------------------------------------------------------------
# TODO   REMOVED MY IMPLEMENTATION OF EFFICIENT CONDITIONING, BECAUSE IT DIDNT WORK OUT FOR SOME REASON I DONT REMEMBER
# TODO <------------------------------------------------------------------------------------------------------

# from models.composit_tasking.custom_layers import quick_scale


# class SpatialAffineEfficientNearestCondBlock(nn.Module):

#     def __init__(self, cfg_txt, in_channels, out_channels, dim_latent_w, ks):
#         super().__init__()

#         # Convolutional layer
#         pd = ks // 2
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=pd)

#         # Spatial conditioning
#         self.spatial_cond = SpatialAffineEfficientNearestCond(cfg_txt=cfg_txt, dim_latent_w=dim_latent_w, n_channels=out_channels)

#         # Leaky relu activation
#         self.lrelu = nn.LeakyReLU()

#     def forward(self, feature_map, latent_w, global_stride, task_id_map):
#         result = self.conv1(feature_map)

#         result = self.spatial_cond(feature_map=result, latent_w_vec=latent_w, global_stride=global_stride, task_id_map=task_id_map)

#         result = self.lrelu(result)

#         return result


# class SpatialAffineEfficientNearestCond(nn.Module):

#     def __init__(self, cfg_txt, dim_latent_w, n_channels):
#         super().__init__()

#         # Parse the configuration text that says which parameter free norm to use
#         # and what filter size to use for computing the affine transformation parameters
#         assert cfg_txt.startswith('cond')
#         parsed = re.search('cond_(\D+)\dx\d_hid(\d)', cfg_txt)
#         param_free_norm_type = str(parsed.group(1))
#         n_hidden = int(parsed.group(2))

#         # Computes intermediate embedding space
#         lin_shared = nn.Linear(dim_latent_w, n_hidden)
#         lin_shared.weight.data.normal_()
#         lin_shared.bias.data.zero_()
#         self.lin_shared = quick_scale(lin_shared)
#         self.lrelu = nn.LeakyReLU()

#         # Choose the appropriate parameter free normalization
#         if param_free_norm_type == 'instance':
#             self.param_free_norm = nn.InstanceNorm2d(num_features=n_channels, affine=False)
#         # elif param_free_norm_type == 'syncbatch':
#         #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
#         elif param_free_norm_type == 'batch':
#             self.param_free_norm = nn.BatchNorm2d(num_features=n_channels, affine=False)
#         else:
#             raise ValueError(f'{param_free_norm_type} is not a recognized param-free norm type for the conditioning')
        
#         # Layers that are going to compute the conditioning affine transformation parameters gamma and beta
#         self.transform_gamma = nn.Linear(n_hidden, n_channels)
#         self.transform_beta = nn.Linear(n_hidden, n_channels)
#         # Bias initialization
#         self.transform_gamma.bias.data[:] = 1
#         self.transform_beta.bias.data[:] = 0
    
#     def forward(self, feature_map, latent_w_vec, global_stride, task_id_map):
#          # Pass the latent code w vectors through a shared fully connected layers

#         downsized_task_maps = []
#         if global_stride == 1:
#             quasi_center = 0
#             downsized_task_maps.append(task_id_map.unsqueeze(1)[:, :, quasi_center::global_stride, quasi_center::global_stride])
#         elif global_stride == 2:
#             quasi_center = 1
#             for delta in [-1, 0]:
#                 downsized_task_maps.append(task_id_map.unsqueeze(1)[:, :, (quasi_center+delta)::global_stride, (quasi_center+delta)::global_stride])
#         else:
#             quasi_center = global_stride // 2
#             for delta in [[0, 0], [-1, 0], [0, -1], [-1, -1]]:
#                 downsized_task_maps.append(task_id_map.unsqueeze(1)[:, :, (quasi_center+delta[0])::global_stride, (quasi_center+delta[1])::global_stride])

#         result = feature_map

#         gamma = torch.zeros(feature_map.shape, dtype = torch.float32).to(feature_map.device)
#         beta = torch.zeros(feature_map.shape, dtype = torch.float32).to(feature_map.device)
#         for task_id in latent_w_vec:
#             for down_task_map in downsized_task_maps:
#                 task_mask = 1.0 * (down_task_map == task_id).type(torch.float32).detach().to(feature_map.device)

#                 curr_transformed_latent_w_vec = self.lin_shared(latent_w_vec[task_id])
#                 curr_transformed_latent_w_vec = self.lrelu(curr_transformed_latent_w_vec)

#                 gamma_curr = self.transform_gamma(curr_transformed_latent_w_vec).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(feature_map.shape[0],1,feature_map.shape[2],feature_map.shape[3])
#                 gamma_curr *= task_mask
#                 gamma += gamma_curr
                
#                 beta_curr = self.transform_beta(curr_transformed_latent_w_vec).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(feature_map.shape[0],1,feature_map.shape[2],feature_map.shape[3])
#                 beta_curr *= task_mask
#                 beta += beta_curr

#         gamma = gamma / len(downsized_task_maps)
#         beta = beta / len(downsized_task_maps)

#         result = result * gamma + beta

#         return result
# TODO <------------------------------------------------------------------------------------------------------
# TODO <------------------------------------------------------------------------------------------------------
