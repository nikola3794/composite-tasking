import torch
import torch.nn as nn

import re


class TaskCompositionBlockV1(nn.Module):
    """
    This block takes the input features and conditions them with respect to
    the Task Palette embedding computed with the task representation block.
    """
    def __init__(self, cfg_txt, in_channels, out_channels, dim_latent_w, ks, **kwargs):
        super().__init__()

        # Convolutional layer
        pd = ks // 2
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=ks, 
            padding=pd
        )

        # Spatial task conditioning
        self.spatial_cond = SpatialAffineCond(
            cfg_txt=cfg_txt, 
            dim_latent_w=dim_latent_w, 
            n_out_ch=out_channels
        )

        # Activation - leaky relu
        self.act = nn.LeakyReLU()

    def forward(self, feature_map, latent_w_tensor, **kwargs):

        result = self.conv1(feature_map)

        result = self.spatial_cond(
            feature_map=result, 
            latent_w_tensor=latent_w_tensor
        )

        result = self.act(result)

        return result
    

class SpatialAffineCond(nn.Module):
    """
    Applies normalization with statitstics in a BatchNorm fashion
    followed with the affine transformation spatial task-wise encoding
    Implmenetation was taken over and modified from:
    https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py
    """

    def __init__(self, cfg_txt, dim_latent_w, n_out_ch):
        super().__init__()

        # Parse the configuration text that says which norm's 
        # statistical normalization to use and which filter size 
        # to use for computing the affine transformation parameters.
        assert cfg_txt.startswith('cond')
        parsed = re.search('cond_(\D+)(\d)x\d_hid(\d+)', cfg_txt)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2
        n_hidden = int(parsed.group(3))

        # Choose the appropriate parameter free normalization
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(
                num_features=n_out_ch, 
                affine=False
            )
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(
                num_features=n_out_ch, 
                affine=False
            )
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        else:
            raise NotImplementedError

        # Computes intermediate embedding space
        self.conv_shared = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_latent_w, 
                out_channels=n_hidden, 
                kernel_size=ks, 
                padding=pw, 
                bias=True
            ),
            nn.LeakyReLU()
        )
        
        # Layers which apply the affine transformation conditioning
        self.affine_beta = torch.nn.Conv2d(
            in_channels=n_hidden, 
            out_channels=n_out_ch, 
            kernel_size=ks, 
            stride=1, 
            padding=pw, 
            bias=True
        )
        self.affine_gamma = torch.nn.Conv2d(
            in_channels=n_hidden, 
            out_channels=n_out_ch, 
            kernel_size=ks, 
            stride=1, 
            padding=pw,
            bias=True
        )
        
        # Bias initialization
        self.affine_beta.bias.data[:] = 0
        self.affine_gamma.bias.data[:] = 1
    
    def forward(self, feature_map, latent_w_tensor):
        # Pass the latent code w map through a conv layer
        spat_w_code = self.conv_shared(latent_w_tensor)

        # Compute affine conditioning tensors
        # (conditioned on the task palette embedding map w)
        spat_gamma = self.affine_gamma(spat_w_code)
        spat_beta = self.affine_beta(spat_w_code)

        return self.param_free_norm(feature_map) * spat_gamma + spat_beta


