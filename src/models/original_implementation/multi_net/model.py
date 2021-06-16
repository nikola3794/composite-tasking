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

from .conditioning_layers import NoCondBlock


class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # Load and check the model configuration dictionary
        self.cfg = cfg

        self._create_decoder()
    
    def _create_decoder(self):

        # Number of channels at the end of each encoder spatial size level
        enc_n_ch = self._get_n_channels_in_enc_levels()

        # TODO <------------------------------------------------------------------------
        # TODO Parametrize this in a better/more automatic way 
        # Number of hidden channels in the conditioning modules, for different levels
        n_ch_cond_hidden_1 = 128
        n_ch_cond_hidden_2 = 128
        n_ch_cond_hidden_3 = 256
        n_ch_cond_hidden_4 = 256
        n_ch_cond_hidden_5 = 256

        # Number of output channels of every decoder level
        n_ch_decoder_5 = enc_n_ch["5"]
        n_ch_decoder_4 = 512
        n_ch_decoder_3 = 256
        n_ch_decoder_2 = 256
        n_ch_decoder_1 = 128
        # TODO <------------------------------------------------------------------------

        # Choose the type of conditioning block
        cond_module = NoCondBlock
        
        # Modules that process skip connections of the encoder:
        # -------------------------------------------------------
        # (Skip connection) Layers wich take the original input image (N, 3, x.H, x.W) and apply task conditioning directly
        self.skip_orig_img_cond_pt0 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_1}", in_channels=3, out_channels=enc_n_ch["1"], 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["dec_conv_ks"]
            )
        self.skip_orig_img_cond_pt1 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_1}", in_channels=enc_n_ch["1"], out_channels=enc_n_ch["1"], 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["dec_conv_ks"]
            )
        # (Skip connection) Layer that takes the ecoder output at spatial dimension (N, n_ch_level_1, x.H/2, x.W/2),
        # adapts it based on the provided target task map and sends it to the decoder at the appropriate location 
        self.skip_1_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_1}", in_channels=enc_n_ch["1"], out_channels=enc_n_ch["1"],
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["skip_conv_ks"]
            )
        # (Skip connection) ...takes (N, n_ch_level_2, x.H/4, x.W/4),
        self.skip_2_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_2}", in_channels=enc_n_ch["2"], out_channels=enc_n_ch["2"], 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["skip_conv_ks"]
            )
        # (Skip connection) ...takes (N, n_ch_level_3, x.H/8, x.W/8),
        self.skip_3_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_3}", in_channels=enc_n_ch["3"], out_channels=enc_n_ch["3"], 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["skip_conv_ks"]
            )
        # (Skip connection) ...takes (N, n_ch_level_4, x.H/16, x.W/16),
        self.skip_4_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_4}", in_channels=enc_n_ch["4"], out_channels=enc_n_ch["4"], 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["skip_conv_ks"]
            )
        # (Skip connection) ...takes (N, n_ch_level_5, x.H/32, x.W/32),
        self.skip_5_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_5}", in_channels=enc_n_ch["5"], out_channels=n_ch_decoder_5, 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["skip_conv_ks"]
            )

        # Decoder:
        # ---------
        # For upsampling feature maps in the Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Layers which take the previous encoder feature map along with the skip connection and
        # pass it through a task conditioning layer
        self.regular_4_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_4}", in_channels=enc_n_ch["4"]+n_ch_decoder_5, out_channels=n_ch_decoder_4, 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["dec_conv_ks"]
            )
        self.regular_3_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_3}", in_channels=enc_n_ch["3"]+n_ch_decoder_4, out_channels=n_ch_decoder_3, 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["dec_conv_ks"]
            )
        self.regular_2_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_2}", in_channels=enc_n_ch["2"]+n_ch_decoder_3, out_channels=n_ch_decoder_2, 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["dec_conv_ks"]
            )
        self.regular_1_cond = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_1}", in_channels=enc_n_ch["1"]+n_ch_decoder_2, out_channels=n_ch_decoder_1, 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["dec_conv_ks"]
            )

        # Final product:
        # ---------------
        # Final task conditioning which takes the output of the Ecnoder as well as the conditioned input image
        # and applies the task conditioning layer
        net_output_ch = 64
        self.final_cond_0 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_ch_cond_hidden_1}", in_channels=enc_n_ch["1"]+n_ch_decoder_1, out_channels=net_output_ch, 
            dim_latent_w=self.cfg["latent_w_dim"], ks=self.cfg["dec_conv_ks"]
            )
        # From feature map to the final prediction
        self.conv_last = nn.Conv2d(in_channels=net_output_ch, out_channels=self.cfg["net_output_ch"], kernel_size=1)
    
    def forward(self, enc_output):
        # Processing skip connections:
        # Process the skip connections with conditioning layers.
        # They will be used in the decoder at the appropriate global stride levels
        # -----------------------------------------------------------------------------
        # Original image - skip connection at global stride 1 
        x_original_cond = self.skip_orig_img_cond_pt0(feature_map=enc_output["0_input"])
        x_original_cond = self.skip_orig_img_cond_pt1(feature_map=x_original_cond)
        # Skip connection after gloal stride 2
        skip_1_cond = self.skip_1_cond(feature_map=enc_output["1_s2"])
        # Skip connection after global stride 4
        skip_2_cond = self.skip_2_cond(feature_map=enc_output["2_s4"])
        # Skip connection after global stride 8
        skip_3_cond = self.skip_3_cond(feature_map=enc_output["3_s8"])
        # Skip connection after global stride 16
        skip_4_cond = self.skip_4_cond(feature_map=enc_output["4_s16"])
        # Skip connection after global stride 32
        skip_5_cond = self.skip_5_cond(feature_map=enc_output["5_s32"])

        # Computing the decoder:
        # ----------------------
        # Decoder level 32->16 (global stride)
        comb_4 = self.upsample(skip_5_cond)
        # Decoder level 16->8
        comb_3 = torch.cat([comb_4, skip_4_cond], dim=1)
        comb_3 = self.regular_4_cond(feature_map=comb_3)
        comb_3 = self.upsample(comb_3)
        # Decoder level 8->4
        comb_2 = torch.cat([comb_3, skip_3_cond], dim=1)
        comb_2 = self.regular_3_cond(feature_map=comb_2)
        comb_2 = self.upsample(comb_2)
        # Decoder level 4->2
        comb_1= torch.cat([comb_2, skip_2_cond], dim=1)
        comb_1 = self.regular_2_cond(feature_map=comb_1)
        comb_1 = self.upsample(comb_1)
        # Decoder level 2->1
        comb_0 = torch.cat([comb_1, skip_1_cond], dim=1)
        comb_0 = self.regular_1_cond(feature_map=comb_0)
        comb_0 = self.upsample(comb_0)
        # Decoder level 1
        # Combine the output of the encoder and pass it through the final task conditioning layer
        comb_final = torch.cat([comb_0, x_original_cond], dim=1)
        comb_final = self.final_cond_0(feature_map=comb_final)

        # Convert from the feature map to the final prediction
        out_logits = self.conv_last(comb_final)


        return out_logits
    
    def _get_n_channels_in_enc_levels(self):
        if self.cfg["encoder_arch"] in ["resnet18", "resnet34"]:
            n_ch = {
                "1": 64,
                "2": 64,
                "3": 128,
                "4": 256,
                "5": 512,
            }
        elif self.cfg["encoder_arch"] in ["resnet50", "resnet101"]:
            n_ch = {
                "1": 64,
                "2": 256,
                "3": 512,
                "4": 1024,
                "5": 2048,
            }
        else:
            raise NotImplementedError

        return n_ch
    
class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Load and check the model configuration dictionary
        self.cfg = cfg

        self._create_encoder()

    def _create_encoder(self):
        # Load the encoder backbone
        if self.cfg["encoder_arch"] == "resnet18":
            base_model = torchvision.models.resnet.resnet18(pretrained=self.cfg["encoder_pre_trained"], progress=False)
        elif self.cfg["encoder_arch"] == "resnet34":
            base_model = torchvision.models.resnet.resnet34(pretrained=self.cfg["encoder_pre_trained"], progress=False)
        elif self.cfg["encoder_arch"] == "resnet50":
            base_model = torchvision.models.resnet.resnet50(pretrained=self.cfg["encoder_pre_trained"], progress=False)
        elif self.cfg["encoder_arch"] == "resnet101":
            base_model = torchvision.models.resnet.resnet101(pretrained=self.cfg["encoder_pre_trained"], progress=False)
        else:
            raise NotImplementedError
        base_layers = list(base_model.children())

        # Encoder layers:
        # -----------
        # Layer output size=(N, n_ch_level_1, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*base_layers[:3])
        # Layer output size=(N, n_ch_level_2, x.H/4, x.W/4)
        self.layer2 = nn.Sequential(*base_layers[3:5]) 
        # Layer output size=(N, n_ch_level_3, x.H/8, x.W/8)
        self.layer3 = base_layers[5]
        # Layer output size=(N, n_ch_level_4, x.H/16, x.W/16)
        self.layer4 = base_layers[6]
        # Layer output size=(N, n_ch_level_5, x.H/32, x.W/32)
        self.layer5 = base_layers[7] 
        
    def forward(self, x):

        # Compute the Encoder and its intermediate results
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        output_levels = {
            "0_input": x,
            "1_s2": layer1,
            "2_s4": layer2,
            "3_s8": layer3,
            "4_s16": layer4,
            "5_s32": layer5,
        }

        return output_levels

class MultiNetV0(nn.Module):

    def __init__(self, cfg, task_z_code_dict):
        super().__init__()
        # Load and check the model configuration dictionary
        self.cfg = cfg
        self._check_cfg()
        self.task_z_code_dict = task_z_code_dict
        
        # Create the decoder
        enc = {}
        dec= {}
        for task_id in self.task_z_code_dict:
            enc[str(task_id)] = Encoder(cfg=cfg)
            dec[str(task_id)] = Decoder(cfg=cfg)
        self.decoders = nn.ModuleDict(dec)
        self.encoders = nn.ModuleDict(enc)

    def _check_cfg(self):
        # Make sure everything that is needed is in the model config file
        assert "cond_cfg_txt" in self.cfg
        assert "n_fc_z_map" in self.cfg
        assert "latent_w_dim" in self.cfg
        assert "skip_conv_ks" in self.cfg
        assert "dec_conv_ks" in self.cfg
        assert "net_output_ch" in self.cfg
        assert "encoder_arch" in self.cfg
        assert "encoder_pre_trained" in self.cfg

    def forward_composite_task(self, x, task_palette, considered_task_ids):

        task_palette = task_palette.unsqueeze(1).repeat(1,3,1,1)

        out_logits = (-1.0 * torch.ones_like(x))
        out_logits = out_logits.to(x.device) # TODO Figure out how to use torch lightnings device
        for task_id in considered_task_ids:
            # Predict every requested task and construct the output
            # based on the provided task palette
            enc_output = self.encoders[str(task_id)](x)
            out_logits_id = self.decoders[str(task_id)](enc_output=enc_output)
            out_logits = torch.where(task_palette==task_id, out_logits_id, out_logits)

        return out_logits
    
    def forward_multi_task(self, x, considered_task_ids):
        out_logits = {}
        for task_id in considered_task_ids:
            # Predict every requested task 
            enc_output = self.encoders[str(task_id)](x)
            out_logits[task_id] = self.decoders[str(task_id)](enc_output=enc_output)

        return out_logits
    
    def forward_single_task(self, x, task_id):
        enc_output = self.encoders[str(task_id)](x)
        out_logits = self.decoders[str(task_id)](enc_output=enc_output)

        return out_logits
    

    def _get_n_channels_in_enc_levels(self):
        if self.cfg["encoder_arch"] in ["resnet18", "resnet34"]:
            n_ch = {
                "1": 64,
                "2": 64,
                "3": 128,
                "4": 256,
                "5": 512,
            }
        elif self.cfg["encoder_arch"] in ["resnet50", "resnet101"]:
            n_ch = {
                "1": 64,
                "2": 256,
                "3": 512,
                "4": 1024,
                "5": 2048,
            }
        else:
            raise NotImplementedError

        return n_ch
        
    def get_encoder_parameters(self):
        """
        Get the parameters of the modules that usually have pre-trained weights (Encoder)
        """
        parameters = []

        for task_id in self.task_z_code_dict:
            parameters.append(self.encoders[str(task_id)].parameters())

        parameters_final = []
        for prm in parameters:
            parameters_final += list(prm)
        return parameters_final
    
    def get_decoder_parameters(self):
        """
        Get the parameters of the modules that are usually training from scratch
        (Decoder + layers on the encoders skip connections)
        """
        parameters = []

        for task_id in self.task_z_code_dict:
            parameters.append(self.decoders[str(task_id)].parameters())

        parameters_final = []
        for prm in parameters:
            parameters_final += list(prm)
        return parameters_final




