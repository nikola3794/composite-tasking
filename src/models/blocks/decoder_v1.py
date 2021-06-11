import torch
import torch.nn as nn

from .non_conditioning_block_v1 import NoConditioningBlockV1
from .task_composition_block_v1 import TaskCompositionBlockV1


class DecoderV1(nn.Module):

    def __init__(self, cfg, enc_n_ch):
        super().__init__()

        # Number of channels at the end of each encoder spatial size level
        self.enc_n_ch = enc_n_ch

        # Load and check the model configuration dictionary
        self.cfg = cfg
        self._check_cfg()

        # Create the decoder architecture
        self._create_decoder()

    def _check_cfg(self):
        assert isinstance(self.enc_n_ch, dict)
        
        # Make sure everything that is needed is in the model config file
        assert self.cfg["which_cond"] in ["task_composition_v1", "no_cond"]
        assert isinstance(self.cfg["which_cond"], str)

        assert isinstance(self.cfg["cond_cfg_txt"], str)

        assert isinstance(self.cfg["latent_w_dim"], int)

        assert isinstance(self.cfg["skip_conv_ks"], int)

        assert isinstance(self.cfg["dec_conv_ks"], int)

        assert isinstance(self.cfg["net_output_ch"], int)

    def _create_decoder(self):
        # Number of hidden channels in the cond. modules, for different levels
        n_hid_ch_cond_1 = 128
        n_hid_ch_cond_2 = 128
        n_hid_ch_cond_3 = 256
        n_hid_ch_cond_4 = 256
        n_hid_ch_cond_5 = 256

        # Number of output channels of every decoder level
        n_out_ch_dec_5 = self.enc_n_ch["5"]
        n_out_ch_dec_4 = 512
        n_out_ch_dec_3 = 256
        n_out_ch_dec_2 = 256
        n_out_ch_dec_1 = 128

        # Choose the type of conditioning block
        if self.cfg["which_cond"] == "task_composition_v1":
            cond_module = TaskCompositionBlockV1
        elif self.cfg["which_cond"] == "no_cond":
            cond_module = NoConditioningBlockV1
        else:
            raise NotImplementedError
        
        # Modules that process skip connections of the encoder:
        # -------------------------------------------------------
        # (Skip connection) Layers wich take the original input image (N, 3, H, W) 
        # and adapts it based on the provided target task map if specified.
        # It then sends it to the decoder at the appropriate location.
        self.skip_orig_img_0 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_1}", 
            in_channels=3, 
            out_channels=self.enc_n_ch["1"], 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["dec_conv_ks"]
        )
        self.skip_orig_img_1 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_1}", 
            in_channels=self.enc_n_ch["1"], 
            out_channels=self.enc_n_ch["1"], 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["dec_conv_ks"]
        )
        # (Skip connection) Layer that takes the ecoder output (N, n_ch_level_1, H/2, W/2) 
        # and adapts it based on the provided target task map if specified.
        # It then sends it to the decoder at the appropriate location.
        self.skip_1 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_1}", 
            in_channels=self.enc_n_ch["1"], 
            out_channels=self.enc_n_ch["1"],
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["skip_conv_ks"]
        )
        # (Skip connection) ...takes (N, n_ch_level_2, H/4, W/4)...
        self.skip_2 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_2}", 
            in_channels=self.enc_n_ch["2"], 
            out_channels=self.enc_n_ch["2"], 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["skip_conv_ks"]
        )
        # (Skip connection) ...takes (N, n_ch_level_3, H/8, W/8)...
        self.skip_3 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_3}", 
            in_channels=self.enc_n_ch["3"], 
            out_channels=self.enc_n_ch["3"], 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["skip_conv_ks"]
        )
        # (Skip connection) ...takes (N, n_ch_level_4, H/16, W/16)...
        self.skip_4 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_4}", 
            in_channels=self.enc_n_ch["4"], 
            out_channels=self.enc_n_ch["4"], 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["skip_conv_ks"]
        )
        # (Skip connection) ...takes (N, n_ch_level_5, H/32, W/32)...
        self.skip_5 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_5}", 
            in_channels=self.enc_n_ch["5"], 
            out_channels=n_out_ch_dec_5, 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["skip_conv_ks"]
        )

        # Decoder:
        # ---------
        # For upsampling feature maps in the Decoder
        self.upsample = nn.Upsample(
            scale_factor=2, 
            mode='bilinear', 
            align_corners=True
        )
        # Layers which take the previous encoder feature map 
        # along with the skip connection.
        # It then passes it through a block that does conditioning if specified.
        self.dec_4 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_4}", 
            in_channels=self.enc_n_ch["4"]+n_out_ch_dec_5, 
            out_channels=n_out_ch_dec_4, 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["dec_conv_ks"]
        )
        self.dec_3 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_3}", 
            in_channels=self.enc_n_ch["3"]+n_out_ch_dec_4, 
            out_channels=n_out_ch_dec_3, 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["dec_conv_ks"]
        )
        self.dec_2 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_2}", 
            in_channels=self.enc_n_ch["2"]+n_out_ch_dec_3, 
            out_channels=n_out_ch_dec_2, 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["dec_conv_ks"]
        )
        self.dec_1 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_1}", 
            in_channels=self.enc_n_ch["1"]+n_out_ch_dec_2, 
            out_channels=n_out_ch_dec_1, 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["dec_conv_ks"]
        )

        # Final product:
        # ---------------
        # Final task conditioning which takes the output of the Ecnoder 
        # as well as the conditioned input image and applies 
        # the task conditioning layer
        self.dec_0 = cond_module(
            cfg_txt=f"{self.cfg['cond_cfg_txt']}_hid{n_hid_ch_cond_1}", 
            in_channels=self.enc_n_ch["1"]+n_out_ch_dec_1, 
            out_channels=64, 
            dim_latent_w=self.cfg["latent_w_dim"], 
            ks=self.cfg["dec_conv_ks"]
        )
        # From feature map to the final prediction
        self.last_conv = nn.Conv2d(
            in_channels=64, 
            out_channels=self.cfg["net_output_ch"], 
            kernel_size=1
        )

    def forward(self, enc_output, latent_w_spat_pyramid=None):
        if self.cfg["which_cond"] == "task_composition_v1":
            return self.forward_composite(
                enc_output=enc_output, 
                latent_w_spat_pyramid=latent_w_spat_pyramid
            )
        elif self.cfg["which_cond"] == "no_cond":
            return self.forward_no_cond(
                enc_output=enc_output
            )
        else:
            raise NotImplementedError

    def forward_composite(self, enc_output, latent_w_spat_pyramid):
        # Processing skip connections:
        # Process the skip connections with conditioning layers.
        # They will be used in the decoder at the appropriate global stride levels
        # ------------------------------------------------------------------------
        # Original image - skip connection at global stride 1 
        x_orig_processed = self.skip_orig_img_0(
            feature_map=enc_output["0_input"], 
            latent_w_tensor=latent_w_spat_pyramid[f"s1"]
        )
        x_orig_processed = self.skip_orig_img_1(
            feature_map=x_orig_processed, 
            latent_w_tensor=latent_w_spat_pyramid[f"s1"]
        )
        # Skip connection at gloal stride 2
        skip_1 = self.skip_1(
            feature_map=enc_output["1_s2"], 
            latent_w_tensor=latent_w_spat_pyramid[f"s2"]
        )
        # Skip connection at global stride 4
        skip_2 = self.skip_2(
            feature_map=enc_output["2_s4"], 
            latent_w_tensor=latent_w_spat_pyramid[f"s4"]
        )
        # Skip connection at global stride 8
        skip_3 = self.skip_3(
            feature_map=enc_output["3_s8"], 
            latent_w_tensor=latent_w_spat_pyramid[f"s8"]
        )
        # Skip connection at global stride 16
        skip_4 = self.skip_4(
            feature_map=enc_output["4_s16"], 
            latent_w_tensor=latent_w_spat_pyramid[f"s16"]
        )
        # Skip connection at global stride 32
        skip_5 = self.skip_5(
            feature_map=enc_output["5_s32"], 
            latent_w_tensor=latent_w_spat_pyramid[f"s32"]
        )

        # Computing the decoder:
        # ----------------------
        # Decoder level 32->16
        comb_4 = self.upsample(skip_5)
        # Decoder level 16->8
        comb_3 = torch.cat([comb_4, skip_4], dim=1)
        comb_3 = self.dec_4(
            feature_map=comb_3, 
            latent_w_tensor=latent_w_spat_pyramid[f"s16"]
        )
        comb_3 = self.upsample(comb_3)
        # Decoder level 8->4
        comb_2 = torch.cat([comb_3, skip_3], dim=1)
        comb_2 = self.dec_3(
            feature_map=comb_2, 
            latent_w_tensor=latent_w_spat_pyramid[f"s8"]
        )
        comb_2 = self.upsample(comb_2)
        # Decoder level 4->2
        comb_1= torch.cat([comb_2, skip_2], dim=1)
        comb_1 = self.dec_2(
            feature_map=comb_1, 
            latent_w_tensor=latent_w_spat_pyramid[f"s4"]
        )
        comb_1 = self.upsample(comb_1)
        # Decoder level 2->1
        comb_0 = torch.cat([comb_1, skip_1], dim=1)
        comb_0 = self.dec_1(
            feature_map=comb_0, 
            latent_w_tensor=latent_w_spat_pyramid[f"s2"]
        )
        comb_0 = self.upsample(comb_0)
        # Decoder level 1
        # Combine the output of the encoder 
        # and pass it through the final task conditioning layer
        comb_final = torch.cat([comb_0, x_orig_processed], dim=1)
        comb_final = self.dec_0(
            feature_map=comb_final, 
            latent_w_tensor=latent_w_spat_pyramid[f"s1"]
        )

        # Convert from the feature map to the final prediction
        return self.last_conv(comb_final)

    def forward_no_cond(self, enc_output):
        # Process the skip connections.
        # They will be used in the decoder at the appropriate global stride levels
        # -----------------------------------------------------------------------------
        # Original image - skip connection at global stride 1 
        x_orig_processed = self.skip_orig_img_0(feature_map=enc_output["0_input"])
        x_orig_processed = self.skip_orig_img_1(feature_map=x_orig_processed)
        # Skip connection at gloal stride 2
        skip_1 = self.skip_1(feature_map=enc_output["1_s2"])
        # Skip connection at global stride 4
        skip_2 = self.skip_2(feature_map=enc_output["2_s4"])
        # Skip connection at global stride 8
        skip_3 = self.skip_3(feature_map=enc_output["3_s8"])
        # Skip connection at global stride 16
        skip_4 = self.skip_4(feature_map=enc_output["4_s16"])
        # Skip connection at global stride 32
        skip_5 = self.skip_5(feature_map=enc_output["5_s32"])

        # Computing the decoder:
        # ----------------------
        # Decoder level 32->16
        comb_4 = self.upsample(skip_5)
        # Decoder level 16->8
        comb_3 = torch.cat([comb_4, skip_4], dim=1)
        comb_3 = self.dec_4(feature_map=comb_3)
        comb_3 = self.upsample(comb_3)
        # Decoder level 8->4
        comb_2 = torch.cat([comb_3, skip_3], dim=1)
        comb_2 = self.dec_3(feature_map=comb_2)
        comb_2 = self.upsample(comb_2)
        # Decoder level 4->2
        comb_1= torch.cat([comb_2, skip_2], dim=1)
        comb_1 = self.dec_2(feature_map=comb_1)
        comb_1 = self.upsample(comb_1)
        # Decoder level 2->1
        comb_0 = torch.cat([comb_1, skip_1], dim=1)
        comb_0 = self.dec_1(feature_map=comb_0)
        comb_0 = self.upsample(comb_0)
        # Decoder level 1
        # Combine the output of the encoder and pass it through the final task layer
        comb_final = torch.cat([comb_0, x_orig_processed], dim=1)
        comb_final = self.dec_0(feature_map=comb_final)

        # Convert from the feature map to the final prediction
        out_logits = self.last_conv(comb_final)

        return out_logits
