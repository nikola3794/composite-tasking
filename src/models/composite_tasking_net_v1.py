import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.encoder_v1 import EncoderV1
from .blocks.decoder_v1 import DecoderV1
from .blocks.task_representation_block_v1 import TaskRepresentationBlockV1


class CompositeTaskingNetV1(nn.Module):

    def __init__(self, cfg, task_z_code_dict):
        super().__init__()

        # Load and check the model configuration dictionary
        self.cfg = cfg
        self._check_cfg()

        # Load the task code dictionary
        self.task_z_code_dict = task_z_code_dict                
        # Extract the vector length of the z code
        self.z_code_dim = self.task_z_code_dict[list(self.task_z_code_dict.keys())[0]].shape[0]

        # Fully connnected network to map the input task code vector z to an intermediate latent space w
        self.task_representation_block = TaskRepresentationBlockV1(
            n_fc=self.cfg["n_fc_z_map"], 
            input_dim=self.z_code_dim, 
            output_dim=self.cfg["latent_w_dim"]
        )
        
        # Create the encoder
        self.encoder = EncoderV1(cfg=self.cfg)

        # Create the decoder
        self.decoder = DecoderV1(
            cfg=self.cfg, 
            enc_n_ch=self.encoder.get_n_channels_in_each_level()
        )
    
    def _check_cfg(self):
        assert isinstance(self.cfg, dict)
        
        assert isinstance(self.cfg["latent_w_dim"], int)

        assert self.cfg["which_cond"] in ["task_composition_v1"]

    def forward(self, x, task_palette, considered_task_ids):
        return self.forward_composite(
            x=x,
            task_palette=task_palette,
            considered_task_ids=considered_task_ids,
        )

    def forward_composite_task(self, x, task_palette, considered_task_ids):
        # Compute the encoder
        enc_output = self.encoder(x=x)

        # Compute w latent task codes, for every considered task
        latent_w_vectors = self.compute_latent_w_for_considered_tasks(
            considered_task_ids=considered_task_ids
        )
        
        # Make a spatial latent w tensor
        # by putting task latent codes w at appropriate spatial locations
        orig_shape = x.shape[0:1] + (self.get_latent_w_dim(), ) + x.shape[2:]
        latent_w_spat = self.construct_spatial_latent_w(
            task_palette=task_palette, 
            latent_w_vectors=latent_w_vectors, 
            orig_shape=orig_shape, 
        )
        
        # Compute the latent w spatial pyramid
        latent_w_spat_pyramid = self.compute_latent_w_spat_pyramid(
            latent_w_spat=latent_w_spat
        )

        # Compute the decoder - final prediction
        out_logits = self.decoder(
            enc_output=enc_output, 
            latent_w_spat_pyramid=latent_w_spat_pyramid
        )

        return out_logits
    
    def forward_multi_task(self, x, considered_task_ids):
        out_logits = {}

        # Compute the encoder
        enc_output = self.encoder(x=x)

        for task_id in considered_task_ids:
            # Compute w latent task codes, for every considered task
            latent_w_vectors = self.compute_latent_w_for_considered_tasks(
                considered_task_ids=considered_task_ids
            )

            # TODO Figure out how to use device from torch lightning
            task_palette = task_id * torch.ones(
                (x.shape[0], x.shape[2], x.shape[3]), 
                dtype=torch.int8,
                device=x.device
            )

            # Make a spatial latent w tensor
            # by putting task latent codes w at appropriate spatial locations
            orig_shape = x.shape[0:1] + (self.get_latent_w_dim(), ) + x.shape[2:]
            latent_w_spat = self.construct_spatial_latent_w(
                task_palette=task_palette, 
                latent_w_vectors=latent_w_vectors, 
                orig_shape=orig_shape, 
            )
            
            # Compute the latent w spatial pyramid
            latent_w_spat_pyramid = self.compute_latent_w_spat_pyramid(
                latent_w_spat=latent_w_spat
            )

            # Compute the decoder - final prediction
            out_logits[task_id] = self.decoder(
                enc_output=enc_output, 
                latent_w_spat_pyramid=latent_w_spat_pyramid
            )

        return out_logits

    def forward_single_task(self, x, task_id):            
        # TODO Figure out how to use device from torch lightning
        task_palette = task_id * torch.ones(
            (x.shape[0], x.shape[2], x.shape[3]), 
            dtype=torch.int8,
            device=x.device
        )

        return self.forward_composite_task(
            x=x, 
            task_palette=task_palette, 
            considered_task_ids=[task_id]
        )

    def compute_latent_w_for_considered_tasks(self, considered_task_ids):
        # (z -->FC--> w)
        # For each considered task, process its code z and obtain the latent code w  
        latent_w_vectors = {}
        for task_id in considered_task_ids:
            latent_w_vectors[task_id] = self.compute_z_to_latent_w(
                z=self.task_z_code_dict[task_id].clone().detach()
            )
        return latent_w_vectors
    
    def compute_z_to_latent_w(self, z):
        return self.task_representation_block(z)
    
    def construct_spatial_latent_w(self, task_palette, latent_w_vectors, orig_shape):
        """
        This is constructed by putting appropriate task latent codes w,
        with respect to the task palette.
        """
        # TODO Figure out how to use device from torch lightning
        dev = list(latent_w_vectors.values())[0].device

        # Expand the calculated latent task codes w into a spatial tensor,
        # so that is correpsonding to the targeted task palette
        task_palette = task_palette.unsqueeze(1)
        latent_w_spat = torch.zeros(orig_shape, dtype=torch.float32).to(dev) 
        for task_id in latent_w_vectors.keys():
            task_mask = (1.0 * (task_palette == task_id)).type(torch.float32).detach().to(dev) 
            curr_w = latent_w_vectors[task_id].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            curr_w = curr_w.repeat(orig_shape[0], 1, orig_shape[2], orig_shape[3]).to(dev) 
            latent_w_spat = torch.where(task_mask == 1.0, curr_w, latent_w_spat)

        return latent_w_spat
    
    # TODO <--------------------------------------
    # Pyramid levels are hardcoded in advance
    # It can be implemented that each implemented architecture
    # gives information about what levels are present.
    # TODO <--------------------------------------
    def compute_latent_w_spat_pyramid(self, latent_w_spat):
        """
        Make a spatial pyramid of latent code w spatial map.
        Smaller sizes are downsampled.
        """
        # Shape of the original latent w spatial tensor
        orig_shape = latent_w_spat.shape

        # Prepare the current task palette of all spatial sizes necessary
        latent_w_spat_pyr = {"s1": latent_w_spat}
        all_glob_strides = [2, 4, 8, 16, 32]
        assert orig_shape[2] % all_glob_strides[-1] == 0
        assert orig_shape[3] % all_glob_strides[-1] == 0
        for glob_stride in [2, 4, 8, 16, 32]:
            prev_shape = latent_w_spat_pyr[f"s{int(glob_stride/2)}"].shape
            # Current pyramid member is interpolated from the previous one
            latent_w_spat_pyr[f"s{glob_stride}"] = F.interpolate(
                latent_w_spat_pyr[f"s{int(glob_stride/2)}"], 
                size=(prev_shape[2]//2, prev_shape[3]//2), 
                mode='bilinear'
            )
        
        return latent_w_spat_pyr

    def get_latent_w_dim(self):
        return self.cfg["latent_w_dim"]
    
    def get_z_code_dim(self):
        return self.z_code_dim
    
    def get_encoder_parameters(self):
        # Returns the encoders parameters
        return self.encoder.parameters()
    
    def get_decoder_parameters(self):
        # Returns the decoders parameters
        return self.decoder.parameters()