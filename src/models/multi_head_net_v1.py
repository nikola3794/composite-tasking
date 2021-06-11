import torch
import torch.nn as nn

from .blocks.encoder_v1 import EncoderV1
from .blocks.decoder_v1 import DecoderV1
from .blocks.task_representation_block_v1 import TaskRepresentationBlockV1


class MultiHeadNetV1(nn.Module):

    def __init__(self, cfg, task_z_code_dict):
        super().__init__()

        # Load and check the model configuration dictionary
        self.cfg = cfg
        self._check_cfg()

        # Load the task code dictionary
        self.task_z_code_dict = task_z_code_dict
        
        # Create the encoder
        self.encoder = EncoderV1(cfg=self.cfg)
        
        # Create a decoder (head) for each task
        dec= {}
        for task_id in self.task_z_code_dict:
            assert self.cfg["which_cond"] in ["no_cond"]
            dec[str(task_id)] = DecoderV1(
                cfg=self.cfg, 
                enc_n_ch=self.encoder.get_n_channels_in_each_level()
            )

        self.decoders = nn.ModuleDict(dec)

    def _check_cfg(self):
        assert isinstance(self.cfg, dict)

        assert isinstance(self.cfg["latent_w_dim"], int)

        assert self.cfg["which_cond"] in ["no_cond"]

    def forward(self, x, task_palette, considered_task_ids, **kwargs):
        return self.forward_multi_task(
            x=x, 
            considered_task_ids=considered_task_ids
        )

    def forward_composite_task(self, x, task_palette, considered_task_ids):
        # Compute the encoder
        enc_output = self.encoder(x=x)

        task_palette = task_palette.unsqueeze(1).repeat(1,3,1,1)

        out_logits = (-1.0 * torch.ones_like(x))
        out_logits = out_logits.to(x.device) # TODO Figure out how to use torch lightnings device
        for task_id in considered_task_ids:
            # Predict every requested task and construct the output
            # based on the provided task palette
            out_logits_id = self.decoders[str(task_id)](enc_output=enc_output)
            out_logits = torch.where(task_palette==task_id, out_logits_id, out_logits)

        return out_logits
    
    def forward_multi_task(self, x, considered_task_ids):
        out_logits = {}

        # Compute the encoder
        enc_output = self.encoder(x=x)

        for task_id in considered_task_ids:
            # Predict every requested task 
            out_logits[task_id] = self.decoders[str(task_id)](enc_output=enc_output)

        return out_logits
    
    def forward_single_task(self, x, task_id):
        enc_output = self.encoder(x=x)
        out_logits = self.decoders[str(task_id)](enc_output=enc_output)

        return out_logits
        
    def get_encoder_parameters(self):
        # Returns the encoders parameters
        return self.encoder.parameters()
    
    def get_decoder_parameters(self):
        # Returns the decoders parameters
        return self.decoders.parameters()
