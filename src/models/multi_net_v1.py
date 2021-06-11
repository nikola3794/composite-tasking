import torch
import torch.nn as nn

from .blocks.encoder_v1 import EncoderV1
from .blocks.decoder_v1 import DecoderV1


class MultiNetV1(nn.Module):

    def __init__(self, cfg, task_z_code_dict):
        super().__init__()

        # Load and check the model configuration dictionary
        self.cfg = cfg
        self._check_cfg()

        # Load the task code dictionary
        self.task_z_code_dict = task_z_code_dict
        
        # Create the encoders and decoders
        # Each task has its own encoder and decoder
        enc = {}
        dec= {}
        for task_id in self.task_z_code_dict:
            # Create the encoder
            enc[str(task_id)] = EncoderV1(cfg=self.cfg)
            assert self.cfg["which_cond"] in ["no_cond"]
            # Create the decoder
            dec[str(task_id)] = DecoderV1(
                cfg=self.cfg, 
                enc_n_ch=enc[str(task_id)].get_n_channels_in_each_level()
            )
        self.decoders = nn.ModuleDict(dec)
        self.encoders = nn.ModuleDict(enc)
    

    def _check_cfg(self):
        assert isinstance(self.cfg, dict)
        
        assert self.cfg["which_cond"] in ["no_cond"]

    def forward_composite_task(self, x, task_palette, considered_task_ids):
        task_palette = task_palette.unsqueeze(1).repeat(1,3,1,1)

        out_logits = (-1.0 * torch.ones_like(x))
        out_logits = out_logits.to(x.device) # TODO Figure out how to use torch lgihtning device
        for task_id in considered_task_ids:
            # Predict every requested task and construct the output
            # based on the provided task palette
            enc_output_id = self.encoders[str(task_id)](x)
            out_logits_id = self.decoders[str(task_id)](enc_output=enc_output_id)
            out_logits = torch.where(task_palette==task_id, out_logits_id, out_logits)

        return out_logits  

    def forward_multi_task(self, x, considered_task_ids):        
        out_logits = {}
        for task_id in considered_task_ids:
            # Predict every requested task and construct the output
            # based on the provided task palette
            enc_output_id = self.encoders[str(task_id)](x)
            out_logits[task_id] = self.decoders[str(task_id)](enc_output=enc_output_id)

        return out_logits 

    def forward_single_task(self, x, task_id):
        assert isinstance(task_id, int)

        enc_output = self.encoders[str(task_id)](x)
        out_logits = self.decoders[str(task_id)](enc_output=enc_output)

        return out_logits

    def get_encoder_parameters(self):
        # Returns the encoders parameters
        return self.encoders.parameters()
    
    def get_decoder_parameters(self):
        # Returns the decoders parameters
        return self.decoders.parameters()





