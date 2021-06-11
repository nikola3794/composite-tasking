import torch
import torch.nn as nn

from .blocks.encoder_v1 import EncoderV1
from .blocks.decoder_v1 import DecoderV1

class SingleTaskingNetV1(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Load and check the model configuration dictionary
        self.cfg = cfg
        self._check_cfg()

        # Create the encoder and decoder
        self.encoder = EncoderV1(cfg=self.cfg)
        self.decoder = DecoderV1(
                cfg=self.cfg, 
                enc_n_ch= self.encoder.get_n_channels_in_each_level()
            )
    
    def _check_cfg(self):
        assert isinstance(self.cfg, dict)
        
        assert self.cfg["which_cond"] in ["no_cond"]

    def forward_composite_task(self, x, task_palette, considered_task_ids):
        raise NotImplementedError  

    def forward_multi_task(self, x, considered_task_ids):        
        raise NotImplementedError

    def forward_single_task(self, x, task_id):
        assert isinstance(task_id, int)

        enc_output = self.encoder(x)
        out_logits = self.decoder(enc_output=enc_output)

        return out_logits

    def get_encoder_parameters(self):
        # Returns the encoders parameters
        return self.encoder.parameters()
    
    def get_decoder_parameters(self):
        # Returns the decoders parameters
        return self.decoder.parameters()





