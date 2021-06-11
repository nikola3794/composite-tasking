import torch
import torch.nn as nn
import torchvision


class EncoderV1(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load and check the model configuration dictionary
        self.cfg = cfg
        self._check_cfg()

        # Create the encoder architecture
        self._create_encoder()

    def _check_cfg(self):
        assert isinstance(self.cfg["encoder_arch"], str)
        assert isinstance(self.cfg["encoder_pre_trained"], bool)

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
        # Layer output size=(N, n_ch_level_1, H/2, W/2)
        self.layer1 = nn.Sequential(*base_layers[:3])
        # Layer output size=(N, n_ch_level_2, H/4, W/4)
        self.layer2 = nn.Sequential(*base_layers[3:5]) 
        # Layer output size=(N, n_ch_level_3, H/8, W/8)
        self.layer3 = base_layers[5]
        # Layer output size=(N, n_ch_level_4, H/16, W/16)
        self.layer4 = base_layers[6]
        # Layer output size=(N, n_ch_level_5, H/32, W/32)
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
    
    def get_n_channels_in_each_level(self):
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
