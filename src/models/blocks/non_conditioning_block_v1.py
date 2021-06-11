import torch
import torch.nn as nn


class NoConditioningBlockV1(nn.Module):
    """
    This is a regular block that applies a convolutonal layer,
    followed by a BatchNorm and activation fuction.
    It serves to run the Composite Tasking Network in a
    regular mode, without any task-wise conditioning for specific pixels.
    """

    def __init__(self, in_channels, out_channels, ks, **kwargs):
        super().__init__()

        # Convolutional layer
        pd = ks // 2
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=ks, 
            padding=pd
        )

        # Regular BatchNorm
        # (no spatial conditioning)
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels, 
            affine=True
        )

        # Activation - leaky relu
        self.act = nn.LeakyReLU()

    def forward(self, feature_map, **kwargs):
        result = self.conv1(feature_map)
        result = self.batch_norm(result)
        result = self.act(result)
        return result

    def get_all_param_except_bn(self):
        return self.conv1.parameters()
    
    def get_bn_parametern(self):
        return self.batch_norm.parameters()