import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
import math
from math import sqrt

import re

import random

class NoCondBlock(nn.Module):

    def __init__(self, cfg_txt, in_channels, out_channels, dim_latent_w, ks):
        super().__init__()

        # Convolutional layer
        pd = ks // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=pd)

        # Batch norm without spatial conditioning
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels, affine=True)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

    def forward(self, feature_map):

        result = self.conv1(feature_map)

        result = self.batch_norm(result)

        result = self.lrelu(result)

        return result
