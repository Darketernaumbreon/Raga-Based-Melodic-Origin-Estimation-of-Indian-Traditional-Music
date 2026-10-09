# Write your CNNBackbone class here
import torch.nn as nn

from .blocks import ConvBlock


class CNNBackbone(nn.Module):

    def __init__(self, pool_kernel=2):
        super().__init__()

        self.block1 = ConvBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, pool_kernel=pool_kernel)
        self.block2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, pool_kernel=pool_kernel)
        self.block3 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, pool_kernel=pool_kernel)
        self.block4 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, pool_kernel=pool_kernel)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x