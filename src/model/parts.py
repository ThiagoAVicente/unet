"""
vcnt 2025
"""

import torch.nn as nn
import torch
from typing import Optional

class DoubleConv(nn.Module):
    """Double Convolution Block with ReLU activations. Uses padding to preserve dimensions and kernel size of 3."""
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Up(nn.Module):
    """Module for upscalling followed by double conv. Uses bilinear upsampling."""
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1:torch.Tensor, x2:Optional[torch.Tensor]=None) -> torch.Tensor:
        x1 = self.up(x1)

        # ensure x1 and x2 have the same spatial dimensions
        if x2 is None or x1.size() != x2.size():
            return self.conv(x1)

        # skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class Bottom(nn.Module):
    """Bottom layer of the U-Net."""
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
