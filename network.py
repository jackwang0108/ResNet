# Torch Library
import torch
import torch.nn as nn

# Standard Library
from typing import *
from pathlib import Path

# Third-Party Library
from colorama import Fore, Style, init

# My Library
from helper import ProjectPath

init(autoreset=True)


class BasicBuildBlock(nn.Module):
    # Attention: Architecture of BasicBuildBlock
    #     x     # [batch, in_channel, height, width]
    #     | \
    #     |  * Residual Path
    #     |  | 
    #     |  Conv2d(in_channels, out_channels, stride) # may be used as dowmsampling
    #     |  |
    #     |  (BatchNorm2d, ReLu)
    #     |  |
    #     |  Conv2d(out_channels, out_channels, stride=1)
    #     |  |
    #     |  (BatchNorm2d)
    #     | /
    #     x + F(x)      # [batch, in_channel, height, width]
    # Notes:
    #   *  1. BasicBuildBlock is used for ResNet18 and ResNet34
    #   *  2. Stride = 2 is used in the first block of layer 3/4/5, an other blocks are 1

    # refer to pytorch implementation
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Optional[int] = 1,
        with_projection: Optional[bool] = None
    ) -> None:
        assert in_channels == out_channels, f"{Fore.RED}BasicBuildBlock do not change input and output channels"
        super(BasicBuildBlock, self).__init__()

        # Residual Path
        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=3, stride=stride, padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        # if with projection, default use type B: using a matrix Ws as linear projection (eqn2), i.e., F(x) + Wsx 
        # in practice, 1x1 conv is the linear projection. Batchnorm is additional
        self.with_prjection = with_projection
        if with_projection is None:
            self.with_prjection = True if in_channels != out_channels or stride == 2 else False
        
        # input and output channel mismatch due to 
        #       1. First block in the layer (output channel of last layer mismatch with current layer)
        #       2. Downsampled layer in layer 3/4/5 by useing stride of 2
        # projection on x is needed to make dimension of x match F(x)'s dimension
        # in detail, channels and [height, width] should be maintained.
        # shape is only changed in the first conv using stride of 2
        if self.with_prjection or stride == 2:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=stride, padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.projection = nn.Sequential()
        
        self.output_relu = nn.ReLU(inplace=True)




