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


class BasicBlock(nn.Module):
    # Attention: Architecture of BasicBuildBlock
    #     x     # [batch, in_channel, height, width]
    #     |
    #     | \
    #     |  * Residual Path
    #     |  | 
    #     |  Conv2d(in_channels, out_channels, stride=stride, ksize=3) # may be used as dowmsampling
    #     |  |
    #     |  (BatchNorm2d, ReLu)
    #     |  |
    #     |  Conv2d(out_channels, out_channels, stride=1, ksize=3)
    #     |  |
    #     |  (BatchNorm2d)
    #     | /
    #     |
    #     x + F(x)      # [batch, in_channel, height, width]
    # Notes:
    #   *  1. BasicBuildBlock is used for ResNet18 and ResNet34
    #   *  2. Stride = 2 is used in the first block of layer 3/4/5, an other blocks are 1

    # refer to pytorch implementation
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        stride: Optional[int] = 1,
        with_projection: Optional[bool] = None
    ) -> None:
        assert in_channels == feature_channels, f"{Fore.RED}BasicBuildBlock do not change input and output channels"
        super(BasicBlock, self).__init__()

        # Residual Path
        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=3, stride=stride, padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels, out_channels=feature_channels,
                kernel_size=3, stride=1, padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels)
        )

        # if with projection, default use type B: using a matrix Ws as linear projection (eqn2), i.e., F(x) + Wsx 
        # in practice, 1x1 conv is the linear projection. Batchnorm is add after all convolution
        self.with_prjection = with_projection
        if with_projection is None:
            self.with_prjection = True if in_channels != feature_channels or stride == 2 else False
        
        # input and output channel mismatch due to 
        #       1. First block in the layer (output channel of last layer mismatch with current layer)
        #       2. Downsampled layer in layer 3/4/5 by useing stride of 2
        # projection on x is needed to make dimension of x match F(x)'s dimension
        # in detail, channels and [height, width] should be maintained.
        # shape is only changed in the first conv using stride of 2
        if self.with_prjection or stride == 2:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=feature_channels,
                    kernel_size=1, stride=stride, padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(feature_channels)
            )
        else:
            self.projection = nn.Sequential()
        
        self.output_relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.residual_path(x)
        x = self.projection(x)
        hx = fx + x
        return self.output_relu(hx)


class BottleneckBlock(nn.Module):
    # Attention: Architecture of BasicBuildBlock
    #     x     # [batch, in_channel, height, width]
    #     |
    #     | \
    #     |  * Residual Path
    #     |  | 
    #     |  Conv2d(in_channels, 64, stride=1, ksize=1) # modified as ResNet v1.5
    #     |  |
    #     |  (BatchNorm2d, ReLu)
    #     |  |
    #     |  Conv2d(64, 64, stride=stride, ksize=3)
    #     |  |
    #     |  (BatchNorm2d, ReLu)
    #     |  |
    #     |  Conv2d(64, feature_channels * self.expansion) # for layer 2/3/4/5, 
    #     | /                                              # feature_channels is 64/128/256/512
    #     |                                                # and final output channel are 256/512/1024/2048
    #     x + F(x)      # [batch, in_channel, height, width]
    # Notes:
    #   *  1. BasicBuildBlock is used for ResNet18 and ResNet34
    #   *  2. Stride = 2 is used in the first block of layer 3/4/5, an other blocks are 1

    expansion: int = 4
    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        stride: Optional[int] = 1,
        with_projection: Optional[bool] = None
    ) -> None:
        super(BottleneckBlock, self).__init__()

        # Residual Path
        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=64,
                kernel_size=1, stride=1, padding=0,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=3, stride=stride, padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=feature_channels * self.expansion,
                kernel_size=1, stride=1, padding=0,
                bias=False
            )
        )

        self.with_prjection = with_projection
        if with_projection is None:
            self.with_prjection = True if in_channels != feature_channels or stride == 2 else False
        
        if self.with_prjection:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=feature_channels * self.expansion,
                    kernel_size=1, stride=stride, padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(feature_channels * self.expansion)
            )
        else:
            self.projection = nn.Sequential()

        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.residual_path(x)
        x = self.projection(x)
        hx = fx + x
        return self.output_relu(hx)


if __name__ == "__main__":
    x = torch.randn(32, 64, 224, 224)
    
    # Test Basic Layer
    # downsample layer
    bb_downsample = BasicBlock(in_channels=64, feature_channels=64, stride=2)
    # normal layer
    bb_normal = BasicBlock(in_channels=64, feature_channels=64, stride=1)
    # Attention: projection layer will never be used for BasicBlock
    # bb_projection = BasicBlock(in_channels=128, feature_channels=)
    print(bb_downsample(x).shape)
    print(bb_normal(x).shape)

    # Test Bottleneck Layer
    # downsample layer
    bb_downsample = BottleneckBlock(in_channels=64, feature_channels=64, stride=2)
    # Attention: normal layer will never be used for BottleneckBlock
    # bb_normal = BottleneckBlock(in_channels=64, feature_channels=64, stride=1, with_projection=True)
    # projection layer
    bb_projection = BottleneckBlock(in_channels=64, feature_channels=64, stride=1, with_projection=True)
    # downsample layer
    print(bb_downsample(x).shape)
    print(bb_normal(x).shape)
