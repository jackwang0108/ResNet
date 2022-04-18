# torch library
import torch
import torch.nn as nn

# standard library
from typing import *

# third-party libraries
from colorama import Fore, init


init(autoreset=True)


class BasicResidualBlock(nn.Module):
    __name__ = "ResidualBlock"

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 with_projection: Union[bool, None] = None) -> None:
        """
        Pytorch implementation of basic residual block for shallow resnet, e.g., resnet18 and resnet34
        Args:
            in_channels (int): in channels
            out_channels (int): out channels
            stride (int): stride of the first 3*3 convolution layer, resnet do not use maxpool, instead, it uses
                convolution as downsample. Set stride to 1 will keep the input shape while set to 2 means downsample
                the image to half
            with_projection (Union[bool, None]): if use project in the shortcut connection, set to None will let the
                model decide.
        """
        super(BasicResidualBlock, self).__init__()
        # Residual path, bias is omitted (as said in original paper)
        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=(3, 3),
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        # shortcut connection
        # with 1D convolution (?)
        if with_projection is None:
            self.with_projection = True if in_channels != out_channels else False
        else:
            self.with_projection = with_projection

        if with_projection or stride == 2:
            # using 2D convolution of 1*1 kernel to match the input and output channels
            # or downsample
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        # output relu
        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use the original notation in paper
        fx = self.residual_path(x)
        x = self.shortcut(x)
        hx = fx + x
        return self.output_relu(hx)


class BottleneckResidualBlock(nn.Module):
    __name__ = "ResidualBlock"

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 with_projection: Union[bool, None] = None) -> None:
        """
        Pytorch implementation of bottleneck residual block for shallow resnet, e.g., resnet18 and resnet34
        Args:
            in_channels (int): in channels
            out_channels (int): out channels
            stride (int): stride of the first 3*3 convolution layer, resnet do not use maxpool, instead, it uses
                convolution as downsample. Set stride to 1 will keep the input shape while set to 2 means downsample
                the image to half
            with_projection (Union[bool, None]): if use project in the shortcut connection, set to None will let the
                model decide.
        """
        super(BottleneckResidualBlock, self).__init__()
        # Residual path, bias is omitted (as said in original paper)
        # using 2D convolution of 1*1 kernel to reduce parameters when process large input, ImageNet for example.
        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut connection
        # with 1D convolution (?)
        if with_projection is None:
            self.with_projection = True if in_channels != out_channels else False
        else:
            self.with_projection = with_projection

        if with_projection or stride == 2:
            # using 2D convolution of 1*1 kernel to match the input and output shape
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        # output relu
        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use the original notation in paper
        fx = self.residual_path(x)
        x = self.shortcut(x)
        hx = fx + x
        return self.output_relu(hx)


class _ResNetBase(nn.Module):
    """ResNet pytorch implementation, offering ResNet variants implementation via __init__ parameters"""
    __name__ = "ResNetBase"

    # cifar like dataset, input images are small
    cifar_like_dataset: List[str] = ["Cifar100", "cifar100"]
    # ImageNet like dataset, input images are large
    imagenet_like_datasets: List[str] = ["tinyImageNet200"]

    def __init__(self, target_dataset: str, num_blocks: List[int], num_class: Union[int, None] = None):
        if not target_dataset in (a := self.cifar_like_dataset + self.imagenet_like_datasets):
            raise NotImplementedError(f"Not implemented for dataset: {target_dataset}, currently available datasets are"
                                      f" {a}")
        
        super(_ResNetBase, self).__init__()

        self.target_dataset: str = target_dataset

        # first transformation layers for different datasets
        # if you want to run on you onw dataset, you need to write the transformation layers
        if self.target_dataset in self.imagenet_like_datasets:
            self.transform = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2),
                nn.MaxPool2d(stride=2)
            )
        elif self.target_dataset in self.cifar_like_dataset:
            self.transform = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            )

        # decide block type
        block_type = BasicResidualBlock if target_dataset in self.cifar_like_dataset else BottleneckResidualBlock

        self.stage1 = self._make_stage(
            block_type=block_type,
            in_channels=64,
            out_channels=64,
            first_stride=1,
            num_blocks=num_blocks[0]
        )
        self.stage2 = self._make_stage(
            block_type=block_type,
            in_channels=64,
            out_channels=128,
            first_stride=2,
            num_blocks=num_blocks[1]
        )
        self.stage3 = self._make_stage(
            block_type=block_type,
            in_channels=128,
            out_channels=256,
            first_stride=2,
            num_blocks=num_blocks[2]
        )
        self.stage4 = self._make_stage(
            block_type=block_type,
            in_channels=256,
            out_channels=512,
            first_stride=2,
            num_blocks=num_blocks[3]
        )
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_class: bool = num_class
        if num_class is not None:
            self.fc = nn.Linear(in_features=512, out_features=num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = self.transform(x)

        x1 = self.stage1(input)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        pooled = self.average_pool(x4)


        feature_map = pooled.view(pooled.shape[0], -1)
        if self.num_class is not None:
            # [batch_size, in_features]
            return self.fc(feature_map)
        return feature_map

    def _make_stage(self, block_type: Union[BasicResidualBlock, BottleneckResidualBlock],
                    in_channels: int, out_channels: int, num_blocks: int, first_stride: int) -> nn.Sequential:
        """
        _make_stage will make a stage. A stage means several basic/bottleneck residual blocks, in original paper, there
            are 4 stages.
        Args:
            block_type (Union[BasicResidualBlock, BottleneckResidualBlock]): type of the block
            in_channels (int): channel of input image of the stage
            out_channels (int): channel of output image of the stage
            num_blocks (int): number of blocks in this stage
            first_stride (int): stride of the the first block in the stage.
        Returns:
            nn.Sequential: stage of the layer
        """
        block_strides: List[int] = [first_stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_idx, stride in enumerate(block_strides):
            blocks.append(
                BasicResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
            )
            if block_idx == 0:
                # only the first block will do downsample and expand channels
                in_channels = out_channels
        return nn.Sequential(*blocks)


class ResNet34(_ResNetBase):
    __name__ = "ResNet34"

    def __init__(self, num_class: int, target_dataset: str):
        assert isinstance(num_class, int), f"{Fore.RED}Classification network must specify predicted class nums"
        super(ResNet34, self).__init__(
            num_class=num_class,
            target_dataset=target_dataset,
            num_blocks=[3, 4, 6, 3],
        )

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        preds: torch.Tensor = self(x)
        result = preds.argmax(dim=1)
        return result



if __name__ == "__main__":
    # [batch_size, channel, height, width]
    images = torch.randn(16, 3, 32, 32)

    # test blocks
    # # test normal
    # brb_keep = BasicResidualBlock(in_channels=3, out_channels=3, stride=1)
    # print(brb_keep(images).shape)
    # # test match channel and downsample
    # brb_down = BasicResidualBlock(in_channels=3, out_channels=6, stride=2)
    # print(brb_down(images).shape)
    #
    # # test normal
    # brb_keep = BottleneckResidualBlock(in_channels=3, out_channels=3, stride=1)
    # print(brb_keep(images).shape)
    #
    # # test match channel and downsample
    # brb_down = BottleneckResidualBlock(in_channels=3, out_channels=6, stride=2)
    # print(brb_down(images).shape)

    # test network
    resnet34 = ResNet34(num_class=100, target_dataset="Cifar100")
    # should be [16, 100]
    print(resnet34(images).shape)
    print(resnet34.inference(images))
