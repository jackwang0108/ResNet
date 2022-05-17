# Torch Library
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# Standard Library
from typing import *
from pathlib import Path
from collections import OrderedDict

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
        # assert in_channels == feature_channels, f"{Fore.RED}BasicBuildBlock do not change input and output channels"
        super(BasicBlock, self).__init__()

        # Residual Path
        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=feature_channels,
                kernel_size=3, stride=stride, padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=feature_channels, out_channels=feature_channels,
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
                in_channels=in_channels, out_channels=feature_channels,
                kernel_size=1, stride=1, padding=0,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels),
            nn.Conv2d(
                in_channels=feature_channels, out_channels=feature_channels,
                kernel_size=3, stride=stride, padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels),
            nn.Conv2d(
                in_channels=feature_channels, out_channels=feature_channels * self.expansion,
                kernel_size=1, stride=1, padding=0,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels * self.expansion),
            nn.ReLU(inplace=True),
        )

        self.with_prjection = with_projection
        if with_projection is None:
            self.with_prjection = True if in_channels != feature_channels * self.expansion or stride == 2 else False
        
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


class ResNet(nn.Module):

    def __init__(
        self,
        num_class: int,
        tiny_image: Optional[bool] = False,
        num_blocks: List[int] = None,
        block_type: Type[Union[BasicBlock, BottleneckBlock]] = None,
        torch_model: Optional[nn.Module] = None
    ) -> None:
        super(ResNet, self).__init__()
        self.name = "ResNet"
        # contruct my model
        if torch_model is None:
            assert not (num_blocks is None or block_type is None), f"{Fore.RED}You must specify block type and num_block, "\
                                                                f"or you should initialize via api like ResNet.resNet50()"
            # input transform
            if tiny_image:
                self.input_transform = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                    nn.BatchNorm2d(num_features=64),
                    nn.ReLU(inplace=True),
                    nn.Sequential()
                )
            else:
                self.input_transform = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(num_features=64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
            
            # body
            self.layer1: nn.Sequential
            self.layer2: nn.Sequential
            self.layer3: nn.Sequential
            self.layer4: nn.Sequential
            feature_channel = 64
            in_channel = 64
            for i in range(4):
                self.__dict__[f"layer{i+1}"] = self._make_layer(
                    block_type=block_type,
                    num_block=num_blocks[i],
                    feature_channel=feature_channel,
                    in_channels=in_channel
                )
                in_channel = feature_channel * block_type.expansion
                feature_channel *= 2
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.feature_extractor = nn.Sequential(OrderedDict({
                "input_transform": self.input_transform,
                "layer1": self.layer1,
                "layer2": self.layer2,
                "layer3": self.layer3,
                "layer4": self.layer4,
                "avgpool": self.avgpool
            }))

            # prediction layer
            # in paper, resnet18/34 final layer feature channel is 512, and output channel is 512 (adaptive 1x1 2d pooled)
            # while resnet50/101/152 final layer feature channel is 512, output channel is 2048 (adaptive 1x1 2d pooled)
            # so, input feature vector should have 512 * 1/4 features, and expandsion of BasicBlock/Bottleneck represents
            # the expansion of last layer feature channel to output channel
            self.classifier = nn.Linear(in_features=512 * block_type.expansion, out_features=num_class)
        else:
            # contruct torch model
            self.num_class = num_class
            self.network = torch_model
            self.input_transform = nn.Sequential(
                self.network.conv1,
                self.network.bn1,
                self.network.relu,
                self.network.maxpool
            )
            self.feature_extractor = nn.Sequential(OrderedDict({
                "input_transform": self.input_transform,
                "layer1": self.network.layer1,
                "layer2": self.network.layer2,
                "layer3": self.network.layer3,
                "layer4": self.network.layer4,
                "avgpool": self.network.avgpool
            }))
            self.classifier = self.network.fc

        # Pytorch init
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
    
    def _make_layer(
        self, 
        num_block: int,
        in_channels: int,
        feature_channel: int,
        block_type: Type[Union[BasicBlock, BottleneckBlock]],
    ) -> nn.Sequential:
        # # feature channel == in channel is only in layer1
        # if feature_channel == in_channels:
        #     dowm_channel_ratio = 1
        # else:
        #     dowm_channel_ratio = 1

        # building layers
        layers = []

        # first block is a downsample block, which uses stride of 2
        # do not use downsample in the first block of first layer, this is known as resnet v1.5
        # and is especially good for tiny imput image
        layers.append(block_type(in_channels=in_channels, feature_channels=feature_channel, stride=1 if feature_channel == 64 else 2))
        # layers.append(block_type(in_channels=in_channels, feature_channels=feature_channel, stride=2))

        for block_idx in range(num_block):
            # first block of the layer has beed added
            if block_idx == 0:
                continue
            layers.append(
                block_type(
                    in_channels=feature_channel * block_type.expansion,
                    feature_channels=feature_channel,
                    stride=1,
                    with_projection=False
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_vector: torch.Tensor = self.feature_extractor(x).flatten(start_dim=1)
        possibility_vector: torch.Tensor = self.classifier(feature_vector)
        return possibility_vector
    
    @staticmethod
    def _get_torch_model(
        network_func: Callable, 
        num_class: int, 
        tiny_image: bool, 
        pretrained: Optional[Union[Path, bool]],
        expansion: int = 1,
    ):
        # load pretrained network
        if isinstance(pretrained, bool):
            network = network_func(pretrained=pretrained)
        else:
            network = network_func()
            if pretrained is not None:
                network.load_state_dict(torch.load(pretrained))
        # remove conv1 (ksize=7) and maxpool to preserve the input
        if tiny_image:
            conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
            maxpool = nn.Sequential()
            network.conv1 = conv1
            network.maxpool = maxpool
        # change prediction layer
        if num_class != 1000:
            fc = nn.Linear(512 * expansion, out_features=num_class)
            network.fc = fc
        return network

    @classmethod
    def resnet18(cls, num_class: int, tiny_image: bool, torch_model: bool, pretrained: Optional[Union[Path, bool]] = None):
        if torch_model:
            network = ResNet._get_torch_model(resnet18, num_class, tiny_image, pretrained)
            obj = cls(num_class=num_class, tiny_image=tiny_image, torch_model=network)
            obj.name = "ResNet18"
            return obj
        else:
            assert pretrained is None or not pretrained or isinstance(pretrained, Path), f"{Fore.RED}Please offer pre-trained parameter path"
            network = cls(
                num_class=num_class, tiny_image=tiny_image,
                num_blocks=[2, 2, 2, 2],
                block_type=BasicBlock,
                torch_model=None
            )
            if pretrained is not None and pretrained:
                network.load_state_dict(torch.load(pretrained))
            network.name = "ResNet18"
            return network
    
    @classmethod
    def resnet34(cls, num_class: int, tiny_image: bool, torch_model: bool, pretrained: Optional[Union[Path, bool]] = None):
        if torch_model:
            network = ResNet._get_torch_model(resnet34, num_class, tiny_image, pretrained)
            obj = cls(num_class=num_class, tiny_image=tiny_image, torch_model=network)
            obj.name = "ResNet34"
            return obj
        else:
            assert pretrained is None or not pretrained or isinstance(pretrained, Path), f"{Fore.RED}Please offer pre-trained parameter path"
            network = cls(
                num_class=num_class, tiny_image=tiny_image,
                num_blocks=[3, 4, 6, 3],
                block_type=BasicBlock,
                torch_model=None
            )
            if pretrained is not None and pretrained:
                network.load_state_dict(torch.load(pretrained))
            network.name = "ResNet34"
            return network

    @classmethod
    def resnet50(cls, num_class: int, tiny_image: bool, torch_model: bool, pretrained: Optional[Union[Path, bool]] = None):
        if torch_model:
            network = ResNet._get_torch_model(resnet50, num_class, tiny_image, pretrained, expansion=4)
            obj = cls(num_class=num_class, tiny_image=tiny_image, torch_model=network)
            obj.name = "ResNet50"
            return obj
        else:
            assert pretrained is None or not pretrained or isinstance(pretrained, Path), f"{Fore.RED}Please offer pre-trained parameter path"
            network = cls(
                num_class=num_class, tiny_image=tiny_image,
                num_blocks=[3, 4, 6, 3],
                block_type=BottleneckBlock,
                torch_model=None
            )
            if pretrained is not None and pretrained:
                    network.load_state_dict(torch.load(pretrained))
            network.name = "ResNet50"
            return network

    @classmethod
    def resnet101(cls, num_class: int, tiny_image: bool, torch_model: bool, pretrained: Optional[Union[Path, bool]] = None):
        if torch_model:
            network = ResNet._get_torch_model(resnet101, num_class, tiny_image, pretrained, expansion=4)
            obj = cls(num_class=num_class, tiny_image=tiny_image, torch_model=network)
            obj.name = "ResNet101"
            return obj
        else:
            assert pretrained is None or not pretrained or isinstance(pretrained, Path), f"{Fore.RED}Please offer pre-trained parameter path"
            network = cls(
                num_class=num_class, tiny_image=tiny_image,
                num_blocks=[3, 4, 23, 3],
                block_type=BottleneckBlock,
                torch_model=None
            )
            if pretrained is not None and pretrained:
                network.load_state_dict(torch.load(pretrained))
            network.name = "ResNet101"
            return network

    @classmethod
    def resnet152(cls, num_class: int, tiny_image: bool, torch_model: bool, pretrained: Optional[Union[Path, bool]] = None):
        if torch_model:
            network = ResNet._get_torch_model(resnet152, num_class, tiny_image, pretrained, expansion=4)
            obj = cls(num_class=num_class, tiny_image=tiny_image, torch_model=network)
            obj.name = "ResNet152"
            return obj
        else:
            assert pretrained is None or not pretrained or isinstance(pretrained, Path), f"{Fore.RED}Please offer pre-trained parameter path"
            network = cls(
                num_class=num_class, tiny_image=tiny_image,
                num_blocks=[3, 8, 36, 3],
                block_type=BottleneckBlock,
                torch_model=None
            )
            if pretrained is not None and pretrained:
                network.load_state_dict(torch.load(pretrained))
            network.name = "ResNet152"
            return network


if __name__ == "__main__":
    # # Test Basic Block
    # x1 = torch.randn(32, 64, 224, 224)
    # # Attention: downsample block is the first block in the layer 2 for resnet18/34
    # bb_downsample = BasicBlock(in_channels=64, feature_channels=64, stride=2)
    # # Attention: normal block is the rest block in the layer 2/3/4/5 for resnet18/34
    # bb_normal = BasicBlock(in_channels=64, feature_channels=64, stride=1)
    # # Attention: inflation downsample block is the actually first block in the layer 3/4/5 for resnet18/34
    # bb_inflation_downsample = BasicBlock(in_channels=64, feature_channels=128, stride=2)
    # print(bb_downsample(x1).shape)
    # print(bb_normal(x1).shape)
    # print(bb_inflation(x1).shape)
    # print(bb_inflation_downsample(x1).shape)

    # Test Bottleneck Layer
    # x1 = torch.randn(32, 64, 224, 224)
    # x2 = torch.randn(32, 256, 112, 112, device="cuda:0")
    # # Attention: downsample block is the first block in the layer 2 for resnet50/101/152
    # bb_downsample = BottleneckBlock(in_channels=64, feature_channels=64, stride=2)
    # # Attention: dowsample channel at 4 times is the block within layers for resnet50/101/152
    # bb_downsample_channel_wi_layer = BottleneckBlock(in_channels=256, feature_channels=64, stride=1).cuda()
    # # Attention: downsample channel block is the first block for resnet50/101/152 between layers
    # bb_downsample_channel_bt_layer = BottleneckBlock(in_channels=256, feature_channels=128, stride=1).cuda()
    # down sample ch
    # print(bb_downsample(x1).shape)
    # print(bb_downsample_channel_wi_layer(x2).shape)
    # print(bb_downsample_channel_bt_layer(x2).shape)

    # test resnet
    tiny = torch.randn(8, 3, 32, 32)
    large = torch.randn(8, 3, 224, 224)
    resnet = ResNet.resnet50(num_class=100, tiny_image=True, torch_model=True, pretrained=False)
    print(resnet)
    print(resnet(tiny).shape)

