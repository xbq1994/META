# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math
import pdb
import torch
from torch import nn

from fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
)
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fastreid.utils import comm
from fastreid.layers.batch_norm import BatchNorm1,BatchNorm2,BatchNorm3,BatchNorm4
import pdb

logger = logging.getLogger(__name__)
model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}


class IN_cal(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super().__init__()

        self.register_buffer('mean_in', torch.zeros(64,num_features))
        self.register_buffer('var_in', torch.zeros(64,num_features))

    def forward(self, x):
        N, C, H, W = x.size()
        x_in = x.view(N, C, -1)

        self.mean_in = x_in.mean(-1)
        self.var_in = x_in.var(-1)
        return x
            
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.relu = nn.ReLU(inplace=False)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.in1 = IN_cal(planes)
        self.bn11 = BatchNorm1(planes)
        self.bn12 = BatchNorm2(planes)
        self.bn13 = BatchNorm3(planes)
        self.bn14 = BatchNorm4(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.in2 = IN_cal(planes)
        self.bn21 = BatchNorm1(planes)
        self.bn22 = BatchNorm2(planes)
        self.bn23 = BatchNorm3(planes)
        self.bn24 = BatchNorm4(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.in3 = IN_cal(planes * self.expansion)
        self.bn31 = BatchNorm1(planes * self.expansion)
        self.bn32 = BatchNorm2(planes * self.expansion)
        self.bn33 = BatchNorm3(planes * self.expansion)
        self.bn34 = BatchNorm4(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, ids=1, data_name=''):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        if ids == 1:
            out = self.bn11(out)
        if ids == 2:
            out = self.bn12(out)
        if ids == 3:
            out = self.bn13(out)
        if ids == 4:
            out = self.bn14(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        if ids == 1:
            out = self.bn21(out)
        if ids == 2:
            out = self.bn22(out)
        if ids == 3:
            out = self.bn23(out)
        if ids == 4:
            out = self.bn24(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3(out)
        if ids == 1:
            out = self.bn31(out)
        if ids == 2:
            out = self.bn32(out)
        if ids == 3:
            out = self.bn33(out)
        if ids == 4:
            out = self.bn34(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)
            if ids == 1:
                residual = self.downsample[2](residual)
            if ids == 2:
                residual = self.downsample[3](residual)
            if ids == 3:
                residual = self.downsample[4](residual)
            if ids == 4:
                residual = self.downsample[5](residual)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_expert(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck_expert, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.in1 = IN_cal(planes)
        self.bn11 = BatchNorm1(planes)
        self.bn12 = BatchNorm2(planes)
        self.bn13 = BatchNorm3(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.in2 = IN_cal(planes)
        self.bn21 = BatchNorm1(planes)
        self.bn22 = BatchNorm2(planes)
        self.bn23 = BatchNorm3(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.in3 = IN_cal(planes * self.expansion)
        self.bn31 = BatchNorm1(planes * self.expansion)
        self.bn32 = BatchNorm2(planes * self.expansion)
        self.bn33 = BatchNorm3(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, ids=1, data_name=''):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        if ids == 1:
            out = self.bn11(out)
        if ids == 2:
            out = self.bn12(out)
        if ids == 3:
            out = self.bn13(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        if ids == 1:
            out = self.bn21(out)
        if ids == 2:
            out = self.bn22(out)
        if ids == 3:
            out = self.bn23(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3(out)
        if ids == 1:
            out = self.bn31(out)
        if ids == 2:
            out = self.bn32(out)
        if ids == 3:
            out = self.bn33(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)
            if ids == 1:
                residual = self.downsample[2](residual)
            if ids == 2:
                residual = self.downsample[3](residual)
            if ids == 3:
                residual = self.downsample[4](residual)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_agg_in(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck_agg_in, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn11 = nn.InstanceNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn21 = nn.InstanceNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn31 = nn.InstanceNorm2d(planes * self.expansion, affine=True)
        self.relu = nn.ReLU(inplace=False)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, ids=1):
        residual = x

        out = self.conv1(x)
        out = self.bn11(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn21(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn31(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_agg_bn(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck_agg_bn, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn11 = BatchNorm4(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn21 = BatchNorm4(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn31 = BatchNorm4(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, ids=1):
        residual = x

        out = self.conv1(x)
        out = self.bn11(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn21(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn31(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)

        return out
                        
class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.in1 = IN_cal(64)
        self.bn11 = BatchNorm1(64) 
        self.bn12 = BatchNorm2(64)
        self.bn13 = BatchNorm3(64)
        self.bn14 = BatchNorm4(64)
        self.relu = nn.ReLU(inplace=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.layer10 = layer1[0]
        self.layer11 = layer1[1]
        self.layer12 = layer1[2]
        layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer20 = layer2[0]
        self.layer21 = layer2[1]
        self.layer22 = layer2[2]
        self.layer23 = layer2[3]
        layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer30 = layer3[0]
        layer3 = self._make_layer_expert(Bottleneck_expert, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer31 = layer3[1]
        self.layer32 = layer3[2]
        self.layer33 = layer3[3]
        self.layer34 = layer3[4]
        self.layer35 = layer3[5]
        layer4 = self._make_layer_expert(Bottleneck_expert, 512, layers[3], last_stride, bn_norm, with_se=with_se)
        self.layer40 = layer4[0]
        self.layer41 = layer4[1]
        self.layer42 = layer4[2]
        
        self.random_init()

        # fmt: off
        if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
        else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
        # fmt: on

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                IN_cal(planes * block.expansion),
                BatchNorm1(planes * block.expansion), 
                BatchNorm2(planes * block.expansion),
                BatchNorm3(planes * block.expansion),
                BatchNorm4(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        return layers

    def _make_layer_expert(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                IN_cal(planes * block.expansion),
                BatchNorm1(planes * block.expansion),
                BatchNorm2(planes * block.expansion),
                BatchNorm3(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        return layers
        
    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x, ids=1, data_name=''):
        x = self.conv1(x)
        x = self.in1(x)
        if ids == 1:
            x = self.bn11(x)
        if ids == 2:
            x = self.bn12(x)
        if ids == 3:
            x = self.bn13(x)
        if ids == 4:
            x = self.bn14(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # layer 1
        x = self.layer10(x, ids=ids, data_name=data_name)
        x = self.layer11(x, ids=ids, data_name=data_name)
        x = self.layer12(x, ids=ids, data_name=data_name)

        # layer 2
        x = self.layer20(x, ids=ids, data_name=data_name)
        x = self.layer21(x, ids=ids, data_name=data_name)
        x = self.layer22(x, ids=ids, data_name=data_name)
        x = self.layer23(x, ids=ids, data_name=data_name)

        # layer 3
        if ids == 4:
            x_inter = self.layer30(x, ids=ids, data_name=data_name)
            return x_inter

        x = self.layer30(x, ids=ids, data_name=data_name)
        x = self.layer31(x, ids=ids, data_name=data_name)
        x = self.layer32(x, ids=ids, data_name=data_name)
        x = self.layer33(x, ids=ids, data_name=data_name)
        x = self.layer34(x, ids=ids, data_name=data_name)
        x = self.layer35(x, ids=ids, data_name=data_name)

        # layer 4
        x = self.layer40(x, ids=ids, data_name=data_name)
        x = self.layer41(x, ids=ids, data_name=data_name)
        x = self.layer42(x, ids=ids, data_name=data_name)

        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResNet_agg(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.inplanes = 64
        super().__init__()

        layer3 = self._make_layer_agg_bn(Bottleneck_agg_bn, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer31 = layer3[1]
        self.layer32 = layer3[2]
        self.layer33 = layer3[3]
        self.layer34 = layer3[4]
        self.layer35 = layer3[5]
        layer4 = self._make_layer_agg_in(Bottleneck_agg_in, 512, layers[3], last_stride, bn_norm, with_se=with_se)
        self.layer40 = layer4[0]
        self.layer41 = layer4[1]
        self.layer42 = layer4[2]
        self.random_init()

        # fmt: off
        if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
        else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
        # fmt: on

    def _make_layer_agg_bn(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm4(planes * block.expansion), 
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        return layers

    def _make_layer_agg_in(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        return layers
        
    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x, ids=1, data_name=''):
        x = self.layer31(x, ids=ids)
        x = self.layer32(x, ids=ids)
        x = self.layer33(x, ids=ids)
        x = self.layer34(x, ids=ids)
        x = self.layer35(x, ids=ids)

        x = self.layer40(x, ids=ids)
        x = self.layer41(x, ids=ids)
        x = self.layer42(x, ids=ids)

        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1]

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    logger.info(f"Loading pretrained model from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))

    return state_dict


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '18x': BasicBlock,
        '34x': BasicBlock,
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage)
    model_agg = ResNet_agg(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                           num_blocks_per_stage, nl_layers_per_stage)
    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                #pdb.set_trace()
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            if with_ibn: key = 'ibn_' + key
            if with_se:  key = 'se_' + key

            state_dict = init_pretrained_weights(key)

        copyWeight(model, state_dict)
        copyWeight(model_agg, state_dict)
    return [model, model_agg]


def copyWeight(model, modelW):
    # copy state_dict to buffers
    curName = []
    for i in modelW.keys():
        curName.append(i)
    tarNames = set()

    for name in modelW.keys():
        # print(name)
        if name.startswith("module"):
            tarNames.add(".".join(name.split(".")[1:]))
        else:
            tarNames.add(name)
    
    for name_t, param_t in model.state_dict().items():
        
        name_t_oral = name_t  # conv1.weight; bn1_1.weight; layer1_0.conv1.weight; layer1_0.bn1_1.weight; layer1_0.bn1_1.running_mean; layer1_0.downsample.2.weight; layer1_0.downsample.2.running_var

# conv1.weight; bn11.weight; layer10.conv1.weight; layer10.bn11.weight; layer10.bn11.running_mean; layer10.downsample.2.weight; layer10.downsample.2.running_var

        if 'layer' in name_t:
            if 'bn' in name_t:
                name_t = name_t.split('.')[0][:-1] + '.' + name_t.split('.')[0][-1] + '.' + name_t.split('.')[-2][:-1] + '.' + name_t.split('.')[-1]
            elif 'downsample' in name_t:
                name_t = name_t.split('.')[0][:-1] + '.' + name_t.split('.')[0][-1] + '.' + name_t.split('.')[1]  + '.' + name_t.split('.')[2]  + '.' + name_t.split('.')[3]
                if '2.weight' in name_t or '3.weight' in name_t or '4.weight' in name_t or '5.weight' in name_t or '2.bias' in name_t or '3.bias' in name_t or '4.bias' in name_t  or '5.bias' in name_t or '2.running' in name_t or '3.running' in name_t or '4.running' in name_t or '5.running' in name_t:
                    name_t = name_t.split('downsample')[0] + 'downsample.1.' + name_t.split('.')[-1]
            else:
                name_t = name_t.split('.')[0][:-1] + '.' + name_t[6:]
        elif 'bn11' in name_t or 'bn12' in name_t or 'bn13' in name_t or 'bn14' in name_t:
            name_t = name_t.split('.')[0][:-1] + '.' + name_t.split('.')[-1]

        module_name_t = 'module.' + name_t

        if name_t in modelW:
            param = modelW[name_t]
            set_param(model, name_t_oral, param)
        elif module_name_t in modelW:
            param = modelW['module.' + name_t]
            set_param(model, name_t_oral, param)
        else:
            if 'num_batches_tracked' not in name_t_oral and 'mean_in' not in name_t_oral and 'var_in' not in name_t_oral:
                print(name_t_oral, name_t)
            continue


def set_param(curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        if name == 'weight':
            curr_mod.weight.data = param
        elif name == 'bias':
            curr_mod.bias.data = param
        elif name == 'running_mean':
            curr_mod.running_mean.data = param
        elif name == 'running_var':
            curr_mod.running_var.data = param
        else:
            pdb.set_trace()