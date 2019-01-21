import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import DetectionCfg
from core.prior_box import PriorBox

Prior = PriorBox(DetectionCfg)


class Bottleneck(nn.Module):
    r"""
    shortcut of Resnet, pre-activation is applied
    """
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, rate=1):
        r"""
        init Bottleneck
        :param in_channel: number of input channels
        :param middle_channel: number of middle channels
        :param stride: stride of middle convolution
        """
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=rate, dilation=rate, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self._init_weight()

    def forward(self, inputs):
        out = self.conv1(F.relu(self.bn1(inputs), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.conv3(F.relu(self.bn3(out), inplace=True))
        out += self.downsample(inputs)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class HeadModule(nn.Module):

    def __init__(self, anchor_num, cls_num):
        r"""
        insert detection head into model
        :param anchor_num: number of anchors in this detection layer
        :param number: number of convolution layers
        :return: detection sequential modules
        """
        super(HeadModule, self).__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True)
                                        )

        self.loc = nn.Conv2d(64, anchor_num*4, kernel_size=1, stride=1, padding=0)
        self.cls = nn.Conv2d(64, anchor_num*cls_num, kernel_size=1, stride=1, padding=0)
        self._init_weight()

    def forward(self, inputs):
        out = self.bottleneck(inputs)
        loc = self.loc(out)
        cls = self.cls(out)

        return loc, cls

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP_module(nn.Module):
    r"""
    Atrous Spatial Pyramid Pooling
    """
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class PerceptionNet(nn.Module):

    def __init__(self, cls_num, number_block):
        r"""
        perception net including detection and segmentation
        :param cls_num: number of classes, including background
        :param number_block: numbers of convolution layers in every resnet bottleneck
        """
        super(PerceptionNet, self).__init__()

        assert len(number_block) == 4

        self.prior = Variable(Prior.forward().cuda())
        self.cls_num = cls_num

        self.in_channels = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)

        # Bottom -> Top
        self.layer1 = self._insert_bottleneck(Bottleneck, 32, number_block[0], stride=1, rate=1)
        self.layer2 = self._insert_bottleneck(Bottleneck, 64, number_block[1], stride=2, rate=1)
        self.layer3 = self._insert_bottleneck(Bottleneck, 64, number_block[2], stride=2, rate=1)
        self.layer4 = self._insert_bottleneck(Bottleneck, 128, number_block[3], stride=2, rate=2)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.down = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        # Lateral connections, reduce_layer reduce aliasing effect
        self.lateral_layer1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.reduce_layer1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lateral_layer2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.reduce_layer2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 16, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        self.aspp1 = ASPP_module(256, 64, rate=1)
        self.aspp2 = ASPP_module(256, 64, rate=6)
        self.aspp3 = ASPP_module(256, 64, rate=12)
        self.aspp4 = ASPP_module(256, 64, rate=18)

        self.detect1 = HeadModule(Prior.number, self.cls_num)
        self.detect2 = HeadModule(Prior.number, self.cls_num)
        self.detect3 = HeadModule(Prior.number, self.cls_num)

        self.last_conv = nn.Sequential(nn.Conv2d(80, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, 3, kernel_size=1, stride=1))

        self.activation = nn.Tanh()

        self._init_weight()

    def forward(self, inputs):

        output = []
        # Bottom -> Top
        conv = self.conv1(inputs)
        conv = self.layer1(conv)

        bottom_feature = F.relu(self.bn3(self.conv3(conv)), inplace=True)

        conv = self.layer2(conv)
        conv = self.layer3(conv)
        c5 = F.relu(self.bn4(self.layer4(conv)), inplace=True)
        c6 = F.relu(self.bn5(self.conv5(c5)), inplace=True)
        c7 = F.relu(self.bn6(self.conv6(c6)), inplace=True)

        # Lateral connections
        f7 = self.down(c7)
        f6 = self._upsample_addition(f7, self.lateral_layer1(c6))
        f6 = self.reduce_layer1(f6)
        f5 = self._upsample_addition(f6, self.lateral_layer2(c5))
        f5 = self.reduce_layer2(f5)

        x1 = self.aspp1(c5)
        x2 = self.aspp2(c5)
        x3 = self.aspp3(c5)
        x4 = self.aspp4(c5)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        x = self._upsample_concat(x, bottom_feature)
        x = self.last_conv(x)
        x = F.interpolate(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)

        output.append(self.detect3(f5))
        output.append(self.detect2(f6))
        output.append(self.detect1(f7))

        loc_logits = torch.cat([x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4) for x, _ in output], dim=1)
        cls_logits = torch.cat([y.permute(0, 2, 3, 1).contiguous().view(y.size(0), -1, self.cls_num)
                                for _, y in output], dim=1)

        return (loc_logits, cls_logits), x

    def _insert_bottleneck(self, block, planes, num_blocks, stride, rate=1):
        r"""
        insert Resnet shortcut into model
        :param block: Bottleneck
        :param channels: middle channels in Bottleneck
        :param number: number of Bottlenecks
        :param stride: stride of first Bottleneck
        :return: sequential of layers
        """
        strides = [stride] + [1]*(num_blocks-2)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride))
            self.in_channels = planes * block.expansion
        layers.append(block(self.in_channels, planes, 1, rate))
        return nn.Sequential(*layers)

    @staticmethod
    def _upsample_addition(a, b):
        r"""
        upsample a, then add b
        :param a:
        :param b:
        :return:
        """
        _, _, H, W = b.size()
        return F.interpolate(a, size=(H, W), mode='bilinear', align_corners=True) + b

    @staticmethod
    def _upsample_concat(a, b):
        r"""
        upsample a, then add b
        :param a:
        :param b:
        :return:
        """
        _, _, H, W = b.size()
        return torch.cat((F.interpolate(a, size=(H, W), mode='bilinear', align_corners=True), b), dim=1)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
