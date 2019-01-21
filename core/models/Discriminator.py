import os
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    r"""
    Discriminator used in segmentation, use PixelDiscriminator as default
    """
    def __init__(self, in_planes, planes, use_sigmoid=True):
        super(Discriminator, self).__init__()

        self.net = [nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(planes, planes * 2, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(planes * 2),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(planes * 2, 1, kernel_size=1, stride=1, padding=0, bias=True)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

        self._init_weight()

    def forward(self, inputs):
        return self.net(inputs)

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
