from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    r"""
    Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['input_dim']
        self.number = len(cfg['aspect_ratios'])*len(cfg['scale_ratios'])*2
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.anchors = cfg['anchors']
        self.aspect_ratios = cfg['aspect_ratios']
        self.scale_ratios = cfg['scale_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[1]), range(f[0])):
                # unit center x,y
                cx = (j + 0.5) / f[0]
                cy = (i + 0.5) / f[1]

                for size_item in self.anchors[k]:
                    s_x = size_item/self.image_size[0]
                    s_y = size_item/self.image_size[1]

                    for ar in self.aspect_ratios:
                        for sr in self.scale_ratios:
                            mean += [cx, cy, s_x*sqrt(ar)*sr, s_y/sqrt(ar)*sr]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            torch.clamp(output, min=0, max=1)
        return output
