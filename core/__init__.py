from .PerceptionNet import PerceptionNet
from .data_loader import BDDLoader
from .loss import detection_loss
from .utils import *
from .detection import Detect
from .augmentations import Augmentation


def detection_collate(batch):
    r"""
    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    :param batch: (tuple) A tuple of tensor images and lists of annotations
    :return: A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
            3) (tensor) batch of segmentation labels stacked on their 0 dim
    """
    targets = []
    imgs = []
    segs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        segs.append(sample[2])
    return torch.stack(imgs, 0), targets, segs
