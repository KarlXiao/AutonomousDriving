import cv2
import numpy as np
from numpy import random
from math import *


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    r"""
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    :param box_a: Multiple bounding boxes, Shape: [num_boxes,4]
    :param box_b: Single bounding box, Shape: [4]
    :return: jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    r"""
    Composes several augmentations together.
    """

    def __init__(self, transforms):
        r"""
        :param transforms: list of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):

        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape

        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class PreResize(object):
    def __init__(self, size=[640, 480]):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size[0], self.size[1]))

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=[640, 480]):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):

        image = cv2.resize(image, (self.size[0], self.size[1]))

        return image, boxes, labels


class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        r"""
        Crop images
        :param image: the image being input during training
        :param boxes: the original bounding boxes in pt form
        :param labels: the class labels for each bbox
        :return: crop images
        """
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self):
        pass

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 1.5)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = 0
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class Rotate(object):
    def __init__(self, min_angle=-10, max_angle=10):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels
            
        height, width, _ = image.shape

        degree = random.uniform(self.min_angle, self.max_angle)

        height_new = int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
        width_new = int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))

        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

        mat_rotation[0, 2] += (width_new-width)/2
        mat_rotation[1, 2] += (height_new-height)/2

        img_rotation = cv2.warpAffine(image, mat_rotation, (width_new, height_new), borderValue=(0, 0, 0))

        for i in range(len(boxes)):
            box = boxes[i]

            x_0 = box[0]*mat_rotation[0][0]+box[1]*mat_rotation[0][1]+mat_rotation[0][2]
            x_2 = box[0]*mat_rotation[0][0]+box[3]*mat_rotation[0][1]+mat_rotation[0][2]
            x_1 = box[2]*mat_rotation[0][0]+box[1]*mat_rotation[0][1]+mat_rotation[0][2]
            x_3 = box[2]*mat_rotation[0][0]+box[3]*mat_rotation[0][1]+mat_rotation[0][2]

            boxes[i][0] = (x_0, x_2)[x_2 < x_0]
            boxes[i][2] = (x_1, x_3)[x_3 > x_1]

            y_0 = box[0]*mat_rotation[1][0]+box[1]*mat_rotation[1][1]+mat_rotation[1][2]
            y_2 = box[0]*mat_rotation[1][0]+box[3]*mat_rotation[1][1]+mat_rotation[1][2]
            y_1 = box[2]*mat_rotation[1][0]+box[1]*mat_rotation[1][1]+mat_rotation[1][2]
            y_3 = box[2]*mat_rotation[1][0]+box[3]*mat_rotation[1][1]+mat_rotation[1][2]

            boxes[i][1] = (y_0, y_1)[y_1 < y_0]
            boxes[i][3] = (y_2, y_3)[y_3 > y_2]

        return img_rotation, boxes, labels


class Augmentation(object):
    def __init__(self, size=[640, 480]):
        self.size = size
        self.augment = Compose([
            PreResize(self.size),
            ConvertFromInts(),
            ToAbsoluteCoords(),
            Rotate(),
            Expand(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
