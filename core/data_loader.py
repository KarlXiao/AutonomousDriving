import os
import json
import cv2
import numpy as np
import torch
import torch.utils.data as data


road_obj = ['car']
# road_obj = ['car', 'bus', 'person', 'truck']
# road_obj = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'traffic light', 'traffic sign', 'train', 'truck']

drivable_area = ['alternative', 'direct']

lane_style = ['crosswalk', 'double other', 'double white', 'double yellow', 'road curb', 'single other', 'single white', 'single yellow']


class BDDLoader(data.Dataset):

    def __init__(self, json_file, im_dir, im_dim):
        if 'val' in json_file:
            self.stage = 'val'
        elif 'train' in json_file:
            self.stage = 'train'
        else:
            self.stage = 'test'

        self.im_dir = im_dir
        self.im_dim = im_dim

        with open(json_file) as f:
            self.data = json.load(f)

        self.num_examples = len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        im_name = item['name']
        im_path = os.path.join(self.im_dir, self.stage, im_name)
        image_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        height, width, channel = image_raw.shape

        seg_label_path = im_path.replace('images', 'labels').replace('.jpg', '_drivable_id.png')
        seg_raw = cv2.imread(seg_label_path)[:, :, 0]

        label = []
        bbox = []
        for sub_item in item['labels']:
            if sub_item['category'] in road_obj:

                box = sub_item['box2d']
                # filter out too small objects
                if (box['y2'] - box['y1']) / height < 0.1:
                    continue
                label.append(road_obj.index(sub_item['category']) + 1)
                bbox.append([box['x1'] / width, box['y1'] / height,
                             (box['x2'] - box['x1']) / width,
                             (box['y2'] - box['y1']) / height])

        image_raw = cv2.resize(image_raw, (self.im_dim[0], self.im_dim[1])) / 255.0
        seg_raw = cv2.resize(seg_raw, (self.im_dim[0], self.im_dim[1]), cv2.INTER_NEAREST)

        if len(label) == 0:
            label.append(0)
            bbox.append([0., 0., 0., 0.])

        labels = np.array(label)
        boxes = np.array(bbox)

        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(image_raw).permute(2, 0, 1).float(), target,\
               torch.from_numpy(seg_raw)

    def __len__(self):
        r"""
        :return:total number of examples in dataset
        """
        return self.num_examples


def drivable2color(seg):
    r"""
    :param seg: segmentation map
    :return: color map
    """
    colors = [[0, 0, 0], [217, 83, 79], [91, 192, 222]]
    color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        color[seg == i] = colors[i]
    return color
