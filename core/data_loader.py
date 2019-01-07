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

    def __init__(self, json_file, im_dir, im_dim, transform=None):
        if 'val' in json_file:
            self.stage = 'val'
        elif 'train' in json_file:
            self.stage = 'train'
        else:
            self.stage = 'test'

        self.im_dir = im_dir
        self.im_dim = im_dim
        self.transform = transform

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

        target = []
        for sub_item in item['labels']:
            if sub_item['category'] in road_obj:

                box = sub_item['box2d']

                # filter out too small objects
                if (box['x2'] - box['x1']) / width < 0.08 or (box['y2'] - box['y1']) / height < 0.08 or \
                    (box['y2'] - box['y1']) / (box['x2'] - box['x1']) > 2.5:
                    continue

                target.append([box['x1'] / width, box['y1'] / height, box['x2'] / width,
                               box['y2'] / height, road_obj.index(sub_item['category']) + 1])

        seg_raw = cv2.resize(seg_raw, (self.im_dim[0], self.im_dim[1]), cv2.INTER_NEAREST)

        if len(target) == 0:
            target.append([0., 0., 0., 0., 0])

        if self.transform is not None:
            target = np.array(target)
            image_raw, boxes, labels = self.transform(image_raw, target[:, :4], target[:, 4])

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        else:
            image_raw = cv2.resize(image_raw, (self.im_dim[0], self.im_dim[1]))

        return torch.from_numpy(image_raw / 255.0).permute(2, 0, 1).float(), target,\
               torch.from_numpy(seg_raw).long()

    def __len__(self):
        r"""
        :return:total number of examples in dataset
        """
        return self.num_examples
