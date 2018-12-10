import os
import json
import argparse
import numpy as np
import cv2
from label import road_obj

parser = argparse.ArgumentParser(description='python code to visualizing data')
parser.add_argument('--im_dir', type=str, required=True)
parser.add_argument('--json_file', type=str, required=True, help='path to json file to parse')
parser.add_argument('--save_dir', type=str, default='data', help='directory to save tfrecords')

args = parser.parse_args()


def run():
    if 'val' in args.json_file:
        data_type = 'val'
    elif 'train' in args.json_file:
        data_type = 'train'

    with open(args.json_file) as f:
        data = json.load(f)
    for item in data:
        im_name = item['name']
        im_path = os.path.join(args.im_dir, 'images', data_type, im_name)
        im = cv2.imread(im_path)

        seg_label_path = im_path.replace('images', 'labels').replace('.jpg', '_drivable_id.png')
        seg = cv2.imread(seg_label_path)

        seg_cor = drivable2color(seg[:, :, 0])

        im[seg_cor > 0] = 0

        for sub_item in item['labels']:
            if sub_item['category'] in road_obj:
                label = road_obj.index(sub_item['category'])+8

                box = sub_item['box2d']
                box = np.array([box['x1'], box['y1'], box['x2'], box['y2']])
                cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 2)
                cv2.putText(im, '{}:{}'.format(label, sub_item['category']),
                            (int(box[0]), int(box[1])), 2, 0.5, (0, 255, 0))

        cv2.imshow('im', seg_cor+im)
        cv2.waitKey(0)


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


if __name__ == '__main__':
    run()
