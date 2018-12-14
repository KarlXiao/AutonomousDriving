import os
import sys
import argparse
import tensorflow as tf
import tqdm
import cv2
import json
import numpy as np
from label import road_obj

FLAGS = None

parser = argparse.ArgumentParser(description='python code to generate tfrecord')
parser.add_argument('--dim', type=list, default=[320, 180], help='input size')
parser.add_argument('--im_dir', type=str, required=True, help='directory restores dataset')
parser.add_argument('--json_file', type=str, required=True, help='path to json file to parse')
parser.add_argument('--save_dir', type=str, default='data', help='directory to save tfrecord')


def _float_feature(value):
    if type(value) == list or type(value) == tuple:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    if type(value) == list or type(value) == tuple:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(unused_argv):
    r"""
    Converts a dataset to tfrecords
    """
    if 'val' in FLAGS.json_file:
        data_type = 'val'
    elif 'train' in FLAGS.json_file:
        data_type = 'train'

    with open(FLAGS.json_file) as f:
        data = json.load(f)

    num_examples = len(data)

    filename = os.path.join(FLAGS.save_dir, data_type + '.tfrecords')
    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in tqdm.trange(num_examples):
            item = data[index]
            im_name = item['name']
            im_path = os.path.join(FLAGS.im_dir, 'images', data_type, im_name)
            image_raw = cv2.imread(im_path)
            height, width, channel = image_raw.shape

            seg_label_path = im_path.replace('images', 'labels').replace('.jpg', '_drivable_id.png')
            seg_raw = cv2.imread(seg_label_path)[:, :, 0]

            image_raw = cv2.resize(image_raw, (FLAGS.dim[0], FLAGS.dim[1]))
            shape = image_raw.shape
            seg_raw = cv2.resize(seg_raw, (FLAGS.dim[0], FLAGS.dim[1]), cv2.INTER_NEAREST)
            image_raw = image_raw.tostring()
            seg_raw = seg_raw.tostring()

            label = []
            bbox = []
            number = 0
            for sub_item in item['labels']:
                if sub_item['category'] in road_obj:
                    label.append(road_obj.index(sub_item['category']) + 1)
                    box = sub_item['box2d']
                    bbox.append(np.array([box['x1']/width, box['y1']/height, box['x2']/width, box['y2']/height]))
                    number += 1

            labels = np.array(label)
            labels = labels.tostring()
            boxes = np.array(bbox)
            boxes = boxes.tostring()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _bytes_feature(labels),
                        'image_raw': _bytes_feature(image_raw),
                        'seg': _bytes_feature(seg_raw),
                        'bbox': _bytes_feature(boxes),
                        'shape': _int64_feature(shape),
                        'number': _int64_feature(number)
                    }))
            writer.write(example.SerializeToString())


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
