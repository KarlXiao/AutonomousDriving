import sys
import argparse
import cv2
import tensorflow as tf
import numpy as np
from label import road_obj

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='TFRecord file to check')


def drivable2color(seg):
    r"""
    :param seg: segmentation map
    :return: color map
    """
    colors = [[0, 0, 0],
              [217, 83, 79],
              [91, 192, 222]]
    color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        color[seg == i] = colors[i]
    return color


def _parse_function(proto):
    r"""
    :param proto: tfrecord files
    :return: parsed data
    """
    features = {
        'label': tf.FixedLenFeature([], tf.string),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'seg': tf.FixedLenFeature([], tf.string),
        'bbox': tf.FixedLenFeature([], tf.string),
        'shape': tf.FixedLenFeature([3], tf.int64),
        'number': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(proto, features)

    shape = parsed_features['shape']
    number = parsed_features['number']
    img = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
    img = tf.reshape(img, [shape[0], shape[1], shape[2]])
    seg = tf.decode_raw(parsed_features['seg'], tf.uint8)
    seg = tf.reshape(seg, [shape[0], shape[1]])
    boxes = tf.decode_raw(parsed_features['bbox'], tf.float64)
    boxes = tf.reshape(boxes, [number, 4])
    labels = tf.decode_raw(parsed_features['label'], tf.int64)

    return img, seg, boxes, labels, number


def main(unparsed):
    r"""check tfrecord data
    :param unparsed:
    :return:
    """
    dataset = tf.data.TFRecordDataset(FLAGS.file)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(1)

    for idx, (images, segs, boxes, labels, numbers) in enumerate(dataset):
        for i, (im, seg, box, label, number) in enumerate(zip(images, segs, boxes, labels, numbers)):
            im = im.numpy()
            seg = seg.numpy()
            box = box.numpy()
            label = label.numpy()
            height, width, channel = im.shape
            seg_cor = drivable2color(seg)
            im[seg_cor > 0] = 0

            for idx in range(number):
                l = label[idx]
                b = box[idx]
                b = [b[0]*width, b[1]*height, b[2]*width, b[3]*height]
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), 2)
                cv2.putText(im, '{}:{}'.format(l, road_obj[l-1]),
                            (int(b[0]), int(b[1])), 2, 0.5, (0, 255, 0))

            cv2.imshow('im', seg_cor+im)
            cv2.waitKey(0)

    return dataset


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
