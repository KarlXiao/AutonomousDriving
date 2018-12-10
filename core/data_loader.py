import tensorflow as tf


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


def create_loader(proto, bach_size, capacity):
    r"""
    create tfrecord loader
    :param proto: str or list, path to tfrecords
    :param bach_size: batch size of data
    :param capacity: shuffle capacity
    :return: dataset
    """
    dataset = tf.data.TFRecordDataset(proto)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=capacity)
    dataset = dataset.batch(bach_size)

    return dataset
