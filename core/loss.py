import tensorflow as tf
import numpy as np
from .utils import match


def focal_loss_sigmoid(prediction, label_, alpha=0.25, gamma=2):
    r"""
    focal loss: https://arxiv.org/abs/1708.02002
    :param prediction: output prediction of net
    :param label_: ground truth labels
    :param alpha: alpha in focal loss, default: 0.25
    :param gamma: gamma in focal loss, default: 2
    :return: focal loss
    """
    sigmoid_p = tf.nn.sigmoid(prediction)
    num = prediction.get_shape().as_list()[-1]
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    labels = tf.one_hot(tf.to_int64(label_) - 1, num)

    # p if y=1, 1-p otherwise
    # for positive samples, (1-p)^gamma, then pos_pred = labels-sigmoid_p
    pos_pred = tf.where(labels > zeros, labels - sigmoid_p, zeros)
    neg_pred = tf.where(labels > zeros, zeros, sigmoid_p)

    loss = - alpha * (pos_pred ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0))\
           - (1 - alpha) * (neg_pred ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_sum(loss) / tf.count_nonzero(label_, dtype=tf.float32)


def focal_loss_softmax(prediction, label_, alpha=0.25, gamma=2):
    r"""
    focal loss: https://arxiv.org/abs/1708.02002
    :param prediction: output prediction of net
    :param label_: ground truth labels
    :param alpha: alpha in focal loss, default: 0.25
    :param gamma: gamma in focal loss, default: 2
    :return: focal loss
    """
    softmax_p = tf.nn.softmax(prediction)
    num = prediction.get_shape().as_list()[-1]
    zeros = tf.zeros_like(softmax_p, dtype=softmax_p.dtype)
    labels = tf.one_hot(tf.to_int64(label_), num)
    weight = 1 / tf.clip_by_value(tf.reduce_sum(labels, axis=(0, 1)), 1, 1e1000000)

    # p if y=1, 1-p otherwise
    # for positive samples, (1-p)^gamma, then pos_pred = labels-sigmoid_p
    pos_pred = tf.where(labels > zeros, labels - softmax_p, zeros)
    neg_pred = tf.where(labels > zeros, zeros, softmax_p)

    loss = - alpha * (pos_pred ** gamma) * tf.log(tf.clip_by_value(softmax_p, 1e-8, 1.0))\
           - (1 - alpha) * (neg_pred ** gamma) * tf.log(tf.clip_by_value(1.0 - softmax_p, 1e-8, 1.0))

    # pos_loss = tf.reduce_sum(pos_loss, axis=(0, 1))
    # neg_loss = tf.reduce_sum(neg_loss, axis=(0, 1))

    loss = tf.reduce_mean(loss, axis=(0, 1))

    return tf.reduce_sum(loss * weight)


def detection_loss(predictions, box_, label_, cfg, images):
    r"""
    loss function of detection
    :param predictions: prediction of net, including classification and localization
    :param box_: ground truth bounding box
    :param label_: ground truth label
    :param iou_thr: threshold of iou, default: 0.5
    :param variance: variance between anchors and bounding box, default: [0.1, 0.2]
    :param neg_pos: negative / positive samples ratios, default: 3
    :return: loss including classification loss and localization loss
    """
    with tf.variable_scope('loss') as scope:

        loc_logits, cls_logits, priors = predictions
        num = loc_logits.get_shape().as_list()[0]
        num_priors = priors.get_shape().as_list()[0]
        loc_t = np.zeros([num, num_priors, 4])
        conf_t = np.zeros([num, num_priors])

        for idx, (boxes, label) in enumerate(zip(box_, label_)):
            im = images[idx]
            match(cfg['iou_thr'], boxes, priors, cfg['variance'], label, loc_t, conf_t, idx, im)

        pos_idx = conf_t > 0
        loc_loss = tf.reduce_mean(tf.losses.mean_squared_error(tf.boolean_mask(loc_t, pos_idx),
                                                               tf.boolean_mask(loc_logits, pos_idx)))

        cls_loss = focal_loss_sigmoid(cls_logits, conf_t)

        cost = cls_loss + loc_loss
        tf.summary.scalar(scope.name + '/loss', cost)
        print('Cls loss: {}, Loc loss: {}'.format(cls_loss.numpy(), loc_loss.numpy()))

    return cost


def OHEM_loss(predictions, box_, label_, cfg, images):
    r"""
    loss function of detection
    :param predictions: prediction of net, including classification and localization
    :param box_: ground truth bounding box
    :param label_: ground truth label
    :param iou_thr: threshold of iou, default: 0.5
    :param variance: variance between anchors and bounding box, default: [0.1, 0.2]
    :param neg_pos: negative / positive samples ratios, default: 3
    :return: loss including classification loss and localization loss
    """
    with tf.variable_scope('loss') as scope:

        loc_logits, cls_logits, priors = predictions
        num = loc_logits.get_shape().as_list()[0]
        num_priors = priors.get_shape().as_list()[0]
        loc_t = np.zeros([num, num_priors, 4])
        conf_t = np.zeros([num, num_priors])

        for idx, (boxes, label) in enumerate(zip(box_, label_)):
            im = images[idx]
            match(cfg['iou_thr'], boxes, priors, cfg['variance'], label, loc_t, conf_t, idx, im)

        pos = tf.cast(conf_t > 0, tf.int64)
        num_pos = np.sum(pos)
        pos_idx = conf_t > 0
        loc_loss = tf.reduce_mean(tf.losses.mean_squared_error(tf.boolean_mask(loc_t, pos_idx),
                                                               tf.boolean_mask(loc_logits, pos_idx)))

        # ====================================================
        #              online hard example mining
        # ====================================================
        label = tf.one_hot(pos, cfg['num_class'], axis=2)
        batch_conf = tf.nn.softmax(cls_logits, dim=-1)
        loss_c = tf.reduce_mean(tf.abs(batch_conf - label), axis=-1).numpy()
        loss_c[pos_idx] = 0
        neg_val = -np.sort(-np.reshape(loss_c, [-1]))[cfg['neg_pos']*num_pos]
        neg_idx = loss_c > neg_val

        neg_loss = tf.losses.softmax_cross_entropy(tf.boolean_mask(label, neg_idx), tf.boolean_mask(cls_logits, neg_idx))
        pos_loss = tf.losses.softmax_cross_entropy(tf.boolean_mask(label, pos_idx), tf.boolean_mask(cls_logits, pos_idx))
        cls_loss = tf.reduce_mean(neg_loss + pos_loss)

        cost = cls_loss + loc_loss
        tf.summary.scalar(scope.name + '/loss', cost)
        print('Cls loss: {}, Loc loss: {}'.format(cls_loss.numpy(), loc_loss.numpy()))

    return cost
