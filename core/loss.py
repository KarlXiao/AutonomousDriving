import tensorflow as tf
import numpy as np
from .utils import match


def focal_loss(prediction, labels, alpha=0.25, gamma=2):
    r"""
    focal loss: https://arxiv.org/abs/1708.02002
    :param prediction: output prediction of net
    :param labels: ground truth labels
    :param alpha: alpha in focal loss, default: 0.25
    :param gamma: gamma in focal loss, default: 2
    :return: focal loss
    """
    sigmoid_p = tf.nn.sigmoid(prediction)
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # p if y=1, 1-p otherwise
    # for positive samples, (1-p)^gamma, then pos_pred = labels-sigmoid_p
    pos_pred = tf.where(labels > zeros, labels - sigmoid_p, zeros)
    neg_pred = tf.where(labels > zeros, zeros, sigmoid_p)

    loss = - alpha * (pos_pred ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
           - (1 - alpha) * (neg_pred ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_sum(loss)


def detection_loss(predictions, box_, label_, images, iou_thr=0.5, variance=[0.1, 0.2], neg_pos=3, use_focal_loss=True):
    r"""
    loss function of detection
    :param predictions: prediction of net, including classification and localization
    :param box_: ground truth bounding box
    :param label_: ground truth label
    :param iou_thr: threshold of iou, default: 0.5
    :param variance: variance between anchors and bounding box, default: [0.1, 0.2]
    :param neg_pos: negative / positive samples ratios, default: 3
    :param use_focal_loss: if use focal loss or not, default: True
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
            match(iou_thr, boxes.numpy(), priors, variance, label, loc_t, conf_t, idx, im)

        pos = tf.cast(conf_t > 0, tf.int64)
        num_pos = np.sum(pos)
        pos_idx = conf_t > 0
        loc_loss = tf.losses.mean_squared_error(tf.boolean_mask(loc_t, pos_idx), tf.boolean_mask(loc_logits, pos_idx))

        if use_focal_loss:
            # ====================================================
            #                   focal loss
            # ====================================================
            cls_loss = focal_loss(cls_logits, label_)
        else:
            # ====================================================
            #              hard example mining
            # ====================================================
            label = tf.one_hot(pos, 2, axis=2)
            batch_conf = tf.nn.softmax(cls_logits, dim=-1)
            loss_c = tf.reduce_mean(tf.abs(batch_conf - label), axis=-1).numpy()
            loss_c[pos_idx] = 0
            neg_val = -np.sort(-np.reshape(loss_c, [-1]))[neg_pos*num_pos]
            neg_idx = loss_c > neg_val

            neg_loss = tf.losses.softmax_cross_entropy(tf.boolean_mask(label, neg_idx), tf.boolean_mask(cls_logits, neg_idx))
            pos_loss = tf.losses.softmax_cross_entropy(tf.boolean_mask(label, pos_idx), tf.boolean_mask(cls_logits, pos_idx))
            cls_loss = neg_loss + pos_loss

        cost = cls_loss + loc_loss
        tf.summary.scalar(scope.name + '/loss', cost)
        print('Cls loss: {}, Loc loss: {}'.format(cls_loss.numpy(), loc_loss.numpy()))

    return cost
