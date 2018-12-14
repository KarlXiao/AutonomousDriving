import tensorflow as tf


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
