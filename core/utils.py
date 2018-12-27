import tensorflow as tf
import numpy as np
import cv2


def point_form(boxes, point_type='xywh'):
    r"""
    representation for comparison to point form ground truth data
    :param boxes: default boxes
    :param point_type:
            xywh: Convert (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
            ccwh: Convert (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    :return: Converted xmin, ymin, xmax, ymax form of boxes
    """

    if point_type == 'xywh':
        output = tf.concat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], 1)
    elif point_type == 'ccwh':
        output = tf.concat([boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2], 1)
    else:
        raise TypeError('point_type must be xywh or ccwh')

    return output


def intersect(box_a, box_b):
    num_a = box_a.get_shape().as_list()[0]
    num_b = box_b.get_shape().as_list()[0]
    try:
        max_xy = np.minimum(tf.tile(tf.expand_dims(box_a[:, 2:], 1), [1, num_b, 1]),
                            tf.tile(tf.expand_dims(box_b[:, 2:], 0), [num_a, 1, 1]))

    except RuntimeError:
        print(box_a)
        print(box_b)
        print(num_a)
        print(num_b)
    min_xy = np.maximum(tf.tile(tf.expand_dims(box_a[:, :2], 1), [1, num_b, 1]),
                        tf.tile(tf.expand_dims(box_b[:, :2], 0), [num_a, 1, 1]))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)

    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    r"""
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    :param box_a: Multiple bounding boxes, Shape: [num_boxes,4]
    :param box_b: Single bounding box, Shape: [4]
    :return: jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    num_a = box_a.get_shape().as_list()[0]
    num_b = box_b.get_shape().as_list()[0]
    inter = intersect(box_a, box_b)
    area_a = tf.tile(tf.expand_dims((box_a[:, 2]-box_a[:, 0]) *
                                    (box_a[:, 3]-box_a[:, 1]), 1), [1, num_b])
    area_b = tf.tile(tf.expand_dims((box_b[:, 2]-box_b[:, 0]) *
                                    (box_b[:, 3]-box_b[:, 1]), 0), [num_a, 1])
    union = area_a + area_b - inter

    return inter / union


def encode(matched, priors, variances):
    r"""
    Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    :param matched: Coords of ground truth for each prior in point-form Shape: [num_priors, 4].
    :param priors: Prior boxes in center-offset form Shape: [num_priors,4]
    :param variances: variance between anchors and bounding box
    :return: encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = matched[:, :2] + matched[:, 2:]/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = matched[:, 2:] / priors[:, 2:]
    g_wh = tf.math.log(g_wh) / variances[1]

    return tf.concat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode(loc, priors, variances):
    r"""
    Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time
    :param loc: location predictions for loc layers, Shape: [num_priors,4]
    :param priors: Prior boxes in center-offset form. Shape: [num_priors,4]
    :param variances: variance between anchors and bounding box
    :return: decoded bounding box predictions
    """
    boxes = tf.concat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * tf.math.exp(loc[:, 2:] * variances[1])), 1)

    return tf.concat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, 2:]], 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, im=None):
    r"""
    Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    :param threshold: The overlap threshold used when mathing boxes
    :param truths: Ground truth boxes, Shape: [num_obj, num_priors]
    :param priors: Prior boxes from priorbox layers, Shape: [n_priors,4]
    :param variances: variance between anchors and bounding box
    :param labels: All the class labels for the image, Shape: [num_obj]
    :param loc_t: Tensor to be filled w/ endcoded location targets
    :param conf_t: Tensor to be filled w/ matched indices for conf preds
    :param idx: current batch index
    :return: The matched indices corresponding to 1)location and 2)confidence preds
    """
    overlaps = jaccard(
        point_form(truths, 'xywh'),
        point_form(priors, 'ccwh')
    )

    matches = truths.numpy()[tf.argmax(overlaps).numpy()]   # Shape: [num_priors,4]
    conf = labels.numpy()[tf.argmax(overlaps).numpy()]      # Shape: [num_priors]
    conf[tf.reduce_max(overlaps, axis=0) < threshold] = 0  # label as background

    # if im != None:
    #     prior_idx = tf.where(tf.reduce_max(overlaps, axis=0) > 0.5).numpy()
    #     img = np.uint8(im.numpy() * 255)
    #     h, w, _ = img.shape
    #     predict = point_form(priors, 'ccwh')
    #     for prior in prior_idx:
    #         box = predict[prior[0], :]
    #         b = [int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)]
    #         cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
    #
    #         for box in truths:
    #             b = [int(box[0] * w), int(box[1] * h), int((box[2]+box[0]) * w), int((box[3]+box[1]) * h)]
    #             cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    #     cv2.imshow('cv', img)
    #     cv2.waitKey(0)

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def nms(boxes, scores, overlap=0.5):
    r"""
    Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object
    :param boxes: The location preds for the img, Shape: [num_priors,4]
    :param scores: The class predscores for the img, Shape:[num_priors]
    :param overlap: The overlap thresh for suppressing unnecessary boxes
    :return: The indices of the kept boxes with respect to num_priors
    """

    if scores.shape == 0:
        return None, 0

    num = scores.shape.as_list()[0]
    scores_copy, idx = tf.nn.top_k(scores, num)
    boxes_copy = tf.gather(boxes, idx)
    keep = np.ones(num)

    for i in range(num):
        if keep[i] == 0:
            continue

        for j in range(i + 1, num):
            box_a = tf.expand_dims(boxes_copy[i], 0)
            box_b = tf.expand_dims(boxes_copy[j], 0)
            iou = jaccard(box_a, box_b)

            if iou.numpy() >= overlap:
                keep[j] = 0

    return tf.boolean_mask(boxes_copy, keep == 1), tf.boolean_mask(scores_copy, keep == 1)
