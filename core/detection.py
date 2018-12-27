from .utils import *


class Detect(object):
    r"""At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh

    def __call__(self, loc_data, conf_data, prior_data, variance):
        r"""

        :param loc_data: (tensor) Loc preds from loc layers. Shape: [batch,num_priors*4]
        :param conf_data: (tensor) Shape: Conf preds from conf layers. Shape: [batch*num_priors,num_classes]
        :param prior_data: (tensor) Prior boxes and variances from priorbox layers. Shape: [1,num_priors,4]
        :param variance: variance between anchors and bounding box
        :return: detection result after nms
        """

        num = loc_data.shape.as_list()[0]  # batch size
        output = []
        conf_preds = tf.transpose(tf.nn.softmax(conf_data, axis=-1), [0, 2, 1])

        # Decode predictions into bboxes.
        for i in range(num):
            # For each class, perform nms
            conf_scores = conf_preds[i]

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl] > self.conf_thresh
                scores = tf.boolean_mask(conf_scores[cl], c_mask)
                if scores.shape == 0:
                    continue
                boxes = tf.boolean_mask(loc_data[i], c_mask)
                prior = tf.boolean_mask(prior_data, c_mask)
                decoded_boxes = decode(boxes, prior, variance)
                # idx of highest scoring and non-overlapping boxes per class
                boxes_nms, scores_nms = nms(point_form(decoded_boxes), scores, self.nms_thresh)
                boxes_nms = tf.clip_by_value(boxes_nms, 0, 1)
                if scores_nms.shape.as_list()[0] == 0:
                    continue
                output.append(tf.concat([boxes_nms, tf.expand_dims(scores_nms, 1)], axis=1).numpy())

        return output
