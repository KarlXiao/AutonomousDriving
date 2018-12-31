from .utils import *


class Detect(object):
    r"""At test time, Detect is the final layer.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, top_k, variance, conf_thresh=0.5, nms_thresh=0.25):
        r"""
        :param num_classes: number of classes
        :param top_k: top k score predicts
        :param variance: variance between anchors and bounding box
        :param conf_thresh: threshold of confidence
        :param nms_thresh: IoU threshold of non-maximum suppression
        """
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance

    def __call__(self, loc_data, conf_data, prior_data):
        r"""
        :param loc_data: (tensor) Loc preds from loc layers. Shape: [batch,num_priors*4]
        :param conf_data: (tensor) Shape: Conf preds from conf layers. Shape: [batch*num_priors,num_classes]
        :param prior_data: (tensor) Prior boxes and variances from priorbox layers. Shape: [1,num_priors,4]
        :return: detection result after nms
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.sigmoid().view(num, num_priors,
                                              self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(0, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                if count == 0:
                    continue
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
