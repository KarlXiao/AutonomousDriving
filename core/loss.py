import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import match, one_hot_embedding
import numpy as np

def focal_loss(prediction, label_, num_classes, alpha=0.25, gamma=2):
    r"""
    focal loss: https://arxiv.org/abs/1708.02002
    :param prediction: output prediction of net
    :param label_: ground truth labels
    :param num_classes: number of classes
    :param alpha: alpha in focal loss, default: 0.25
    :param gamma: gamma in focal loss, default: 2
    :return: focal loss
    """
    mask = label_ == 0
    num_neg = torch.sum(mask, dim=1, dtype=torch.float32)

    t = one_hot_embedding(label_.data.cpu() - 1, num_classes)
    mask = mask.unsqueeze(2).expand_as(t)

    t[mask] = 0
    t = Variable(t).cuda()
    num_pos = torch.sum(t, dim=(1, 2))
    num_pos = torch.clamp(num_pos, min=1, max=torch.max(num_pos))

    p = prediction.sigmoid()
    pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
    w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    num_pos = num_pos.unsqueeze(1).unsqueeze(1).expand_as(w)
    num_neg = num_neg.unsqueeze(1).unsqueeze(1).expand_as(w)
    w = w * (1 - pt).pow(gamma) / num_pos.expand_as(w)
    pos_weight = - torch.log(1e-8 + torch.sum(t, dim=1).unsqueeze(1).expand_as(w) / num_neg)

    return F.binary_cross_entropy_with_logits(prediction, t, w, reduction='sum', pos_weight=pos_weight)


def detection_loss(predictions, targets, cfg):
    r"""
    loss function of detection
    :param predictions: prediction of net, including classification and localization
    :param targets: ground truth including boxes and labels
    :param cfg: configuration including:
            iou_thr: threshold of iou, default: 0.5
            variance: variance between anchors and bounding box, default: [0.1, 0.2]
            neg_pos: negative / positive samples ratios, default: 3
    :return: loss including classification loss and localization loss
    """
    loc_logits, cls_logits, priors = predictions
    num = loc_logits.size(0)
    num_priors = priors.size(0)
    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)

    for idx, target in enumerate(targets):
        boxes = target[:, :4]
        label = target[:, -1]
        match(cfg['iou_thr'], boxes, priors, cfg['variance'], label, loc_t, conf_t, idx)

    loc_t = loc_t.cuda()
    conf_t = conf_t.cuda()
    loc_t = Variable(loc_t, requires_grad=False)
    conf_t = Variable(conf_t, requires_grad=False)

    pos = conf_t > 0

    pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_logits)
    loc_p = loc_logits[pos_idx].view(-1, 4)
    loc_t = loc_t[pos_idx].view(-1, 4)
    loc_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

    cls_loss = focal_loss(cls_logits, conf_t,  cfg['num_class'])

    # N = num_pos.data.sum().float()
    loc_loss /= num
    cls_loss /= num

    return cls_loss, loc_loss
