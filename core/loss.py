import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import match, one_hot_embedding


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
    pos_weight[t == 0] = 1

    return - torch.sum(w * pos_weight * (t * torch.log(p) + (1 - t) * torch.log(1 - p)))


def detection_loss(predictions, targets, cfg, priors):
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
    loc_logits, cls_logits = predictions
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

    num_pos = pos.sum().float()
    loc_loss /= (num_pos + 1)
    cls_loss /= (num_pos + 1)

    return cls_loss, loc_loss


def segmentation_loss(x, segs, num_cls):
    n, c, h, w = x.size()
    log_p = x.transpose(1, 2).transpose(2, 3)
    log_p = log_p[segs.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = segs >= 0
    target_mask = segs[mask]
    weight = torch.bincount(target_mask).float()
    weight = - torch.log(1e-8 + weight / torch.sum(weight))
    # in case some labels are not included in current batch
    if len(weight) < num_cls:
        weight = torch.ones(num_cls).cuda()

    loss = F.cross_entropy(log_p, target_mask, weight=weight)

    return loss


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
        # self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
