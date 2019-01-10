import torch
import numpy as np


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
        output = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], 1)
    elif point_type == 'ccwh':
        output = torch.cat([boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2], 1)
    else:
        raise TypeError('point_type must be xywh or ccwh')

    return output


def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
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
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


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
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode(loc, priors, variances):
    r"""
    Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time
    :param loc: location predictions for loc layers, Shape: [num_priors,4]
    :param priors: Prior boxes in center-offset form. Shape: [num_priors,4]
    :param variances: variance between anchors and bounding box
    :return: decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
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
        truths,
        point_form(priors, 'ccwh')
    )

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]

    conf = labels[best_truth_idx]             # Shape: [num_priors]

    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def nms(boxes, scores, overlap=0.5, top_k=50):
    r"""
    Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object
    :param boxes: The location preds for the img, Shape: [num_priors,4]
    :param scores: The class predscores for the img, Shape:[num_priors]
    :param overlap: The overlap thresh for suppressing unnecessary boxes
    :param top_k: The number of top scores
    :return: The indices of the kept boxes with respect to num_priors
    """

    keep = scores.new(scores.size(0)).zero_().long()
    count = 0
    if boxes.numel() == 0:
        return keep, count
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def one_hot_embedding(labels, num_classes):
    r"""
    Embedding labels to one-hot form.
    :param labels: (LongTensor) class labels, sized [N,].
    :param num_classes: (int) number of classes.
    :return: (tensor) encoded labels, sized [N,#classes].
    """
    y = torch.eye(num_classes)  # [D,D]

    return y[labels]            # [N,D]


def drivable2color(seg):
    r"""
    :param seg: segmentation map
    :return: color map
    """
    colors = [[0, 0, 0], [217, 83, 79], [91, 192, 222]]
    color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        color[seg == i] = colors[i]
    return color
