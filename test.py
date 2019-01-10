import sys
import torch
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
import argparse
import cv2
from core import PerceptionNet, BDDLoader, detection_collate, Detect, drivable2color
from config import DetectionCfg as cfg


parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default='checkpoint/Perception_44.pth', help='model directory for finetune training')
parser.add_argument('--json', type=str, default='data/BDD100K/labels/bdd100k_labels_images_val.json', help='tfrecords to load')
parser.add_argument('--im_dir', type=str, default='data/BDD100K/images', help='images directory')

parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers used in data loading')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda to train model')

args = parser.parse_args()


def val():

    train_dataset = BDDLoader(args.json, args.im_dir, cfg['input_dim'])

    data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  collate_fn=detection_collate, shuffle=True)

    model = PerceptionNet(cfg['num_class'], [2, 3, 5, 2])

    print('Loading {}...'.format(args.resume))
    model.load_weights(args.resume)

    detect = Detect(cfg['num_class'], 50, cfg['variance'], cfg['conf_thr'])

    if args.cuda:
        model = model.cuda()

    #######################################################################################

    for iteration, (images, targets, segs) in enumerate(data_loader):

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        out, features = model(images)
        detections = detect(out[0].data, out[1].data, model.prior)
        seg = features.data.max(1)[1].cpu().numpy()[0]
        seg_cor = drivable2color(seg)

        img = np.transpose(images[0].data.cpu().numpy(), (1, 2, 0))*255
        img = img.copy()
        img[seg_cor > 0] = 0
        img += seg_cor
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.5:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                j += 1

                cv2.putText(img, "Score:{:.2f}".format(score), (int(pt[0]), int(pt[1])), 1, 2.0, (255, 0, 0), 1)
                cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 3)

        for target in targets[0]:
            box = target[:4]
            box *= scale
            label = int(target[4])
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

        count = round(iteration / len(data_loader) * 50)
        sys.stdout.write('{}/{}: [{}{}]\r'.format(iteration + 1, len(data_loader), '#' * count, ' ' * (50 - count)))

        cv2.imshow('cv', img.astype(np.uint8))
        cv2.waitKey(0)

    sys.stdout.write('\n')


if __name__ == '__main__':

    val()
