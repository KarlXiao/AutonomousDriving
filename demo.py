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
parser.add_argument('--resume', type=str, default='checkpoint/rest3463_config_iou0.4_lr5e-4_notanh/Perception_45.pth', help='model directory for finetune training')
parser.add_argument('--json', type=str, default='data/BDD100K/labels/bdd100k_labels_images_val.json', help='tfrecords to load')
parser.add_argument('--im_dir', type=str, default='data/BDD100K/images', help='images directory')

parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers used in data loading')

args = parser.parse_args()


def val():

    train_dataset = BDDLoader(args.json, args.im_dir, cfg['input_dim'])

    data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  collate_fn=detection_collate, shuffle=False)

    model = PerceptionNet(cfg['num_class'], [3, 4, 6, 3])

    print('Loading {}...'.format(args.resume))
    model.load_weights(args.resume)

    detect = Detect(cfg['num_class'], 50, cfg['variance'], cfg['conf_thr'])

    model = model.cuda()

    #######################################################################################

    for iteration, (images, targets, segs) in enumerate(data_loader):

        images = Variable(images.cuda())

        out, features = model(images)
        detections = detect(out[0].data, out[1].data, model.prior)
        seg = features.data.max(1)[1].cpu().numpy()[0]
        seg_cor = drivable2color(seg)
        seg_cor_ = drivable2color(segs[0, :, :].numpy())

        img = np.transpose(images[0].data.cpu().numpy(), (1, 2, 0))*255
        img = img.copy()
        img_ = img.copy()

        img[seg_cor > 0] = 0
        img += seg_cor
        img_[seg_cor_ > 0] = 0
        img_ += seg_cor_

        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.5:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                j += 1

                cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 3)

        for target in targets[0]:
            box = target[:4]
            box *= scale
            cv2.rectangle(img_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

        count = round(iteration / len(data_loader) * 50)
        sys.stdout.write('{}/{}: [{}{}]\r'.format(iteration + 1, len(data_loader), '#' * count, ' ' * (50 - count)))

        im = np.concatenate((img, img_), axis=1)

        cv2.imwrite('images/{}.jpg'.format(iteration), im.astype(np.uint8))

    sys.stdout.write('\n')


if __name__ == '__main__':

    val()
