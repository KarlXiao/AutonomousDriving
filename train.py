import os
import sys
import torch
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
import argparse
from core import PerceptionNet, BDDLoader, detection_loss, detection_collate
from config import DetectionCfg as cfg


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='checkpoint', help='directory to save checkpoint')
parser.add_argument('--log', type=str, default='log/', help='directory to save training log')
parser.add_argument('--resume', type=str, default=None, help='model directory for finetune training')
parser.add_argument('--json', type=str, default='data/BDD100K/labels/bdd100k_labels_images_train.json', help='tfrecords to load')
parser.add_argument('--im_dir', type=str, default='data/BDD100K/images', help='images directory')

parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--epoch', type=int, default=1600, help='number of training epoch')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='learning rate decay step')
parser.add_argument('--momentum', default=0.9, type=float, help='learning rate decay rate')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma update for optimizer')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers used in data loading')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda to train model')

args = parser.parse_args()


def train():

    best_loss = np.inf

    train_dataset = BDDLoader(args.json, args.im_dir, cfg['input_dim'])

    data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  collate_fn=detection_collate, shuffle=True)

    model = PerceptionNet(cfg['num_class'], [1, 1, 1, 1])

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load_weights(args.resume)
    else:
        model.apply(weights_init)

    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #######################################################################################

    for epoch in np.arange(args.epoch):

        if epoch in cfg['lr_steps']:
            step_index = cfg['lr_steps'].index(epoch) + 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        average_loc = 0.0
        average_cls = 0.0
        for iteration, (images, targets, segs) in enumerate(data_loader):

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

            out, fetures = model(images)

            optimizer.zero_grad()
            cls_loss, loc_loss = detection_loss(out, targets, cfg)
            loss = cls_loss + loc_loss
            loss.backward()
            optimizer.step()

            average_cls = ((average_cls * iteration) + cls_loss.item()) / (iteration + 1)
            average_loc = ((average_loc * iteration) + loc_loss.item()) / (iteration + 1)
            count = round(iteration / len(data_loader) * 50)
            sys.stdout.write('[Epoch {}], {}/{}: [{}{}] Avg_loc loss: {:.4}, Avg_conf loss:{:.4}\r'.format(
                epoch, iteration + 1, len(data_loader), '#' * count, ' ' * (50 - count), average_loc, average_cls))

        sys.stdout.write('\n')
        average_loss = average_cls + average_loc

        if best_loss > average_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'Perception_{}.pth'.format(epoch)))
            print('Epoch: {} model is saved'.format(epoch))


def weights_init(m):
    r"""
    weight init function
    """
    if isinstance(m, torch.nn.Conv2d):
        init.xavier_normal_(m.weight.data)


def adjust_learning_rate(optimizer, gamma, step):
    r"""
    Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    Adapted from PyTorch Imagenet example:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param optimizer:optimizer
    :param gamma:learning rate decay
    :param step:specified step
    :return:
    """
    lr = args.lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train()
