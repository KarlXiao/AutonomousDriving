import sys
import torch.utils.data as data
import argparse
from core import *
from config import DetectionCfg as cfg
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='checkpoint', help='directory to save checkpoint')
parser.add_argument('--resume', type=str, default=None, help='perception net path for finetune training')
parser.add_argument('--json', type=str, default='data/BDD100K/labels/bdd100k_labels_images_train.json', help='json file to load')
parser.add_argument('--im_dir', type=str, default='data/BDD100K/images', help='images directory')

parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--epoch', type=int, default=300, help='number of training epoch')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='weight rate decay')
parser.add_argument('--gamma', default=0.5, type=float, help='gamma update for optimizer')
parser.add_argument('--lr_decay', type=int, default=10, help='multiply by a gamma every lr_decay iterations')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in data loading')
parser.add_argument('--log', default='default', type=str, help='training log')
parser.add_argument('--start_epoch', default=1, type=int, help='iterations starting from this value')

args = parser.parse_args()


def train():

    best_loss = np.inf
    writer = SummaryWriter(os.path.join('runs', args.log))
    dummy_input = torch.rand(1, 3, cfg['input_dim'][1], cfg['input_dim'][0])

    train_dataset = BDDLoader(args.json, args.im_dir, cfg['input_dim'])

    data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  collate_fn=detection_collate, shuffle=True)

    model = BaseModel(args, cfg)
    writer.add_graph(model.perception, (dummy_input, ))

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load(args.resume)

    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #######################################################################################

    for epoch in np.arange(args.start_epoch, args.epoch):

        writer.add_scalar('Train/learning rate', optimizer.param_groups[0]['lr'], epoch)

        average_loc, average_cls, average_seg, average_dis = [0.0]*4

        writer.add_scalar('Train/learning rate', model.optimizer_P.param_groups[0]['lr'], epoch)

        for iteration, (images, targets, segs) in enumerate(data_loader):

            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]
            segs = Variable(segs.cuda())

            model.optimize(images, targets, segs)

            average_cls = ((average_cls * iteration) + model.loss_P_cls.item()) / (iteration + 1)
            average_loc = ((average_loc * iteration) + model.loss_P_loc.item()) / (iteration + 1)
            average_seg = ((average_seg * iteration) + model.loss_P_seg.item()) / (iteration + 1)
            average_dis = ((average_dis * iteration) + model.loss_D.item()) / (iteration + 1)

            writer.add_scalar('Train/cls loss', average_cls, (epoch - 1) * len(data_loader) + iteration)
            writer.add_scalar('Train/loc loss', average_loc, (epoch - 1) * len(data_loader) + iteration)
            writer.add_scalar('Train/seg loss', average_seg, (epoch - 1) * len(data_loader) + iteration)
            writer.add_scalar('Train/dis loss', average_dis, (epoch - 1) * len(data_loader) + iteration)

            count = round(iteration / len(data_loader) * 50)
            sys.stdout.write('[Epoch {}], {}/{}: [{}{}] Avg_loc loss: {:.4}, Avg_conf loss:{:.4}, '
                             'Avg_seg loss:{:.4}, Avg_dis loss:{:.4}\r'.format(epoch, iteration + 1, len(data_loader),
                                                           '#' * count, ' ' * (50 - count), average_loc,
                                                           average_cls, average_seg, average_dis))

            del images, targets, segs

        sys.stdout.write('\n')
        average_loss = average_cls + average_loc + average_seg + average_dis

        for key, param in model.perception.named_parameters():
            writer.add_histogram(key, param.clone(), epoch)

        if best_loss > average_loss:
            best_loss = average_loss
            model.save(os.path.join(args.save_dir, args.log), epoch)
            print('Epoch: {} model is saved'.format(epoch))

        model.update_learning_rate()


if __name__ == '__main__':

    if not os.path.exists(os.path.join(args.save_dir, args.log)):
        os.makedirs(os.path.join(args.save_dir, args.log))

    train()
