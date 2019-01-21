import os
import numpy as np
from .models import Discriminator, PerceptionNet
from .loss import *


class BaseModel(nn.Module):
    def __init__(self, args, cfg):
        super(BaseModel, self).__init__()

        self.args = args
        self.cfg = cfg

        self.perception = PerceptionNet(cfg['num_class'], [3, 4, 6, 3])
        self.discriminator = Discriminator(6, 64, True)

        self.optimizer_P = torch.optim.Adam(self.perception.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer = [self.optimizer_P, self.optimizer_D]

        self.criterionGAN = GANLoss().cuda()

        self.scheduler = []
        self.logits = None
        self.features = None
        self.inputs = None

        self.setup()

    def forward(self, inputs):
        self.inputs = inputs
        self.logits, self.features = self.perception(inputs)

    def backward_D(self, segs):
        r"""
        compute discriminator loss
        :param segs: segmentation labels
        """
        fake = torch.cat((self.inputs, self.features), 1)
        pred_fake = self.discriminator(fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        mask = np.zeros(segs.shape)[:, np.newaxis, :, :].repeat(3, 1)
        for i in range(3):
            mask[:, i, :, :] = (segs[:, :, :] == i).data.cpu().numpy()
        mask_tensor = torch.from_numpy(mask).float().cuda()

        real = torch.cat((self.inputs, mask_tensor), 1)
        pred_real = self.discriminator(real)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_P(self, targets, segs):
        r"""
        compute perception loss
        :param segs: segmentation labels
        """
        # First, Perception net should fake the discriminator
        fake = torch.cat((self.inputs, self.features), 1)
        pred_fake = self.discriminator(fake)
        self.loss_P_seg = self.criterionGAN(pred_fake, True) + segmentation_loss(self.features, segs, 3)

        self.loss_P_cls, self.loss_P_loc = detection_loss(self.logits, targets, self.cfg, self.perception.prior)

        self.loss_P = self.loss_P_seg + self.loss_P_cls + self.loss_P_loc

        self.loss_P.backward()

    def optimize(self, inputs, targets, segs):
        r"""
        optimize parameters in perception net and discriminator
        """
        self.forward(inputs)
        # First, optimize discriminator
        self.set_requires_grad(self.discriminator, True)
        self.optimizer_D.zero_grad()
        self.backward_D(segs)
        self.optimizer_D.step()

        # Second, optimize perception net
        self.set_requires_grad(self.discriminator, False)
        self.optimizer_P.zero_grad()
        self.backward_P(targets, segs)
        self.optimizer_P.step()

    def setup(self):
        for optimizer in self.optimizer:
            self.scheduler.append(torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_decay, self.args.gamma))

    def update_learning_rate(self):
        for scheduler in self.scheduler:
            scheduler.step()

    def save(self, path, epoch):
        r"""
        save perception net and discriminator
        :param path: directory to save models
        :param epoch: epoch count to save
        """
        torch.save(self.perception.state_dict(), os.path.join(path, 'Perception_{}.pth'.format(epoch)))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'Discriminator_{}.pth'.format(epoch)))

    def load(self, path):
        r"""
        load perception net
        :param path: path to perception net
        """
        self.perception.load_weights(path)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        r"""
        set parameter requires_grad=False to avoid computation
        :param nets: nets to be set
        :param requires_grad: require gradient or not, default: False
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
