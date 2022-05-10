import os
import torch
from torch.optim import Adam
from models.unet import UNet

from .closure.rei_end2end import closure_rei_end2end
from .closure.rei_end2end_ct import closure_rei_end2end_ct
from .closure.ei_end2end import closure_ei_end2end
from .closure.supervised import closure_supervised

from utils.nn import adjust_learning_rate
from utils.logger import get_timestamp, LOG


class REI(object):
    def __init__(self, in_channels, out_channels, img_width, img_height, dtype, device):
        """
        Robust Equivariant Imaging
        """
        super(REI, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_width = img_width
        self.img_height = img_height

        self.dtype = dtype
        self.device = device

    def train_rei(self, dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                  schedule, pretrained, lr_cos, save_ckp, tau, report_psnr, args):

        save_path = './ckp/{}_rei_{}_{}_sigma{}_gamma{}'\
            .format(get_timestamp(), physics.name,
                    physics.noise_model['noise_type'],
                    physics.noise_model['sigma'],
                    physics.noise_model['gamma'])
        if physics.name=='ct':
            save_path += '_I0_{}'.format(physics.I0)
            closure_rei = closure_rei_end2end_ct
        else:
            closure_rei = closure_rei_end2end

        os.makedirs(save_path, exist_ok=True)

        generator = UNet(in_channels=self.in_channels,
                         out_channels=self.out_channels,
                         compact=4, residual=True,
                         circular_padding=True, cat=True).to(self.device)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location=self.device)
            generator.load_state_dict(checkpoint['state_dict'])

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])

        log = LOG(save_path, filename='training_loss',
                  field_name=['epoch', 'loss_sure', 'loss_req', 'loss_total', 'psnr', 'mse'])

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)

            loss = closure_rei(generator, dataloader, physics, transform, optimizer,
                               criterion, alpha, tau, self.dtype, self.device, report_psnr)

            log.record(epoch + 1, *loss)
            print('{}\tEpoch[{}/{}]\tsure={:.4e}\treq={:.4e}\tloss={:.4e}\tpsnr={:.4f}\tmse={:.4e}'
                  .format(get_timestamp(), epoch, epochs, *loss))

            if save_ckp:
                if epoch >0 and epoch % ckp_interval == 0:
                    state = {'epoch': epoch,
                             'state_dict': generator.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'args':args.__dict__}
                    torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))

            if epoch + 1 == epochs:
                state = {'epoch': epoch,
                         'state_dict': generator.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'args':args.__dict__}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
        log.close()

    def train_ei(self, dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                  schedule, pretrained, lr_cos, save_ckp, report_psnr, args):

        save_path = './ckp/{}_ei_{}_{}_sigma{}_gamma{}' \
            .format(get_timestamp(), physics.name,
                    physics.noise_model['noise_type'],
                    physics.noise_model['sigma'],
                    physics.noise_model['gamma'])
        if physics.name == 'ct':
            save_path += '_I0_{}'.format(physics.I0)

        os.makedirs(save_path, exist_ok=True)

        generator = UNet(in_channels=self.in_channels,
                         out_channels=self.out_channels,
                         compact=4, residual=True,
                         circular_padding=True, cat=True).to(self.device)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location=self.device)
            generator.load_state_dict(checkpoint['state_dict'])

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])

        log = LOG(save_path, filename='training_loss',
                  field_name=['epoch', 'loss_mc', 'loss_eq', 'loss_total', 'psnr',
                              'mse'])

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)

            loss = closure_ei_end2end(generator, dataloader, physics, transform, optimizer,
                               criterion, alpha, self.dtype, self.device, report_psnr)

            log.record(epoch + 1, *loss)
            print(
                '{}\tEpoch[{}/{}]\tmc={:.4e}\teq={:.4e}\tloss={:.4e}\tpsnr={:.4f}\tmse={:.4e}'
                .format(get_timestamp(), epoch, epochs, *loss))

            if save_ckp:
                if epoch > 0 and epoch % ckp_interval == 0:
                    state = {'epoch': epoch,
                             'state_dict': generator.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'args': args.__dict__}
                    torch.save(state,
                               os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))

            if epoch + 1 == epochs:
                state = {'epoch': epoch,
                         'state_dict': generator.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'args': args.__dict__}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
        log.close()

    def train_sup(self, dataloader, physics, epochs, lr, ckp_interval,
                  schedule, pretrained, lr_cos, save_ckp, report_psnr, args):

        save_path = './ckp/{}_sup_{}_{}_sigma{}_gamma{}' \
            .format(get_timestamp(), physics.name,
                    physics.noise_model['noise_type'],
                    physics.noise_model['sigma'],
                    physics.noise_model['gamma'])
        if physics.name == 'ct':
            save_path += '_I0_{}'.format(physics.I0)

        os.makedirs(save_path, exist_ok=True)

        generator = UNet(in_channels=self.in_channels,
                         out_channels=self.out_channels,
                         compact=4, residual=True,
                         circular_padding=True, cat=True).to(self.device)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location=self.device)
            generator.load_state_dict(checkpoint['state_dict'])

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])

        log = LOG(save_path, filename='training_loss',
                  field_name=['epoch', 'loss_total', 'psnr', 'mse'])

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)

            loss = closure_supervised(generator, dataloader, physics, optimizer,
                                      criterion, self.dtype, self.device, report_psnr)

            log.record(epoch + 1, *loss)
            print(
                '{}\tEpoch[{}/{}]\tloss={:.4e}\tpsnr={:.4f}\tmse={:.4e}'
                .format(get_timestamp(), epoch, epochs, *loss))

            if save_ckp:
                if epoch > 0 and epoch % ckp_interval == 0:
                    state = {'epoch': epoch,
                             'state_dict': generator.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'args': args.__dict__}
                    torch.save(state,
                               os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))

            if epoch + 1 == epochs:
                state = {'epoch': epoch,
                         'state_dict': generator.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'args': args.__dict__}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
        log.close()