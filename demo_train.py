import torch
import argparse

from rei.rei import REI

from dataset.mridb import MRIData
from dataset.cvdb import CVDB_CVPR
from dataset.ctdb import CTData

from physics.mri import MRI
from physics.inpainting import Inpainting
from physics.ct import CT

from transforms.shift import Shift
from transforms.rotate import Rotate


'''
# --------------------------------------------
# training code for REI
# --------------------------------------------
# Dongdong Chen (d.chen@ed.ac.uk)
# github: https://github.com/edongdongchen/EI
#         https://github.com/edongdongchen/REI
#
# Reference:
@inproceedings{chen2021equivariant,
  title     = {Equivariant Imaging: Learning Beyond the Range Space},
  author    = {Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {4379-4388}
}
@inproceedings{chen2022robust,
  title     = {Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements},
  author    = {Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
# --------------------------------------------
'''


parser = argparse.ArgumentParser(description='REI')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--schedule', nargs='+', type=int,
                    help='learning rate schedule (when to drop lr by 10x),'
                         'default [2000, 3000, 4000] for CT,'
                         'default [500, 1000, 1500] for inpainting')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run '
                         '(default 3000 for CT, 500 for inpainting and MRI)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate '
                                       '(default 5e-4 for CT, 1e-3 for inpainting)',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-8)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 2 for CT, 1 for inpainting)')
parser.add_argument('--ckp-interval', default=500, type=int,
                    help='save checkpoints interval epochs (default: 1000 for CT, 500 for inpainting)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# rei specific configs:
parser.add_argument('--n-trans', default=3, type=int,
                    help='number of transformations for EI')
parser.add_argument('--alpha-req', default=1.0, type=float,
                    help='equivariance strength (default: 1 for inpainting and mri, 1000 for CT)')
parser.add_argument('--alpha-sure', default=1.0, type=float,
                    help='sure strength (default: 1 for inpainting and mri, 1e-5 for CT)')
parser.add_argument('--alpha-eq', default=1.0, type=float,
                    help='equivariance strength (default: 1 for inpainting and mri, 1000 for CT)')
parser.add_argument('--alpha-mc', default=1.0, type=float,
                    help='mc strength (default: 1 for inpainting and mri, 1e-5 for CT)')
parser.add_argument('--tau', default=1e-2, type=float,
                    help='small positive number (default: 1e-2 for inpainting and mri, 10 for CT)')

# inverse problem task configs:
parser.add_argument('--task', default='inpainting', type=str,
                    help="inverse problems=['ct', 'inpainting', 'mri'] (default: 'mri')")
parser.add_argument('--acceleration', default=4, type=int,
                    help='acceleration ratio for MRI task (default: 4)')
parser.add_argument('--ct-views', default=50, type=int,
                    help='number of radon views for CT task (default: 50)')
parser.add_argument('--ct-I0', default=1e5, type=float,
                    help='CT intensity I0 (default: 1e5)')
parser.add_argument('--mask-rate', default=0.3, type=float,
                    help='mask rate for Inpainting task (default: 0.3)')

# noise model
parser.add_argument('--noise-type', default='g', type=str,
                    help="noise type (default: 'g' (gaussian), 'p' (poisson), "
                         "'mpg' (mixed poisson and gaussian))")
parser.add_argument('--noise-sigma', default=0.1, type=float,
                    help='guassian noise std (default: 0.1)')
parser.add_argument('--noise-gamma', default=0.1, type=float,
                    help='poisson noise scale (default: 0.1)')

def main():
    args = parser.parse_args()

    device=f'cuda:{args.gpu}'

    pretrained = None
    lr_cos = False
    save_ckp = True
    report_psnr = True

    if args.task == 'mri':
        acceleration = 4
        tau=1e-2
        epochs = 500
        ckp_interval = 50
        schedule = [300]

        batch_size = 2
        lr = {'G': 5e-4, 'WD': 1e-8}
        alpha = {'req': 1, 'sure': 1}

        noise_model = {'noise_type':'g', # gaussian
                      'sigma':args.noise_sigam,
                      'gamma':0}

        dataloader = torch.utils.data.DataLoader(dataset=MRIData(mode='train'),
                                                 batch_size=batch_size, shuffle=True)

        transform = Rotate(n_trans=2, random_rotate=True)
        physics = MRI(acceleration=acceleration, device=device, noise_model=noise_model)


        rei = REI(in_channels=2, out_channels=2, img_width=320, img_height=320,
                  dtype=torch.float, device=device)

        rei.train_rei(dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                      schedule, pretrained, lr_cos, save_ckp, tau, report_psnr, args)

    if args.task == 'inpainting':
        mask_rate = 0.3
        tau = 1e-2
        epochs = 500
        ckp_interval = 100
        schedule = [100, 200, 300, 400]
        batch_size = 1
        lr = {'G': 1e-4, 'WD': 1e-8}
        alpha = {'req': 1, 'sure': 1}

        noise_model = {'noise_type':'p', # poisson
                      'sigma':0,
                      'gamma':args.noise_gamma}

        dataloader = CVDB_CVPR(dataset_name='Urban100', mode='train', batch_size=batch_size,
                               shuffle=True, crop_size=(512, 512), resize=True)

        transform = Shift(n_trans=3)

        physics = Inpainting(img_heigth=256, img_width=256,
                             mask_rate=mask_rate, device=device, noise_model=noise_model)

        rei = REI(in_channels=3, out_channels=3, img_width=256, img_height=256,
                  dtype=torch.float, device=device)

        rei.train_rei(dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                      schedule, pretrained, lr_cos, save_ckp, tau, report_psnr, args)

    if args.task == 'ct':
        n_views = 50 # number of views
        tau = 10 # SURE

        epochs = 3000
        ckp_interval = 100
        schedule = [1000, 2000]

        batch_size = 2
        lr = {'G': 5e-4, 'WD': 1e-8}
        alpha = {'req': 1e3, 'sure': 1e-5}

        # define a MPG noise model
        I0 = 1e5
        noise_sigam = 30
        noise_model = {'noise_type': 'mpg', # mixed poisson-gaussian
                       'sigma': noise_sigam,
                       'gamma': 1}

        dataloader = torch.utils.data.DataLoader(
            dataset=CTData(mode='train'), batch_size=batch_size, shuffle=True)


        transform = Rotate(n_trans=2, random_rotate=True)

        physics = CT(256, n_views, circle=False, device=device, I0=I0,
                     noise_model=noise_model)

        rei = REI(in_channels=1, out_channels=1, img_width=256, img_height=256,
                  dtype=torch.float, device=device)

        rei.train_rei(dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                      schedule, pretrained, lr_cos, save_ckp, tau, report_psnr, args)

if __name__ == '__main__':
    main()