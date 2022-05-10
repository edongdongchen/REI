import torch
import argparse

from rei.rei import REI
from dataset.ctdb import CTData
from physics.ct import CT

from transforms.rotate import Rotate


parser = argparse.ArgumentParser(description='REI')
# inverse problem configs:
parser.add_argument('--task', default='ct', type=str,
                    help="inverse problems=['ct', 'inpainting', 'mri'] (default: 'ct')")

def main():
    args = parser.parse_args()

    device='cuda:1'

    pretrained = None
    lr_cos = False
    save_ckp = True
    report_psnr = True

    n_views = 50
    tau = 10

    epochs = 3000
    ckp_interval = 100
    schedule = [1000, 2000]

    batch_size = 2
    lr = {'G': 5e-4, 'WD': 1e-8}
    alpha = {'req': 1e3, 'sure': 1e-5}

    I0 = 1e5
    noise_sigam = 30
    noise_model = {'noise_type': 'mpg',
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