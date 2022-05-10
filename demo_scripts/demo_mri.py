import torch
import argparse

from rei.rei import REI

from dataset.mridb import MRIData
from physics.mri import MRI
from transforms.rotate import Rotate


parser = argparse.ArgumentParser(description='REI')
# inverse problem configs:
parser.add_argument('--task', default='mri', type=str,
                    help="inverse problems=['ct', 'inpainting', 'mri'] (default: 'mri')")

def main(cuda=0, sigma=0.1):
    args = parser.parse_args()

    device=f'cuda:{cuda}'

    pretrained = None
    lr_cos = False
    save_ckp = True
    report_psnr = True

    acceleration = 4
    tau=1e-2
    epochs = 500
    ckp_interval = 100
    schedule = [300]

    batch_size = 2
    lr = {'G': 5e-4, 'WD': 1e-8}
    alpha = {'req': 1, 'sure': 1}

    noise_model = {'noise_type':'g',
                  'sigma':sigma,
                  'gamma':0}

    dataloader = torch.utils.data.DataLoader(dataset=MRIData(mode='test'),
                                             batch_size=batch_size, shuffle=True)

    transform = Rotate(n_trans=2, random_rotate=True)

    physics = MRI(acceleration=acceleration, device=device, noise_model=noise_model)


    rei = REI(in_channels=2, out_channels=2, img_width=320, img_height=320,
              dtype=torch.float, device=device)

    rei.train_rei(dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                  schedule, pretrained, lr_cos, save_ckp, tau, report_psnr, args)


if __name__ == '__main__':
    main(cuda=0, sigma=0.01)
    main(cuda=0, sigma=0.05)
    main(cuda=0, sigma=0.1)
    main(cuda=0, sigma=0.2)