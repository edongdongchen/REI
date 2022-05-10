import torch
import argparse

from rei.rei import REI

from dataset.cvdb import CVDB_CVPR
from physics.inpainting import Inpainting
from transforms.shift import Shift



parser = argparse.ArgumentParser(description='REI')
# inverse problem configs:
parser.add_argument('--task', default='inpainting', type=str,
                    help="inverse problems=['ct', 'inpainting', 'mri'] (default: 'inpainting')")

def main(cuda=0, gamma=0.01):
    args = parser.parse_args()

    device=f'cuda:{cuda}'

    pretrained = None
    lr_cos = False
    save_ckp = True
    report_psnr = True


    mask_rate = 0.3
    tau = 1e-2
    epochs = 500
    ckp_interval = 100
    schedule = [100, 200, 300, 400]
    batch_size = 1
    lr = {'G': 1e-4, 'WD': 1e-8}
    alpha = {'req': 1, 'sure': 1}

    noise_model = {'noise_type':'p',
                  'sigma':0,
                  'gamma':gamma}

    dataloader = CVDB_CVPR(dataset_name='Urban100', mode='train', batch_size=batch_size,
                           shuffle=True, crop_size=(512, 512), resize=True)

    transform = Shift(n_trans=3)

    physics = Inpainting(img_heigth=256, img_width=256,
                         mask_rate=mask_rate, device=device, noise_model=noise_model)

    rei = REI(in_channels=3, out_channels=3, img_width=256, img_height=256,
              dtype=torch.float, device=device)

    rei.train_rei(dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                  schedule, pretrained, lr_cos, save_ckp, tau, report_psnr, args)



if __name__ == '__main__':
    main(cuda=0, gamma=0.01)
    main(cuda=0, gamma=0.05)
    main(cuda=0, gamma=0.1)