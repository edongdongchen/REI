import torch
import numpy as np
from models.unet import UNet

from dataset.mridb import MRIData
from dataset.cvdb import CVDB_CVPR
from dataset.ctdb import CTData

from physics.mri import MRI
from physics.inpainting import Inpainting
from physics.ct import CT

from utils.metric import cal_psnr, cal_psnr_complex


def test_mri(net_name, net_ckp, sigma, device):
    acceleration = 4

    noise_model = {'noise_type': 'g',
                   'sigma': sigma,
                   'gamma': 0}

    unet = UNet(in_channels=2, out_channels=2, compact=4, residual=True,
                circular_padding=True, cat=True).to(device)

    dataloader = torch.utils.data.DataLoader(dataset=MRIData(mode='test'), batch_size=1, shuffle=False)

    forw = MRI(acceleration=acceleration, device=device, noise_model=noise_model)

    psnr_fbp, psnr_net = [],[]

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(torch.float).to(device)

        y = forw.A(x, add_noise=True)

        fbp = forw.A_dagger(y)

        psnr_fbp.append(cal_psnr_complex(fbp, x))

        checkpoint = torch.load(net_ckp, map_location=device)
        unet.load_state_dict(checkpoint['state_dict'])
        unet.to(device).eval()
        x_net = unet(fbp)

        psnr_net.append(cal_psnr_complex(x_net, x))

    print('AVG-PSNR (acceleration={}x\tnoise_level={})\t A^+y={:.3f} + {:.3f}\t{}={:.3f} + {:.3f}'.format(
        acceleration,noise_model['sigma'],np.mean(psnr_fbp),np.std(psnr_fbp), net_name, np.mean(psnr_net), np.std(psnr_net)))


def test_inpainting(net_name, net_ckp,  gamma, device):
    mask_rate=0.3
    noise_model = {'noise_type': 'p',
                   'sigma': 0,
                   'gamma': gamma}

    unet = UNet(in_channels=3, out_channels=3, compact=4, residual=True,
                circular_padding=True, cat=True).to(device)

    dataloader = CVDB_CVPR(dataset_name='Urban100', mode='test', batch_size=1,
                           shuffle=False, crop_size=(512, 512), resize=True)

    forw = Inpainting(img_heigth=256, img_width=256,
                         mask_rate=mask_rate, device=device, noise_model=noise_model)

    psnr_fbp, psnr_net = [],[]

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(torch.float).to(device)

        y = forw.A(x, add_noise=True)

        fbp = forw.A_dagger(y)

        psnr_fbp.append(cal_psnr(fbp, x))

        checkpoint = torch.load(net_ckp, map_location=device)
        unet.load_state_dict(checkpoint['state_dict'])
        unet.to(device).eval()
        x_net = unet(fbp)

        psnr_net.append(cal_psnr(x_net, x))

    print('AVG-PSNR (mask_rate={}\tnoise_level={})\t A^+y={:.3f} + {:.3f}\t{}={:.3f} + {:.3f}'.format(
        mask_rate, noise_model['gamma'],np.mean(psnr_fbp),np.std(psnr_fbp), net_name, np.mean(psnr_net), np.std(psnr_net)))


def test_ct(net_name, net_ckp, device):
    radon_view = 50
    I0 = 1e5
    sigma = 30

    noise_model = {'noise_type': 'mpg',
                   'sigma': sigma,
                   'gamma': 1}

    unet = UNet(in_channels=1, out_channels=1, compact=4, residual=True,
                circular_padding=True, cat=True).to(device)


    dataloader = torch.utils.data.DataLoader(
        dataset=CTData(mode='train',
                       root_dir=f'../dataset/CT/CT100_{256}x{256}.mat'),
        batch_size=2, shuffle=True)

    radon_view = radon_view
    forw = CT(256, radon_view, circle=False, device=device, I0=I0, noise_model=noise_model)

    # normalize the input
    f = lambda fbp: unet((fbp - forw.MIN) / (forw.MAX - forw.MIN)) \
                    * (forw.MAX - forw.MIN) + forw.MIN

    psnr_fbp, psnr_net = [],[]

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(torch.float).to(device)

        x = x * (forw.MAX - forw.MIN) + forw.MIN
        y = forw.A(x, add_noise=True)
        fbp = forw.iradon(torch.log(forw.I0 / y))


        psnr_fbp.append(cal_psnr(fbp, x))

        checkpoint = torch.load(net_ckp, map_location=device)
        unet.load_state_dict(checkpoint['state_dict'])
        unet.to(device).eval()
        x_net = f(fbp)

        psnr_net.append(cal_psnr(x_net, x))

    print('AVG-PSNR (views={}\tI0={}\tsigma={})\t FBP={:.3f} + {:.3f}\t{}={:.3f} + {:.3f}'.format(
        radon_view,I0,sigma, np.mean(psnr_fbp),np.std(psnr_fbp), net_name, np.mean(psnr_net), np.std(psnr_net)))

if __name__ == '__main__':
    device = 'cuda:3'
    net_ckp_mri = './mri.pt'
    net_ckp_ipt = './ipt.pt'
    net_ckp_ct = './ct.pt'
    test_mri(net_name='rei',net_ckp=net_ckp_mri, sigma=0.1, device=device)
    test_inpainting(net_name='rei',net_ckp=net_ckp_ipt, gamma=0.05, device=device)
    test_ct(net_name='rei',net_ckp=net_ckp_ct, device=device)