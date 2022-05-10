import torch
import fastmri
from fastmri.data import transforms as T

class MRI():
    def __init__(self, acceleration=4, device='cpu', noise_model=None):
        mask = torch.load(f'../physics/mask_{acceleration}x.pth.tar')['mask']

        self.mask = mask.to(device)
        self.mask_func = lambda shape, seed: self.mask

        self.name = 'mri'
        self.acceleration = acceleration
        self.compress_ratio = 1 / acceleration

        if self.noise_mode is None:
            self.noise_model = {'noise_type':'g',
                                'sigma':0.01,
                                'gamma':0}
        else:
            self.noise_model = noise_model

    def apply_mask(self, y):
        y, _ = T.apply_mask(y, self.mask_func)
        return y

    def noise(self, y):
        if self.noise_model is not None:
            if self.noise_model['noise_type']=='g':
                n = torch.randn_like(y) * self.noise_model['sigma']
                n = fastmri.fft2c(n)
                y = y + n
            elif self.noise_model['noise_type']=='p':
                u = y
                z = torch.poisson(u / self.noise_model['gamma'])
                y = self.noise_model['gamma'] * z
            elif self.noise_model['noise_type']=='mpg':
                u = y
                z = torch.poisson(u / self.noise_model['gamma'])
                n = torch.randn_like(y) * self.noise_model['sigma']
                n = fastmri.fft2c(n)
                y = self.noise_model['gamma'] * z + n
            else:
                print('only Guassian (g), Poisson (p), and Mixed Poisson-Gaussian (mpg) '
                      'are supported!')
        return y

    def A(self, x, add_noise=False):
        y = fastmri.fft2c(x.permute(0, 2, 3, 1))
        if add_noise:
            y = self.noise(y)
        y, _ = T.apply_mask(y, self.mask_func)
        return y.permute(0,3,1,2)

    def A_dagger(self, y):
        x = fastmri.ifft2c(y.permute(0,2,3,1))
        return x.permute(0,3,1,2)