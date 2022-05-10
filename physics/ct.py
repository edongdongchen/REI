import torch
import numpy as np
from .radon import Radon, IRadon


class CT():
    def __init__(self, img_width, radon_view, uniform=True, circle = False, device='cuda:0', I0=1e5, noise_model=None):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)

        self.name='ct'
        self.I0 = I0

        # used for normalzation input
        self.MAX = 0.032 / 5
        self.MIN = 0

        if noise_model is None:
            self.noise_model = {'noise_type':'mpg',
                                'sigma':30,
                                'gamma':1}
        else:
            self.noise_model = noise_model

    def noise(self, m):
        if self.noise_model['gamma']>0:
            m = self.noise_model['gamma'] * torch.poisson(m / self.noise_model['gamma'])
            if self.noise_model['sigma'] > 0:
                noise = torch.randn_like(m) * self.noise_model['sigma']
                m = m + noise
        return m

    def A(self, x, add_noise=False):
        m = self.I0 * torch.exp(-self.radon(x)) # clean GT measurement
        if add_noise:
            m = self.noise(m)
        return m

    def A_dagger(self, y):
        return self.iradon(y)