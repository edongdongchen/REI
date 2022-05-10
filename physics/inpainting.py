import os
import torch

class Inpainting():
    def __init__(self, img_heigth=512, img_width=512, mask_rate=0.3, device='cuda:0', noise_model=None):
        self.name='inpainting'
        self.mask_rate = mask_rate
        self.compress_ratio = 1 - mask_rate
        self.noise_model = noise_model

        if img_heigth==256 and img_width==256:
            mask_path = '../dataset/masks/mask_random{}.pt'.format(mask_rate)
        else:
            mask_path = '../dataset/masks/mask_{}x{}_random{}.pt'.format(img_heigth, img_width, mask_rate)
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path).to(device)
        else:
            self.mask = torch.ones(img_heigth, img_width, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
            torch.save(self.mask, mask_path)

    def noise(self, x):
        y = x
        if self.noise_model is not None:
            if self.noise_model['noise_type']=='g' and self.noise_model['sigma']>0:
                n = torch.randn_like(x) * self.noise_model['sigma']
                y = x + n
                y = torch.einsum('kl,ijkl->ijkl', self.mask, y)
            elif self.noise_model['noise_type']=='p' and self.noise_model['gamma']>0:
                u = x
                z = torch.poisson(u / self.noise_model['gamma'])
                y = self.noise_model['gamma'] * z
                y = torch.einsum('kl,ijkl->ijkl', self.mask, y)
            elif self.noise_model['noise_type']=='mpg' and self.noise_model['sigma']>0 and self.noise_model['gamma']>0:
                n = torch.randn_like(x) * self.noise_model['sigma']
                u = x
                z = torch.poisson(u / self.noise_model['gamma'])
                y = self.noise_model['gamma'] * z + n
                y = torch.einsum('kl,ijkl->ijkl', self.mask, y)
        return y

    def A(self, x, add_noise=False):
        y = self.noise(x) if add_noise else x
        y = torch.einsum('kl,ijkl->ijkl', self.mask, y)
        return y

    def A_dagger(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)