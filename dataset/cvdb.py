import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset

import os

def CVDB_CVPR(dataset_name='Urban100', mode='train', batch_size=1, shuffle=True, crop_size=(512, 512), resize=False):
    if dataset_name=='Urban100':
        if os.path.exists('../dataset/Urban100/'):
            imgs_path = '../dataset/Urban100/'
        else:
            imgs_path = './dataset/Urban100/'
    if resize:
        transform_data = transforms.Compose([transforms.CenterCrop(crop_size),
                                             transforms.Resize(int(crop_size[0]/2)),
                                             transforms.ToTensor()])
    else:
        transform_data = transforms.Compose([transforms.CenterCrop(crop_size),
                                             transforms.ToTensor()])
    if mode == 'train':
        imgs_path = imgs_path + 'train/'
    if mode == 'test':
        imgs_path = imgs_path + 'test/'

    dataset = datasets.ImageFolder(imgs_path, transform=transform_data, target_transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader