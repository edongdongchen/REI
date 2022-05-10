import torch
from torch.utils.data.dataset import Dataset
import scipy.io as scio

class CTData(Dataset):
    """CT dataset."""
    def __init__(self, mode='train', root_dir='../dataset/CT/CT100_256x256.mat', sample_index=None):
        # the original CT100 dataset can be downloaded from
        # https://www.kaggle.com/kmader/siim-medical-images
        # the images are resized and saved in Matlab.

        mat_data = scio.loadmat(root_dir)
        x = torch.from_numpy(mat_data['DATA'])

        if mode=='train':
             self.x = x[0:90]
        if mode=='test':
            self.x = x[90:100,...]

        self.x = self.x.type(torch.FloatTensor)

        if sample_index is not None:
            self.x = self.x[sample_index].unsqueeze(0)

    def __getitem__(self, index):
        x = self.x[index]
        return x

    def __len__(self):
        return len(self.x)