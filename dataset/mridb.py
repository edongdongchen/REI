import torch
from torch.utils.data.dataset import Dataset

class MRIData(Dataset):
    """fastMRI dataset (knee subset)."""
    def __init__(self, mode='train',
                 root_dir='../dataset/mri/fastmri_knee_4865_norm_single_slice.pt',
                 sample_index=None, tag=900):

        x = torch.load(root_dir)
        x = x.squeeze()

        if mode == 'train':
            self.x = x[:tag]
        if mode == 'test':
            self.x = x[tag:, ...]

        self.x = torch.stack([self.x, torch.zeros_like(self.x)], dim=1)

        if sample_index is not None:
            self.x = self.x[sample_index].unsqueeze(0)

    def __getitem__(self, index):
        x = self.x[index]
        return x

    def __len__(self):
        return len(self.x)