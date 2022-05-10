import torch
import numpy as np
import kornia as dgm
import random

class Rotate():
    def __init__(self, n_trans, random_rotate=False):
        self.n_trans = n_trans
        self.random_rotate = random_rotate
    def apply(self, x):
        return rotate_dgm(x, self.n_trans, self.random_rotate)

def rotate_dgm(data, n_trans=5, random_rotate=False):
    if random_rotate:
        theta_list = random.sample(list(np.arange(1, 360)), n_trans)
    else:
        theta_list = np.arange(10, 360, int(360 / n_trans))

    data = torch.cat([data if theta == 0 else dgm.rotate(data, torch.Tensor([theta]).type_as(data))
                      for theta in theta_list], dim=0)
    return data