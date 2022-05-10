import torch
import numpy as np

# --------------------------------
# metric
# --------------------------------
def cal_psnr(a, b, mask=None):
    # a: prediction
    # b: ground-truth
    if mask is None:
        alpha = np.sqrt(a.shape[-1] * a.shape[-2])  # a.shape[-1]*a.shape[-2]
        return 20*torch.log10(alpha*torch.norm(b, float('inf'))/torch.norm(b-a, 2)).detach().cpu().numpy()
    else:
        alpha = np.sqrt(mask[mask>0].numel())
        return 20*torch.log10(alpha*torch.norm(b[mask>0], float('inf'))/torch.norm(b[mask>0]-a[mask>0], 2)).detach().cpu().numpy()

def cal_psnr_complex(a, b):
    """
    first permute the dimension, such that the last dimension of the tensor is 2 (real, imag)
    :param a: shape [N,2,H,W]
    :param b: shape [N,2,H,W]
    :return: psnr value
    """
    a = complex_abs(a.permute(0,2,3,1))
    b = complex_abs(b.permute(0,2,3,1))
    return cal_psnr(a,b)


def cal_mse(a, b, mask=None):
    # a: prediction
    # b: ground-truth
    if mask is None:
        return torch.nn.MSELoss()(a, b).item()
    else:
        return torch.nn.MSELoss()(a[mask>0], b[mask>0]).item()


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()