import torch
import numpy as np
from utils.metric import cal_psnr, cal_mse, cal_psnr_complex


def closure_supervised(net, dataloader, physics, optimizer,
                       criterion, dtype, device, reportpsnr=False):
    loss_seq, psnr_seq, mse_seq = [], [], []

    cal_psnr_fn = cal_psnr_complex if physics.name in ['mri'] else cal_psnr

    if physics.name in ['ct']:
        norm = lambda x: (x - physics.MIN) / (physics.MAX - physics.MIN)
        f = lambda fbp: net(norm(fbp)) * (physics.MAX - physics.MIN) + physics.MIN
    else:
        f = lambda y: net(physics.A_dagger(y))

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x

        if len(x.shape) == 5:
            N, n_crops, C, H, W = x.shape
            x = x.view(N * n_crops, C, H, W)

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(dtype).to(device)

        if physics.name in ['ct']:
            x = x * (physics.MAX - physics.MIN) + physics.MIN
            meas0 = physics.A(x, add_noise=True)
            s_mpg = torch.log(physics.I0 / meas0)
            fbp_mpg = physics.iradon(s_mpg)
            x1 = f(fbp_mpg)

            loss = criterion(norm(x1), norm(x))

        else:
            y0 = physics.A(x, add_noise=True)
            x1 = f(y0)

            loss = criterion(x1, x)

        loss_seq.append(loss.item())

        if reportpsnr:
            psnr_seq.append(cal_psnr_fn(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_closure = [np.mean(loss_seq)]

    if reportpsnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure