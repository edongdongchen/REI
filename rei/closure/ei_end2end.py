import torch
import numpy as np
from utils.metric import cal_psnr, cal_mse, cal_psnr_complex


def closure_ei_end2end(net, dataloader, physics, transform, optimizer,
                       criterion, alpha, dtype, device, report_psnr):

    loss_mc_seq, loss_eq_seq, loss_seq, psnr_seq, mse_seq = [], [], [], [], []
    cal_psnr_fn = cal_psnr_complex if physics.name in ['mri'] else cal_psnr

    if physics.name in ['ct']:
        norm = lambda x: (x - physics.MIN) / (physics.MAX - physics.MIN)
        f = lambda fbp: net(norm(fbp)) * (physics.MAX - physics.MIN) + physics.MIN
    else:
        f = lambda y: net(physics.A_dagger(y))

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        x = x.unsqueeze(1) if len(x.shape)==3 else x
        x = x.type(dtype).to(device) # GT

        if physics.name in ['ct']:
            x = x * (physics.MAX - physics.MIN) + physics.MIN

            meas0 = physics.A(x, add_noise=True)

            s_mpg = torch.log(physics.I0 / meas0)
            fbp_mpg = physics.iradon(s_mpg)

            x1 = f(fbp_mpg)
            meas1 = physics.A(x1)

            loss_mc = alpha['mc'] * criterion(meas1, meas0)

            # EI: x2, x3
            x2 = transform.apply(x1)
            meas2 = physics.A(x2)
            s2 = torch.log(physics.I0 / meas2)
            fbp_2 = physics.iradon(s2)
            x3 = f(fbp_2)

            loss_eq = alpha['eq'] * criterion(norm(x3), norm(x2))

        else:
            y0 = physics.A(x, add_noise=True)
            x1 = f(y0)
            y1 = physics.A(x1)

            loss_mc = alpha['mc'] * criterion(y1, y0)

            # EI: x2, x3
            x2 = transform.apply(x1)
            x3 = f(physics.A(x2))
            # loss EI
            loss_eq = alpha['eq'] * criterion(x3, x2)

        loss = loss_mc + loss_eq

        loss_mc_seq.append(loss_mc.item())
        loss_eq_seq.append(loss_eq.item())
        loss_seq.append(loss.item())

        if report_psnr:
            psnr_seq.append(cal_psnr_fn(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_closure = [np.mean(loss_mc_seq), np.mean(loss_eq_seq), np.mean(loss_seq)]
    if report_psnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure