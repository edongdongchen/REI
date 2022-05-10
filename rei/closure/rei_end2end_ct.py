import torch
import numpy as np
from utils.metric import cal_psnr, cal_mse

def closure_rei_end2end_ct(net, dataloader, physics, transform, optimizer,
                        criterion, alpha, tau, dtype, device, reportpsnr=False,):
    loss_sure_seq, loss_req_seq, loss_seq, psnr_seq, mse_seq = [], [], [], [], []

    assert physics.name=='ct', 'This scripts only work for MPG noise in the CT task!'

    norm = lambda x: (x - physics.MIN) / (physics.MAX - physics.MIN)
    f = lambda fbp: net(norm(fbp)) * (physics.MAX - physics.MIN) + physics.MIN

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape)==5:
            N,n_crops,C,H,W =x.shape
            x = x.view(N*n_crops, C,H,W)
        if len(x.shape)==3:
            x = x.unsqueeze(1)
        x = x.type(dtype).to(device) # GT
        x = x * (physics.MAX - physics.MIN) + physics.MIN # normalize data

        meas0 = physics.A(x, add_noise=True)

        s_mpg = torch.log(physics.I0 / meas0)
        fbp_mpg = physics.iradon(s_mpg)

        x1 = f(fbp_mpg)

        meas1 = physics.A(x1)

        # SURE-based unbiased estimator to the clean measurement consistency loss
        if physics.noise_model['noise_type'] == 'mpg':
            sigma2 = physics.noise_model['sigma'] ** 2
            b1 = torch.randn_like(meas0)
            b2 = torch.rand_like(meas0) > 0.5
            b2 = (2 * b2.int() - 1) * 1.0  # binary [-1, 1]

            fbp_2 = physics.iradon(torch.log(physics.I0 / (meas0 + tau * b1)))
            fbp_2p = physics.iradon(torch.log(physics.I0 / (meas0 + tau * b2)))
            fbp_2n = physics.iradon(torch.log(physics.I0 / (meas0 - tau * b2)))

            meas2 = physics.A(f(fbp_2))
            meas2p = physics.A(f(fbp_2p))
            meas2n = physics.A(f(fbp_2n))

            K = meas0.shape[0]  # batch size
            m = meas0.shape[-1] * meas0.shape[-2] * meas0.shape[-3] # dimension of y

            loss_A = torch.sum((meas1 - meas0).pow(2)) / (K * m) - sigma2
            loss_div1 = 2 / (tau * K * m) * ((b1 * (physics.noise_model['gamma'] * meas0 + sigma2)) * (meas2 - meas1)).sum()
            loss_div2 = 2 * sigma2 * physics.noise_model['gamma'] / (tau ** 2 * K * m) * (b2 * (meas2p + meas2n - 2 * meas1)).sum()

            loss_sure = alpha['sure'] * (loss_A + loss_div1 + loss_div2)

        # REQ (EI with noisy input)
        x2 = transform.apply(x1)
        meas_x2 = physics.A(x2, add_noise=True)
        fbp_x2 = physics.iradon(torch.log(physics.I0 / meas_x2))
        x3 = f(fbp_x2)

        # compute loss_req
        loss_req = alpha['req'] * criterion(norm(x3), norm(x2))

        loss = loss_sure + loss_req

        loss_sure_seq.append(loss_sure.item())
        loss_req_seq.append(loss_req.item())
        loss_seq.append(loss.item())

        if reportpsnr:
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_closure = [np.mean(loss_sure_seq), np.mean(loss_req_seq), np.mean(loss_seq)]

    if reportpsnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure