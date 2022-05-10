import torch
import numpy as np
from utils.metric import cal_psnr, cal_mse, cal_psnr_complex

def closure_rei_end2end(net, dataloader, physics, transform, optimizer,
                        criterion, alpha, tau, dtype, device, reportpsnr=False):
    assert physics.name in ['mri', 'inpainting'], \
        'This scripts only work ' \
        'for Gaussian noise (e.g. in MRI) ' \
        'and Poission noise (e.g. in Inpainting)!'

    loss_sure_seq, loss_req_seq, loss_seq, psnr_seq, mse_seq = [], [], [], [], []

    cal_psnr_fn = cal_psnr_complex if physics.name in ['mri'] else cal_psnr
    f = lambda y: net(physics.A_dagger(y))

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape)==5:
            N,n_crops,C,H,W =x.shape
            x = x.view(N*n_crops, C,H,W)
        if len(x.shape)==3:
            x = x.unsqueeze(1)
        x = x.type(dtype).to(device) # GT

        y0 = physics.A(x, add_noise=True)
        x0 = physics.A_dagger(y0) #A^+y, or FBP in CT

        x1 = net(x0)
        y1 = physics.A(x1)

        # SURE-based unbiased estimator to the clean measurement consistency loss
        if physics.noise_model['noise_type']=='g':
            sigma2 = physics.noise_model['sigma'] ** 2
            # generate a random vector b
            b = torch.randn_like(x0)
            if physics.name in ['mri', 'inpainting']:
                b = physics.A(b)

            y2 = physics.A(net(physics.A_dagger(y0 + tau * b)))

            # compute batch size K
            K = y0.shape[0]
            # compute n (dimension of x)
            n = y0.shape[-1]*y0.shape[-2]*y0.shape[-3]

            # compute m (dimension of y)
            if physics.name=='mri':
                m = n /physics.acceleration # dim(y)
            if physics.name == 'inpainting':
                m = n * (1 - physics.mask_rate)

            # compute loss_sure
            loss_sure = torch.sum((y1 - y0).pow(2)) / (K * m) - sigma2 \
                        + (2 * sigma2 / (tau *m * K)) * (b * (y2 - y1)).sum()

        if physics.noise_model['noise_type'] == 'p':
            # generate a random vector b
            b = torch.rand_like(y0) > 0.5
            b = (2 * b.int() - 1) * 1.0  # binary [-1, 1]
            b = physics.A(b * 1.0)
            if physics.name in ['mri', 'inpainting']:
                b = physics.A(b)

            y2 = physics.A(net(physics.A_dagger(y0 + tau * b)))

            # compute batch size K
            K = y0.shape[0]
            # compute n (dimension of x)
            n = y0.shape[-1]*y0.shape[-2]*y0.shape[-3]

            # compute m (dimension of y)
            if physics.name=='mri':
                m = n /physics.acceleration # dim(y)
            if physics.name == 'inpainting':
                m = n * (1 - physics.mask_rate)

            loss_sure = torch.sum((y1 - y0).pow(2)) / (K * m) \
                        - physics.noise_model['gamma'] * y0.sum() / (K * m) \
                        + 2 * physics.noise_model['gamma'] / (tau * K * m) * ((b * y0) * (y2 - y1)).sum()

        # REQ (EI with noisy input)
        x2 = transform.apply(x1)
        x3 = f(physics.A(x2, add_noise=True))
        # compute loss_req
        loss_req = alpha['req'] * criterion(x3, x2)

        loss = loss_sure + loss_req

        loss_sure_seq.append(loss_sure.item())
        loss_req_seq.append(loss_req.item())
        loss_seq.append(loss.item())

        if reportpsnr:
            psnr_seq.append(cal_psnr_fn(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    loss_closure = [np.mean(loss_sure_seq), np.mean(loss_req_seq), np.mean(loss_seq)]

    if reportpsnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure