from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import show_image, plot_l2, write_output, verbose
import time as tm

# Constraints for image pixel values
BOXMIN = -0.5
BOXMAX = 0.5
# calculations to constrain tanh() within [BOXMIN, BOXMAX]
TO_MUL = .5*(BOXMAX - BOXMIN)
TO_ADD = .5*(BOXMAX + BOXMIN)

EARLY_STOP = True # stop optimizing the loss early

def loss(const, conf, adv_sample, input, logits, targeted, target):
    """
        Calculating the loss with f_6 objective function,
        refer to the paper for details
    """
    N = input.size(0)
    max_ = torch.tensor([0]*N).to(input.device)

    f_i = torch.stack([
            max([logits[i][j] for j in range(len(logits[0])) if j != target[i]]) for i in range(N)
            ])
    f_t =  torch.stack([logits[i][target[i]] for i in range(N)])
    fx = (f_i - f_t) if targeted else (f_t - f_i) + conf
    obj = torch.max(fx, max_)
    l = torch.norm(adv_sample-input)**2 + const*obj
    return l, fx

def attack_all(
        net,
        sampleloader,
        targeted,
        classes,
        dataname,
        **kwargs
    ):
    best_atck_all = []
    l2_all = []
    const_all = []
    indices_all = []
    cnt_adv = 0
    cnt_all = 0

    device = next(net.parameters()).device

    for bidx, batch in enumerate(sampleloader):
        inputs = batch[0].to(device)
        targets = batch[1].to(device) if targeted else None

        start_time = tm.time()
        vals = l2attack(net, inputs, targeted, targets, **kwargs)
        total_time = tm.time()-start_time

        best_atck = vals[0]
        l2_all += vals[1]
        const_all += vals[2]
        indices = vals[3]
        cnt_adv += len(indices)
        cnt_all += len(best_atck)

        print("\n=> Attack took %f mins"%(total_time/60))
        print(f"Found attack for {len(indices)}/{len(best_atck)} samples.")

        for i in indices:
            label = net.predict(inputs[i])[0][0]
            lab_atck = net.predict(best_atck[i])[0][0]
            fname = 'targeted' if targeted else 'untargeted/' + "cwl2/" + dataname
            show_image(i, (best_atck[i], lab_atck), (inputs[i], label),
                         classes, fname=fname, l2=vals[1][i])

    ## DEBUG:
    lr = kwargs['lr'] if 'lr' in kwargs.keys() else 0.01
    iterations = kwargs['max_iterations'] if 'max_iterations' in kwargs.keys() else 1000
    kwargs = dict(lr=lr,iterations=iterations)

    write_output(cnt_all, cnt_adv, const_all, l2_all,
                dataname, net.__class__.__name__, **kwargs)


def l2attack(
        net,
        x, # size (N,d...)
        targeted,
        target=None, # size (N,C)
        max_const=1e10,
        min_const=1e-3,
        init_const=1e-2,
        conf=0,
        bin_steps=10,
        max_iterations=1000,
        lr=0.01
    ):
    """
        Input tensor X of N images and Target of N targets (if targeted),
        returns attack, l2, const and attack time.
    """
    def bin_search_const(const_i, max_const_i, min_const_i, fx):
        """
            Binary search for const in range
            [min_const, max_const].

            Return smallest const for which f(x) moves to 0
            or end search if const is found.
        """
        end_iters = False
        if fx >= 0:
            if const_i*10 > max_const:
                end_iters = True
                return const_i, max_const_i, min_const_i, end_iters
            if max_const_i == max_const:
                # no successful attack found yet
                const_i *= 10
            else:
                min_const_i = const_i
                const_i = .5*(max_const_i+min_const_i)
        if fx < 0:
            max_const_i = const_i
            const_i = .5*(max_const_i+min_const_i)
        return const_i, max_const_i, min_const_i, end_iters

    N = x.size(0)

    device = x.device

    adv_samples = []
    adv_targets = []
    l2_distances = []
    const_vals = []
    indices = []

    if not targeted:
        target, _ = net.predict(x)

    max_const_n = torch.tensor([max_const]*N).to(device).float()
    min_const_n = torch.tensor([min_const]*N).to(device).float()
    const_n = torch.tensor([init_const]*N).to(device).float()
    w_n = torch.zeros_like(x).requires_grad_(True).float() # init

    found_atck_n = [False]*N
    best_atck_n = x.clone().detach()
    best_const_n = torch.zeros(N)
    best_l2_n = torch.zeros(N)
    best_w_n = torch.zeros_like(x).to(device)

    inv_input = torch.atanh((x-TO_ADD)/TO_MUL)
    eps = torch.tensor(np.random.uniform(-0.001, 0.001, x.shape)).to(device) # random noise in range [-0.03, 0.03]
    w_n = (inv_input+eps).clone().detach().requires_grad_(True)

    for _ in range(bin_steps):
        params = [{'params': w_n}]
        optimizer = torch.optim.Adam(params, lr=lr)
        # prev_loss_sum = np.inf
        for i in range(max_iterations+1):
            adv_sample_n = TO_MUL*torch.tanh(w_n)+TO_ADD
            _, logits_n = net.predict(adv_sample_n, logits=True)
            loss_n, fx_n = loss(const_n, conf, adv_sample_n, x, logits_n, targeted, target)

            # early stop if loss changes less than 0.0001 and 20% iterations done
            # if EARLY_STOP and loss_n.sum() > prev_loss_sum*.9999 and i > max_iterations/5:
            #     print("Early stop")
            #     print(i)
            #     break
            # prev_loss_sum = loss_n.sum()

            loss_n.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        # update parameters
        w_n = w_n.clone().detach()
        eps.data = torch.tensor(np.random.uniform(-0.001, 0.001, w_n.shape), device=device)

        for i in range(N):
            # store best attack so far
            if fx_n[i] < 0:
                found_atck_n[i] = True
                if i not in indices:
                    indices.append(i)
                best_atck_n[i] = adv_sample_n[i].detach().clone()
                best_const_n[i] = const_n[i].clone().item()
                best_l2_n[i] = torch.norm(adv_sample_n[i] - x[i]).item()
                best_w_n[i] = w_n[i].clone().detach()

            vals = bin_search_const(const_n[i].item(), max_const_n[i].item(), min_const_n[i].item(), fx_n[i])
            const_n[i], max_const_n[i], min_const_n[i], _ = vals
            # set w for next SGD iteration
            if found_atck_n[i]:
                w_n[i] = best_w_n[i]+eps[i]
            else:
                w_n[i] = inv_input[i]+eps[i]
        w_n = w_n.requires_grad_(True)

    return best_atck_n, best_l2_n, best_const_n, indices
