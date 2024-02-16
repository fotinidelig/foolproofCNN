from PIL import Image
from datetime import datetime
import os
import torch
import numpy as np
from .utils import *
import time as tm
from experiments.models.utils import predict

'''
    *Disclaimer*
    Using:
    - random initialization (in each binary step)
    - no early stopping
    - tanh rounding with noise to avoid NaN values
    - predict() function from BasicModelClass which uses
      model in inference mode (model.eval())
'''

def loss(
        const,
        conf,
        adv_sample,
        input,
        logits,
        targeted,
        target
    ):
    """
    Calculating the loss with f_6 objective function,
    refer to the paper for details
    """
    device = input.device
    N = input.size(0)
    max_ = torch.tensor([0]*N).to(device)
    f_i = torch.stack([
        max([logits[i][j] for j in range(len(logits[0])) if j != target[i]]) for i in range(N)
    ]).to(device)
    f_t =  torch.stack([logits[i][target[i]] for i in range(N)])
    fx = (f_i - f_t) if targeted else (f_t - f_i) + conf
    obj = torch.max(fx, max_)
    l = torch.norm(adv_sample-input)**2 + const*obj
    return l, fx


def cwattack(
        model,
        x, # size (N,d...)
        targeted,
        target=None, # size (N,C)
        max_const=1e10,
        min_const=1e-3,
        init_const=1e-2,
        conf=0,
        bin_steps=10,
        max_iterations=1000,
        lr=0.01,
        x_min = -0.5,
        x_max = 0.5,
        random_init = True
    ):
    """
        Input tensor X of N images and Target of N targets (if targeted),
        returns attack, l2, const and attack time.
    """
    def bin_search_const(const_i, max_const_i, min_const_i, fx):
        """
            Binary search for const in range
            [min_const_i, max_const_i].
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
    indices = []

    # calculations to constrain tanh() within [BOXMIN, BOXMAX]
    TO_MUL = .5*(x_max - x_min)
    TO_ADD = .5*(x_max + x_min)

    if not targeted:
        target, _ = predict(model, x)

    max_const_n = torch.tensor([max_const]*N, device=device).float()
    min_const_n = torch.tensor([min_const]*N, device=device).float()
    const_n = torch.tensor([init_const]*N, device=device).float()

    found_atck_n = [False]*N
    best_atck_n = x.clone().detach()
    best_l2_n = torch.full((N,), np.inf, device=device)
    best_const_n = torch.zeros(N, device=device)
    best_w_n = torch.zeros_like(x, device=device)

    eps = torch.tensor(np.random.uniform(-0.02, 0.02, x.shape), device=device) # random noise in range [-0.03, 0.03]
    input = (x+eps).clamp(min=x_min, max=x_max) # control for arctanh NaN values
    inv_input = torch.atanh((input-TO_ADD)/TO_MUL).to(device)
    w_n = inv_input.clone().detach().requires_grad_(True).to(device)

    for _ in range(bin_steps):
        params = [{'params': w_n}]
        optimizer = torch.optim.Adam(params, lr=lr)
        cur_fx_n = torch.full((N,), np.inf)
        for i in range(max_iterations+1):
            adv_n = TO_MUL*torch.tanh(w_n)+TO_ADD
            _, logits_n = predict(model, adv_n, logits=True)
            loss_n, fx_n = loss(const_n, conf, adv_n, x, logits_n, targeted, target)

            loss_n.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            # update best results
            for k in range(N):
                if fx_n[k] < 0:
                    cur_fx_n[k] = fx_n[k]
                    found_atck_n[k] = True
                    if i not in indices:
                        indices.append(i)
                    l2_i = torch.norm(adv_n[k] - x[k])
                    if l2_i < best_l2_n[k]:
                        best_l2_n[k]  = l2_i
                        best_atck_n[k] = adv_n[k].detach().clone()
                        best_const_n[k] = const_n[k].clone()
                        best_w_n[k] = w_n[k].clone().detach()

        # update parameters
        w_n = w_n.clone().detach()
        if random_init:
            eps.data = torch.tensor(np.random.uniform(-0.02, 0.02, w_n.shape), device=device)
        else:
            eps.data = torch.zeros_like(w_n)
        for i in range(N):
            # update const
            vals = bin_search_const(const_n[i].item(), max_const_n[i].item(), min_const_n[i].item(), cur_fx_n[i])
            const_n[i], max_const_n[i], min_const_n[i], _ = vals
            # update w for next outer step
            if found_atck_n[i]:
                w_n[i] = best_w_n[i]+eps[i]
            else:
                w_n[i] = inv_input[i]+eps[i]
        w_n = w_n.requires_grad_(True)

    best_l2_n = torch.norm((best_atck_n-x).view(x.shape[0], -1), dim=1).tolist()
    return best_atck_n, best_l2_n, best_const_n


def cw_attack_all(
        model,
        model_name,
        sampleloader,
        targeted,
        dataset,
        classes,
        **kwargs
    ):
    device = next(model.parameters()).device
    folder = 'advimages/targeted/cw/' if targeted else 'advimages/untargeted/cw/'
    show_image = show_image_function(classes, folder)

    best_atck = []
    const_all = 0
    successful = 0
    distance = 0
    cnt_all = 0

    for bidx, batch in enumerate(sampleloader):
        inputs = batch[0].to(device)
        labels,_ = predict(model, inputs)
        targets = batch[1].to(device) if targeted else None

        start_time = tm.time()
        vals = cwattack(model, inputs, targeted, targets, **kwargs)
        batch_time = tm.time()-start_time

        output = vals[0] # (N, C, H, W)
        best_atck += output # (N, C, H, W) tensor == N x (C, H, W) list
        cnt_all += len(output)
        cnt = 0
        for i in range(len(output)):
            if succeeded(model, output[i], labels[i], targets[i] if targeted else None):
                const_all += vals[2][i]
                distance += vals[1][i]
                cnt += 1
                target = predict(model, output[i])[0][0]
                unique_idx = int(i + cnt_all - len(output) + target) # index of image across all batches + attack label
                show_image(unique_idx, (output[i], target), (inputs[i], labels[i]),
                        l2=vals[1][i], with_perturb=True)

        print("\n=> Attack took %f mins"%(batch_time/60))
        print(f"Found attack for {cnt}/{len(output)} samples.")
        successful += cnt
    # Logs
    lr = kwargs['lr'] if 'lr' in kwargs.keys() else 0.01
    iterations = kwargs['max_iterations'] if 'max_iterations' in kwargs.keys() else 1000
    mean_const = const_all/successful
    mean_distance = distance/successful
    kwargs = dict(lr=lr, iterations=iterations)
    kwargs['mean_const'] = mean_const
    kwargs['mean_distance'] = mean_distance
    kwargs['Attack'] = 'C&W'
    kwargs['total_cnt'] = len(best_atck)
    kwargs['adv_cnt'] = successful
    kwargs['dataset'] = dataset
    kwargs['model'] = model_name
    write_attack_log(**kwargs)

    best_atck = torch.stack(best_atck, dim=0).detach()
    return best_atck
