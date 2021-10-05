import torch
from torch import nn
import numpy as np

from .fgsm import fgsm, clip


def pgd(
    net: nn.Module,
    x: torch.tensor,
    eps: float,
    alpha: float,
    norm,
    n_iters: int,
    targeted = False,
    target = None,
    x_min = -.5,
    x_max = .5
):
    assert norm in [np.inf, 1, 2], "To run PGD attack, norm must be np.inf, 1, or 2."
    assert alpha < eps, "Inner step alpha must be smaller than outer step eps"
    assert len(x.size()) == 4, "Function accepts elements in batches"

    net.eval()
    device = next(net.parameters()).device

    adv_x = x.clone().detach().requires_grad_(True).float()

    # start from a random point near x
    rand = torch.zeros_like(x).uniform_(-eps, eps).float()
    rand = clip(rand, eps, norm, pgd_proj=True)
    adv_x = adv_x + rand
    adv_x = torch.clamp(adv_x, x_min, x_max)

    for _ in range(n_iters):
        alpha_x = fgsm(net, adv_x, alpha, norm, targeted, target, x_min, x_max)
        perturb = alpha_x - x
        perturb = clip(perturb, eps, norm, pgd_proj=True)
        adv_x = x + perturb
        adv_x = torch.clamp(adv_x, x_min, x_max)

    return adv_x

def pgd_inf(
    net: nn.Module,
    x: torch.tensor,
    eps = 0.031,
    alpha = 0.007,
    n_iters = 40,
    targeted = False,
    target = None,
    x_min = -.5,
    x_max = .5
):
    assert alpha < eps, "Inner step 'alpha' must be smaller than outer step 'eps'."
    assert len(x.size()) == 4, "Function accepts elements in batches."

    net.eval()
    device = next(net.parameters()).device

    adv_x = x.clone().detach()

    # start from a random point near x
    rand = torch.zeros_like(x).uniform_(-eps, eps).float()
    adv_x = adv_x + rand
    adv_x = torch.clamp(adv_x, x_min, x_max).to(device)

    if not targeted:
        target = torch.argmax(net.forward(x))

    criterion = nn.CrossEntropyLoss()
    for _ in range(n_iters):
        adv_x = adv_x.clone().detach().requires_grad_(True)
        pred = net.forward(adv_x)
        loss = criterion(pred, target)
        if targeted:
            loss = -loss
        loss.backward()
        eta = alpha*adv_x.grad.data.sign()
        perturb = adv_x + eta - x
        clipped = torch.clamp(perturb, -eps, eps)
        adv_x = x + clipped
        adv_x = torch.clamp(adv_x, x_min, x_max)

    return adv_x

def attack_all(
    net,
    sampleloader,
    targeted,
    classes,
    dataname,
    eps,
    alpha,
    norm,
    n_iters,
    **kwargs
    ):
    best_atck_all = []
    l2_all = []
    cnt_adv = 0
    cnt_all = 0

    device = next(net.parameters()).device

    for bidx, batch in enumerate(sampleloader):
        inputs = batch[0].to(device)
        targets = batch[1].to(device) if targeted else None

        start_time = tm.time()
        vals = pgd(net, inputs, eps, alpha, norm, n_iters, targeted, targets, **kwargs)
        total_time = tm.time()-start_time

        best_atck = vals[0]
        l2_all += vals[1]
        cnt_all += len(best_atck)

        indices = []
        for i, advimg in enumerate(best_atck):
            label = net.predict(inputs[i])[0][0]
            if succeeded(net, advimg, label, targets[i] if targeted else None):
                indices.append(i)
                cnt_adv += len(indices)

                print("\n=> Attack took %f mins"%(total_time/60))
                print(f"Found attack for {len(indices)}/{len(best_atck)} samples.")

                for i in indices:
                    label = net.predict(inputs[i])[0][0]
                    lab_atck = net.predict(best_atck[i])[0][0]
                    fname = 'targeted' if targeted else 'untargeted/' + "pgd/" + dataname
                    show_image(i, (best_atck[i], lab_atck), (inputs[i], label),
                    classes, fname=fname, l2=vals[1][i])

                    write_output(cnt_all, cnt_adv, [], l2_all, dataname,
                    net.__class__.__name__, iterations=n_iters)


def succeeded(net, adv_x, label, target=None):
    x_pred = net.predict(adv_x)[0][0]
    success = (target == x_pred) if target else (label != x_pred)
    if success:
        print("=> Found attack.")
    else:
        print("=> Didn't find attack.")
    return success
