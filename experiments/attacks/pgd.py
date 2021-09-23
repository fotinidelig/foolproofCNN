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

    adv_x = x.clone().detach().requires_grad_(True).float()

    # start from a random point near x
    rand = torch.zeros_like(x).uniform_(-eps, eps)
    rand = clip(rand, eps, norm, pgd_proj=True)
    adv_x = adv_x + rand
    adv_x = torch.clamp(adv_x, x_min, x_max)

    for _ in range(n_iters):
        alpha_x = fgsm(net, adv_x, alpha, norm, targeted, target, x_min, x_max)
        perturb = x - alpha_x
        perturb = clip(perturb, eps, norm, pgd_proj=True)
        adv_x = x + perturb
        adv_x = torch.clamp(adv_x, x_min, x_max)
    return adv_x
