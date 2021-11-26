import torch
from torch import nn
import numpy as np
from .utils import *

def clip(x, eps, norm, proj=False):
    '''
        Using norms np.inf, 1, or 2

        If proj == True, project if x's norm exceeds eps.
    '''
    assert len(x.size()) == 4, "Function accepts elements in batches."

    use_gpu()

    N = x.size(0)
    _one = torch.tensor([1.0]*N)
    zero_thres = torch.tensor([1e-12]*N)

    if norm == np.inf:
        if proj:
            clipped = torch.clamp(x, -eps, eps)
        else:
            clipped = eps*torch.sign(x)
    elif norm == 1:
        one_norm = torch.maximum(torch.sum(x.view(N, -1), dim=1), zero_thres)
        clipped = x*(eps/one_norm).unsqueeze(1) if not proj else x*torch.min(_one, eps/one_norm).unsqueeze(1)
    elif norm == 2:
        sq_norm = torch.maximum(torch.norm(x.view(N, -1), dim=1), zero_thres)
        clipped = x*(eps/sq_norm).unsqueeze(1) if not proj else x*torch.min(_one, eps/sq_norm).unsqueeze(1)
    return clipped

def fgsm(
    model,
    x,
    eps,
    norm,
    targeted=False,
    target=None,
    x_min = -.5,
    x_max = .5
    ):
    assert norm in [np.inf, 1, 2], "To run FGSM attack, norm must be np.inf, 1, or 2."
    assert len(x.size()) == 4, "Function accepts elements in batches."

    model.eval()

    if not targeted:
        logits = model.forward(x)
        target = torch.argmax(logits, dim=len(logits.shape)-1)

    x = x.clone().detach().requires_grad_(True).float()
    criterion = nn.CrossEntropyLoss()
    x_pred = model.forward(x)
    loss = criterion(x_pred, target)
    if targeted:
        loss = -loss
    loss.backward()
    perturb = clip(x.grad, eps, norm)
    adv_x = x + perturb
    adv_x = torch.clamp(adv_x, x_min, x_max)
    return adv_x
