import torch
from torch import nn
import numpy as np

def N_mul(x, a):
    '''Multiply tensor of (N,d...) dimensions
       with tensor of N dimensions
    '''
    assert x.shape[0] == a.shape[0], "Input tensors must have same 0 dimension"
    ret = torch.stack([x[i]*a[i] for i in range(x.shape[0])])
    return ret

def clip(x, eps, norm, pgd_proj=False):
    '''
        Using norms np.inf, 1, or 2

        If pgd_proj == True, project only if x's norm exceeds eps.
    '''
    assert len(x.size()) == 4, "Function accepts elements in batches"

    red_dims = tuple(range(1, len(x))) # red_dims = (1,2,3)
    min_ = torch.tensor([1.0]*x.size(0))
    zero_thres = torch.tensor([1e-12]*x.size(0)) # zero_thres.shape = N

    if norm == np.inf:
        if pgd_proj:
            clipped = torch.clamp(x, -eps, eps)
        else:
            clipped = eps*torch.sign(x)
    elif norm == 1:
        one_norm = torch.maximum(torch.sum(x, dim=red_dims), zero_thres)
        clipped = N_mul(x, eps/one_norm) if not pgd_proj else N_mul(x, torch.min(min_, eps/one_norm))
    elif norm == 2:
        sq_norm = torch.maximum(torch.norm(x, dim=red_dims), zero_thres)
        clipped = N_mul(x, eps/sq_norm) if not pgd_proj else N_mul(x, torch.min(min_, eps/sq_norm))
    return clipped

def fgsm(
    net,
    x,
    eps,
    norm,
    targeted=False,
    target=None,
    x_min = -.5,
    x_max = .5
    ):
    assert norm in [np.inf, 1, 2], "To run FGSM attack, norm must be np.inf, 1, or 2."
    assert len(x.size()) == 4, "Function accepts elements in batches"

    net.eval()

    if not targeted:
        target = torch.argmax(net.forward(x))

    x = x.clone().detach().requires_grad_(True).float()
    criterion = nn.CrossEntropyLoss()
    x_pred = net.forward(x)
    loss = criterion(x_pred, target)
    if targeted:
        loss = -loss
    loss.backward()

    perturb = clip(x.grad, eps, norm)
    adv_x = x + perturb
    adv_x = torch.clamp(adv_x, x_min, x_max)
    return adv_x
