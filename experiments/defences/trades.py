import time
import numpy as np
import torch
from torch import nn
from experiments.models.utils import learning_curve

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

def train_trades(
        net,
        _lambda,
        norm,
        trainloader,
        lr = .01,
        lr_decay = 1, # set to 1 for no effect
        epochs = 40,
        momentum = .9,
        weight_decay = 5e-4,
        **kwargs
    ):

    # PGD paramters
    eps = 0.031
    alpha = 0.007
    iters = 40

    net.train()
    device = next(net.parameters()).device

    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = momentum,
                                 nesterov = True, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader)*epochs) # for WideResNet
    criterion = nn.CrossEntropyLoss().to(device)
    batch_size = trainloader.batch_size
    loss_p_epoch = []

    for epoch in range(epochs):
        start_time = time.time()
        for bidx, batch in enumerate(trainloader):
            x = batch[0].to(device)
            y = batch[1].to(device)

            loss = trades_loss(net, optimizer, x, y, eps, alpha, norm, _lambda, iters)
            # loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        loss_p_epoch.append(float(loss.item()))
        epoch_time = time.time()-start_time
        cur_lr = optimizer.param_groups[0]["lr"]
        print("=> [EPOCH %d] LOSS = %.4f, LR = %.4f, TIME = %.4f mins"%
                (epoch, loss.item(), cur_lr, epoch_time/60))
    if kwargs['filename']:
        learning_curve(np.arange(epochs), loss_p_epoch, "all", lr, batch_size, kwargs['filename'])
    torch.save(net.state_dict(), f"pretrained/TRADES_{net.__class__.__name__}.pt")


def trades_loss(
    net,
    optimizer,
    x,
    y,
    eps,
    alpha,
    norm,
    _lambda,
    iters,
    x_min = -0.5,
    x_max = 0.5,
    ):

    device = x.device

    # start from a random point near x
    adv_x = x.clone().detach()
    rand = torch.zeros_like(x).uniform_(-eps, eps).float()
    adv_x = adv_x + rand
    adv_x = torch.clamp(adv_x, x_min, x_max).to(device)

    net.eval()
    set_requires_grad(net, False)
    true_prob = nn.Softmax(dim=1)(net(x))
    for _ in range(iters):
        adv_x = adv_x.clone().detach().requires_grad_(True)
        pred = net.forward(adv_x)
        adv_prob = nn.LogSoftmax(dim=1)(pred) # log softmax for predictionts, softmax for ground-truth
        loss = nn.KLDivLoss()(adv_prob, true_prob)
        loss.backward()
        if norm == np.inf:
            eta = alpha*adv_x.grad.sign()
            perturb = adv_x + eta - x
            perturb = torch.clamp(perturb, -eps, eps)
            adv_x = torch.clamp(x+perturb, x_min, x_max)
        elif norm == 2:
            eta = clip(adv_x.grad, alpha, norm)
            perturb = adv_x + eta - x
            perturb = clip(perturb, eps, norm, True)
            adv_x = torch.clamp(x+perturb, x_min, x_max)
        else:
            raise RuntimeError(f"Invalid norm was passed:{norm}, expected 2 or np.inf norm")

    set_requires_grad(net, True)
    net.train()
    optimizer.zero_grad()

    pred = net(x)
    loss_nat = nn.CrossEntropyLoss()(pred, y).mean()
    true_prob = nn.Softmax(dim=1)(pred)

    pred = net(adv_x)
    adv_prob = nn.LogSoftmax(dim=1)(pred)
    loss_rob = nn.KLDivLoss()(adv_prob, true_prob).mean()
    loss = loss_nat + loss_rob/_lambda

    loss.backward()

    return loss

# from https://github.com/pytorch/pytorch/issues/2655#issuecomment-333501083
def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
