import time
import numpy as np
import torch
import os
from torch import nn
from experiments.models.utils import learning_curve, validate_model
from experiments.attacks.fgsm import clip


def train_trades(
    model,
    _lambda,
    trainloader,
    valloader = None,
    norm = np.inf,
    lr = .01,
    lr_decay = 1, # set to 1 for no effect
    epochs = 40,
    momentum = .9,
    weight_decay = 5e-4,
    model_name = None,
    eps = 0.031,
    alpha = 0.007,
    iters = 30,
    **kwargs
    ):

    model.train()
    device = next(model.parameters()).device

    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum,
                                 nesterov = True, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader)*epochs) # for WideResmodel
    criterion = nn.CrossEntropyLoss().to(device)
    batch_size = trainloader.batch_size
    loss_p_epoch = []
    val_loss_p_epoch = []

    for epoch in range(epochs):
        start_time = time.time()
        losses = 0
        for i, batch in enumerate(trainloader):
            x = batch[0].to(device)
            y = batch[1].to(device)

            loss = trades_loss(model, optimizer, x, y, eps, alpha, norm, _lambda, iters)
            losses += float(loss)

            optimizer.step()
            optimizer.zero_grad()

            del x, y
        epoch_time = time.time()-start_time
        print("=> [EPOCH %d] LOSS = %.4f, LR = %.4f, TIME = %.4f mins"%
        (epoch, losses/(i+1), optimizer.param_groups[0]["lr"], epoch_time/60))

        if valloader:
            val_loss = validate_model(model, valloader, device)
            val_loss_p_epoch.append(val_loss)
            print("=> [EPOCH %d] VAL LOSS = %.4f"%(epoch, val_loss))


        loss_p_epoch.append(losses/(i+1))

        scheduler.step()

    if 'l_curve_name' in kwargs.keys():
        learning_curve(np.arange(epochs), loss_p_epoch, val_loss_p_epoch, "all", lr, batch_size, kwargs['l_curve_name'])
    if not os.path.isdir('models'):
        os.makedirs('models')
    model_name = model.__class__.__name__ if not model_name else model_name
    torch.save(model.state_dict(), f"pretrained/{model_name}.pt")


def trades_loss(
    model,
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
    rand = torch.empty(x.shape).normal_(mean=0,std=1).to(device)
    adv_x = adv_x + rand
    adv_x = torch.clamp(adv_x, x_min, x_max).to(device)

    model.eval()
    set_requires_grad(model, False)
    true_prob = nn.Softmax(dim=1)(model(x))
    for _ in range(iters):
        adv_x = adv_x.clone().detach().requires_grad_(True)
        pred = model.forward(adv_x)
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

    ## DEBUG:
    # i = np.random.randint(x.shape[0]) # random image
    # o = np.random.randint(100) # do 50% of epochs
    # classes = [str(i) for i in range(10)]
    # _l2 = torch.norm(adv_x[i] - x[i])
    # if o < 50:
    #     lab_atck = model.predict(adv_x[i])[0][0]
    #     fname = 'trades_test'
    #     show_image(i, (adv_x[i], lab_atck), (x[i], y[i]),
    #              classes, fname="trades_adv", l2=_l2)

    set_requires_grad(model, True)
    model.train()
    optimizer.zero_grad()

    pred = model(x)
    loss_nat = nn.CrossEntropyLoss()(pred, y)
    # true_prob = nn.Softmax(dim=1)(pred)

    pred = model(adv_x)
    adv_prob = nn.LogSoftmax(dim=1)(pred)
    loss_rob = nn.KLDivLoss()(adv_prob, true_prob).mean()
    loss = loss_nat + loss_rob/_lambda
    loss.backward()
    return loss.item()

# from https://github.com/pytorch/pytorch/issues/2655#issuecomment-333501083
def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
