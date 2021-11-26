import time
import torch
from torch import nn
import numpy as np
from .utils import *
from .fgsm import fgsm, clip


def pgd(
    model: nn.Module,
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
    assert norm in [np.inf, 2], "To run PGD attack, np.inf or 2 norm is accepted (L1 norm not used/implemented)."
    assert alpha < eps, "Inner step `alpha` must be smaller than outer step `eps`"
    assert len(x.size()) == 4, "Function accepts elements in batches"

    model.eval()
    use_gpu()

    adv_x = x.clone().requires_grad_(True).float()

    # start from a random point near x
    rand = torch.zeros_like(x).uniform_(-eps, eps).float()
    rand = clip(rand, eps, norm, proj=True)
    adv_x = adv_x + rand
    adv_x = torch.clamp(adv_x, x_min, x_max)
    for _ in range(n_iters):
        alpha_x = fgsm(model, adv_x, alpha, norm,
                    targeted, target, x_min, x_max)
        perturb = alpha_x - x
        perturb = clip(perturb, eps, norm, proj=True)
        adv_x = x + perturb
        adv_x = torch.clamp(adv_x, x_min, x_max)

    return adv_x, torch.norm((adv_x-x).view(-1, 1), dim=1).tolist()


def pgd_attack_all(
    model,
    sampleloader,
    targeted,
    dataset,
    classes,
    eps,
    alpha,
    norm,
    n_iters,
    **kwargs
    ):

    use_gpu()
    folder = 'advimages/'+'targeted/' if targeted else 'untargeted/' + "pgd/"
    show_image = show_image_function(classes, folder)

    best_atck = []
    labels = []
    distance = 0
    successful = 0

    for bidx, batch in enumerate(sampleloader):
        inputs = batch[0]
        labels = model.predict(inputs)[0]
        targets = batch[1] if targeted else None

        start_time = time.time()
        vals = pgd_inf(model, inputs, eps, alpha, norm, n_iters, targeted, targets, **kwargs)
        total_time = time.time()-start_time
        output = vals[0]
        best_atck += output
        successful += len(output)

        cnt = 0
        for i in range(len(output)):
            if succeeded(model, output[i], labels[i], targets[i] if targeted else None):
                cnt +=1
                target = model.predict(output[i])[0][0]
                show_image(i, (output[i], target), (inputs[i], labels[i]),
                          l2=vals[1][i])

        print("\n=> Attack took %f mins"%(total_time/60))
        print(f"Found attack for {cnt}/{len(output)} samples.")

        # TODO Logs
        # write_attack_log(len(best_atck), successful, dataset,
                    # model.__class__.__name__, **kwargs)
    return torch.stack(best_atck).detach()


def succeeded(model, adv_x, label, target=None):
    x_pred, _ = model.predict(adv_x)
    success = (target == x_pred[0]) if target else (label != x_pred[0])
    if success:
        print("=> Found attack.")
    else:
        print("=> Didn't find attack.")
    return success
