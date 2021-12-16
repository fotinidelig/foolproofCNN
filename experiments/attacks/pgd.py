import time
import torch
from torch import nn
import numpy as np
from .utils import *
from experiments.models.utils import predict
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
    rand = torch.zeros_like(x)
    rand = rand.uniform_(-alpha, alpha).float()
    rand = clip(rand, alpha, norm, proj=True)
    adv_x = adv_x + rand
    adv_x = torch.clamp(adv_x, x_min, x_max)

    for _ in range(n_iters):
        alpha_x = fgsm(model, adv_x, alpha, norm,
                    targeted, target, x_min, x_max)
        perturb = alpha_x - x
        perturb = clip(perturb, eps, norm, proj=True)
        adv_x = x + perturb
        adv_x = torch.clamp(adv_x, x_min, x_max)

    return adv_x, torch.norm((adv_x-x).view(x.shape[0], -1), dim=1).tolist()


def pgd_attack_all(
    model,
    model_name,
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
    device = next(model.parameters()).device
    folder = 'advimages/targeted/pgd/' if targeted else 'advimages/untargeted/pgd/'
    show_image = show_image_function(classes, folder)

    best_atck = []
    l_inf = 0
    distance = 0
    successful = 0
    for bidx, batch in enumerate(sampleloader):
        inputs = batch[0].to(device)
        labels, _ = predict(model, inputs)
        targets = batch[1].to(device) if targeted else None

        start_time = time.time()
        vals = pgd(model, inputs, eps, alpha, norm, n_iters, targeted, targets, **kwargs)
        total_time = time.time()-start_time
        output = vals[0]
        best_atck += output

        cnt = 0
        for i in range(len(output)):
            if succeeded(model, output[i], labels[i], targets[i] if targeted else None):
                cnt +=1
                distance += vals[1][i]
                l_inf_i = (output[i]-inputs[i]).view(-1)
                l_inf += max(l_inf_i)
                target = predict(model, output[i])[0][0]
                show_image(i, (output[i], target), (inputs[i], labels[i]),
                          l2=vals[1][i])
            else:
                for _ in range(4): # 4 retries for failed attacks
                    target = torch.stack([targets[i]]) if targeted else None
                    retry = pgd(model, torch.stack([inputs[i]]), eps, alpha, norm, n_iters, targeted, target, **kwargs)
                    if succeeded(model, retry[0][0], labels[i], targets[i] if targeted else None):
                        cnt +=1
                        distance += retry[1][0]
                        l_inf_i = (retry[0][0]-inputs[i]).view(-1)
                        l_inf += max(l_inf_i)
                        target = predict(model, retry[0][0])[0][0]
                        show_image(i, (retry[0][0], target), (inputs[i], labels[i]),
                               l2=retry[1][0])
                        break
        successful += cnt
        print("\n=> Attack took %f mins"%(total_time/60))
        print(f"Found attack for {cnt}/{len(output)} samples.")

    ## LOGS
    mean_distance = distance/successful
    kwargs = dict()
    kwargs['mean_distance'] = mean_distance
    kwargs['Attack'] = 'PGD'
    kwargs['total_cnt'] = len(best_atck)
    kwargs['adv_cnt'] = successful
    kwargs['dataset'] = dataset
    kwargs['model'] = model_name
    kwargs['iters'] = n_iters
    kwargs['epsilon'] = eps
    kwargs['alpha'] = alpha
    if norm == np.inf:
        kwargs['l_inf_distance'] = l_inf/successful
    write_attack_log(**kwargs)
    return torch.stack(best_atck).detach()
