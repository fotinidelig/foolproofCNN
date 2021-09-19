import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from typing import Optional
from .utils import BaseAttack, plot_l2, write_output, verbose
import torch.backends.cudnn as cudnn
import time as tm
import sklearn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

# Constraints for image pixel values
BOXMIN = -0.5
BOXMAX = 0.5
# calculations to constrain tanh() within [BOXMIN, BOXMAX]
TO_MUL = .5*(BOXMAX - BOXMIN)
TO_ADD = .5*(BOXMAX + BOXMIN)

def l2_ball(l2, shape):
    '''
        Return random vector of shape
        and l2 norm
    '''
    xrand = np.random.uniform(0, 1, shape)
    xsum = np.sum(xrand**2)
    xrand = xrand*np.sqrt(l2**2/xsum)
    return torch.tensor(xrand)

def show_image(idx, adv_img, img, classes, fname, l2=None):
    plt.clf()
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

    if l2:
        fig.suptitle(f'L2 distance: {l2:.3f}', fontsize=16)

    adv_img, target = adv_img
    img, label = img
    images = list(map(lambda x: x.cpu().detach(), [img, adv_img]))
    npimgs = list(map(lambda x: x + .5, images)) # un-normalize
    npimgs = list(map(lambda x: x.numpy(), npimgs))
    npimgs = list(map(lambda x: np.round(x*255).astype(int), npimgs))
    kwargs = img.size()[0]==1 and {'cmap': 'gray'} or {}
    ax1.set_title("Original: Class %s"%classes[label])
    pl1=ax1.imshow(np.transpose(npimgs[0], (1, 2, 0)), **kwargs)
    ax2.set_title("Perturbed: Class %s"%classes[target])
    pl2=ax2.imshow(np.transpose(npimgs[1], (1, 2, 0)), **kwargs)

    if not os.path.isdir('advimages'):
        os.makedirs('advimages')
    plt.savefig(f"advimages/{fname}_sample_{idx}_{classes[target]}.png")


class L2Attack(BaseAttack):
    def __init__(
        self,
        net,
        dataset,
        init_const,
        conf: Optional[int] = 0,
        iterations: Optional[int] = 10000,
        max_const: Optional[float] = 1e10,
        min_const: Optional[float] = 1e-3,
    ):
        super(L2Attack, self).__init__(init_const, conf, iterations, max_const, min_const)
        self.net = net
        self.classes = dataset.classes
        self.data_name = dataset.__class__.__name__
        self.advset = []

    def loss(self, const, adv_sample, input, logits, target):
        """
            Calculating the loss with f_6 objective function,
            refer to the paper for details
        """
        f_i = max([logits[i] for i in range(len(logits)) if i != target])
        f_t =  logits[target]
        fx = f_i - f_t + self.conf
        obj = max([fx, 0])
        l = torch.norm(adv_sample-input)**2 + const*obj
        return l, fx

    def attack(self, sampleloader, samplelabs):
        bin_steps = 10
        lr = .01
        batch_size = sampleloader.batch_size
        total_samples = 0
        adv_samples = []
        adv_targets = []
        l2_distances = []
        const_vals = []

        # eps = np.finfo(float).eps
         # small l2 noise for gradient descent
        for bidx, batch in enumerate(sampleloader):
            inputs = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)

            for idx, (input, target) in enumerate(zip(inputs, targets), 0):
                total_samples+=1
                found_atck = False # init
                const = self.init_const # init
                max_const = self.max_const # init
                min_const = self.min_const # init
                w = torch.zeros(input.size(), requires_grad=True, device=device) # init
                start_time = tm.time()
                for step in range(bin_steps):
                    # random initialization of SGD
                    inv_input = torch.atanh((input-TO_ADD)/TO_MUL)
                    eps = np.random.uniform(-0.03, 0.03, input.shape) # random noise in range [-0.03, 0.03]
                    if found_atck:
                        w = (w+eps).clone().detach().requires_grad_(True).to(device)
                    else:
                        w = (inv_input+eps).clone().detach().requires_grad_(True).to(device)

                    params = [{'params': w}]
                    optimizer = torch.optim.Adam(params, lr=lr)

                    # optimization loop
                    start1=tm.time()
                    for i in range(self.iterations+1):
                        adv_sample = TO_MUL*torch.tanh(w)+TO_ADD
                        _, logits = self.net.predict(adv_sample.float(), logits=True) # net weights and input must be same dtype, aka float32
                        loss, fx = self.loss(const, adv_sample, input, logits[0], target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad() # always do zero_grad() after step()

                    const, max_const, min_const, end_iters = self.bin_search_const(const, max_const, min_const, fx) # update const for next iteration
                    if end_iters:
                        break
                    if fx < 0:
                        found_atck = True
                        best_atck = adv_sample.detach().clone()
                        best_const = const
                        best_l2 = torch.norm(adv_sample - input).item()

                torch.cuda.synchronize()
                total_time = tm.time()-start_time

                if found_atck:
                    input_l2 = torch.norm(input)
                    label = samplelabs[total_samples-1]
                    show_image(total_samples-1, (best_atck, target), (input, label),
                                 self.classes, fname=self.data_name, l2=best_l2)
                    adv_samples.append(best_atck)
                    adv_targets.append(target)
                    l2_distances.append(float(best_l2))
                    const_vals.append(best_const)

                if verbose:
                    print("\n=> Attack took %f mins"%(total_time/60))
                    if found_atck:
                        print("=> Found attack with CONST = %.3f, L2 = %.3f"%(best_const,best_l2))
                    else:
                        print("=> Didn't find attack, CONST = %.3f."%const)

        if len(adv_samples) > 0:
            plot_l2(l2_distances, self.iterations)
            write_output(total_samples, adv_samples, const_vals, l2_distances)
            adv_dataset = torch.utils.data.TensorDataset(torch.stack(adv_samples), torch.tensor(adv_targets))
            self.advset = adv_dataset
