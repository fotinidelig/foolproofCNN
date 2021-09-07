import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from typing import Optional
from .utils import BaseAttack
import time as tm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ID = torch.cuda.current_device()
print("=> CUDA ID: %d"%ID)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Constraints for image pixel values
BOXMIN = -.5
BOXMAX = .5
# calculations to constrain tanh() within [BOXMIN, BOXMAX]
TO_MUL = .5*(BOXMAX - BOXMIN)
TO_ADD = .5*(BOXMAX + BOXMIN)

class L2Attack(BaseAttack):
    def __init__(
        self,
        const,
        conf: Optional[int] = 0,
        iterations: Optional[int] = 10000,
        max_const: Optional[float] = 1e10,
        min_const: Optional[float] = 1e-3,
    ):
        super(L2Attack, self).__init__(const, conf, iterations, max_const, min_const)

    def loss(self, w, input, logits, target):
        """
            Calculating the loss with f_6 objective function,
            refer to the paper for details
        """
        f_i = max([logits[i] for i in range(len(logits)) if i != target])
        f_t =  logits[target]
        fx = f_i - f_t + self.conf
        obj = max([fx, 0])
        l = torch.norm(TO_MUL*torch.tanh(w)+TO_ADD-input)**2 + self.const*obj
        return l, fx

    # TODO: check effect of discretization
    # def discretization(self, image):
    #     """
    #         Returns a perturbed image with pixel values scaled to
    #         be valid after the attack
    #     """

    def attack(self, net, samples, targets):
        if not isinstance(samples, list):
            samples = [samples]
        if not isinstance(targets, list):
            targets = [targets]

        bin_steps = 10
        lr = .01

        w = torch.zeros((3, 32, 32), dtype=float, device=device, requires_grad=True)
        params = [{'params': w}]
        optimizer = torch.optim.Adam(params, lr=lr)
        adv_samples = []

        for idx, ((input, label), target) in enumerate(zip(samples, targets), 0):
            input = input.to(device)
            target = torch.tensor(target).to(device)
            label = torch.tensor(label).to(device)
            start_time = tm.time()

            found_atck = False # important initialisation
            for iteration in range(bin_steps):
                for i in range(self.iterations):
                    adv_sample = TO_MUL*torch.tanh(w)+TO_ADD
                    logits = net.forward(adv_sample.float())[0] # net weights and input must be same dtype, aka float32
                    loss, fx = self.loss(w,input,logits,target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad() # always do zero_grad() after step()
                self.const = self.bin_search_const(self.const, fx) # update const for next iteration
                if fx <= 0:
                    found_atck = True
                    best_atck = deepcopy(adv_sample.float().data) # can't copy non-leaf tensors
                    best_const = self.const
                    best_l2 = torch.norm(adv_sample - input)
            total_time = tm.time()-start_time
            print("=> Attack took %f mins"%(total_time/60))
            if found_atck:
                input_l2 = torch.norm(input)
                print("=> Found attack with CONST = %.3f. and L2 = %.4f"%(best_const, best_l2/input_l2*100),"%")
                print(best_atck.max(), best_atck.min())
                self.show_image(idx, (best_atck, target), (input, label))
                adv_samples.append()
            else:
                print("=> Didn't find attack.")

        adv_dataset = torch.utils.data.TensorDataset(torch.stack(adv_samples), torch.tensor(targets))
        self.advset = adv_dataset

    def show_image(self, idx, adv_img ,img = None):
        plt.clf()
        fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

        adv_img, target = adv_img
        img, label = img

        images = list(map(lambda x: x.cpu().detach(), [img, adv_img]))
        images = list(map(lambda x: x*.5+.5, images)) # un-normalize
        npimgs = list(map(lambda x: x.numpy(), images))
        npimgs = list(map(lambda x: np.round(x*255).astype(int), npimgs))

        ax1.set_title("Original: Class %s"%classes[label])
        pl1=ax1.imshow(np.transpose(npimgs[0], (1, 2, 0)))
        ax2.set_title("Perturbed: Class %s"%classes[target])
        pl2=ax2.imshow(np.transpose(npimgs[1], (1, 2, 0)))

        if not os.path.isdir('advimages'):
            os.makedirs('advimages')
        plt.savefig("advimages/sample_%d.png"%idx)

    def test(net):
        print("=> Testing success of attack")
        loader = torch.utils.data.DataLoader(self.advset)
        net.test(loader)
