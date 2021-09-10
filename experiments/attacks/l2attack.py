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
BOXMIN = 0
BOXMAX = 1
# calculations to constrain tanh() within [BOXMIN, BOXMAX]
TO_MUL = .5*(BOXMAX - BOXMIN)
TO_ADD = .5*(BOXMAX + BOXMIN)

class L2Attack(BaseAttack):
    def __init__(
        self,
        init_const,
        conf: Optional[int] = 0,
        iterations: Optional[int] = 10000,
        max_const: Optional[float] = 1e10,
        min_const: Optional[float] = 1e-3,
    ):
        super(L2Attack, self).__init__(init_const, conf, iterations, max_const, min_const)

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

    # TODO: check effect of discretization
    # def discretization(self, image):
    #     """
    #         Returns a perturbed image with pixel values scaled to
    #         be valid after the attack
    #     """

    def attack(self, net, sampleloader, samplelabs):
        bin_steps = 10
        lr = .01
        batch_size = sampleloader.batch_size
        adv_samples = []
        adv_targets = []
        eps = np.finfo(float).eps
        for bidx, batch in enumerate(sampleloader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            for idx, (input, target) in enumerate(zip(inputs, targets), 0):
                start_time = tm.time()
                found_atck = False # init
                const = self.init_const # init
                max_const = self.max_const # init
                min_const = self.min_const # init
                w = torch.zeros((3,32,32), requires_grad=True, device=device) # init
                for step in range(bin_steps):
                    # Initialize w for gradient descent,
                    # needs to start near the previous attack or input
                    if found_atck:
                        w = w.clone().detach().requires_grad_(True).to(device)
                    else:
                        inv_input = torch.atanh((input+eps-TO_ADD)/TO_MUL)
                        w = inv_input.clone().detach().requires_grad_(True).to(device)
                    params = [{'params': w}]
                    optimizer = torch.optim.Adam(params, lr=lr)

                    # optimization loop
                    start1=tm.time()
                    for i in range(self.iterations+1):
                        adv_sample = TO_MUL*torch.tanh(w)+TO_ADD
                        _, logits = net.predict(adv_sample, logits=True) # net weights and input must be same dtype, aka float32
                        loss, fx = self.loss(const, adv_sample, input, logits[0], target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad() # always do zero_grad() after step()
                    # end1=tm.time()-start1
                    # print("iteration_time=%.2f"%(end1))

                    const, max_const, min_const, end_iters = self.bin_search_const(const, max_const, min_const, fx) # update const for next iteration
                    if end_iters:
                        break
                    if fx < 0:
                        found_atck = True
                        best_atck = adv_sample.detach().clone()
                        best_const = const
                        best_l2 = torch.norm(adv_sample - input)
                torch.cuda.synchronize()
                total_time = tm.time()-start_time
                # print("=> Attack took %f mins"%(total_time/60))
                if found_atck:
                    input_l2 = torch.norm(input)
                    # print("=> Found attack with CONST = %.3f"%(best_const))
                    label = samplelabs[idx+bidx*batch_size]
                    self.show_image(idx+bidx*batch_size, (best_atck, target), (input, label))
                    preds, probs = net.predict(best_atck)
                    adv_samples.append(best_atck)
                    adv_targets.append(target)
                else:
                    print("=> Didn't find attack, CONST = %.3f."%const)

        adv_dataset = torch.utils.data.TensorDataset(torch.stack(adv_samples), torch.tensor(adv_targets))
        self.advset = adv_dataset

    def show_image(self, idx, adv_img ,img = None):
        plt.clf()
        fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

        adv_img, target = adv_img
        img, label = img

        images = list(map(lambda x: x.cpu().detach(), [img, adv_img]))
        npimgs = list(map(lambda x: x.numpy(), images))
        npimgs = list(map(lambda x: np.round(x*255).astype(int), npimgs))

        ax1.set_title("Original: Class %s"%classes[label])
        pl1=ax1.imshow(np.transpose(npimgs[0], (1, 2, 0)))
        ax2.set_title("Perturbed: Class %s"%classes[target])
        pl2=ax2.imshow(np.transpose(npimgs[1], (1, 2, 0)))

        if not os.path.isdir('advimages'):
            os.makedirs('advimages')
        plt.savefig("advimages/sample_%d.png"%idx)

    def test(self, net):
        print("\n=> Testing success of attack")
        loader = torch.utils.data.DataLoader(self.advset)
        net._test(loader)
