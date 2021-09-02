import torch
import numpy as np
from typing import Optional
from .utils import BaseAttack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def loss(self, w, x, logits, target):
        """
            Implementing the f_6 objective function,
            refer to the corresponding C&W paper
        """
        f_i = max([logits[i] for i in range(len(logits)) if i != target])
        f_t =  logits[target]
        fx = f_i - f_t + self.conf
        obj = max([f_i-f_t, -1*self.conf])
        l = ((1/2*(torch.tanh(w)-1)-x)**2).sum() + self.const*obj
        return l, fx

    def attack(self, net, samples, targets):
        if not isinstance(samples, list):
            samples = [samples]
        if not isinstance(targets, list):
            targets = [targets]

        w = torch.zeros((3, 32, 32), dtype=float, device=device, requires_grad=True)
        params = [{'params': w}]
        optimizer = torch.optim.Adam(params, lr=.01)
        adv_samples = []

        bin_steps = 9
        for idx, ((sample, label), target) in enumerate(zip(samples, targets), 0):
            found_atck = False
            prev_fx = 1 # random value > 0 for initialization
            for iteration in range(bin_steps):
                optimizer.zero_grad() # always do zero_grad() before optimization
                for i in range(self.iterations):
                    sample = sample.to(device)
                    adv_sample = sample + w
                    logits = net.forward(adv_sample.float(), only_logits=True)[0] # net weights and input must be same dtype, aka float32
                    loss, fx = self.loss(w,adv_sample,logits,target)
                    loss.backward()
                    optimizer.step()
                delta = .5*(torch.tanh(w)-1)
                self.const, end_iters = self.bin_search_const(self.const, fx, prev_fx) # update const for next iteration
                prev_fx = fx
                if fx <= 0:
                    found_atck = True
                    best_atck = copy.deepcopy(sample+delta)
                    best_const = self.const
                    best_fx = fx
                if end_iters:
                    print(best_fx <= 0)
                    break
            if found_atck:
                self.show_image(idx, best_atck, sample)
                print("=> Found attack with CONST = %.3f."%self.best_const)
                adv_samples.append(best_atck.float())
            else:
                print("=> Didn't find attack.")

        adv_dataset = torch.utils.data.TensorDataset(torch.stack(adv_samples), torch.tensor(targets))
        self.advset = adv_dataset

    def show_image(self, idx, adv_img ,img = None):
        plt.clf()
        fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

        # first load images to cpu and detach
        adv_img, target = adv_img
        img, label = img
        images = list(map(lambda x: x.cpu().detach(), [img, adv_img]))
        images = list(map(lambda x: x/2+.5, images))
        npimgs = list(map(lambda x: x.numpy(), images))
        ax1.set_title("Original: Class %s"%classes[label])
        pl1=ax1.imshow(np.transpose(npimgs[0], (1, 2, 0)))
        ax2.set_title("Perturbed: Class %s"%classes[target])
        pl2=ax2.imshow(np.transpose(npimgs[1], (1, 2, 0)))
        plt.savefig("advset/sample_%d.png"%idx)

    def test(net):
        print("=> Testing success of attack")
        loader = torch.utils.data.DataLoader(self.advset)
        net.test(loader)
