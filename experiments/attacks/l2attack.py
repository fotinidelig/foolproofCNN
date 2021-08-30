import torch
import numpy as np
from .utils import BaseAttack

class L2Attack(BaseAttack):
    def __init__(
        self,
        const,
        conf,
        iterations
    ):
        super(L2Attack, self).__init__(const, conf, iterations)

    def loss(self, w, x, logits, target):
        f_i = max([logits[i] for i in range(len(logits)) if i != target])
        f_t =  logits[target]
        obj = max([f_i-f_t, -1*self.conf])
        l = ((1/2*(torch.tanh(w)-1)-x)**2).sum() + self.const*obj
        return l

    def attack(self, samples, targets):
        if not isinstance(samples, list):
            samples = [samples]
        if not isinstance(targets, list):
            targets = [targets]

        w = torch.zeros((3, 32, 32), dtype=float, device=device, requires_grad=True)
        w = w.to(device)
        params = [{'params': w}]
        optimizer = torch.optim.Adam(params, lr=.01)
        adv_samples = []

        for (sample, label), target in zip(samples, targets):
            optimizer.zero_grad() # always do zero_grad() before optimization
            for i in range(self.iterations):
                sample = sample.to(device)
                adv_sample = sample + w
                logits = net.forward(adv_sample.float(), only_logits=True)[0] # net weights and input must be same dtype, aka float32
                loss = self.loss(w,adv_sample,logits,target)
                loss.backward()
                optimizer.step()
            delta = .5*(torch.tanh(w)-1)
            adv_sample = sample+delta
            self.show_image(sample, adv_sample)
            adv_samples.append(adv_sample.float())

        adv_dataset = torch.utils.data.TensorDataset(torch.stack(adv_samples), torch.tensor(targets))
        self.advset = adv_dataset

    def show_image(self, adv_img ,img = None):
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

    def test(net):
        print("=> Testing success of attack")
        loader = torch.utils.data.DataLoader(self.advset)
        net.test(loader)
