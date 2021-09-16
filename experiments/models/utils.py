import matplotlib.pyplot as plt
import time
import os
import numpy as np
from typing import Optional
import torch
import torchvision
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BasicConv2D(nn.Module):
    ## Conv + ReLU layers
    def __init__(self, i_channels, o_channels, kernel_size, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(i_channels, o_channels, kernel_size=kernel_size, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class BasicLinear(nn.Module):
    ## Linear + ReLU Layers
    def __init__(self, i_features, o_features, **kwargs):
        super(BasicLinear, self).__init__()
        self.fc = nn.Linear(i_features, o_features, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        return out

class BasicResBlock(nn.Module):
    ## BN -> RELU -> CONV
    def __init__(self, i_channels, o_channels, kernel_size, stride = 1, no_id_map = False):
        super(BasicResBlock, self).__init__()

        self.sameInOut = (i_channels == o_channels)

        bn1 = nn.BatchNorm2d(i_channels)
        relu1 = nn.ReLU(inplace=True)
        conv1 = nn.Conv2d(i_channels, o_channels, kernel_size=kernel_size, padding=1, stride=stride)
        bn2 = nn.BatchNorm2d(o_channels)
        relu2 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(o_channels, o_channels, kernel_size=kernel_size, padding=1, stride=1)
        self.seqRes = nn.Sequential(bn1, relu1, conv1, bn2, relu2, conv2)
        self.id_conv = nn.Conv2d(i_channels, o_channels, 1, stride=stride)

    def forward(self, x):
        out = self.seqRes(x)
        out += self.sameInOut and x or self.id_conv(x)
        return out

class WideResBlock(nn.Module):
    def __init__(self, n_blocks, i_channels, o_channels, kernel_size, stride=1):
        super(WideResBlock, self).__init__()
        self.blocks = []
        self.blocks.append(BasicResBlock(i_channels, o_channels, kernel_size, stride=stride).to(device))

        for i in range(n_blocks-1):
            self.blocks.append(BasicResBlock(o_channels, o_channels, kernel_size, stride=1).to(device))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError


    def predict(self, samples, logits = False):
        # turn on evaluation mode, aka don't use dropout
        # check if x batch of samples or sample
        if len(samples.size()) < 4:
            samples = torch.reshape(samples, (1, *samples.size()))

        self.eval()
        logs = self.forward(samples)
        if logits:
            return torch.argmax(logs, 1), logs
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logs)
        return torch.argmax(probs, 1), probs


    def _train(
        self,
        trainloader,
        lr = .01,
        lr_decay = 1, # set to 1 for no effect
        epochs = 50,
        momentum = .9,
        weight_decay = 5e-4,
        **kwargs
    ):
        # turn on training mode, necessary for dropout/batch_norm layers
        self.train()

        optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum = momentum,
                                     nesterov = True, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader)*epochs)
        criterion = nn.CrossEntropyLoss().to(device)
        batch_size = trainloader.batch_size
        loss_p_epoch = []

        for epoch in range(epochs):
            iters = []
            losses = []
            start_time = time.time()
            for i, batch in enumerate(trainloader, 0):
                data = batch[0].to(device)
                targets = batch[1].to(device)
                pred = self.forward(data)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                iters.append(i)
                losses.append(float(loss.item()))
            loss_p_epoch.append(float(loss.item()))
            epoch_time = time.time()-start_time
            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            print("=> Epoch %d, loss = %.4f"%(epoch, loss))
            print("=> [EPOCH TRAINING] %.4f mins."%(epoch_time/60))
            # if epoch%5 == 0:
            #     learning_curve(iters, losses, epoch, cur_lr)
        if kwargs['filename']:
            learning_curve(np.arange(epochs), loss_p_epoch, "all", lr, batch_size, kwargs['filename'])
        if not os.path.isdir('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), f"pretrained/{self.__class__.__name__}.pt")


    def _test(self, testloader):
        accuracy = 0
        for i, (samples, targets) in enumerate(testloader, 0):
            samples = samples.to(device)
            targets = targets.to(device)
            labels, probs = self.predict(samples)
            accuracy += sum([int(labels[j])==int(targets[j]) for j in range(len(samples))])

        total = testloader.batch_size * (i+1)
        print(accuracy, total)
        accuracy = float(accuracy/total)
        print("**********************")
        print("Test accuracy: %.2f"%(accuracy*100),"%")


def learning_curve(iters, losses, epoch, lr, batch_size, filename):
    plt.clf()
    plt.title("Training Curve (batch_size={}, lr={}), epoch={}".format(batch_size, lr, epoch))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(iters, losses)
    plt.savefig(f"training_plots/learning_curve_{filename}.png")

## Debug-friendly-functions
def print_named_weights_sum(model, p_name):
    for name, param in model.named_parameters():
        if p_name in name:
            print("PARAMETER: %s"%name)
            print(param.sum().cpu().data)

# change fc2 layer to the desired layer name
def debug_activations(model):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.fc2.register_forward_hook(get_activation('fc2'))
