import matplotlib.pyplot as plt
import time
import os
import numpy as np
from typing import Optional, Callable
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
    def __init__(self, num_blocks, i_channels, o_channels, kernel_size, **kwargs):
        super(BasicResBlock, self).__init__()
        self.num_blocks = num_blocks

        def block(ni, no):
            layers = [
                nn.BatchNorm2d(ni),
                nn.ReLU(),
                nn.Conv2d(ni, no, kernel_size=kernel_size, **kwargs),
                nn.BatchNorm2d(no),
                nn.ReLU(),
                nn.Conv2d(no, no, kernel_size=kernel_size, **kwargs),
            ]
            return layers

        self.block1 = block(i_channels, o_channels)
        self.block2 = block(o_channels, o_channels)

    def forward(self, x):
        def identity(x, block):
            channels = x.size()[1]
            return nn.ReLU(nn.BatchNorm2d(channels))

        def forward_block(x, block):
            out = x
            for layer in block:
                out = layer(out)
            return out

        out = forward_block(x, self.block1)

        for i in range(self.num_blocks-1):
            out = forward_block(out, self.block2)
        return out + identity(x)


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
        **kwargs
    ):
        # turn on training mode, necessary for dropout/batch_norm layers
        self.train()

        optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum = momentum, nesterov = True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_decay)
        criterion = nn.CrossEntropyLoss()
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
