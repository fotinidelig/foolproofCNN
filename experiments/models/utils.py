import numpy as np
from typing import Optional, Callable
import torch
import torchvision
from torch import nn


class BasicConv2D(nn.Module):
    ## Conv + ReLU layers
    def __init__(self, i_channels, o_channels, kernel_size, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(i_channels, o_channels, kernel_size=kernel_size, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class BasicLinear(nn.Module):
    ## Linear + ReLU Layers
    def __init__(self, i_features, o_features, **kwargs):
        super(BasicLinear, self).__init__()
        self.fc = nn.Linear(i_features, o_features, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def predict(self, samples, **kwargs):
        self.eval()
        raise NotImplementedError

    def _train(self, trainloader):
        self.train()
        raise NotImplementedError

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
