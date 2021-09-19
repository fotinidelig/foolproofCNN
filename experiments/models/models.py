import os
from torch import nn
import torch.nn.functional as F
import torch
import torchvision
from .utils import BasicConv2D, BasicLinear, WideResBlock, BasicModel, verbose
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional


## Remember to use GPU for training and move dataset & model to GPU memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.rcParams["font.family"] = "serif"

###############
## CWCIFAR10 ##
###############

class CWCIFAR10(BasicModel):
    """
        The model architecture for CIFAR10 that Nicholas Carlini and David Wagner used in
        'Towards evaluating the robustness of Neural Networks'.
        A convolutional NN, using classes BasicConv2D and BasicLinear.

        All computations (forward, predict, _train, _test) are done in batches,
        reshape is used in the case of one (lonely) sample.

        Achieved accuracy ~77%
    """
    def __init__(self, **kwargs):
        super(CWCIFAR10, self).__init__()
        self.conv11 = BasicConv2D(3, 64, 3, stride=(1,1), **kwargs) # CIFAR sample dimensions are (3,32,32)
        self.conv12 = BasicConv2D(64, 64, 3, stride=(1,1), **kwargs)
        self.conv21 = BasicConv2D(64, 128, 3, stride=(1,1), **kwargs)
        self.conv22 = BasicConv2D(128, 128, 3, stride=(1,1), **kwargs)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = BasicLinear(128*5*5, 256)
        self.fc2 = BasicLinear(256, 256)
        self.fc3 = BasicLinear(256, 10)
        self.dropout = nn.Dropout(p=.5)
        if verbose:
            print("\n", self)


    def forward(self, x):
        # check if x batch of samples or sample
        if len(x.size()) < 4:
            x = torch.reshape(x,(1,*(x.size())))

        out = self.conv11(x)
        out = self.conv12(out)
        out = self.mp(out)
        out = self.conv21(out)
        out = self.conv22(out)
        out = self.mp(out)
        N, C, W, H = (*(out.shape),)
        out = torch.reshape(out, (N, C*W*H))
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        logits = self.fc3(out) # logits layer

        # Don't use softmax layer since it is incorporated in torch.nn.CrossEntropyLoss()
        return logits

#############
## CWMNIST ##
#############

class CWMNIST(CWCIFAR10):
    '''
        Similar implementation to CWCIFAR10,
        as described in the same paper.
        Achieved accuracy ~98%
    '''
    def __init__(self, **kwargs):
        super(CWCIFAR10, self).__init__()
        self.conv11 = BasicConv2D(1, 32, 3, stride=(1,1), **kwargs) # MNIST sample dimensions are (3,28,28)
        self.conv12 = BasicConv2D(32, 32, 3, stride=(1,1), **kwargs)
        self.conv21 = BasicConv2D(32, 64, 3, stride=(1,1), **kwargs)
        self.conv22 = BasicConv2D(64, 64, 3, stride=(1,1), **kwargs)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = BasicLinear(64*4*4, 200)
        self.fc2 = BasicLinear(200, 200)
        self.fc3 = BasicLinear(200, 10)
        self.dropout = nn.Dropout(p=.5)
        if verbose:
            print("\n", self)

#################
## Wide ResNet ##
#################

class WideResNet(BasicModel):
    '''
        Wide Residual Network architecture,
        see paper arxiv.org/abs/1605.07146
        by S Zagoruyko, N Komodakis
    '''
    def __init__(
        self,
        i_channels: Optional[int] = 3,
        depth: Optional[int] = 16,
        width: Optional[int] = 1
    ):
        super(WideResNet, self).__init__()
        assert (depth-4)%6 == 0, 'depth should be 6n+4'
        N = int((depth-4)/6)
        widths = [width*i for i in [16,32,64]]
        widths = [16]+widths

        # group 1
        self.conv1 = nn.Conv2d(i_channels, widths[0], kernel_size=3, padding=1)

        self.group2 = WideResBlock(N, widths[0], widths[1], 3)
        self.group3 = WideResBlock(N, widths[1], widths[2], 3, stride=2)
        self.group4 = WideResBlock(N, widths[2], widths[3], 3, stride=2)
        self.bn1 = nn.BatchNorm2d(widths[3])
        self.fc = BasicLinear(64*width, 10)
        if verbose:
            print("\n", self)

    def forward(self, x):
        # check if x batch of samples or sample
        if len(x.size()) < 4:
            x = torch.reshape(x,(1,*(x.size())))

        out = self.conv1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.avg_pool2d(out, 8)
        N, C, W, H = (*(out.shape),)
        out = out.view(-1, C*W*H)
        logits = self.fc(out)

        # Don't use softmax layer since it is incorporated in torch.nn.CrossEntropyLoss()
        return logits
