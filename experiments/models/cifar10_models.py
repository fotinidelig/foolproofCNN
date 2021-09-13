import os
from torch import nn
import torch.nn.functional as F
import torch
import torchvision
from .utils import BasicConv2D, BasicLinear, BasicResBlock, BasicModel
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional

## Remember to use GPU for training and move dataset & model to GPU memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.rcParams["font.family"] = "serif"

class CWCIFAR10(BasicModel):
    """
        The model architecture for CIFAR10 that Nicholas Carlini and David Wagner used in
        'Towards evaluating the robustness of Neural Networks'.
        A convolutional NN, using classes BasicConv2D and BasicLinear.

        All computations (forward, predict, _train, _test) are done in batches,
        reshape is used in the case of one (lonely) sample.

        Achieved accuracy ~77%
    """
    def __init__(self):
        super(CWCIFAR10, self).__init__()
        self.conv11 = BasicConv2D(3,64,3,stride=(1,1)) # CIFAR sample dimensions are (3,32,32)
        self.conv12 = BasicConv2D(64,64,3,stride=(1,1))
        self.conv21 = BasicConv2D(64,128,3,stride=(1,1))
        self.conv22 = BasicConv2D(128,128,3,stride=(1,1))
        self.mp = nn.MaxPool2d(2)
        self.fc1 = BasicLinear(128*5*5,256)
        self.fc2 = BasicLinear(256,256)
        self.fc3 = BasicLinear(256,10)
        self.dropout = nn.Dropout(p=.5)
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

        self.conv1 = BasicResBlock(N, i_channels, o_channels=16, kernel_size=3, padding=1)
        self.conv2 = BasicResBlock(N, 16, 16*width, 3, padding=1)
        self.conv3 = BasicResBlock(N, 16*width, 32*width, 3, padding=1)
        self.conv4 = BasicResBlock(N, 32*width, 64*width, 3, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.ap = nn.AvgPool2d(8)
        self.fc = BasicLinear(64*width, 10)

    def forward(self, x):
        # check if x batch of samples or sample
        if len(x.size()) < 4:
            x = torch.reshape(x,(1,*(x.size())))

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.mp(out)
        out = self.conv3(out)
        out = self.mp(out)
        out = self.conv4(out)
        out = self.ap(out)
        N, C, W, H = (*(out.shape),)
        out = torch.reshape(out, (N, C*W*H))
        logits = self.fc(out)

        # Don't use softmax layer since it is incorporated in torch.nn.CrossEntropyLoss()
        return logits
