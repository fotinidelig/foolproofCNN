# -*- coding: utf-8 -*-
"""Models-Carlini-et-al.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hxd3sOU8PYOLGffB1a-3kjl18Aq0Bzkq
"""

## Imports
import argparse

import numpy as np
import os
import random
import matplotlib.pyplot as plt
from typing import Optional, Callable
import time

# import experiments
from experiments.models.cifar10_models import CWCIFAR10
from experiments.attacks.l2attack import L2Attack
from experiments.utils import ExtendedCIFAR10

import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch import nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='Run model or/and attack on CIFAR10.')
parser.add_argument('--pre-trained', dest='pretrained', action='store_const',
                     const=True, default=False,
                     help='use the pre-trained model stored in ./models/')
args = parser.parse_args()

############
## GLOBAL ##
############

NUM_WORKERS = 2
BATCH_SIZE = 128

##################
## Load Dataset ##
##################

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])

trainset = ExtendedCIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

testset = ExtendedCIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("\n Dataset: %s \n Trainset: %d samples\n Testset: %d samples\n BATCH_SIZE: %d \n Classes: %d"%
      ("Cifar10",trainset.__len__(),testset.__len__(), BATCH_SIZE, len(classes)))

############
## Train ##
############

## Remember to use GPU for training and move dataset & model to GPU memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=> Using device: %s"%device)

net = CWCIFAR10()
net = net.to(device)

if args.pretrained:
    print("\n=> Using pre trained model.")
    net.load_state_dict(torch.load("models/CWCIFAR10.pt"))
else:
    print("\n=> Training...")
    start_time = time.time()
    net._train(trainloader)
    train_time = time.time() - start_time
    print("\n=> [TOTAL TRAINING] %.4f mins."%(train_time/60))

with torch.no_grad():
    net._test(testloader)

############
## Attack ##
############

CONST = 1 # initial minimization constance
CONF = 0 # defines the classification confidence

N_SAMPLES = 100
MAX_ITERATIONS = 10000

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

sampleloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                         shuffle=True, num_workers=NUM_WORKERS)

# Load samples for the attack
samples = []
dataiter = iter(sampleloader)
for i in range(N_SAMPLES):
    data = dataiter.next()
    data[0] = torch.reshape(data[0],(3,32,32))
    data[1] = int(data[1][0])
    samples.append(data)

# sampleset = torch.utils.data.Dataset(samples, batch_size=20, shuffle=True, num_workers=NUM_WORKERS)
print("\n=> Running attack with %d samples"%N_SAMPLES)
attack = L2Attack(CONST, CONF, MAX_ITERATIONS)

target = 1 # target class
attack.attack(samples,[target for i in range(len(samples))])

with torch.no_grad():
    attack.test(net)

## Use all classes as targets
# for target,_ in enumerate(classes, 0):
#     attack.attack(samples,[target for i in range(len(samples))])
#     attack.test(net)
