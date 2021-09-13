## Imports
import argparse
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from typing import Optional, Callable
import time

# import experiments
from experiments.models.cifar10_models import CWCIFAR10, WideResNet
from experiments.attacks.l2attack import L2Attack
from experiments.datasets.all import load_cifar10, load_mnist

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='Run model or/and attack on CIFAR10.')
parser.add_argument('--pre-trained', dest='pretrained', action='store_const', const=True,
                     default=False, help='use the pre-trained model stored in ./models/')
parser.add_argument('--dataset', dest='dataset', default='cifar10', choices=['cifar10', 'mnist'],
                     help='define dataset to train or attack')
args = parser.parse_args()

############
## GLOBAL ##
############

NUM_WORKERS = 2
BATCH_SIZE = 128

##################
## Load Dataset ##
##################

if args.dataset == 'cifar10':
    print("=> Loading CIFAR10 dataset")
    trainset, trainloader, testset, testloader = load_cifar10(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
if args.dataset == 'mnist':
    print("=> Loading MNIST dataset")
    trainset, trainloader, testset, testloader = load_mnist(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

###########
## Train ##
###########

## Remember to use GPU for training and move dataset & model to GPU memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.empty_cache()
    print("=> Using device: %s"%device)
else:
    raise RuntimeError("=> CUDA not available, abord.")

# net = CWCIFAR10()
net = WideResNet(i_channels=3, depth=40, width=2)
net = net.to(device)

if args.pretrained:
    print("\n=> Using pretrained model.")
    net.load_state_dict(torch.load("pretrained/WideResNet.pt"))
else:
    print("\n=> Training...")
    start_time = time.time()
    net._train(trainloader, lr_decay=.9, filename="40_2")
    train_time = time.time() - start_time
    print("\n=> [TOTAL TRAINING] %.4f mins."%(train_time/60))

with torch.no_grad():
    net._test(testloader)

############
## Attack ##
############

CONST = 0.01 # initial minimization constance
CONF = 0 # defines the classification confidence

N_SAMPLES = 20
MAX_ITERATIONS = 1000


sampleloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                         shuffle=True, num_workers=NUM_WORKERS)

# Load samples for the attack
sample_imgs = []
sample_labs = []
dataiter = iter(sampleloader)
for i in range(N_SAMPLES):
    data = dataiter.next()
    data[0] = torch.reshape(data[0],(3,32,32))
    data[1] = int(data[1][0])
    sample_imgs.append(data[0])
    sample_labs.append(data[1])
target = 2 # target class
sampleset = torch.utils.data.TensorDataset(torch.stack(sample_imgs),
                                            torch.tensor([target for i in range(len(sample_imgs))]))
sampleloader = torch.utils.data.DataLoader(sampleset, batch_size=10,
                                            shuffle=False, num_workers=NUM_WORKERS, pin_memory = True)

print("\n=> Running attack with %d samples"%N_SAMPLES)
attack = L2Attack(CONST, CONF, MAX_ITERATIONS)

torch.cuda.empty_cache() # empty cache before attack

attack.attack(net, sampleloader, sample_labs)

# with torch.no_grad():
#     attack.test(net)

## Use all classes as targets
# for target,_ in enumerate(classes, 0):
#     attack.attack(samples,[target for i in range(len(samples))])
#     attack.test(net)
