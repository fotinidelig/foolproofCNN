## Imports
import argparse
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from typing import Optional, Callable
import time

# import experiments
from experiments.models.cifar10_models import CWCIFAR10, WideResNet, CWMNIST
from experiments.attacks.l2attack import L2Attack
from experiments.datasets.all import load_cifar10, load_mnist

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

############
## PARSER ##
############

parser = argparse.ArgumentParser(description='Run model or/and attack on CIFAR10.')
parser.add_argument('--pre-trained', dest='pretrained', action='store_const', const=True,
                     default=False, help='use the pre-trained model stored in ./models/')
parser.add_argument('--dataset', dest='dataset', default='cifar10', choices=['cifar10', 'mnist'],
                     help='define dataset to train on or attack with')
parser.add_argument('--model', dest='model', default='cwcifar10', choices=['cwcifar10', 'cwmnist', 'wideresnet'],
                     help='define the model architecture you want to use')
parser.add_argument('--attack', dest='run_attack',  action='store_const', const=True,
                     default=False, help='run attack on defined model')
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

if args.model == "cwcifar10":
    net = CWCIFAR10()
if args.model == "cwmnist":
    net = CWMNIST()
else:
    net = WideResNet(i_channels=3, depth=40, width=2)
net = net.to(device)

if args.pretrained:
    print("\n=> Using pretrained model.")
    net.load_state_dict(torch.load(f"pretrained/{net.__class__.__name__}.pt"))
else:
    print("\n=> Training...")
    start_time = time.time()
    net._train(trainloader, lr=.1, lr_decay=.9, filename="22_2")
    train_time = time.time() - start_time
    print("\n=> [TOTAL TRAINING] %.4f mins."%(train_time/60))

with torch.no_grad():
    net._test(testloader)

############
## Attack ##
############

def run_attack(sampleloader, const, conf, n_samples, max_iterations, n_classes):
    input_imgs = []
    input_labs = []
    dataiter = iter(sampleloader)
    for i in range(n_samples):
        data = dataiter.next()
        data[0] = torch.reshape(data[0],data[0].size()[1:])
        data[1] = int(data[1][0])
        input_imgs.append(data[0])
        input_labs.append(data[1])
    attack = L2Attack(net, const, conf, max_iterations)
    for i in range(n_classes):
        print("\n=> Running attack with %d samples"%n_samples)
        torch.cuda.empty_cache() # empty cache before attack
        target = i # target class
        inputset = TensorDataset(torch.stack(input_imgs),
                                                    torch.tensor([target for i in range(len(input_imgs))]))
        inputloader = DataLoader(inputset, batch_size=10,
                                                    shuffle=False, num_workers=NUM_WORKERS, pin_memory = True)
        attack.attack(inputloader, input_labs)

    with torch.no_grad():
        advloader = DataLoader(advset, batch_size=10,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory = True)
        net._test(attack.advset)
    return attack

if args.run_attack:
    sampleloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                             shuffle=True, num_workers=NUM_WORKERS)
    attack = run_attack(sampleloader, const=.01, conf=0, n_samples=20, max_iterations=1000, len(trainset.classes))
