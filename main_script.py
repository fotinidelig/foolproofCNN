#!/usr/bin/env python3

## Imports
import argparse
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from typing import Optional, Callable
import time

# import experiments
from experiments.models.utils import write_output
from experiments.models.models import CWCIFAR10, WideResNet, CWMNIST
from experiments.attacks.l2attack import attack_all
from experiments.utils import load_data

import torch
import torchvision
from torchvision.datasets import CIFAR10, MNIST
from torch import nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset


def run_attack(net, device, targeted, sampleloader, n_samples, batch, n_classes, **kwargs):
    input_imgs = []
    input_labs = []
    dataiter = iter(sampleloader)

    net = net.to(device)
    i = 0
    while i < n_samples:
        data = dataiter.next()
        img = torch.reshape(data[0],data[0].size()[1:])
        label = int(data[1][0])

        if net.predict(img.to(device))[0][0] == label:
            i+=1
            input_imgs.append(img)
            input_labs.append(label)

    iterations = 1 if not targeted else n_classes

    for i in range(iterations):
        print(f"\n=> Running attack with {n_samples} samples.")
        # torch.cuda.empty_cache() # empty cache before attack
        target = i if targeted else -1 # target class

        inputset = TensorDataset(torch.stack(input_imgs),
                                torch.tensor([target for i in range(len(input_imgs))]))
        inputloader = DataLoader(inputset, batch_size=batch,
                                shuffle=False, num_workers=2)

        dataname = sampleloader.dataset.__class__.__name__
        classes = sampleloader.dataset.classes

        attack_all(net, inputloader, targeted, classes, dataname,
                    **kwargs)

    # with torch.no_grad():
    #     advloader = DataLoader(attack.advset, batch_size=10,
    #                             shuffle=False, num_workers=2)
    #     net._test(advloader)
    return None


def main():
    ############
    ## PARSER ##
    ############

    parser = argparse.ArgumentParser(description='Run model or/and attack on CIFAR10 and MNIST.')
    parser.add_argument('--pre-trained', dest='pretrained', action='store_const', const=True,
                         default=False, help='use the pre-trained model stored in ./models/')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'],
                         help='define dataset to train on or attack with')
    parser.add_argument('--augment', action='store_const', const=True,
                          default=False, help='apply augmentation on dataset')
    # Model
    parser.add_argument('--model', default='cwcifar10', choices=['cwcifar10', 'cwmnist', 'wideresnet'],
                         help='define the model architecture you want to use')
    parser.add_argument('--epochs', default=35, type=int,
                         help='training epochs')
    parser.add_argument('--lr', default=0.01, type=float,
                         help='initial learning rate')
    parser.add_argument('--lr-decay',dest='lr_decay', default=0.95, type=float,
                         help='learning rate (exponential) decay')
    # WideResNet
    parser.add_argument('--depth', default=40, type=int,
                        help='total number of conv layers in a WideResNet')
    parser.add_argument('--width', default=2, type=int,
                        help='width of a WideResNet')
    # Attack
    parser.add_argument('--attack', dest='run_attack',  action='store_const', const=True,
                         default=False, help='run attack on defined model')
    parser.add_argument('--cpu', action='store_const', const=True,
                         default=False, help='run attack on cpu, not cuda')
    parser.add_argument('--n_samples', default=20, type=int,
                        help='number of samples to attack')
    parser.add_argument('--a_batch', default=2, type=int,
                       help='batch size for attack')
    parser.add_argument('--targeted', action='store_const', const=True,
                         default=False, help='run targeted attack on all classes')
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
        trainset, trainloader, testset, testloader = load_data(CIFAR10, batch_size=BATCH_SIZE, augment=args.augment)
    if args.dataset == 'mnist':
        print("=> Loading MNIST dataset")
        trainset, trainloader, testset, testloader = load_data(MNIST, batch_size=BATCH_SIZE, augment=args.augment)

    ###########
    ## Train ##
    ###########

    ## Remember to use GPU for training and move dataset & model to GPU memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("=> Using device: %s."%device)
    else:
        print("Using CPU instead.")

    if args.model == "cwcifar10":
        net = CWCIFAR10()
    if args.model == "cwmnist":
        net = CWMNIST()
    if args.model == "wideresnet":
        net = WideResNet(i_channels=3, depth=args.depth, width=args.width)

    net = net.to(device)
    if args.pretrained:
        print("\n=> Using pretrained model.")
        net.load_state_dict(torch.load(f"pretrained/{net.__class__.__name__}.pt", map_location=torch.device('cpu')))
    else:
        print("\n=> Training...")
        start_time = time.time()
        net._train(trainloader, epochs=args.epochs, lr=args.lr, lr_decay=args.lr_decay, filename="cifar10")
        train_time = time.time() - start_time
        print("\n=> [TOTAL TRAINING] %.4f mins."%(train_time/60))

    with torch.no_grad():
        accuracy = net._test(testloader)
        write_output(net, accuracy, args.lr, args.lr_decay)

    ############
    ## Attack ##
    ############
    if args.cpu:
        device = 'cpu'
    if args.run_attack:
        n_classes=len(trainset.classes)
        # n_classes=3
        sampleloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                 shuffle=True, num_workers=NUM_WORKERS)
        attack = run_attack(net, device, args.targeted, sampleloader, n_samples=args.n_samples,
                            batch=args.a_batch, n_classes=n_classes)
                            # lr=0.005, max_iterations=2000)

if __name__ == "__main__":
    main()
