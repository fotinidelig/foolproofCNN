#!/usr/bin/env python3


import argparse
import numpy as np
import time

from experiments.models.utils import write_train_output, train, calc_accuracy
from experiments.models.models import CWCIFAR10, WideResNet, CWMNIST
from experiments.utils import load_data, x_max_min

import torch
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader, TensorDataset

## READ CONFIGURATION PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
low_fname = config.get('general','rob_accuracy_low')
high_fname = config.get('general','rob_accuracy_high')


def main():
    ############
    ## PARSER ##
    ############

    parser = argparse.ArgumentParser(description='''
                                            With this script you can train any custom model from the ./experiments/model directory
                                            on CIFAR10 or MNIST datasets.
                                            All imported modules can be found in the ./experiments/ directory.
                                            ''')

    parser.add_argument('--pre-trained', dest='pretrained', action='store_const', const=True,
                         default=False, help='use the pre-trained model stored in ./pretrained/')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'],
                         help='define dataset to train on.')
    parser.add_argument('--augment', action='store_const', const=True,
                         default=False, help='apply augmentation on dataset')
    # Filter
    parser.add_argument('--filter', default='high', choices=['high', 'low', 'band'],
                         help='filter dataset images in frequency space. Default "high"')
    parser.add_argument('--threshold', default=str(0), type=str,
                         help='filter threshold. Use "," in case of two values. Default 0 (no filtering)')
    # Model
    parser.add_argument('--model', default='cwcifar10', choices=['cwcifar10', 'cwmnist', 'wideresnet'],
                         help='define the model architecture you want to use')
    parser.add_argument('--epochs', default=35, type=int,
                         help='training epochs')
    parser.add_argument('--lr', default=0.01, type=float,
                         help='initial learning rate')
    parser.add_argument('--lr-decay',dest='lr_decay', default=0.95, type=float,
                         help='learning rate (exponential) decay')
    parser.add_argument('--model_name', default=None, type=str,
                         help='name to save or load model from')
    # WideResNet
    parser.add_argument('--depth', default=40, type=int,
                         help='total number of conv layers in a WideResNet (def. 40)')
    parser.add_argument('--width', default=2, type=int,
                         help='width of a WideResNet (def. 2)')
    args = parser.parse_args()

    ############
    ## GLOBAL ##
    ############

    NUM_WORKERS = 2
    BATCH_SIZE = 128

    ##################
    ## Load Dataset ##
    ##################

    threshold = (*[int(val) for val in args.threshold.split(',')],)

    if args.dataset == 'cifar10':
        print("=> Loading CIFAR10 dataset")
        trainset, trainloader, testset, testloader = load_data(CIFAR10, batch_size=BATCH_SIZE, augment=args.augment,
                                                                filter=args.filter, threshold=threshold)
    if args.dataset == 'mnist' or args.model =='cwmnist':
        print("=> Loading MNIST dataset")
        trainset, trainloader, testset, testloader = load_data(MNIST, batch_size=BATCH_SIZE, augment=args.augment,
                                                                filter=args.filter, threshold=threshold)

    ###########
    ## Train ##
    ###########

    ## Remember to use GPU for training and move dataset & model to GPU memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("=> Using device: %s"%device)
    else:
        print("=> Using device: CPU")

    if args.model == "cwcifar10":
        net = CWCIFAR10()
    if args.model == "cwmnist":
        net = CWMNIST()
    if args.model == "wideresnet":
        net = WideResNet(i_channels=3, depth=args.depth, width=args.width)

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = net.__class__.__name__
    print(f"Model Name: {model_name}")

    net = net.to(device)
    train_time = 0
    if args.pretrained:
        print("\n=> Using pretrained model.")
        net.load_state_dict(torch.load(f"pretrained/{model_name}.pt", map_location=torch.device('cpu')))
    else:
        print("\n=> Training...")
        start_time = time.time()
        train(net, trainloader, epochs=args.epochs, lr=args.lr, lr_decay=args.lr_decay,
                 model_name=model_name, l_curve_name=model_name)
        train_time = time.time() - start_time
        print("\n=> [TOTAL TRAINING] %.4f mins."%(train_time/60))

    accuracy = calc_accuracy(net, testloader)

    out_args = dict(LR=args.lr, LR_Decay=args.lr_decay, Runtime=train_time/60)
    if args.model == 'wideresnet':
        out_args['depth'] = args.depth
        out_args['width'] = args.width

    ## only when filter is applied
    if args.threshold != '0':
        _, _, _, filtered_testloader = load_data(CIFAR10, batch_size=BATCH_SIZE, augment=args.augment,
                                                    filter=args.filter, threshold=threshold, filter_test=True)
        accuracy_filtered = calc_accuracy(net, filtered_testloader)
        out_args['filter'] = f"{args.filter}, threshold: {threshold}"
        out_args['accuracy_filtered_dataset'] = f'{accuracy_filtered*100}%'
        if args.filter == 'low':
            f = open(low_fname, 'a+')
        elif args.filter == 'high':
            f = open(high_fname, 'a+')
        f.write(f"{args.threshold}, {accuracy}\n")
        f.close()

    # accuracy_train = calc_accuracy(net, trainloader)
    # out_args['accuracy_trainset'] = f'{accuracy_train*100}%'
    write_train_output(net, model_name, accuracy, **out_args)

if __name__ == "__main__":
    main()