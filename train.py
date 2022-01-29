#!/usr/bin/env python3


import argparse
import numpy as np
import time

from experiments.models.utils import write_train_output, train, calc_accuracy
from experiments.utils import load_wrap, load_model
from experiments.parser import parser

import torch
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader, TensorDataset

## READ CONFIGURATION PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
low_fname = config.get('general','accuracy_low')
high_fname = config.get('general','accuracy_high')
band_fname = config.get('general','accuracy_band')


def main():
    ############
    ## PARSER ##
    ############

    args = parser(train=True, attack=False)
    threshold = (*[int(val) for val in args.threshold.split(',')],) if args.threshold else None
    input_size = (*[int(val) for val in args.input_size.split(',')],)
    output_size = (*[int(val) for val in args.output_size.split(',')],) if args.output_size else None

    ############
    ## GLOBAL ##
    ############

    NUM_WORKERS = 2
    BATCH_SIZE = 128

    ##################
    ## Load Dataset ##
    ##################

    trainset, trainloader, testset, testloader, validloader = load_wrap(BATCH_SIZE, args.root, args.dataset,
                                                            args.model, args.augment, args.filter, threshold,
                                                            input_size=input_size, output_size=output_size,
                                                            validation=True)
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

    classes = len(testset.classes)
    model, model_name = load_model(args.model, classes, args)
    model = model.to(device)
    
    train_time = 0
    if args.pretrained:
        print("\n=> Using pretrained model.")
        model.load_state_dict(torch.load(f"pretrained/{model_name}.pt", map_location=torch.device('cpu')))
    else:
        print("\n=> Training...")
        start_time = time.time()
        train(model, trainloader, validloader, epochs=args.epochs, lr=args.lr, lr_decay=args.lr_decay,
                 model_name=model_name, l_curve_name=model_name)
        train_time = time.time() - start_time
        print("\n=> [TOTAL TRAINING] %.4f mins."%(train_time/60))

    ##############
    ## Evaluate ##
    ##############

    accuracy = calc_accuracy(model, testloader)

    out_args = dict(LR=args.lr, LR_Decay=args.lr_decay, Runtime=train_time/60)
    if args.model == 'wideresnet':
        out_args['depth'] = args.depth
        out_args['width'] = args.width

    ## only when filter is applied
    if args.threshold:
        _, _, _, filtered_testloader = load_wrap(BATCH_SIZE, args.root, args.dataset, args.model,
                                                    args.augment, args.filter, threshold, filter_test=True,
                                                    input_size=input_size, output_size=output_size)

        accuracy_filtered = calc_accuracy(model, filtered_testloader)
        out_args['filter'] = f"{args.filter}, threshold: {threshold}"
        out_args['accuracy_filtered_dataset'] = f'{accuracy_filtered*100}%'
        if args.filter == 'low':
            f = open(low_fname, 'a+')
        elif args.filter == 'high':
            f = open(high_fname, 'a+')
        else:
            f = open(band_fname, 'a+')
        f.write(f"{threshold}, {accuracy}\n")
        f.close()

    # accuracy_train = calc_accuracy(net, trainloader)
    # out_args['accuracy_trainset'] = f'{accuracy_train*100}%'
    write_train_output(model, model_name, accuracy, **out_args)

if __name__ == "__main__":
    main()
