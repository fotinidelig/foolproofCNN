#!/usr/bin/env python3

## Imports
import numpy as np
import time
import os

# import experiments
from experiments.models.utils import write_train_output, calc_accuracy
from experiments.defences.trades import train_trades
from experiments.utils import load_data_wrapper, load_model
from experiments.parser import parser

import torch
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader, TensorDataset


def main():
    ############
    ## PARSER ##
    ############

    args = parser(advtrain=True, attack=False)
    norm = 2 if args.norm == '2' else np.inf
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

    trainset, trainloader, testset, testloader, validloader = load_data_wrapper(BATCH_SIZE, args.root, args.dataset,
                                                                                args.augment,
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
        train_trades(model, args._lambda, trainloader, validloader, norm, lr=args.lr, lr_decay=args.lr_decay,
                    eps = args.epsilon, alpha = args.alpha, iters = args.iters, epochs=args.epochs,
                    model_name=model_name, l_curve_name=model_name)
        train_time = time.time() - start_time
        print("\n=> [TOTAL TRAINING] %.4f mins."%(train_time/60))

    accuracy = calc_accuracy(model, testloader)
    out_args = dict(LR=args.lr, Lambda=args._lambda, Runtime=train_time/60)
    write_train_output(model, model_name, accuracy, **out_args)

if __name__ == "__main__":
    main()
