#!/usr/bin/env python3

import argparse
import numpy as np
import time

from experiments.models.utils import predict, calc_accuracy
from experiments.attacks.cwattack import cw_attack_all
from experiments.attacks.boundary_attack_wrapper import boundary_attack_all
from experiments.attacks.pgd import pgd_attack_all
from experiments.attacks.utils import modified_frequencies, equal_samples
from experiments.utils import load_data_wrapper, load_model
from experiments.parser import parser

## DEBUG
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

import torch
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, TensorDataset

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

    args = parser(train=False, attack=True)
    input_size = (*[int(val) for val in args.input_size.split(',')],)
    output_size = (*[int(val) for val in args.output_size.split(',')],) if args.output_size else None

    ############
    ## GLOBAL ##
    ############

    BATCH_SIZE = 128
    NUM_WORKERS = 2

    ##################
    ## Load Dataset ##
    ##################


    trainset, trainloader, testset, testloader = load_data_wrapper(BATCH_SIZE, args.root, args.dataset,
                                                                   False, input_size=input_size,
                                                                   output_size=output_size)

    ###########
    ## Model ##
    ###########

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("=> Using device: %s"%device)
    else:
        print("=> Using device: CPU")

    classes = len(testset.classes)
    model, model_name = load_model(args.model, classes, args)
    model = model.to(device)
    model.eval() # only used for inference
    print("\n=> Using pretrained model.")
    model.load_state_dict(torch.load(f"pretrained/{model_name}.pt", map_location=torch.device('cpu')))
    acuracy = calc_accuracy(model, testloader)
    print(f"Model accuracy = {acuracy*100}% (on natural data)")

    ############
    ## Attack ##
    ############

    assert args.samples % args.batch == 0, "`batch` must divide `samples`"
    n_classes=len(trainset.classes)

    if args.attack =='cw':
        attack_func = cw_attack_all
        atck_args = dict(lr=args.lr, max_iterations=args.iters, conf=0.01)
    elif args.attack == 'pgd':
        attack_func = pgd_attack_all
        norm = 2 if args.norm == '2' else np.inf
        atck_args = dict(eps=args.epsilon, alpha=args.alpha,
                        n_iters=args.iters, x_min=-.5, x_max=.5, norm=norm)
    else:
        attack_func = boundary_attack_all
        atck_args = dict(steps=1000)

    sampleloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)

    input_imgs = []
    input_labs = []
    dataiter = iter(sampleloader)
    device = next(model.parameters()).device
    i = 0

    if args.samples == 0:
        input_imgs, input_labs = equal_samples(n_classes, trainloader,
                                               model, device)
    else:
        while i < args.samples:
            data = dataiter.next()
            img = torch.reshape(data[0],data[0].size()[1:])
            label = int(data[1][0])

            if predict(model, img.to(device))[0][0] == label:
                i+=1
                input_imgs.append(img)
                input_labs.append(label)

    iterations = 1 if not args.targeted else n_classes
    dataset = sampleloader.dataset.__class__.__name__
    classes = sampleloader.dataset.classes

    for i in range(iterations):
        print(f"\n=> Running attack with {args.samples} samples.")
        target = i if args.targeted else -1 # target class
        inputset = TensorDataset(torch.stack(input_imgs),
                                torch.tensor([target for i in range(len(input_imgs))]))
        inputloader = DataLoader(inputset, batch_size=args.batch, shuffle=False)

        advimgs = attack_func(model, model_name, inputloader, args.targeted, dataset, classes,**atck_args)

        modified_frequencies(torch.stack(input_imgs), advimgs.detach())


if __name__ == "__main__":
    main()
