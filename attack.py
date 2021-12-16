#!/usr/bin/env python3

import argparse
import numpy as np
import time

from experiments.models.utils import predict, calc_accuracy
from experiments.models.models import CWCIFAR10, WideResNet, CWMNIST, resnet_model
from experiments.attacks.cwattack import cw_attack_all
from experiments.attacks.boundary_attack_wrapper import boundary_attack_all
from experiments.attacks.pgd import pgd_attack_all
from experiments.attacks.utils import modified_frequencies
from experiments.utils import load_wrap
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

    ##################
    ## Load Dataset ##
    ##################

    BATCH_SIZE = 128
    NUM_WORKERS = 2
    threshold = (*[int(val) for val in args.threshold.split(',')],) if args.threshold else None
    trainset, trainloader, testset, testloader = load_wrap(BATCH_SIZE, args.root, args.dataset, args.model,
                                                            False, None, None)

    ###########
    ## Model ##
    ###########

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("=> Using device: %s"%device)
    else:
        print("=> Using device: CPU")

    if args.model == "cwcifar10":
        model = CWCIFAR10()
    if args.model == "cwmnist":
        model = CWMNIST()
    if args.model == "wideresnet":
        model = WideResNet(i_channels=3, depth=args.depth, width=args.width)
    if args.model == "resnet":
        classes = len(trainset.classes)
        model = resnet_model(args.layers, classes, True, True)

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = model.__class__.__name__
    print(f"Model Name: {model_name}")

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
    elif args.attack == 'boundary':
        attack_func = boundary_attack_all
        atck_args = dict(steps=1000)

    sampleloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)

    input_imgs = []
    input_labs = []
    dataiter = iter(sampleloader)
    device = next(model.parameters()).device
    i = 0
    while i < args.samples:
        data = dataiter.next()
        img = torch.reshape(data[0],data[0].size()[1:])
        label = int(data[1][0])

        if predict(model, img.to(device))[0][0] == label:
            i+=1
            input_imgs.append(img)
            input_labs.append(label)

    iterations = 1 if not args.targeted else n_classes
    cnt1, cnt2 = 0,0
    for i in range(iterations):
        print(f"\n=> Running attack with {samples} samples.")
        target = i if targeted else -1 # target class

        inputset = TensorDataset(torch.stack(input_imgs),
                                torch.tensor([target for i in range(len(input_imgs))]))
        inputloader = DataLoader(inputset, batch_size=args.batch, shuffle=False)

        dataset = sampleloader.dataset.__class__.__name__
        classes = sampleloader.dataset.classes
        advimgs = attack_func(model, model_name, inputloader, targeted, dataset, classes,**atck_args)

        modified_frequencies(torch.stack(input_imgs), advimgs.detach())
if __name__ == "__main__":
    main()
