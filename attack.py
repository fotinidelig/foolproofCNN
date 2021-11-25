#!/usr/bin/env python3

import argparse
import numpy as np
import time

from experiments.models.utils import train, calc_accuracy
from experiments.models.models import CWCIFAR10, WideResNet, CWMNIST
from experiments.attacks.cwattack import cw_attack_all
from experiments.attacks.boundary_attack_wrapper import boundary_attack_all
from experiments.attacks.pgd import pgd_attack_all,pgd_attack_all_inf
from experiments.attacks.utils import frequency_l1_diff
from experiments.utils import load_data, x_max_min

## DEBUG
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

import torch
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader, TensorDataset

## READ CONFIGURATION PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
low_fname = config.get('general','rob_accuracy_low')
high_fname = config.get('general','rob_accuracy_high')

def run_attack(model, attack_func, targeted, sampleloader, samples, batch, n_classes, **kwargs):
    input_imgs = []
    input_labs = []
    dataiter = iter(sampleloader)
    device = next(model.parameters()).device

    model.eval()

    i = 0
    while i < samples:
        data = dataiter.next()
        img = torch.reshape(data[0],data[0].size()[1:])
        label = int(data[1][0])

        if model.predict(img.to(device))[0][0] == label:
            i+=1
            input_imgs.append(img)
            input_labs.append(label)

    iterations = 1 if not targeted else n_classes
    cnt1, cnt2 = 0,0
    for i in range(iterations):
        print(f"\n=> Running attack with {samples} samples.")
        target = i if targeted else -1 # target class

        inputset = TensorDataset(torch.stack(input_imgs),
                                torch.tensor([target for i in range(len(input_imgs))]))
        inputloader = DataLoader(inputset, batch_size=batch, shuffle=False)

        dataset = sampleloader.dataset.__class__.__name__
        classes = sampleloader.dataset.classes
        advimgs = attack_func(model, inputloader, dataset, targeted, classes,**kwargs)

        frequency_l1_diff(torch.stack(input_imgs), advimgs.detach())
    return None


def main():
    ############
    ## PARSER ##
    ############

    parser = argparse.ArgumentParser(description='''
                                            Run an attack from the ./experiments/attacks directory.
                                            All imported modules can be found in the ./experiments/ directory.
                                            ''')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'],
                         help='define dataset to train on or attack with')
    # Filter
    parser.add_argument('--filter', default='high', choices=['high', 'low', 'band'],
                         help='filter dataset images in frequency space. Default "high"')
    parser.add_argument('--threshold', default=str(0), type=str,
                         help='filter threshold. Use "," in case of two values. Default 0 (no filtering)')
    # Model
    parser.add_argument('--model', default='cwcifar10', choices=['cwcifar10', 'cwmnist', 'wideresnet'],
                         help='model architecture. Default cwcifar10')
    parser.add_argument('--model_name', default=None, type=str,
                         help='name of saved model (if different from model). Default None')
    # Attack
    parser.add_argument('--cpu', action='store_const', const=True,
                         default=False, help='run attack on cpu, not cuda')
    parser.add_argument('--attack', default='cw', type=str, choices=['cw', 'pgd', 'boundary'],
                          help='attack method. Default C&W attack (cw).')
    parser.add_argument('--samples', default=100, type=int,
                         help='number of samples to attack')
    parser.add_argument('--lr', default=0.01, type=float,
                         help="learning rate for attack optimization" )
    parser.add_argument('--batch', default=50, type=int,
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

    threshold = (*[int(val) for val in args.threshold.split(',')],)

    if args.dataset == 'cifar10':
        print("=> Loading CIFAR10 dataset")
        trainset, trainloader, testset, testloader = load_data(CIFAR10, batch_size=BATCH_SIZE, augment=False,
                                                            filter=args.filter, threshold=threshold)
    if args.dataset == 'mnist' or args.model =='cwmnist':
        print("=> Loading MNIST dataset")
        trainset, trainloader, testset, testloader = load_data(MNIST, batch_size=BATCH_SIZE, augment=False,
                                                            filter=args.filter, threshold=threshold)

    ###########
    ## Train ##
    ###########

    ## Remember to use GPU for training and move dataset & model to GPU memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.cpu and device == 'cuda':
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

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = model.__class__.__name__
    print(f"Model Name: {model_name}")

    model = model.to(device)
    train_time = 0
    print("\n=> Using pretrained model.")
    model.load_state_dict(torch.load(f"pretrained/{model_name}.pt", map_location=torch.device('cpu')))

    acuracy = calc_accuracy(model, testloader)
    print(f"Model accuracy = {acuracy*100}% (on natural data)")

    ############
    ## Attack ##
    ############

    assert args.samples % args.batch == 0, "batch must divide samples"
    n_classes=len(trainset.classes)

    if args.attack =='cw':
        attack_func = cw_attack_all # to run cw attack
        atck_args = dict(lr=args.lr, max_iterations=800, save_attacks=False, conf=0.01)
    elif args.attack == 'pgd':
        attack_func = pgd_attack_all # to run pgd attack
        atck_args = dict(eps=0.03, alpha=0.007, n_iters=60,x_min=-.5,x_max=.5,norm=np.inf)
    elif args.attack == 'boundary':
        attack_func = boundary_attack_all
        atck_args = dict(steps=1000)

    sampleloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)
    attack = run_attack(model, attack_func, args.targeted, sampleloader, samples=args.samples,
                            batch=args.batch, n_classes=n_classes, **atck_args)

if __name__ == "__main__":
    main()
