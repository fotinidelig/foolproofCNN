#!/usr/bin/env python3

## Imports
import argparse
import numpy as np
import time

# import experiments
from experiments.models.utils import write_train_output, train, test
from experiments.models.models import CWCIFAR10, WideResNet, CWMNIST
from experiments.attacks.l2attack import attack_all
from experiments.utils import load_data

import torch
from torchvision.datasets import CIFAR10, MNIST
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
    return None


def calc_accuracy(net, testloader):
    with torch.no_grad():
        net.eval()
        accuracy = 0
        total = 0
        device = next(net.parameters()).device
        for i, (samples, targets) in enumerate(testloader, 0):
            total += samples.size(0)
            samples = samples.to(device)
            targets = targets.to(device)
            out=net(samples)
            labels = torch.argmax(out, dim=1)
            accuracy += sum([int(labels[j])==int(targets[j]) for j in range(len(samples))])

        accuracy = float(accuracy/total)
        return accuracy


def main():
    ############
    ## PARSER ##
    ############

    parser = argparse.ArgumentParser(description='''
                                            With this script you can train any custom model from the ./experiments/model directory
                                            on CIFAR10 or MNIST datasets,
                                            or run an attack from the ./experiments/attacks directory.
                                            All imported modules can be found in the ./experiments/ directory.
                                            ''')

    parser.add_argument('--pre-trained', dest='pretrained', action='store_const', const=True,
                         default=False, help='use the pre-trained model stored in ./pretrained/')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'],
                         help='define dataset to train on or attack with')
    parser.add_argument('--filter', default='high', choices=['high', 'low', 'band'],
                         help='filter dataset images in frequency space. Default "high"')
    parser.add_argument('--threshold', default=str(0), type=str,
                        help='filter threshold. Use "," for more than one values. Default 0 (for no filtering)')
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
    parser.add_argument('--model_name', default=None, type=str,
                         help='name to save or load model from')
    # WideResNet
    parser.add_argument('--depth', default=40, type=int,
                        help='total number of conv layers in a WideResNet')
    parser.add_argument('--width', default=2, type=int,
                        help='width of a WideResNet')
    # Attack
    parser.add_argument('--attack',  action='store_const', const=True,
                         default=False, help='run attack on defined model')
    parser.add_argument('--cpu', action='store_const', const=True,
                         default=False, help='run attack on cpu, not cuda')
    parser.add_argument('--n_samples', default=100, type=int,
                        help='number of samples to attack')
    parser.add_argument('--a_lr', default=0.01, type=float,
                       help="learning rate for attack optimization" )
    parser.add_argument('--a_batch', default=50, type=int,
                       help='batch size for attack')
    parser.add_argument('--targeted', action='store_const', const=True,
                         default=False, help='run targeted attack on all classes')
    args = parser.parse_args()
    threshold = (*[int(val) for val in args.threshold.split(',')],)

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
        print("=> Using device: %s."%device)
    else:
        print("Using CPU instead.")

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

    ## only when filter is applied
    if args.threshold != 0:
        _, _, _, filtered_testloader = load_data(CIFAR10, batch_size=BATCH_SIZE, augment=args.augment,
                                                    filter=args.filter, threshold=threshold, filter_test=True)
        accuracy_filtered = calc_accuracy(net, filtered_testloader)
        out_args['filter'] = f"{args.filter}, threshold: {threshold}"
        out_args['accuracy_filtered_dataset'] = f'{accuracy_filtered*100}%'

    if not args.attack:
        write_train_output(net, model_name, accuracy, **out_args)

    ############
    ## Attack ##
    ############
    if args.cpu:
        device = 'cpu'
    if args.attack:
        assert args.n_samples % args.a_batch == 0, "a_batch must divide n_samples"
        n_classes=len(trainset.classes)
        # n_classes=3
        sampleloader = DataLoader(testset, batch_size=1,
                                                 shuffle=True, num_workers=NUM_WORKERS)
        attack = run_attack(net, device, args.targeted, sampleloader, n_samples=args.n_samples,
                            batch=args.a_batch, n_classes=n_classes,
                            lr=args.a_lr, max_iterations=1000, save_attacks=True)

if __name__ == "__main__":
    main()
