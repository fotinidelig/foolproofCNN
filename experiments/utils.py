import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, DatasetFolder
from experiments.fourier.gauss_filtering import *
from experiments.fourier.box_filtering import *
from experiments.fourier.analysis import FourierFilter


def normalize(data: torch.tensor):
    '''
        Calculates mean and std of a dataset
        for each channel, and returns
        a normalization transformation.
    '''
    dims = (0,1,2) if len(data.shape) == 3 else 0
    mean = data.float().mean(dims)
    std =  data.float().std(dims)
    return T.Normalize((*mean,), (*std,))

def filtering(filter, threshold):
    if filter == 'low':
        filter = xLPG # one of xLPG, xLP
    elif filter == 'high':
        filter = xHP
    elif filter == 'band' and isinstance(threshold, tuple):
        filter = xBP_smooth # one of xBP, xBP_smooth
    else:
        raise ValueError('''filter must be one of ("low", "high", "band")
                        and threshold must be tuple (a, b) if filter=="band"''')
    return FourierFilter(filter, threshold)

def load_data(
    dataclass=None,
    root='./data',
    batch_size=128,
    num_workers=2,
    augment=False,
    filter=None,
    threshold: Union[int, tuple]=None,
    empirical_norm=False,
    filter_test=False,
    output_size: tuple=None
):
    '''
        Loads datasets from torchvision.datasets or custom
        datapath and converts
        image range from [0,1] to [-0.5,0.5].
        A filter can be applied to the training or
        validation set.
        Also resizing is applied.
    '''

    print("Loading dataset...")
    print("Resizing:", output_size)
    print("Augmentation:", augment)
    print("Filtering:",filter,threshold)

    if empirical_norm: # only for CIFAR10 and (Tiny) ImageNet
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    else:
        mean = (.5,.5,.5) if dataclass != MNIST else (.5)
        std = (1,1,1) if dataclass != MNIST else (1)
        normalize = T.Normalize(mean, std)

    filterT = filtering(filter, threshold) if filter else T.Lambda(lambda t: t)

    downsize = T.Resize(output_size) if output_size else T.Lambda(lambda t: t)

    transform_test = T.Compose([T.ToTensor(), normalize, downsize])
    transform_standard = T.Compose([T.ToTensor(), normalize, downsize, filterT])
    transform_augment = T.Compose([
            T.ToTensor(),
            normalize,
            downsize,
            filterT,
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip()])

    # choose augmented or standard train set
    transform_train = transform_augment if augment else transform_standard

    # apply filter to test set
    if filter_test:
        transform_test = transform_standard

    if dataclass:
        trainset = dataclass(root=root, train=True,
                           download=True, transform=transform_train)
        testset = dataclass(root=root, train=False,
                           download=True, transform=transform_test)
    else:
        trainset = ImageFolder(root+'train', transform=transform_train)
        testset = ImageFolder(root+'test', transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                       shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainset, trainloader, testset, testloader


def load_wrap(batch_size, root, dataset, model, augment, filter, threshold, **kwargs):
    if root:
        print(f"=> Loading dataset from path {root}")
        trainset, trainloader, testset, testloader = load_data(None, root=root, batch_size=batch_size,# output_size=(128,128),
                                                 augment=augment, filter=filter, threshold=threshold, **kwargs)
    else:
        if dataset == 'cifar10':
            print("=> Loading CIFAR10 dataset")
            trainset, trainloader, testset, testloader = load_data(CIFAR10, root='./data', batch_size=batch_size,
                                                 augment=augment, filter=filter, threshold=threshold, **kwargs)
        if dataset == 'mnist' or model =='cwmnist':
            print("=> Loading MNIST dataset")
            trainset, trainloader, testset, testloader = load_data(MNIST, root='./data', batch_size=batch_size,
                                                 augment=augment, filter=filter, threshold=threshold, **kwargs)

    return trainset, trainloader, testset, testloader


def x_max_min(loader):
    '''
        Calculate max & min of dataset after
        mormalization transform is applied.

        Transform should be:
        [T.transform_X, T.Normalize(mean std), ...]
    '''
    mean = loader.dataset.transform.transforms[1].mean
    std = loader.dataset.transform.transforms[1].std

    x_max = np.array([1.,1.,1.])
    x_max = (x_max-mean)/std
    x_max = x_max.max()

    x_min = np.array([0,0,0])
    x_min = (x_min-mean)/std
    x_min = x_min.min()

    print(f"Max: {x_max} Min: {x_min}")
    return x_max, x_min
