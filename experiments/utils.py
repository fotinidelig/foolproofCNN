import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, DatasetFolder
from experiments.fourier.analysis import filterImage, xLP, xHP, xBP
from experiments.fourier.fourier_transform import FourierFilter

def show_sample(
    dataset,
    index: Optional[int] = None,
    count: Optional[int] = None,
):
    if (index != None):
        assert index < dataset.__len__(), "IndexError: Index out of bounds"
        indices = [index]
    else:
        indices = [np.random.choice(dataset.__len__())]
        if (count != None):
            assert count < dataset.__len__(), "RuntimeError: Requested too many elements"
            indices = np.random.choice(dataset.__len__(), count, replace = False)

    images = [dataset[i][0] for i in indices]
    labels = [dataset[i][1] for i in indices]

    for img in images:
        img = img + .5 # un-normalize
        npimg = img.numpy()
        npimg = np.round(nping*255).astype(int)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(len(images))))

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

def load_data(
    dataclass,
    augment,
    batch_size = 128,
    num_workers = 2,
    root = './data',
    filter = Optional[str],
    threshold = Optional[int, tuple],
    filter_test = False
):
    '''
        Loads CIFAR10 or MNIST datasets and converts
        image range from [0,1] to [-0.5,0.5].
        A filter can be applied to the training or
        validation set.
    '''
    if filter == 'low':
        filter = xLP
    elif filter == 'high':
        filter = xHP
    elif filter == 'band' and isinstance(threshold, tuple):
        filter = xBP

    if filter:
        fourier_transform = FourierFilter(filter, threshold)
    else:
        fourier_transform = FourierFilter(xHP, 1) # only for sanity

    # set normalization parameters for each data class
    mean = (.5,.5,.5) if dataclass != MNIST else (.5)
    std = (1,1,1) if dataclass != MNIST else (1)

    # create transforms for train and test set
    transform_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    transform_standard = T.Compose([T.ToTensor(),fourier_transform, T.Normalize(mean, std)])
    transform_augment = T.Compose([
            T.ToTensor(),
            fourier_transform,
            T.Normalize(mean, std),
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            T.RandomHorizontalFlip()])

    # choose augmented or standard train set
    transform_train = transform_augment if augment else transform_standard

    # apply filter to test set
    if filter_test:
        transform_test = transform_standard

    trainset = dataclass(root=root, train=True,
                       download=True, transform=transform_train)
    testset = dataclass(root=root, train=False,
                       download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                       shuffle=False, num_workers=num_workers, pin_memory=True)
    print("\n Dataset: %s \n Trainset: %d samples\n Testset: %d samples\n BATCH_SIZE: %d \n Classes: %d \n"%
          (dataclass.__name__,trainset.__len__(),testset.__len__(), batch_size, len(trainset.classes)))

    return trainset, trainloader, testset, testloader

## TODO: Create new dataset class for tiny-imagenet-200
def dataset_folder_loader(
    path = './data/tiny-imagenet-200/',
    batch_size = 128,
    input_size = (64, 64),
    train = True,
    test = True,
    val = False,
):
    '''
        Loads tiny-imagenet-200 dataset, which should be
        downloaded from https://image-net.org/download-images.php
        and extracted in path.
    '''
    # pre-calculated mean and std values
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])

    transform = T.Compose([
                T.RandomResizedCrop(input_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # normalize])
                T.Normalize([0.5, 0.5, 0.5], [1, 1, 1])])

    train_images = ImageFolder(path+'train', transform=transform)
    # print(train_images.classes, train_images.class_to_idx, train_images.targets[500:520]) ## DEBUG:
    train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
    try:
        test_images = ImageFolder(path+'test', transform=transform)
        test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=True)
    except FileNotFoundError as err:
        print(f"Excepted error:\n{err}")
        return train_images, train_loader, None, None

    if val:
        val_images = ImageFolder(path+'val', transform=transform)
        val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=True)
        return train_images, train_loader, test_images, test_loader, val_images, val_loader

    return train_images, train_loader, test_images, test_loader
