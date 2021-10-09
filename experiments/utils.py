import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

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

def load_data(dataclass, augment, batch_size = 128, num_workers = 2, root = './data'):
    '''
        Loads CIFAR10 or MNIST datasets and converts
        image range from [0,1] to [-0.5,0.5]
    '''
    mean = (.5,.5,.5) if dataclass != MNIST else (.5)
    std = (1,1,1) if dataclass != MNIST else (1)
    transform_standard = T.Compose([T.ToTensor(),T.Normalize(mean, std)])
    transform_augment = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            T.RandomHorizontalFlip()])

    transform = transform_augment if augment else transform_standard

    trainset = dataclass(root=root, train=True,
                       download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = dataclass(root=root, train=False,
                       download=True, transform=transform_standard)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                       shuffle=False, num_workers=num_workers, pin_memory=True)

    print("\n Dataset: %s \n Trainset: %d samples\n Testset: %d samples\n BATCH_SIZE: %d \n Classes: %d \n"%
          (dataclass.__name__,trainset.__len__(),testset.__len__(), batch_size, len(trainset.classes)))

    return trainset, trainloader, testset, testloader

## TODO: Create new dataset class for tiny-imagenet-200
def tiny_imagenet_loader(path = './data/tiny-imagenet-200/', batch_size = 128):
    '''
        Loads tiny-imagenet-200 dataset, which should be
        downloaded from https://image-net.org/download-images.php
        and extracted in path.
    '''
    # pre-calculated mean and std values
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    transform = T.Compose([T.ToTensor(),T.Normalize(mean, std)])
    val_images = ImageFolder(path+'val', transform=transform)
    val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=True)
    train_images = ImageFolder(path+'train', transform=transform)
    train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
    test_images = ImageFolder(path+'test', transform=transform)
    test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=True)
    print(dataloader.dataset.__class__.__name__) ## DEBUG:

    return train_images, train_loader, test_images, test_loader
