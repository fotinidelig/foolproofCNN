# import experiments
import torch
import torchvision
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms


def load_cifar10(batch_size = 128, num_workers = 1, root = './data'):
    trainset = CIFAR10(root=root, train=True,
                                    download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = CIFAR10(root=root, train=False,
                                           download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)

    print("\n Dataset: %s \n Trainset: %d samples\n Testset: %d samples\n BATCH_SIZE: %d \n Classes: %d \n"%
          ("Cifar10",trainset.__len__(),testset.__len__(), batch_size, len(trainset.classes)))

    return trainset, trainloader, testset, testloader

def load_mnist(batch_size = 128, num_workers = 1, root = './data'):
    trainset = MNIST(root=root, train=True,
                                    download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = MNIST(root=root, train=False,
                                           download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)

    print("\n Dataset: %s \n Trainset: %d samples\n Testset: %d samples\n BATCH_SIZE: %d \n Classes: %d \n"%
          ("MNIST",trainset.__len__(),testset.__len__(), batch_size, len(trainset.classes)))

    return trainset, trainloader, testset, testloader
