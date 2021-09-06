import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable

import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch import nn

class ExtendedCIFAR10(CIFAR10):
    """
    An extension to support showing image samples
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):

        super(ExtendedCIFAR10, self).__init__(
            root,
            train,
            transform,
            target_transform,
            download
        )

        self.classes = list(self.class_to_idx.keys())

    def show_sample(
        self,
        index: Optional[int] = None,
        count: Optional[int] = None,
    ):
        if (index != None):
            assert index < self.__len__(), "IndexError: Index out of bounds"
            indices = [index]
        else:
            indices = [np.random.choice(self.__len__())]
            if (count != None):
                assert count < self.__len__(), "RuntimeError: Requested too many elements"
                indices = np.random.choice(self.__len__(), count, replace = False)

        images = [self[i][0] for i in indices]
        labels = [self[i][1] for i in indices]

        for img in images:
            img = img/2 + .5 # un-normalize
            npimg = img.numpy()
            npimg = np.round(nping*255).astype(int)
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        print(' '.join('%5s' % self.classes[labels[j]] for j in range(len(images))))
