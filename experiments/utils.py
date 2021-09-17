import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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
        for each channels, and returns
        a normalization transformation.
    '''
    dims = (0,1,2) if len(data.shape) == 3 else 0
    mean = data.float().mean(dims)
    std =  data.float().std(dims)
    return transforms.Normalize((*mean,), (*std,))

def load_data(dataclass, batch_size = 128, num_workers = 1, root = './data'):
    '''
        Loads CIFAR10 or MNIST datasets and converts
        image range from [0,1] to [-0.5,0.5]
    '''
    mean = (.5,.5,.5) if dataclass != MNIST else (.5)
    std = (1,1,1) if dataclass != MNIST else (1)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
    trainset = dataclass(root=root, train=True,
                       download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = dataclass(root=root, train=False,
                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                       shuffle=False, num_workers=num_workers, pin_memory=True)

    print("\n Dataset: %s \n Trainset: %d samples\n Testset: %d samples\n BATCH_SIZE: %d \n Classes: %d \n"%
          ("Cifar10",trainset.__len__(),testset.__len__(), batch_size, len(trainset.classes)))

    return trainset, trainloader, testset, testloader
