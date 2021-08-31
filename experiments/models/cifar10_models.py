import os
from torch import nn
import torch.nn.functional as F
import torch
import torchvision
from .utils import BasicConv2D, BasicLinear
import numpy as np
import matplotlib.pyplot as plt
import time

## Remember to use GPU for training and move dataset & model to GPU memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.rcParams["font.family"] = "serif"

class CWCIFAR10(nn.Module):
    """
        The model architecture for CIFAR10 that Nicholas Carlini and David Wagner used in
        'Towards evaluating the robustness of Neural Networks'.
        A convolutional NN, using classes BasicConv2D and BasicLinear.
    """
    def __init__(self):
        super(CWCIFAR10, self).__init__()
        self.conv11 = BasicConv2D(3,64,3,stride=(1,1)) # CIFAR sample dimensions are (3,32,32)
        self.conv12 = BasicConv2D(64,64,3,stride=(1,1))
        self.conv21 = BasicConv2D(64,128,3,stride=(1,1))
        self.conv22 = BasicConv2D(128,128,3,stride=(1,1))
        self.mp = nn.MaxPool2d(2)
        self.fc1 = BasicLinear(128*5*5,256)
        self.fc2 = BasicLinear(256,256)
        self.fc3 = BasicLinear(256,10)

        self.batch_size = 128
        print("\n", self)


    def forward(self, x, only_logits = False):
        if len(x.shape) == 3:
            x = torch.reshape(x,(1,*(x.shape))) # convolution input must be 4-dimensional
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.mp(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.mp(x)
        N, C, W, H = (*(x.shape),)
        x = torch.reshape(x, (N, C*W*H))
        # x = F.dropout(x, p = .5)
        x = self.fc1(x)
        x = F.dropout(x, p = .5)
        x = self.fc2(x)
        logits = self.fc3(x) # logits layer

        # Don't use softmax layer since it is incorportaed in torch.nn.CrossEntropyLoss()

        return logits


    def _train(self, trainloader):

        lr = .01
        lr_decay = 1 #set to 1 for no effect
        epochs = 50
        momentum = .9
        batch_size = 128

        # turn on training mode, necessary for dropout layers
        self.train()

        def learning_curve(iters, losses, epoch, lr):
            plt.clf()
            plt.title("Training Curve (batch_size={}, lr={}), epoch={}".format(batch_size, lr, epoch))
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.plot(iters, losses)
            # plt.show()
            plt.savefig(f"training_plots/learning_{epoch}.png")

        optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum = momentum, nesterov = True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in np.arange(epochs):
            iters = []
            losses = []
            acc_loss = 0
            start_time = time.time()
            for i, batch in enumerate(trainloader, 0):
                optimizer.zero_grad()
                data = batch[0].to(device)
                targets = batch[1].to(device)
                pred = self.forward(data)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()
                iters.append(i)
                losses.append(float(loss.item()))
                acc_loss+=float(loss)
            epoch_time = time.time()-start_time
            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            print("=> Epoch %d, accumulated loss = %.4f"%(epoch, acc_loss/self.batch_size))
            print("=> [EPOCH TRAINING] %.4f mins."%(epoch_time/60))
            if epoch%5 == 0:
                learning_curve(iters, losses, epoch, cur_lr)

        if not os.path.isdir('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), "models/CWCIFAR10.pt")

    def _test(self, testloader):

        # turn on evaluation mode, aka don't use dropout
        self.eval()

        accuracy = 0
        for i, (samples, targets) in enumerate(testloader, 0):
            losses = []
            samples = samples.to(device)
            targets = targets.to(device)
            pred = self.forward(samples)
            _, labels = torch.max(pred, 1)
            accuracy += sum([int(labels[j])==int(targets[j]) for j in range(len(labels))])

        total = self.batch_size * i
        accuracy = float(accuracy)/total
        print("*******************")
        print("Test accuracy: %.2f"%(accuracy*100),"%")
