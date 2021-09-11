import os
from torch import nn
import torch.nn.functional as F
import torch
import torchvision
from .utils import BasicConv2D, BasicLinear, BasicModel
import numpy as np
import matplotlib.pyplot as plt
import time

## Remember to use GPU for training and move dataset & model to GPU memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.rcParams["font.family"] = "serif"

class CWCIFAR10(BasicModel):
    """
        The model architecture for CIFAR10 that Nicholas Carlini and David Wagner used in
        'Towards evaluating the robustness of Neural Networks'.
        A convolutional NN, using classes BasicConv2D and BasicLinear.

        All computations (forward, predict, _train, _test) are done in batches,
        reshape is used in the case of one (lonely) sample.
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
        self.dropout = nn.Dropout(p=.5)
        print("\n", self)


    def forward(self, x):
        # check if x batch of samples or sample
        if len(x.size()) < 4:
            x = torch.reshape(x,(1,*(x.size())))
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.mp(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.mp(x)
        N, C, W, H = (*(x.shape),)
        x = torch.reshape(x, (N, C*W*H))
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        logits = self.fc3(x) # logits layer

        # Don't use softmax layer since it is incorporated in torch.nn.CrossEntropyLoss()

        return logits

    def predict(self, samples, logits = False):
        # turn on evaluation mode, aka don't use dropout
        # check if x batch of samples or sample
        if len(samples.size()) < 4:
            samples = torch.reshape(samples, (1, *samples.size()))

        self.eval()
        logs = self.forward(samples)
        if logits:
            return torch.argmax(logs, 1), logs
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logs)
        return torch.argmax(probs, 1), probs


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
                data = batch[0].to(device)
                targets = batch[1].to(device)
                pred = self.forward(data)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                iters.append(i)
                losses.append(float(loss.item()))
                acc_loss+=float(loss)
            epoch_time = time.time()-start_time
            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            print("=> Epoch %d, accumulated loss = %.4f"%(epoch, acc_loss/batch_size))
            print("=> [EPOCH TRAINING] %.4f mins."%(epoch_time/60))
            if epoch%5 == 0:
                learning_curve(iters, losses, epoch, cur_lr)

        if not os.path.isdir('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), "pretrained/CWCIFAR10.pt")

class WideResNet():
    def __init__(self):
        super(WideResNet, self).__init__()
