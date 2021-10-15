import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch import nn

## READ CONFIGURATION PARAMETERS
# from config import config_params
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
verbose = config.getboolean('general','verbose')
train_fname = config.get('general','train_fname')
attack_fname = config.get('general','attack_fname')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BasicConv2D(nn.Module):
    ## Conv + ReLU layers
    def __init__(self, i_channels, o_channels, kernel_size, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(i_channels, o_channels, kernel_size=kernel_size, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class BasicLinear(nn.Module):
    ## Linear + ReLU Layers
    def __init__(self, i_features, o_features, **kwargs):
        super(BasicLinear, self).__init__()
        self.fc = nn.Linear(i_features, o_features, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        return out

class BasicResBlock(nn.Module):
    ## BN -> RELU -> CONV (x2)
    def __init__(self, i_channels, o_channels, kernel_size, stride = 1):
        super(BasicResBlock, self).__init__()

        self.sameInOut = (i_channels == o_channels)

        bn1 = nn.BatchNorm2d(i_channels)
        relu1 = nn.ReLU(inplace=True)
        conv1 = nn.Conv2d(i_channels, o_channels, kernel_size=kernel_size, padding=1, stride=stride)
        bn2 = nn.BatchNorm2d(o_channels)
        relu2 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(o_channels, o_channels, kernel_size=kernel_size, padding=1, stride=1)
        self.seqRes = nn.Sequential(bn1, relu1, conv1, bn2, relu2, conv2)
        self.id_conv = nn.Conv2d(i_channels, o_channels, 1, stride=stride)
        self.id_bn = nn.BatchNorm2d(i_channels)
        # Initialization as found in vision::torchvision::models::resnet.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.seqRes(x)
        out += x if self.sameInOut else self.id_conv(F.relu(self.id_bn(x)))
        return out

class WideResBlock(nn.Module):
    def __init__(self, n_blocks, i_channels, o_channels, kernel_size, stride=1):
        super(WideResBlock, self).__init__()
        self.blocks = []
        self.blocks.append(BasicResBlock(i_channels, o_channels, kernel_size, stride=stride).to(device))

        for i in range(n_blocks-1):
            self.blocks.append(BasicResBlock(o_channels, o_channels, kernel_size, stride=1).to(device))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError


    def predict(self, samples, logits = False):
        # check if x batch of samples or sample
        if len(samples.size()) < 4:
            samples = torch.reshape(samples, (1, *samples.size()))

        self.eval()
        logs = self.forward(samples.float())
        if logits:
            return torch.argmax(logs, 1), logs
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logs)
        return torch.argmax(probs, 1), probs


def train(
    model,
    trainloader,
    lr = .01,
    lr_decay = 1, # set to 1 for no effect
    epochs = 40,
    momentum = .9,
    weight_decay = 5e-4,
    params_to_update = None,
    model_name = None,
    **kwargs
):
    # turn on training mode, necessary for dropout/batch_norm layers
    model.train()

    if not params_to_update:
        params_to_update = model.parameters()

    optimizer = torch.optim.SGD(params_to_update, lr = lr, momentum = momentum,
                                 nesterov = True, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader)*epochs) # for WideResNet
    criterion = nn.CrossEntropyLoss().to(device)
    batch_size = trainloader.batch_size
    loss_p_epoch = []
    total_inputs = 0

    for epoch in range(epochs):
        iters = []
        losses = []
        start_time = time.time()
        for i, batch in enumerate(trainloader, 0):
            total_inputs += batch[0].size(0)
            data = batch[0].to(device)
            targets = batch[1].to(device)
            pred = model(data.float())
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iters.append(i)
            losses.append(float(loss.item())*batch_size)

        # if epoch%5 == 0:
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        loss_p_epoch.append(sum(losses)/total_inputs)
        epoch_time = time.time()-start_time
        if verbose:
            print("=> [EPOCH %d] LOSS = %.4f, LR = %.4f, TIME = %.4f mins"%
                    (epoch, loss_p_epoch[-1], cur_lr, epoch_time/60))
        # if epoch%5 == 0:
        #     learning_curve(iters, losses, epoch, cur_lr)
    if kwargs['filename']:
        learning_curve(np.arange(epochs), loss_p_epoch, "all", lr, batch_size, kwargs['filename'])
    if not os.path.isdir('models'):
        os.makedirs('models')
    model_name = model.__class__.__name__ if not model_name else model_name
    torch.save(model.state_dict(), f"pretrained/{model_name}.pt")


def test(model, testloader):
    model.cpu()
    accuracy = 0
    for i, (samples, targets) in enumerate(testloader, 0):
        samples = samples
        targets = targets
        labels, probs = model.predict(samples)
        accuracy += sum([int(labels[j])==int(targets[j]) for j in range(len(samples))])

    total = testloader.batch_size * (i+1)
    accuracy = float(accuracy/total)
    return accuracy

def write_output(model, accuracy, **kwargs):
    f = open(train_fname, 'a')
    outputf = dict(file=f)
    print("<==>", **outputf)
    print(datetime.now(), **outputf)
    print(f"Model {model.__class__.__name__}", **outputf)
    for key, val in kwargs.items():
        print(f"{key}: {val}", **outputf)
    print("Test accuracy: %.2f"%(accuracy*100),"%", **outputf)
    print("=><=", **outputf)

def learning_curve(iters, losses, epoch, lr, batch_size, filename):
    plt.clf()
    plt.rcParams["font.family"] = "serif"
    plt.title("Training Curve (batch_size={}, lr={}), epoch={}".format(batch_size, lr, epoch))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(iters, losses)
    plt.savefig(f"training_plots/learning_curve_{filename}.png")

## Debug-friendly-functions
def print_named_weights_sum(model, p_name = None):
    for name, param in model.named_parameters():
        if  not p_name or p_name in name:
            print("PARAMETER: %s"%name)
            print(param.sum().cpu().data)

# change fc2 layer to the desired layer name
def debug_activations(model, layer):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model[layer].register_forward_hook(get_activation(layer))
