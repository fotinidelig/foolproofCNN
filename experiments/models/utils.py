import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import nn

## READ CONFIGURATION PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
verbose = config.getboolean('general','verbose')
train_fname = config.get('general','train_fname')

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

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

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
        self.convLayers = nn.Sequential(bn1, relu1, conv1, bn2, relu2, conv2)

        self.id_conv = nn.Conv2d(i_channels, o_channels, 1, stride=stride)
        self.id_bn = nn.BatchNorm2d(i_channels)
        self.resLayers = nn.Sequential(self.id_bn, nn.ReLU(inplace=True), self.id_conv)
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
        out = self.convLayers(x)
        out += x if self.sameInOut else self.resLayers(x)
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


def train(
    model,
    trainloader,
    valloader=None,
    lr = .01,
    lr_decay = 1, # set to 1 for no effect
    epochs = 40,
    momentum = .9,
    weight_decay = 5e-4,
    params_to_update = None,
    model_name = None,
    **kwargs
):
    '''
        Optimizations used in training:
        *automatic mixed precision (amp) of model weights
        *cuDNN autotuner for convolution computations
        *num_workers>0 and pin_memory=True in DataLoaders
        *param.grad=None instead of optimizer.zero_grad()
    '''
    # turn on training mode, necessary for dropout/batch_norm layers
    model.train()

    if not params_to_update:
        params_to_update = model.parameters()

    optimizer = torch.optim.SGD(params_to_update, lr = lr, momentum = momentum,
                                 nesterov = True, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    batch_size = trainloader.batch_size
    loss_p_epoch = []
    val_loss_p_epoch = []
    total_inputs = 0

    for epoch in range(epochs):
        iters = []
        losses = 0
        start_time = time.time()
        for i, batch in enumerate(trainloader, 0):
            total_inputs += batch[0].size(0)
            data = batch[0].to(device)
            targets = batch[1].to(device)
            pred = model(data.float())
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad = None
            iters.append(i)
            losses += float(loss.item())

        epoch_time = time.time()-start_time
        loss_p_epoch.append(losses/(i+1))
        if verbose:
            print("=> [EPOCH %d] LOSS = %.4f, LR = %.4f, TIME = %.4f mins"%
            (epoch, loss_p_epoch[-1], optimizer.param_groups[0]["lr"], epoch_time/60))
        if valloader:
            val_loss = validate_model(model, valloader, device)
            val_loss_p_epoch.append(val_loss)
            print("=> [EPOCH %d] VAL LOSS = %.4f"%(epoch, val_loss))
        scheduler.step()

    if 'l_curve_name' in kwargs.keys():
        learning_curve(np.arange(epochs), loss_p_epoch, val_loss_p_epoch, "all", lr, batch_size, kwargs['l_curve_name'])
    if not os.path.isdir('models'):
        os.makedirs('models')
    model_name = model.__class__.__name__ if not model_name else model_name
    torch.save(model.state_dict(), f"pretrained/{model_name}.pt")


def predict(model, samples, logits = False):
    model.eval()
    # check if x batch of samples or sample
    if len(samples.size()) < 4:
        samples = torch.reshape(samples, (1, *samples.size()))

    logs = model(samples.float())
    if logits:
        return torch.argmax(logs, 1), logs
    probs = torch.nn.Softmax(dim=1)(logs)
    return torch.argmax(probs, 1), probs

def calc_accuracy(model, testloader):
    with torch.no_grad():
        model.eval()
        accuracy = 0
        total = 0
        device = next(model.parameters()).device
        for i, (samples, targets) in enumerate(testloader, 0):
            total += samples.size(0)
            samples = samples.to(device)
            targets = targets.to(device)
            labels = predict(model, samples)[0]

            accuracy += sum([int(labels[j])==int(targets[j]) for j in range(len(samples))])

    accuracy = float(accuracy/total)
    return accuracy

def validate_model(model, valloader, device):
    model.eval()
    loss = 0
    for i, (samples, labels) in enumerate(valloader):
        out = model(samples.to(device).float())
        loss += nn.CrossEntropyLoss()(out, labels.to(device)).item()
    model.train()
    return loss/(i+1)

def write_train_output(model, model_name, accuracy, **kwargs):
    f = open(train_fname, 'a')
    outputf = dict(file=f)
    print("<==>", **outputf)
    print(datetime.now(), **outputf)
    print(f"Model {model_name}", **outputf)
    for key, val in kwargs.items():
        print(f"{key}: {val}", **outputf)
    print("Test accuracy: %.2f"%(accuracy*100),"%", **outputf)
    print("=><=", **outputf)

def learning_curve(iters, losses, val_losses, epoch, lr, batch_size, filename):
    plt.rcParams["font.family"] = "serif"
    plt.title("Training Curve (batch_size={}, lr={}), epoch={}".format(batch_size, lr, epoch))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(iters, losses, label="train loss")
    if val_losses != [] and val_losses:
        plt.plot(iters, val_losses, label="val loss")
        plt.legend()
    if not os.path.isdir('training_plots'):
        os.makedirs('training_plots')
    plt.savefig(f"training_plots/{filename}.png")

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
