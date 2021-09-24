import os
import torch
import torchvision
from torch import nn
from typing import Optional, Callable
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

## READ CONFIGURATION PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
verbose = config.getboolean('general','verbose')
attack_fname = config.get('general','attack_fname')

def write_output(total_cnt, adv_cnt, const_list, l2_list, dataset):
    success_rate = float(adv_cnt)/total_cnt
    cosnt_mean = np.mean(const_list) if len(const_list) > 0 else -1
    cosnt_l2 = np.mean(l2_list) if len(l2_list) > 0 else -1

    f = open(attack_fname, 'a')
    kwargs = dict(file=f)

    print("*********", **kwargs)
    print(datetime.now(), **kwargs)
    print("=> Stats:", **kwargs)
    print(f"Dataset: {dataset}", **kwargs)
    print(f"Success Rate: {(success_rate*100):.2f}% || {adv_cnt}/{total_cnt}", **kwargs)
    print(f"Mean const: {cosnt_mean:.3f}", **kwargs)
    print(f"Mean l2: {cosnt_l2:.2f}", **kwargs)


def show_image(idx, adv_img, img, classes, fname='', l2=None):
    plt.clf()
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

    if l2:
        fig.suptitle(f'L2 distance: {l2:.3f}', fontsize=16)

    adv_img, target = adv_img
    img, label = img
    images = list(map(lambda x: x.cpu().detach(), [img, adv_img]))
    npimgs = list(map(lambda x: x + .5, images)) # un-normalize
    npimgs = list(map(lambda x: x.numpy(), npimgs))
    npimgs = list(map(lambda x: np.round(x*255).astype(int), npimgs))

    kwargs = img.size()[0]==1 and {'cmap': 'gray'} or {}
    ax1.set_title("Original: Class %s"%classes[label])
    pl1=ax1.imshow(np.transpose(npimgs[0], (1, 2, 0)), **kwargs)
    ax2.set_title("Perturbed: Class %s"%classes[target])
    pl2=ax2.imshow(np.transpose(npimgs[1], (1, 2, 0)), **kwargs)

    path = f"advimages/{fname}/H{datetime.now().hour}/"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(f"{path}sample_{idx}_{classes[target]}.png")


def plot_l2(l2_list, iterations):
    mean_l2 = [np.mean(l2_list)]*len(l2_list)
    x = np.arange(len(l2_list))
    plt.clf()
    plt.title("L2 distance from input")
    plt.xlabel("Sample")
    plt.ylabel("L2 distance")
    plt.plot(x, l2_list, label='l2', marker='o')
    plt.plot(x, mean_l2, label="mean", linestyle="--")
    legend = plt.legend(loc='upper right')
    plt.savefig(f"{datetime.now()}_l2_distance.png")
