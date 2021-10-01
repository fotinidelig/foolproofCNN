import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

## READ CONFIGURATION PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
verbose = config.getboolean('general','verbose')
attack_fname = config.get('general','attack_fname')

def write_output(total_cnt, adv_cnt, const_list, l2_list, dataset, model, lr, iterations, time):
    success_rate = float(adv_cnt)/total_cnt
    cosnt_mean = sum(const_list)/adv_cnt
    l2_mean = sum(l2_list)/adv_cnt

    f = open(attack_fname, 'a')
    kwargs = dict(file=f)

    print("*********", **kwargs)
    print(datetime.now(), **kwargs)
    print("=> Stats:", **kwargs)
    print(f"Total time: {time}", **kwargs)
    print(f"Dataset: {dataset}", **kwargs)
    print(f"Model: {model}", **kwargs)
    print(f"lr: {lr} iterations: {iterations}", **kwargs)
    print(f"const_all{const_list}", **kwargs)
    print(f"Success Rate: {(success_rate*100):.2f}% || {adv_cnt}/{total_cnt}", **kwargs)
    print(f"Mean const: {cosnt_mean:.3f}", **kwargs)
    print(f"Mean l2: {l2_mean:.2f}", **kwargs)


def img_pipeline(imgs):
    images = list(map(lambda x: x.cpu().detach(), imgs))
    npimgs = list(map(lambda x: x + .5, images)) # un-normalize
    npimgs = list(map(lambda x: x.numpy(), npimgs))
    npimgs = list(map(lambda x: np.round(x*255).astype(int), npimgs))
    return npimgs

def show_image(idx, adv_img, img, classes, fname='', l2=None, with_perturb=False):
    plt.clf()
    ncols = 3 if with_perturb else 2
    fig, ax = plt.subplots(nrows=1,ncols=ncols, dpi=300, figsize=(7, 7))
    if l2:
        fig.suptitle(f'L2 distance: {l2:.3f}', fontsize=16)

    adv_img, target = adv_img
    img, label = img
    npimgs = img_pipeline([img, adv_img])

    kwargs = img.size()[0]==1 and {'cmap': 'gray'} or {}
    ax[0].set_title("Original: Class %s"%classes[label])
    pl1=ax[0].imshow(np.transpose(npimgs[0], (1, 2, 0)), **kwargs)
    ax[2 if with_perturb else 1].set_title("Perturbed: Class %s"%classes[target])
    pl2=ax[2 if with_perturb else 1].imshow(np.transpose(npimgs[1], (1, 2, 0)), **kwargs)

    if with_perturb:
        perturb = npimgs[1] - npimg[0]
        ax[1].set_title("Perturbation")
        pl2=ax[1].imshow(np.transpose(perturb, (1, 2, 0)), **kwargs)

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

def show_in_grid(adv_imgs, classes, fname='', **kwargs):
    '''
        adv_imgs.shape = (N,M) or (C,C) for all classes to all classes
        SO, adv_imgs[i] represents a row of M images in the subplot
        Grid has NxM images
    '''
    def set_axis_style(ax):
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticklabels([])

    plt.clf()
    N = len(adv_imgs)
    M = len(adv_imgs[0])
    fig, ax = plt.subplots(nrows=N,ncols=M, figsize=(M,N), dpi=300)

    xy_style = [set_axis_style(ax[i][j]) for i in range(N) for j in range(M)]
    row_0_style = [ax[0][c].set_xlabel(classes[c]) for c in range(M)]
    if "hide_col_label" not in kwargs.keys():
        col_0_style = [ax[c][0].set_ylabel(classes[c]) for c in range(N)]

    if "suptitle" in kwargs.keys():
        fig.suptitle(kwargs["suptitle"])

    # iterate over natural image rows
    pl = [0]*N
    kwargs = adv_imgs[0][0].size()[0]==1 and {'cmap': 'gray'} or {}
    for n, c_imgs in enumerate(adv_imgs):
        c_imgs = img_pipeline(c_imgs)
        pl[n] = [ax[n][i].imshow(np.transpose(c_imgs[i], (1, 2, 0)), **kwargs) for i in range(M)]

    path = f"advimages/grid"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(f"{path}/{fname}.png")
