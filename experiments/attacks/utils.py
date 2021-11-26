import torch
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from experiments.fourier.analysis import toDFT, visDFT

## READ CONFIGURATION PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
verbose = config.getboolean('general','verbose')
attack_fname = config.get('general','attack_fname')


use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor
                                         if torch.cuda.is_available() and x
                                         else torch.FloatTensor)

def write_attack_log(total_cnt, adv_cnt, dataset, model, time, **kwargs):
    success_rate = float(adv_cnt)/total_cnt

    f = open(attack_fname, 'a')
    outputf = dict(file=f)

    print("*********", **outputf)
    print(datetime.now(), **outputf)
    print("=> Stats:", **outputf)
    print(f"Total time: {time}", **outputf)
    print(f"Dataset: {dataset}", **outputf)
    print(f"Model: {model}", **outputf)
    print(f"Success Rate: {(success_rate*100):.2f}% || {adv_cnt}/{total_cnt}", **outputf)
    for key, val in kwargs.items():
        print(f"{key}: {val}", **outputf)

def ToValidImg(x):
    return (x-x.min())/(x.max()-x.min())

def img_pipeline(imgs):
    images = list(map(lambda x: x.cpu().detach(), imgs))
    npimgs = list(map(lambda x: ToValidImg(x), images)) # un-normalize
    npimgs = list(map(lambda x: x.numpy(), npimgs))
    npimgs = list(map(lambda x: np.round(x*255).astype(int), npimgs))
    return npimgs


def save_images(images, path, fnames):
    if len(images.shape) != 4:
        images = [images]
    if not isinstance(fnames, list):
        fnames = [fnames]
    for image, fname in zip(images,fnames):
        if not os.path.isdir(path):
            os.makedirs(path)
        image = ToValidImg(image)
        save_image(image, path+fname+'.png')

def show_image_function(classes, folder):
    def show_image(idx, adv_img, img, l2=None, with_perturb=False):
        def set_axis_style(ax):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)


        ncols = 3 if with_perturb else 2
        fig, ax = plt.subplots(nrows=1,ncols=ncols, dpi=300)
        plt.rc('font', family='serif')

        set_axis_style(ax[0])
        set_axis_style(ax[2 if with_perturb else 1])

        if l2:
            fig.suptitle(f'L2 distance: {l2:.5f}', fontsize=16)

        adv_img, target = adv_img
        img, label = img
        npimgs = img_pipeline([img, adv_img])

        kwargs = img.size()[0]==1 and {'cmap': 'gray'} or {}
        kwargs['vmax'] = 255
        kwargs['vmin'] = 0

        ax[0].set_title("Class: %s"%classes[label])
        ax[0].imshow(np.transpose(npimgs[0], (1, 2, 0)), **kwargs)
        ax[2 if with_perturb else 1].set_title("Class: %s"%classes[target])
        ax[2 if with_perturb else 1].imshow(np.transpose(npimgs[1], (1, 2, 0)), **kwargs)

        if with_perturb:
            set_axis_style(ax[1])
            perturb = npimgs[1] - npimgs[0]
            perturb = ToValidImg(perturb)
            ax[1].set_title("Perturbation")
            ax[1].imshow(np.transpose(perturb, (1, 2, 0)), **kwargs)

        if not os.path.isdir(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}sample_{idx}_{classes[target]}.png")
        plt.show()
    return show_image

def plot_l2(l2_list, iterations):
    mean_l2 = [np.mean(l2_list)]*len(l2_list)
    x = np.arange(len(l2_list))
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

    N = len(adv_imgs)
    M = len(adv_imgs[0])
    fig, ax = plt.subplots(nrows=N,ncols=M, figsize=(M,N), dpi=300)

    plt.rcParams["font.family"] = "serif"

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


def frequency_l1_diff(x_ben, x_adv, **kwargs):
    device = x_adv.device
    if len(x_ben.shape) == 3:
        x_ben = torch.view(1,*x_ben.shape()).to(device)
        x_adv = torch.view(1,*x_adv.shape())

    x_ben = x_ben.to(device)
    N, C, H, W = x_ben.shape
    _, f_ben_amps, f_ben_phase = toDFT(x_ben)
    _, f_adv_amps, f_adv_phase = toDFT(x_adv)

    diff_per_img = torch.sum(abs(f_ben_amps-f_adv_amps), dim=(1,2,3))
    percentage_per_frequency = torch.zeros((N, H, W)).to(device)
    for i in range(N):
        percentage_per_frequency[i] = torch.sum(abs((f_ben_amps[i]-f_adv_amps[i])), dim=(0,))
        percentage_per_frequency[i] /= diff_per_img[i]
        percentage_per_frequency[i] *= 100
    diff_per_freq = torch.median(percentage_per_frequency, dim=0)[0]

    fig, ax = plt.subplots(1,1)
    # plot = ax.imshow(ToValidImg(20*np.log10(diff_per_freq.to('cpu'))), cmap='magma')
    plot = ax.imshow(ToValidImg(diff_per_freq.to('cpu')), cmap='magma')
    ax.set_title("Amplitude Distortion %")

    ax.xaxis.set_ticks(range(1, W+1, 5))
    ax.xaxis.set_ticklabels(list(range(-((W-1)//2), W//2, 5)))
    ax.yaxis.set_ticks(range(1, H+1, 5))
    ax.yaxis.set_ticklabels(list(range(-((H-1)//2), H//2, 5)))

    fig.colorbar(plot,ax=ax)
    plt.savefig("frequency_l1_diff.png")
    plt.show()
