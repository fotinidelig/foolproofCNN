import torch
from torchvision.utils import save_image
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from experiments.fourier.analysis import toDFT, vizDFT
from experiments.models.utils import predict

## READ CONFIGURATION PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
verbose = config.getboolean('general','verbose')
attack_fname = config.get('general','attack_fname')


use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor
                                         if torch.cuda.is_available() and x
                                         else torch.FloatTensor)

def succeeded(model, adv_x, label, target=None):
    '''
        model: target model
        adv_x: single adversarial image
        label: true image label
        target: target image label if not None

        Returns:
        true: if adv_x is adversarial
        false: otherwise
    '''
    assert len(adv_x.shape) == 3, "adv_x must be single 3-dimensional image"
    pred, _ = predict(model, adv_x)
    success = (target == pred[0]) if target else (label != pred[0])
    return success

def write_attack_log(total_cnt, adv_cnt, dataset, model, **kwargs):
    success_rate = float(adv_cnt)/total_cnt

    f = open(attack_fname, 'a')
    outputf = dict(file=f)

    print("*********", **outputf)
    print(datetime.now(), **outputf)
    print("=> Stats:", **outputf)
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

        # if l2:
        #     fig.suptitle(f'L2 distance: {l2:.5f}', fontsize=16)

        adv_img, target = adv_img
        img, label = img
        npimgs = img_pipeline([img, adv_img])
        kwargs = img.size()[0]==1 and {'cmap': 'gray'} or {}

        ax[0].set_title("%s"%classes[label])
        ax[0].imshow(np.transpose(npimgs[0], (1, 2, 0)), **kwargs)
        ax[2 if with_perturb else 1].set_title("%s"%classes[target])
        ax[2 if with_perturb else 1].imshow(np.transpose(npimgs[1], (1, 2, 0)), **kwargs)

        if with_perturb:
            set_axis_style(ax[1])
            perturb = npimgs[1] - npimgs[0]
            perturb = ToValidImg(perturb)
            ax[1].set_title("Perturbation")
            ax[1].imshow(np.transpose(perturb, (1, 2, 0)), **kwargs)

        if not os.path.isdir(folder):
            os.makedirs(folder)
        # plt.savefig(f"{folder}sample_{idx}_{classes[target]}.png", bbox_inches='tight')
        plt.savefig(f"{folder}sample_{idx}_{classes[target]}.svg", bbox_inches='tight')
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
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])

    N = len(adv_imgs)
    M = len(adv_imgs[0])
    fig, ax = plt.subplots(nrows=N,ncols=M, figsize=(M,N), dpi=300)

    plt.rcParams["font.family"] = "serif"

    xy_style = [set_axis_style(ax[i][j]) for i in range(N) for j in range(M)]
    for c in range(M):
        ax[0][c].set_xlabel(classes[c])

    if "hide_col_label" not in kwargs.keys():
        col_0_style = [ax[c][0].set_ylabel(classes[c]) for c in range(N)]

    if "suptitle" in kwargs.keys():
        fig.suptitle(kwargs["suptitle"])
    # iterate over natural image rows

    pl = [0]*N
    kwargs = adv_imgs[0][0].size()[0]==1 and {'cmap': 'gray'} or {}
    for n, c_imgs in enumerate(adv_imgs):
        c_imgs = img_pipeline(c_imgs)
        if M==N:
            pl[n] = [ax[i][n].imshow(np.transpose(c_imgs[i], (1, 2, 0)), **kwargs) for i in range(M)]
        else:
            pl[n] = [ax[n][i].imshow(np.transpose(c_imgs[i], (1, 2, 0)), **kwargs) for i in range(M)]

    path = f"advimages/grid"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(f"{path}/{fname}.png", bbox_inches='tight')
    plt.savefig(f"{path}/{fname}.svg",bbox_inches='tight')


def modified_frequencies(x_ben, x_adv, **kwargs):
    '''
        Expects `x_ben`, `x_adv` in shape (<N,> C, H, W)
    '''
    device = x_adv.device
    if len(x_ben.shape) == 3:
        x_ben = torch.view(1,*x_ben.shape()).to(device)
        x_adv = torch.view(1,*x_adv.shape())

    x_ben = x_ben.to(device)
    N, C, H, W = x_ben.shape
    _, f_ben_amps, f_ben_phase = toDFT(x_ben)
    _, f_adv_amps, f_adv_phase = toDFT(x_adv)

    diff_per_img = torch.sum(abs(f_ben_amps-f_adv_amps), dim=(1,2,3))
    ratio_freq_total = torch.zeros((N, H, W)).to(device)
    for i in range(N):
        ratio_freq_total[i] = torch.sum(abs(f_ben_amps[i]-f_adv_amps[i]), dim=(0,))
        ratio_freq_total[i] /= diff_per_img[i]
    diff_per_freq = torch.median(ratio_freq_total, dim=0)[0]

    fig, ax = plt.subplots(1,1, dpi=300)
    # plot = ax.imshow(ToValidImg(20*np.log10(diff_per_freq.to('cpu'))), cmap='magma')


    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.colorbar(plot,ax=ax)
    # plt.savefig("frequency_l1_diff.png", bbox_inches='tight')
    plt.savefig("frequency_l1_diff.svg", bbox_inches='tight')
    plt.show()


def imshow_all_subimg(input, adv, model, classes):
    def set_axis_style(ax):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    fig, ax = plt.subplots(dpi=300)
    set_axis_style(ax)

    if len(input.shape) == 3:
        input = torch.view(1,*input.shape()).to('cpu')
        adv = torch.view(1,*adv.shape()).to('cpu')

    input = input.to('cpu')
    adv = adv.to('cpu')
    N, C, H, W = input.shape
    _, f_ben_amps, f_ben_phase = toDFT(input)
    _, f_adv_amps, f_adv_phase = toDFT(adv)

    pert = ToValidImg(adv-input)


    for i in range(len(input)):
        class_id, _ = predict(model, torch.stack([input[i],adv[i]]))
        ax.imshow(ToValidImg(input[i]).permute(1,2,0))
        plt.savefig(f"img_{i}_{class_id[0]}.svg", bbox_inches='tight')
        vizDFT(f_ben_amps[i].permute(1,2,0),f"img_{i}_freq","")
        fig, ax = plt.subplots(dpi=300)
        set_axis_style(ax)
        ax.imshow(ToValidImg(adv[i]).permute(1,2,0))
        plt.savefig(f"img_{i}_adv_{class_id[1]}.svg", bbox_inches='tight')
        vizDFT(f_adv_amps[i].permute(1,2,0),f"img_{i}_adv_freq","")
        fig, ax = plt.subplots(dpi=300)
        set_axis_style(ax)
        ax.imshow(pert[i].permute(1,2,0))
        plt.savefig(f"img_{i}_pert.svg", bbox_inches='tight')
        plt.show()


def equal_samples(n_classes, dataloader, model, device):
    # creates list with n_samples-images from n_classes-classes
    counters = dict()
    n_samples = 50
    imgs = []
    labs = []
    for batch in dataloader:
        for im, lab in zip(batch[0],batch[1]):
            if predict(model, im.to(device))[0][0] != lab:
                continue
            if str(lab) not in counters.keys():
                counters[str(lab)] = 0
            if counters[str(lab)] < n_samples:
                imgs.append(im)
                labs.append(lab)
                counters[str(lab)] += 1
            if sum([counters[k] == n_samples for k in counters.keys()]) == n_classes:
                break
    assert len(list(counters.keys())) == n_classes, "Not all classes are represented"
    assert sum([counters[k] == n_samples for k in counters.keys()]) == n_classes, "Not all classes with n_samples"
    return imgs, labs
