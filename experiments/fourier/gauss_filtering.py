import numpy as np
import torch
from scipy import signal
from .box_filtering import xHP
from torch.fft import fftshift

## Frequency filtering with gaussian kernels
## All filters expect images of shape (C, H, W)


def HWC_to_CHW(image, inverse = False):
    if not inverse and torch.argmin(torch.tensor(image.size())) == 2:
        image = image.transpose(1,2).transpose(1,0)
    if inverse and torch.argmin(torch.tensor(image.size())) == 0:
        image = image.transpose(1,0).transpose(1,2)
    return image

def gauss_kernel(thres_x, thres_y, H, W):
    kernel = np.zeros((H, W))
    kernel_x = signal.gaussian(W, thres_x)
    kernel_y = signal.gaussian(H, thres_y)
    kernel = np.outer(kernel_y, kernel_x)
    return kernel

def xLPG(amps, threshold):
    '''
    Low-Pass Gaussian filter
    '''
    amps = HWC_to_CHW(amps)
    H = amps.shape[1]
    W = amps.shape[2]

    assert threshold <= H/2, f"Filter value too large: max {int(H/2)}, threshold {threshold}"
    assert threshold <= W/2, f"Filter value too large: max {int(W/2)}, threshold {threshold}"

    kernel = gauss_kernel(threshold, threshold, H, W)
    filtered = amps*kernel
    return filtered

def xHPG(amps, threshold):
    '''
    High-Pass Gaussian filter
    '''
    amps = HWC_to_CHW(amps)
    H = amps.shape[1]
    W = amps.shape[2]

    assert threshold <= H/2, f"Filter value too large: max {int(H/2)}, threshold {threshold}"
    assert threshold <= W/2, f"Filter value too large: max {int(W/2)}, threshold {threshold}"

    kernel = gauss_kernel(W-threshold, H-threshold, H, W)
    kernel = fftshift(torch.tensor(kernel))
    filtered = amps*kernel
    return filtered

def xBPG(amps, left, right):
    '''
    Band-Pass filter
    using high-pass Gaussian filter
    and low-pass Gaussian filter
    '''
    amps = HWC_to_CHW(amps)
    H = amps.shape[1]
    W = amps.shape[2]

    assert left <= right, "when calling xBP(amps, left, right) left <= right should hold"
    assert left <= H/2 and right <= H/2, f"Filter value too large: max {int(H/2)}, left {left}, right {right}"
    assert left <= W/2 and right <= W/2, f"Filter value too large: max {int(W/2)}, left {left}, right {right}"

    small_kernel_x = np.zeros(shape=(left+right,))
    small_kernel_x = signal.gaussian(left+right+W, right-left)
    small_kernel_y = np.zeros(shape=(left+right,))
    small_kernel_y = signal.gaussian(left+right+H, right-left)

    half_x = small_kernel_x[:W]
    kernel_x = half_x.copy()
    kernel_x += half_x[::-1]
    half_y = small_kernel_y[:H]
    kernel_y = half_y.copy()
    kernel_y += half_y[::-1]
    kernel = np.outer(kernel_y, kernel_x)
    filtered = amps*kernel
    return filtered

def xBP_smooth(amps, left, right):
    '''
    Band-Pass filter
    using high-pass box filter
    and low-pass Gaussian filter
    '''
    filteredH = xLPG(amps, right)
    filtered = xHP(filteredH, left)
    return filtered
