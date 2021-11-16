'''
    Implementation of RGB Image Filtering
    in frequency space.
    Using torch.fft module.
'''

from typing import Union
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2, fftshift, ifftshift
from scipy import signal

plt.rcParams["font.family"] = "serif"

def HWC_to_CHW(image, inverse = False):
    if not inverse and torch.argmin(torch.tensor(image.size())) == 2:
        image = image.transpose(1,2).transpose(1,0)
    if inverse and torch.argmin(torch.tensor(image.size())) == 0:
        image = image.transpose(1,0).transpose(1,2)
    return image

def toDFT(
    images: torch.tensor # shape (<N>, C, H, W) or (<N>, H, W, C)
):
    if len(images.shape) == 4:
        images = torch.stack([HWC_to_CHW(x) for x in images])
    else:
        images = HWC_to_CHW(images)

    transformed = fftshift(fft2(images), dim=list(range(len(images.shape)))[-2:]) # move origin to center
    amps = transformed.abs()
    phase = transformed.angle() # in rads
    return transformed, amps, phase

def fromDFT(
    amps: torch.tensor,
    phase: torch.tensor,
):
    if len(amps.shape) == 4:
        amps = torch.stack([HWC_to_CHW(x) for x in amps])
        phase = torch.stack([HWC_to_CHW(x) for x in phase])
    else:
        amps, phase = list(map(lambda x: HWC_to_CHW(x),[amps, phase]))

    complex = amps*(np.cos(phase)+1j*np.sin(phase))
    inversed = ifft2(ifftshift(complex, dim=list(range(len(amps.shape)))[-2:]))
    return inversed.abs()

## Filtering - High, Low, Band Pass
## All filters expect images of shape (C, H, W)
def xLP(amps, threshold):
    amps = HWC_to_CHW(amps)
    H = amps.shape[1]
    W = amps.shape[2]

    center = (int(H/2), int(W/2))

    assert center[0]-threshold >= 0, f"Filter value too large: center0 {center[0]} threshold {threshold}"
    assert center[1]-threshold >= 0, f"Filter value too large: center1 {center[1]} threshold {threshold}"

    if threshold == center[0]:
        return amps

    filtered = amps.clone()
    x = (center[0]-threshold+1, center[0]+threshold)
    y = (center[1]-threshold+1, center[1]+threshold)
    filtered[:,x[0]:x[1], y[0]:y[1]] = 0
    return amps-filtered

def xHP(amps, threshold):
    amps = HWC_to_CHW(amps)
    H = amps.shape[1]
    W = amps.shape[2]

    center = (int(H/2), int(W/2))

    assert center[0]-threshold >= 0, f"Filter value too large: center0 {center[0]} threshold {threshold}"
    assert center[1]-threshold >= 0, f"Filter value too large: center1 {center[1]} threshold {threshold}"

    if threshold == center[0]:
        return torch.zeros_like(amps)

    filtered = amps.clone()
    # filter out low frequencies in range threshold - 1
    x = (center[0]-threshold+2, center[0]+threshold-1)
    y = (center[1]-threshold+2, center[1]+threshold-1)
    filtered[:,x[0]:x[1], y[0]:y[1]] = 0
    return filtered

def xBP(amps, thresholdL, thresholdH):
    filteredH = xLP(amps, thresholdH)
    filtered = xHP(filteredH, thresholdL)
    return filtered

def filterImage(images: torch.tensor, filter, threshold = Union[int, tuple]):
    '''
        Takes images as input and applies filter
        (one of xLP, xHP, xBP) to them.
        Images must be of shape (C, H, W)
    '''
    if filter == xBP and not isinstance(threshold, tuple):
        raise("RuntimeError: if filter = xBP, threshold must be tuple of size 2")
    orig_shape = images.shape
    _, amps, phases = toDFT(images)
    threshold = threshold if isinstance(threshold, tuple) else (threshold,)
    amps = filter(amps, *threshold)
    if (amps.shape != orig_shape):
        phases = HWC_to_CHW(phases)

    images = fromDFT(amps, phases)
    images = (images-images.min())/(images.max()-images.min())
    return images

def visDFT(
    amps: torch.tensor,
    fname = None
):
    def normalize(amplitudes):
        amplitudes = 20*np.log10(amplitudes)
        amplitudes = (amplitudes-amplitudes.min())/(amplitudes.max()-amplitudes.min())
        return amplitudes

    amps = normalize(amps)

    plt.clf()
    fig, axis = plt.subplots(1, 1, dpi=300)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(amps)
    axis.set_title("Amplitude(dB)")
    plt.show()
