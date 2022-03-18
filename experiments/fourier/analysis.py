'''
    Implementation of RGB Image
    frequency analysis tools.

    Using torch.fft module.
'''

from typing import Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from .box_filtering import xBP
from .gauss_filtering import xBP_smooth
from torch.fft import fft2, ifft2, fftshift, ifftshift

plt.rcParams["font.family"] = "serif"

class FourierFilter(object):
    def __init__(self, filterFun, threshold = Union[int, tuple]):
        self.filterFun = filterFun
        self.threshold = threshold

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: (Low, High or Band pass)Filtered tensor in frequency domain
        """
        return filterImage(tensor, self.filterFun, self.threshold)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def HWC_to_CHW(image, inverse = False):
    '''
    Transpose single image from Hight-Width-Channels 
    to Channels-Hight-Width
    '''
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


def filterImage(images: torch.tensor, filter, threshold = Union[int, tuple]):
    '''
        Takes images as input and applies filter
        (one of xLP, xHP, xBP) to them.
        Images must be of shape (C, H, W)
    '''
    if filter in [xBP, xBP_smooth] and not isinstance(threshold, tuple):
        raise ValueError("If `filter` argument is xBP or xBP_smooth, threshold must be tuple of size 2")
    orig_shape = images.shape
    _, amps, phases = toDFT(images)
    threshold = threshold if isinstance(threshold, tuple) else (threshold,)
    amps = filter(amps, *threshold)
    if (amps.shape != orig_shape):
        phases = HWC_to_CHW(phases)

    images = fromDFT(amps, phases)
    images = (images-images.min())/(images.max()-images.min())
    return images

def vizDFT(
    amps: torch.tensor,
    fname = None,
    title = None
):
    '''
        Plots 'amps' (i.e. image spectrum) in log-scale
        Saved as .svg if 'fname!=None'.
    '''
    def normalize(amplitudes):
        amplitudes = 20*np.log10(amplitudes)
        amplitudes = (amplitudes-amplitudes.min())/(amplitudes.max()-amplitudes.min())
        return amplitudes

    amps = normalize(amps)

    fig, axis = plt.subplots(1, 1, dpi=300)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(amps)
    axis.set_title(title if title != None else "Amplitude(dB)")
    if fname:
        plt.savefig(f"{fname}.svg", bbox_inches='tight')
    plt.show()
