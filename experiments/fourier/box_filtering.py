import numpy as np
import torch

## Frequency filtering with box (square) kernels
## All filters expect images of shape (C, H, W)


def HWC_to_CHW(image, inverse = False):
    if not inverse and torch.argmin(torch.tensor(image.size())) == 2:
        image = image.transpose(1,2).transpose(1,0)
    if inverse and torch.argmin(torch.tensor(image.size())) == 0:
        image = image.transpose(1,0).transpose(1,2)
    return image


def xLP(amps, threshold):
    amps = HWC_to_CHW(amps)
    H = amps.shape[1]
    W = amps.shape[2]

    center = (int(H/2), int(W/2))

    assert threshold <= H/2, f"Filter value too large: max {int(H/2)}, threshold {threshold}"
    assert threshold <= W/2, f"Filter value too large: max {int(W/2)}, threshold {threshold}"

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

    assert threshold <= H/2, f"Filter value too large: max {int(H/2)}, threshold {threshold}"
    assert threshold <= W/2, f"Filter value too large: max {int(W/2)}, threshold {threshold}"

    if threshold == center[0]:
        return torch.zeros_like(amps)

    filtered = amps.clone()
    # filter out low frequencies in range threshold - 1
    x = (center[0]-threshold+2, center[0]+threshold-1)
    y = (center[1]-threshold+2, center[1]+threshold-1)
    filtered[:,x[0]:x[1], y[0]:y[1]] = 0
    return filtered

def xBP(amps, left, right):
    filteredH = xLP(amps, right)
    filtered = xHP(filteredH, left)
    return filtered
