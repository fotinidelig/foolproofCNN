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

def write_output(total_samples, adv_samples, const_list, l2_list, dataset):
    success_rate = float(len(adv_samples))/total_samples
    cosnt_mean = np.mean(const_list) if len(const_list) > 0 else -1
    cosnt_l2 = np.mean(l2_list) if len(l2_list) > 0 else -1

    f = open(attack_fname, 'a')
    kwargs = dict(file=f)

    print("*********", **kwargs)
    print(datetime.now(), **kwargs)
    print("=> Stats:", **kwargs)
    print(f"Dataset: {dataset}", **kwargs)
    print(f"Success Rate: {(success_rate*100):.2f}% || {len(adv_samples)}/{total_samples}", **kwargs)
    print(f"Mean const: {cosnt_mean:.3f}", **kwargs)
    print(f"Mean l2: {cosnt_l2:.2f}", **kwargs)

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


class BaseAttack():
    def __init__(
        self,
        init_const: int,
        conf: int,
        iterations: int,
        max_const: float,
        min_const: float
    ):
        self.init_const = init_const
        self.conf = conf
        self.iterations = iterations
        self.advset = []
        self.max_const = max_const
        self.min_const = min_const

    def loss(self, w, input, **kwargs):
        raise NotImplementedError

    def bin_search_const(self, const, max_const, min_const, fx):
        """
            Binary search for const in range
            [min_const, max_const].

            Return smallest const for which f(x) moves to 0
            or end search if const is found.
        """
        end_iters = False
        if fx >= 0:
            if const*10 > self.max_const:
                end_iters = True
                return const, end_iters
            if max_const == self.max_const:
                # no successful attack found yet
                const *= 10
            else:
                min_const = const
                const = .5*(max_const+min_const)
        if fx < 0:
            max_const = const
            const = .5*(max_const+min_const)
        return const, max_const, min_const, end_iters

    def attack(self, samples, targets):
        raise NotImplementedError
