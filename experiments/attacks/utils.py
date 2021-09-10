import torch
import torchvision
from torch import nn
from typing import Optional, Callable

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

    def test_attack():
        raise NotImplementedError
