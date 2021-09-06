import torch
import torchvision
from torch import nn
from typing import Optional, Callable

MAX_CONST = 1e10

class BaseAttack():
    def __init__(
        self,
        const: int,
        conf: int,
        iterations: int,
        max_const: float,
        min_const: float
    ):
        self.const = const
        self.conf = conf
        self.iterations = iterations
        self.advset = []
        self.max_const = max_const
        self.min_const = min_const

    def loss(self, w, input, **kwargs):
        raise NotImplementedError

    def bin_search_const(self, const, fx):
        """
            Binary search for const in range
            [min_const, max_const].

            Return smallest const for which f(x) moves to 0
            or end search if const is found.
        """
        end_iters = False
        if fx > 0:
            if self.max_const == MAX_CONST:
                # no successful attack found yet
                const *= 10
            else:
                self.min_const = const
                const = .5*(self.max_const+self.min_const)
        if fx <= 0:
            self.max_const = const
            const = .5*(self.max_const+self.min_const)
        return const

    def attack(self, samples, targets):
        raise NotImplementedError

    def test_attack():
        raise NotImplementedError
