import torch
import torchvision
from torch import nn
from typing import Optional, Callable

class BaseAttack():
    def __init__(
        self,
        const: int,
        conf: Optional[int] = 0,
        iterations: Optional[int] = 10000
    ):
        self.const = const
        self.conf = conf
        self.iterations = iterations
        self.advset = []

    def loss(self, w, x, **kwargs):
        raise NotImplementedError

    def attack(self, samples, targets):
        raise NotImplementedError

    def test_attack():
        raise NotImplementedError  
