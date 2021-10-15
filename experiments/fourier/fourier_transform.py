from .analysis import filterImage
from typing import Union

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
