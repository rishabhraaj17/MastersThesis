from typing import Any

import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d

import numpy as np
import skimage.measure


def min_pool2d_numpy(x, kernel_size):
    if not isinstance(kernel_size, tuple):
        if x.ndim == 2:
            kernel_size = (kernel_size, kernel_size)
        else:
            kernel_size = (kernel_size, kernel_size, 3)
    return skimage.measure.block_reduce(x, kernel_size, np.min)


def min_pool2d(x, kernel_size, stride=None, padding=0, dilation=1,
               return_indices=False, ceil_mode=False):
    return -1 * max_pool2d(-x, kernel_size, stride, padding, dilation,
                           return_indices, ceil_mode)


class MinPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                       return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x, iterations: int = 1):
        out = x
        for _ in range(iterations):
            out = -1 * self.max_pool2d(-out)
        return out

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


if __name__ == '__main__':
    xx = torch.randn((1, 32, 32))
    yy = xx.squeeze().numpy()
    m_pool2d = MinPool2D(3)
    # out1 = m_pool2d(xx)
    out1 = torch.from_numpy(min_pool2d_numpy(yy, 3)).unsqueeze(0)
    out2 = min_pool2d(xx, 3)
    print(torch.allclose(out1, out2))
