from typing import Any

import torch.nn as nn


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
