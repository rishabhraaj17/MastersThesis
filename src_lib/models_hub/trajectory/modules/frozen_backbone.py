from typing import Optional, Callable, List, Tuple

import torch
from mmcls.models import ResNet_CIFAR, GlobalAveragePooling
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub import Base


class ResNetBackbone(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None,
                 out_channel: int = 32, resnet_depth: int = 18, froze_bn: bool = True):
        super(ResNetBackbone, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.out_channel = out_channel  # or 64 for global
        self.backbone = ResNet_CIFAR(
            depth=resnet_depth,  # or 34 for global
            num_stages=4,
            out_indices=(3,),
            style='pytorch',
            norm_eval=froze_bn
        )
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, stride=1),
            GlobalAveragePooling()
        )

    @classmethod
    def from_config(cls, config: DictConfig, train_dataset: Dataset = None, val_dataset: Dataset = None,
                    desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                    additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None,
                    out_channel: int = 32, resnet_depth: int = 18, froze_bn: bool = True):
        return ResNetBackbone(config=config, train_dataset=train_dataset, val_dataset=val_dataset,
                              desired_output_shape=desired_output_shape, loss_function=loss_function,
                              additional_loss_functions=additional_loss_functions, collate_fn=collate_fn,
                              out_channel=out_channel, resnet_depth=resnet_depth, froze_bn=froze_bn)

    def forward(self, x):
        out = self.backbone(x)
        out = self.projector(out)
        return out


if __name__ == '__main__':
    m = ResNetBackbone.from_config({}, resnet_depth=34)
    inp = torch.randn((1, 3, 720, 680))
    o = m(inp)
    print()
