from typing import Tuple, Optional, Callable

import torch
from mmseg.models import HRNet, FCNHead
from mmseg.ops import resize
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub.base import Base


class HRNetwork(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(HRNetwork, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        norm_cfg = dict(type='BN', requires_grad=True)
        self.backbone = HRNet(
            norm_cfg=norm_cfg,
            norm_eval=False,
            extra=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(4,),
                    num_channels=(64,)),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='BASIC',
                    num_blocks=(4, 4),
                    num_channels=(18, 36)),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='BASIC',
                    num_blocks=(4, 4, 4),
                    num_channels=(18, 36, 72)),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    block='BASIC',
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(18, 36, 72, 144)))
        )
        self.head = FCNHead(
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            channels=sum([18, 36, 72, 144]),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=False,
        )

        self.head_corrector = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        )

        self.use_correctors = True
        self.align_corners = False  # replace it

        self.init_weights(pretrained=None)  # fix me

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)

        out = resize(
            input=out,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        out = self.head_corrector(out)
        return out

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])


if __name__ == '__main__':
    m = HRNetwork({}, None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
