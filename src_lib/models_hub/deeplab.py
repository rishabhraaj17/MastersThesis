from typing import Tuple, Optional, Callable

import torch
from mmseg.models import EncoderDecoder, ResNetV1c, DepthwiseSeparableASPPHead, FCNHead
from mmseg.ops import resize
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large

from src_lib.models_hub.base import Base


class DeepLabV3(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(DeepLabV3, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        if self.config.deep_lab_v3.backbone == 'r50':
            self.net = deeplabv3_resnet50(pretrained=self.config.deep_lab_v3.pretrained,
                                          num_classes=self.config.deep_lab_v3.num_classes,
                                          aux_loss=self.config.deep_lab_v3.aux_loss)
        elif self.config.deep_lab_v3.backbone == 'r101':
            self.net = deeplabv3_resnet101(pretrained=self.config.deep_lab_v3.pretrained,
                                           num_classes=self.config.deep_lab_v3.num_classes,
                                           aux_loss=self.config.deep_lab_v3.aux_loss)
        elif self.config.deep_lab_v3.backbone == 'mnv3_large':
            self.net = deeplabv3_mobilenet_v3_large(pretrained=self.config.deep_lab_v3.pretrained,
                                                    num_classes=self.config.deep_lab_v3.num_classes,
                                                    aux_loss=self.config.deep_lab_v3.aux_loss)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.config.deep_lab_v3.aux_loss:
            out = self.net(x)
            return out['out'], out['aux']
        return self.net(x)['out'],

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])


class DeepLabV3Plus(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(DeepLabV3Plus, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = True  # replace it

        norm_cfg = dict(type='BN', requires_grad=True)
        self.backbone = ResNetV1c(
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=norm_cfg,
                norm_eval=False,
                style='pytorch',
                contract_dilation=True
            )
        self.head = DepthwiseSeparableASPPHead(
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=256,
                c1_channels=48,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        self.aux_head = FCNHead(
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )

        self.head_corrector = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        )
        self.aux_head_corrector = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        )
        self.use_correctors = True
        self.with_aux_head = True  # replace it

        self.init_weights(pretrained=None)  # fix me

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()
        if self.with_aux_head:
            if isinstance(self.aux_head, nn.ModuleList):
                for a_head in self.aux_head:
                    a_head.init_weights()
            else:
                self.aux_head.init_weights()

    def forward(self, x):
        feats = self.backbone(x)
        out1 = self.head(feats)
        out2 = self.aux_head(feats)

        out1 = resize(
            input=out1,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        out2 = resize(
            input=out2,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.use_correctors:
            out1 = self.head_corrector(out1)
            out2 = self.aux_head_corrector(out2)
        return out1, out2

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])


if __name__ == '__main__':
    m = DeepLabV3Plus({}, None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
