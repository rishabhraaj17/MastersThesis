from typing import Tuple, Optional, Callable

import torch
from mmseg.models import ResNetV1c, DAHead, FCNHead
from mmseg.ops import resize
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub.base import Base
from src_lib.models_hub.utils import Up


class DANet(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(DANet, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = self.config.danet.align_corners

        norm_cfg = dict(type=self.config.danet.norm.type,
                        requires_grad=self.config.danet.norm.requires_grad)
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
        self.head = DAHead(
            in_channels=2048,
            in_index=3,
            channels=512,
            pam_channels=64,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=self.align_corners,
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

        if self.config.danet.use_up_module:
            self.head_corrector = nn.ModuleList([
                nn.Sequential(
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       use_double_conv=self.config.danet.up.use_double_conv),
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       use_double_conv=self.config.danet.up.use_double_conv),
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       as_last_layer=True,
                       use_double_conv=self.config.danet.up.use_double_conv)
                ),
                nn.Sequential(
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       use_double_conv=self.config.danet.up.use_double_conv),
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       use_double_conv=self.config.danet.up.use_double_conv),
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       as_last_layer=True,
                       use_double_conv=self.config.danet.up.use_double_conv)
                ),
                nn.Sequential(
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       use_double_conv=self.config.danet.up.use_double_conv),
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       use_double_conv=self.config.danet.up.use_double_conv),
                    Up(in_ch=self.config.danet.up.in_ch,
                       out_ch=self.config.danet.up.out_ch,
                       use_conv_trans2d=self.config.danet.up.use_convt2d,
                       bilinear=self.config.danet.up.bilinear,
                       channels_div_factor=self.config.danet.up.ch_div_factor,
                       as_last_layer=True,
                       use_double_conv=self.config.danet.up.use_double_conv)
                )
            ])
            self.aux_head_corrector = nn.Sequential(
                Up(in_ch=self.config.danet.up.in_ch,
                   out_ch=self.config.danet.up.out_ch,
                   use_conv_trans2d=self.config.danet.up.use_convt2d,
                   bilinear=self.config.danet.up.bilinear,
                   channels_div_factor=self.config.danet.up.ch_div_factor,
                   use_double_conv=self.config.danet.up.use_double_conv),
                Up(in_ch=self.config.danet.up.in_ch,
                   out_ch=self.config.danet.up.out_ch,
                   use_conv_trans2d=self.config.danet.up.use_convt2d,
                   bilinear=self.config.danet.up.bilinear,
                   channels_div_factor=self.config.danet.up.ch_div_factor,
                   use_double_conv=self.config.danet.up.use_double_conv),
                Up(in_ch=self.config.danet.up.in_ch,
                   out_ch=self.config.danet.up.out_ch,
                   use_conv_trans2d=self.config.danet.up.use_convt2d,
                   bilinear=self.config.danet.up.bilinear,
                   channels_div_factor=self.config.danet.up.ch_div_factor,
                   as_last_layer=True,
                   use_double_conv=self.config.danet.up.use_double_conv)
            )
        else:
            self.head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
            self.aux_head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )

        self.use_correctors = self.config.danet.use_correctors
        self.with_aux_head = self.config.danet.with_aux_head

        self.init_weights(pretrained=self.config.danet.pretrained)

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

        if not self.config.danet.use_up_module:
            out1 = [resize(
                input=o1,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for o1 in out1]
            out2 = resize(
                input=out2,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        if self.use_correctors:
            out1 = [hc(o1) for o1, hc in zip(out1, self.head_corrector)]  # pam_cam, pam, cam
            out2 = self.aux_head_corrector(out2)
        return out1, out2

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])


if __name__ == '__main__':
    m = DANet(OmegaConf.load('../../src/position_maps/config/model/model.yaml'), None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
