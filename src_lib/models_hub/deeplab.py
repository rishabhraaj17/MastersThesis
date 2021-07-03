from typing import Tuple, Optional, Callable

import torch
from mmseg.models import ResNetV1c, DepthwiseSeparableASPPHead, FCNHead, ASPPHead, UNet
from mmseg.ops import resize
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large

from src_lib.models_hub.base import Base
from src_lib.models_hub.utils import Up, UpProject


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

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


class DeepLabV3Plus(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(DeepLabV3Plus, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = self.config.deep_lab_v3_plus.align_corners
        self.with_aux_head = self.config.deep_lab_v3_plus.with_aux_head
        self.use_correctors = self.config.deep_lab_v3_plus.use_correctors

        norm_cfg = dict(type=self.config.deep_lab_v3_plus.norm.type,
                        requires_grad=self.config.deep_lab_v3_plus.norm.requires_grad)
        self.backbone = ResNetV1c(
            depth=self.config.deep_lab_v3_plus.resnet_depth,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True
        )
        if self.config.deep_lab_v3_plus.aspp_head:
            self.head = ASPPHead(
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                dropout_ratio=0.1,
                num_classes=self.config.deep_lab_v3_plus.head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        else:
            self.head = DepthwiseSeparableASPPHead(
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=256,
                c1_channels=48,
                dropout_ratio=0.1,
                num_classes=self.config.deep_lab_v3_plus.head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        if self.with_aux_head:
            self.aux_head = FCNHead(
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=self.config.deep_lab_v3_plus.aux_head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )

        up_block = UpProject if self.config.deep_lab_v3_plus.use_fcrn_up_project else Up

        if self.config.deep_lab_v3_plus.use_up_module:
            self.head_corrector = nn.Sequential(
                up_block(in_ch=self.config.deep_lab_v3_plus.head_corrector.in_ch[0],
                         out_ch=self.config.deep_lab_v3_plus.head_corrector.out_ch[0],
                         use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                         bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                         channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                         use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                         skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv),
                up_block(in_ch=self.config.deep_lab_v3_plus.head_corrector.in_ch[1],
                         out_ch=self.config.deep_lab_v3_plus.head_corrector.out_ch[1],
                         use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                         bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                         channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                         as_last_layer=True,
                         use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                         skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv)
            )
            if self.with_aux_head:
                self.aux_head_corrector = nn.Sequential(
                    up_block(in_ch=self.config.deep_lab_v3_plus.aux_head_corrector.in_ch[0],
                             out_ch=self.config.deep_lab_v3_plus.aux_head_corrector.out_ch[0],
                             use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                             bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                             channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                             use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                             skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv),
                    up_block(in_ch=self.config.deep_lab_v3_plus.aux_head_corrector.in_ch[1],
                             out_ch=self.config.deep_lab_v3_plus.aux_head_corrector.out_ch[1],
                             use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                             bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                             channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                             use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                             skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv),
                    up_block(in_ch=self.config.deep_lab_v3_plus.aux_head_corrector.in_ch[2],
                             out_ch=self.config.deep_lab_v3_plus.aux_head_corrector.out_ch[2],
                             use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                             bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                             channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                             as_last_layer=True,
                             use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                             skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv)
                )
        elif self.use_correctors:
            self.head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
            if self.with_aux_head:
                self.aux_head_corrector = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
                )
        else:
            self.head_corrector = nn.Identity()
            if self.with_aux_head:
                self.aux_head_corrector = nn.Identity()

        self.init_weights(pretrained=self.config.deep_lab_v3_plus.pretrained)

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
        if self.with_aux_head:
            out2 = self.aux_head(feats)

        if not self.config.deep_lab_v3_plus.use_up_module:
            out1 = resize(
                input=out1,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            if self.with_aux_head:
                out2 = resize(
                    input=out2,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)

        if self.use_correctors:
            out1 = self.head_corrector(out1)
            if self.with_aux_head:
                out2 = self.aux_head_corrector(out2)
        if self.with_aux_head:
            return out1, out2
        return out1

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


class UNetDeepLabV3Plus(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(UNetDeepLabV3Plus, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = self.config.deep_lab_v3_plus.align_corners
        self.with_aux_head = self.config.deep_lab_v3_plus.with_aux_head
        self.use_correctors = self.config.deep_lab_v3_plus.use_correctors

        norm_cfg = dict(type=self.config.deep_lab_v3_plus.norm.type,
                        requires_grad=self.config.deep_lab_v3_plus.norm.requires_grad)
        self.backbone = UNet(
            in_channels=3,
            base_channels=64,
            num_stages=5,
            strides=(1, 1, 1, 1, 1),
            enc_num_convs=(2, 2, 2, 2, 2),
            dec_num_convs=(2, 2, 2, 2),
            downsamples=(True, True, True, True),
            enc_dilations=(1, 1, 1, 1, 1),
            dec_dilations=(1, 1, 1, 1),
            with_cp=False,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(type='InterpConv'),
            norm_eval=False
        )
        self.head = ASPPHead(
            in_channels=64,
            in_index=4,
            channels=16,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=self.align_corners
        )
        if self.with_aux_head:
            self.aux_head = FCNHead(
                in_channels=128,
                in_index=3,
                channels=64,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )

        up_block = UpProject if self.config.deep_lab_v3_plus.use_fcrn_up_project else Up

        if self.config.deep_lab_v3_plus.use_up_module:
            self.head_corrector = nn.Sequential(
                up_block(in_ch=self.config.deep_lab_v3_plus.head_corrector.in_ch[0],
                         out_ch=self.config.deep_lab_v3_plus.head_corrector.out_ch[0],
                         use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                         bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                         channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                         use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                         skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv),
                up_block(in_ch=self.config.deep_lab_v3_plus.head_corrector.in_ch[1],
                         out_ch=self.config.deep_lab_v3_plus.head_corrector.out_ch[1],
                         use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                         bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                         channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                         as_last_layer=True,
                         use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                         skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv)
            )
            if self.with_aux_head:
                self.aux_head_corrector = nn.Sequential(
                    up_block(in_ch=self.config.deep_lab_v3_plus.aux_head_corrector.in_ch[0],
                             out_ch=self.config.deep_lab_v3_plus.aux_head_corrector.out_ch[0],
                             use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                             bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                             channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                             use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                             skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv),
                    up_block(in_ch=self.config.deep_lab_v3_plus.aux_head_corrector.in_ch[1],
                             out_ch=self.config.deep_lab_v3_plus.aux_head_corrector.out_ch[1],
                             use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                             bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                             channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                             use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                             skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv),
                    up_block(in_ch=self.config.deep_lab_v3_plus.aux_head_corrector.in_ch[2],
                             out_ch=self.config.deep_lab_v3_plus.aux_head_corrector.out_ch[2],
                             use_conv_trans2d=self.config.deep_lab_v3_plus.up.use_convt2d,
                             bilinear=self.config.deep_lab_v3_plus.up.bilinear,
                             channels_div_factor=self.config.deep_lab_v3_plus.up.ch_div_factor,
                             as_last_layer=True,
                             use_double_conv=self.config.deep_lab_v3_plus.up.use_double_conv,
                             skip_double_conv=self.config.deep_lab_v3_plus.up.skip_double_conv)
                )
        elif self.use_correctors:
            self.head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
            if self.with_aux_head:
                self.aux_head_corrector = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
                )
        else:
            self.head_corrector = nn.Identity()
            if self.with_aux_head:
                self.aux_head_corrector = nn.Identity()

        self.init_weights(pretrained=self.config.deep_lab_v3_plus.pretrained)

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
        if self.with_aux_head:
            out2 = self.aux_head(feats)

        if not self.config.deep_lab_v3_plus.use_up_module:
            out1 = resize(
                input=out1,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            if self.with_aux_head:
                out2 = resize(
                    input=out2,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)

        if self.use_correctors:
            out1 = self.head_corrector(out1)
            if self.with_aux_head:
                out2 = self.aux_head_corrector(out2)
        if self.with_aux_head:
            return out1, out2
        return out1

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


if __name__ == '__main__':
    m = DeepLabV3Plus(OmegaConf.load('../../src/position_maps/config/model/model.yaml'), None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
