from typing import Tuple, Optional, Callable

import torch
from mmseg.models import HRNet, FCNHead, OCRHead, ResNetV1c
from mmseg.ops import resize
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub.base import Base
from src_lib.models_hub.utils import Up

CONFIG = {
    'hr18': {
        'stage1': {
            'num_modules': 1,
            'num_branches': 1,
            'num_blocks': (4,),
            'num_channels': (64,),
        },
        'stage2': {
            'num_modules': 1,
            'num_branches': 2,
            'num_blocks': (4, 4),
            'num_channels': (18, 36),
        },
        'stage3': {
            'num_modules': 4,
            'num_branches': 3,
            'num_blocks': (4, 4, 4),
            'num_channels': (18, 36, 72),
        },
        'stage4': {
            'num_modules': 3,
            'num_branches': 4,
            'num_blocks': (4, 4, 4, 4),
            'num_channels': (18, 36, 72, 144),
        },
        'fcn_head': {
            'in_channels': [18, 36, 72, 144],
            'in_index': (0, 1, 2, 3),
            'channels': sum([18, 36, 72, 144]),
        },
        'ocr_head': {
            'in_channels': [18, 36, 72, 144],
            'in_index': (0, 1, 2, 3),
            'channels': 512,
            'ocr_channels': 256,
        }
    },
    'hr18_small': {
        'stage1': {
            'num_modules': 1,
            'num_branches': 1,
            'num_blocks': (2,),
            'num_channels': (64,),
        },
        'stage2': {
            'num_modules': 1,
            'num_branches': 2,
            'num_blocks': (2, 2),
            'num_channels': (18, 36),
        },
        'stage3': {
            'num_modules': 3,
            'num_branches': 3,
            'num_blocks': (2, 2, 2),
            'num_channels': (18, 36, 72),
        },
        'stage4': {
            'num_modules': 2,
            'num_branches': 4,
            'num_blocks': (2, 2, 2, 2),
            'num_channels': (18, 36, 72, 144),
        },
        'fcn_head': {
            'in_channels': [18, 36, 72, 144],
            'in_index': (0, 1, 2, 3),
            'channels': sum([18, 36, 72, 144]),
        },
        'ocr_head': {
            'in_channels': [18, 36, 72, 144],
            'in_index': (0, 1, 2, 3),
            'channels': 512,
            'ocr_channels': 256,
        }
    },
    'hr48': {
        'stage1': {
            'num_modules': 1,
            'num_branches': 1,
            'num_blocks': (4,),
            'num_channels': (64,),
        },
        'stage2': {
            'num_modules': 1,
            'num_branches': 2,
            'num_blocks': (4, 4),
            'num_channels': (48, 96),
        },
        'stage3': {
            'num_modules': 4,
            'num_branches': 3,
            'num_blocks': (4, 4, 4),
            'num_channels': (48, 96, 192),
        },
        'stage4': {
            'num_modules': 3,
            'num_branches': 4,
            'num_blocks': (4, 4, 4, 4),
            'num_channels': (48, 96, 192, 384),
        },
        'fcn_head': {
            'in_channels': [48, 96, 192, 384],
            'in_index': (0, 1, 2, 3),
            'channels': sum([48, 96, 192, 384]),
        },
        'ocr_head': {
            'in_channels': [48, 96, 192, 384],
            'in_index': (0, 1, 2, 3),
            'channels': 512,
            'ocr_channels': 256,
        }
    },
    'hr48_small': {
        'stage1': {
            'num_modules': 1,
            'num_branches': 1,
            'num_blocks': (2,),
            'num_channels': (64,),
        },
        'stage2': {
            'num_modules': 1,
            'num_branches': 2,
            'num_blocks': (2, 2),
            'num_channels': (48, 96),
        },
        'stage3': {
            'num_modules': 3,
            'num_branches': 3,
            'num_blocks': (2, 2, 2),
            'num_channels': (48, 96, 192),
        },
        'stage4': {
            'num_modules': 2,
            'num_branches': 4,
            'num_blocks': (2, 2, 2, 2),
            'num_channels': (48, 96, 192, 384),
        },
        'fcn_head': {
            'in_channels': [48, 96, 192, 384],
            'in_index': (0, 1, 2, 3),
            'channels': sum([48, 96, 192, 384]),
        },
        'ocr_head': {
            'in_channels': [48, 96, 192, 384],
            'in_index': (0, 1, 2, 3),
            'channels': 512,
            'ocr_channels': 256,
        }
    },
}


class OCRNet(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(OCRNet, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = self.config.ocrnet.align_corners
        self.num_stages = self.config.ocrnet.num_stages
        self.use_correctors = self.config.ocrnet.use_correctors

        norm_cfg = dict(type=self.config.ocrnet.norm.type,
                        requires_grad=self.config.ocrnet.norm.requires_grad)
        self.backbone = HRNet(
            norm_cfg=norm_cfg,
            norm_eval=False,
            extra=dict(
                stage1=dict(
                    num_modules=CONFIG[self.config.ocrnet.model]['stage1']['num_modules'],
                    num_branches=CONFIG[self.config.ocrnet.model]['stage1']['num_branches'],
                    num_blocks=CONFIG[self.config.ocrnet.model]['stage1']['num_blocks'],
                    num_channels=CONFIG[self.config.ocrnet.model]['stage1']['num_channels'],
                    block='BOTTLENECK',
                ),
                stage2=dict(
                    num_modules=CONFIG[self.config.ocrnet.model]['stage2']['num_modules'],
                    num_branches=CONFIG[self.config.ocrnet.model]['stage2']['num_branches'],
                    num_blocks=CONFIG[self.config.ocrnet.model]['stage2']['num_blocks'],
                    num_channels=CONFIG[self.config.ocrnet.model]['stage2']['num_channels'],
                    block='BASIC',
                ),
                stage3=dict(
                    num_modules=CONFIG[self.config.ocrnet.model]['stage3']['num_modules'],
                    num_branches=CONFIG[self.config.ocrnet.model]['stage3']['num_branches'],
                    num_blocks=CONFIG[self.config.ocrnet.model]['stage3']['num_blocks'],
                    num_channels=CONFIG[self.config.ocrnet.model]['stage3']['num_channels'],
                    block='BASIC',
                ),
                stage4=dict(
                    num_modules=CONFIG[self.config.ocrnet.model]['stage4']['num_modules'],
                    num_branches=CONFIG[self.config.ocrnet.model]['stage4']['num_branches'],
                    num_blocks=CONFIG[self.config.ocrnet.model]['stage4']['num_blocks'],
                    num_channels=CONFIG[self.config.ocrnet.model]['stage4']['num_channels'],
                    block='BASIC',
                ))
        )
        self.fcn_head = FCNHead(
            in_channels=CONFIG[self.config.ocrnet.model]['fcn_head']['in_channels'],
            in_index=CONFIG[self.config.ocrnet.model]['fcn_head']['in_index'],
            channels=CONFIG[self.config.ocrnet.model]['fcn_head']['channels'],
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=self.align_corners,
        )
        self.ocr_head = OCRHead(
            in_channels=CONFIG[self.config.ocrnet.model]['ocr_head']['in_channels'],
            in_index=CONFIG[self.config.ocrnet.model]['ocr_head']['in_index'],
            input_transform='resize_concat',
            channels=CONFIG[self.config.ocrnet.model]['ocr_head']['channels'],
            ocr_channels=CONFIG[self.config.ocrnet.model]['ocr_head']['ocr_channels'],
            dropout_ratio=-1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=self.align_corners,
        )

        if self.config.ocrnet.use_up_module:
            self.fcn_head_corrector = nn.Sequential(
                Up(in_ch=self.config.ocrnet.up.in_ch,
                   out_ch=self.config.ocrnet.up.out_ch,
                   use_conv_trans2d=self.config.ocrnet.up.use_convt2d,
                   bilinear=self.config.ocrnet.up.bilinear,
                   channels_div_factor=self.config.ocrnet.up.ch_div_factor,
                   use_double_conv=self.config.ocrnet.up.use_double_conv,
                   skip_double_conv=self.config.ocrnet.up.skip_double_conv),
                Up(in_ch=self.config.ocrnet.up.in_ch,
                   out_ch=self.config.ocrnet.up.out_ch,
                   use_conv_trans2d=self.config.ocrnet.up.use_convt2d,
                   bilinear=self.config.ocrnet.up.bilinear,
                   channels_div_factor=self.config.ocrnet.up.ch_div_factor,
                   as_last_layer=True,
                   use_double_conv=self.config.ocrnet.up.use_double_conv,
                   skip_double_conv=self.config.ocrnet.up.skip_double_conv)
            )
            self.ocr_head_corrector = nn.Sequential(
                Up(in_ch=self.config.ocrnet.up.in_ch,
                   out_ch=self.config.ocrnet.up.out_ch,
                   use_conv_trans2d=self.config.ocrnet.up.use_convt2d,
                   bilinear=self.config.ocrnet.up.bilinear,
                   channels_div_factor=self.config.ocrnet.up.ch_div_factor,
                   use_double_conv=self.config.ocrnet.up.use_double_conv,
                   skip_double_conv=self.config.ocrnet.up.skip_double_conv),
                Up(in_ch=self.config.ocrnet.up.in_ch,
                   out_ch=self.config.ocrnet.up.out_ch,
                   use_conv_trans2d=self.config.ocrnet.up.use_convt2d,
                   bilinear=self.config.ocrnet.up.bilinear,
                   channels_div_factor=self.config.ocrnet.up.ch_div_factor,
                   as_last_layer=True,
                   use_double_conv=self.config.ocrnet.up.use_double_conv,
                   skip_double_conv=self.config.ocrnet.up.skip_double_conv)
            )
        elif self.use_correctors:
            self.fcn_head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
            self.ocr_head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.fcn_head_corrector = nn.Identity()
            self.ocr_head_corrector = nn.Identity()
            print('Correctors not in use!')

        self.init_weights(pretrained=self.config.ocrnet.pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.fcn_head.init_weights()
        self.ocr_head.init_weights()

    def forward(self, x):
        feats = self.backbone(x)
        out = self.fcn_head(feats)
        out_ocr = self.ocr_head(feats, out)

        if not self.config.ocrnet.use_up_module:
            out = resize(
                input=out,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            out_ocr = resize(
                input=out_ocr,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        out = self.fcn_head_corrector(out)
        out_ocr = self.ocr_head_corrector(out_ocr)
        return out, out_ocr

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


class OCRResNet(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(OCRResNet, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = self.config.ocrnet.align_corners
        self.num_stages = self.config.ocrnet.num_stages
        self.use_correctors = self.config.ocrnet.use_correctors

        norm_cfg = dict(type=self.config.ocrnet.norm.type,
                        requires_grad=self.config.ocrnet.norm.requires_grad)
        self.backbone = ResNetV1c(
            depth=self.config.ocrnet.resnet_depth,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True
        )
        self.fcn_head = FCNHead(
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=self.align_corners,
        )
        self.ocr_head = OCRHead(
            in_channels=2048,
            in_index=3,
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=False,
        )

        if self.config.ocrnet.use_up_module:
            self.fcn_head_corrector = nn.Sequential(
                Up(in_ch=self.config.ocrnet.up.in_ch,
                   out_ch=self.config.ocrnet.up.out_ch,
                   use_conv_trans2d=self.config.ocrnet.up.use_convt2d,
                   bilinear=self.config.ocrnet.up.bilinear,
                   channels_div_factor=self.config.ocrnet.up.ch_div_factor,
                   use_double_conv=self.config.ocrnet.up.use_double_conv,
                   skip_double_conv=self.config.ocrnet.up.skip_double_conv),
                Up(in_ch=self.config.ocrnet.up.in_ch,
                   out_ch=self.config.ocrnet.up.out_ch,
                   use_conv_trans2d=self.config.ocrnet.up.use_convt2d,
                   bilinear=self.config.ocrnet.up.bilinear,
                   channels_div_factor=self.config.ocrnet.up.ch_div_factor,
                   as_last_layer=True,
                   use_double_conv=self.config.ocrnet.up.use_double_conv,
                   skip_double_conv=self.config.ocrnet.up.skip_double_conv)
            )
            self.ocr_head_corrector = nn.Sequential(
                Up(in_ch=self.config.ocrnet.up.in_ch,
                   out_ch=self.config.ocrnet.up.out_ch,
                   use_conv_trans2d=self.config.ocrnet.up.use_convt2d,
                   bilinear=self.config.ocrnet.up.bilinear,
                   channels_div_factor=self.config.ocrnet.up.ch_div_factor,
                   use_double_conv=self.config.ocrnet.up.use_double_conv,
                   skip_double_conv=self.config.ocrnet.up.skip_double_conv),
                Up(in_ch=self.config.ocrnet.up.in_ch,
                   out_ch=self.config.ocrnet.up.out_ch,
                   use_conv_trans2d=self.config.ocrnet.up.use_convt2d,
                   bilinear=self.config.ocrnet.up.bilinear,
                   channels_div_factor=self.config.ocrnet.up.ch_div_factor,
                   as_last_layer=True,
                   use_double_conv=self.config.ocrnet.up.use_double_conv,
                   skip_double_conv=self.config.ocrnet.up.skip_double_conv)
            )
        elif self.use_correctors:
            self.fcn_head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
            self.ocr_head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.fcn_head_corrector = nn.Identity()
            self.ocr_head_corrector = nn.Identity()
            print('Correctors not in use!')

        self.init_weights(pretrained=self.config.ocrnet.pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.fcn_head.init_weights()
        self.ocr_head.init_weights()

    def forward(self, x):
        feats = self.backbone(x)
        out = self.fcn_head(feats)
        out_ocr = self.ocr_head(feats, out)

        if not self.config.ocrnet.use_up_module:
            out = resize(
                input=out,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            out_ocr = resize(
                input=out_ocr,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        out = self.fcn_head_corrector(out)
        out_ocr = self.ocr_head_corrector(out_ocr)
        return out, out_ocr

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


if __name__ == '__main__':
    m = OCRResNet(OmegaConf.load('../../src/position_maps/config/model/model.yaml'), None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
