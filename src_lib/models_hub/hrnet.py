from typing import Tuple, Optional, Callable

import torch
from mmpose.models import TopdownHeatmapSimpleHead
from mmseg.models import HRNet, FCNHead
from mmseg.ops import resize
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub.base import Base
from src_lib.models_hub.utils import Up

HEAD_CONFIG = {
    'zero': {
        'num_deconv_layers': 0,
        'num_deconv_filters': (256, 256),
        'num_deconv_kernels': (4, 4),
        'extra': dict(final_conv_kernel=1, )
    },
    'two_four': {
        'num_deconv_layers': 2,
        'num_deconv_filters': (256, 256),
        'num_deconv_kernels': (4, 4),
        'extra': dict(final_conv_kernel=1, )
    },
    'two_two': {
        'num_deconv_layers': 2,
        'num_deconv_filters': (256, 256),
        'num_deconv_kernels': (2, 2),
        'extra': dict(final_conv_kernel=1, )
    },
    'two_one': {
        'num_deconv_layers': 2,
        'num_deconv_filters': (256, 256),
        'num_deconv_kernels': (1, 1),
        'extra': dict(final_conv_kernel=1, )
    }
}

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
    },
    'hr32': {
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
            'num_channels': (32, 64),
        },
        'stage3': {
            'num_modules': 4,
            'num_branches': 3,
            'num_blocks': (4, 4, 4),
            'num_channels': (32, 64, 128),
        },
        'stage4': {
            'num_modules': 3,
            'num_branches': 4,
            'num_blocks': (4, 4, 4, 4),
            'num_channels': (32, 64, 128, 256),
        },
    },
    'hr32_small': {
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
            'num_channels': (32, 64),
        },
        'stage3': {
            'num_modules': 3,
            'num_branches': 3,
            'num_blocks': (2, 2, 2),
            'num_channels': (32, 64, 128),
        },
        'stage4': {
            'num_modules': 2,
            'num_branches': 4,
            'num_blocks': (2, 2, 2, 2),
            'num_channels': (32, 64, 128, 256),
        },
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
    },
}


class HRNetwork(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(HRNetwork, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = self.config.hrnet.align_corners
        self.use_correctors = self.config.hrnet.use_correctors

        norm_cfg = dict(type=self.config.hrnet.norm.type,
                        requires_grad=self.config.hrnet.norm.requires_grad)
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
            align_corners=self.align_corners,
        )

        if self.config.hrnet.use_up_module:
            self.head_corrector = nn.Sequential(
                Up(in_ch=self.config.hrnet.up.in_ch,
                   out_ch=self.config.hrnet.up.out_ch,
                   use_conv_trans2d=self.config.hrnet.up.use_convt2d,
                   bilinear=self.config.hrnet.up.bilinear,
                   channels_div_factor=self.config.hrnet.up.ch_div_factor,
                   use_double_conv=self.config.hrnet.up.use_double_conv,
                   skip_double_conv=self.config.hrnet.up.skip_double_conv),
                Up(in_ch=self.config.hrnet.up.in_ch,
                   out_ch=self.config.hrnet.up.out_ch,
                   use_conv_trans2d=self.config.hrnet.up.use_convt2d,
                   bilinear=self.config.hrnet.up.bilinear,
                   channels_div_factor=self.config.hrnet.up.ch_div_factor,
                   as_last_layer=True,
                   use_double_conv=self.config.hrnet.up.use_double_conv,
                   skip_double_conv=self.config.hrnet.up.skip_double_conv)
            )
        elif self.use_correctors:
            self.head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.head_corrector = nn.Identity()

        self.init_weights(pretrained=self.config.hrnet.pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)

        if not self.config.hrnet.use_up_module:
            out = resize(
                input=out,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        out = self.head_corrector(out)
        return out

    def calculate_loss(self, pred, target):
        return self.loss_function(pred, target)

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = pred.sigmoid() if apply_sigmoid else pred
        return weight_factor * loss_function(pred.sigmoid(), target)


class HRPoseNetwork(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(HRPoseNetwork, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = self.config.hrposenet.align_corners
        self.use_correctors = self.config.hrposenet.use_correctors

        norm_cfg = dict(type=self.config.hrposenet.norm.type,
                        requires_grad=self.config.hrposenet.norm.requires_grad)
        self.backbone = HRNet(
            in_channels=3,
            extra=dict(
                stage1=dict(
                    num_modules=CONFIG[self.config.hrposenet.model]['stage1']['num_modules'],
                    num_branches=CONFIG[self.config.hrposenet.model]['stage1']['num_branches'],
                    num_blocks=CONFIG[self.config.hrposenet.model]['stage1']['num_blocks'],
                    num_channels=CONFIG[self.config.hrposenet.model]['stage1']['num_channels'],
                    block='BOTTLENECK',
                ),
                stage2=dict(
                    num_modules=CONFIG[self.config.hrposenet.model]['stage2']['num_modules'],
                    num_branches=CONFIG[self.config.hrposenet.model]['stage2']['num_branches'],
                    num_blocks=CONFIG[self.config.hrposenet.model]['stage2']['num_blocks'],
                    num_channels=CONFIG[self.config.hrposenet.model]['stage2']['num_channels'],
                    block='BASIC',
                ),
                stage3=dict(
                    num_modules=CONFIG[self.config.hrposenet.model]['stage3']['num_modules'],
                    num_branches=CONFIG[self.config.hrposenet.model]['stage3']['num_branches'],
                    num_blocks=CONFIG[self.config.hrposenet.model]['stage3']['num_blocks'],
                    num_channels=CONFIG[self.config.hrposenet.model]['stage3']['num_channels'],
                    block='BASIC',
                ),
                stage4=dict(
                    num_modules=CONFIG[self.config.hrposenet.model]['stage4']['num_modules'],
                    num_branches=CONFIG[self.config.hrposenet.model]['stage4']['num_branches'],
                    num_blocks=CONFIG[self.config.hrposenet.model]['stage4']['num_blocks'],
                    num_channels=CONFIG[self.config.hrposenet.model]['stage4']['num_channels'],
                    block='BASIC',
                ),
            )
        )
        self.head = TopdownHeatmapSimpleHead(
            in_channels=32,
            out_channels=1,
            num_deconv_layers=HEAD_CONFIG[self.config.hrposenet.head_conf]['num_deconv_layers'],
            num_deconv_filters=HEAD_CONFIG[self.config.hrposenet.head_conf]['num_deconv_filters'],
            num_deconv_kernels=HEAD_CONFIG[self.config.hrposenet.head_conf]['num_deconv_kernels'],
            extra=HEAD_CONFIG[self.config.hrposenet.head_conf]['extra'],
            align_corners=self.align_corners,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
        )

        if self.config.hrposenet.use_up_module:
            self.head_corrector = nn.Sequential(
                Up(in_ch=self.config.hrposenet.up.in_ch,
                   out_ch=self.config.hrposenet.up.out_ch,
                   use_conv_trans2d=self.config.hrposenet.up.use_convt2d,
                   bilinear=self.config.hrposenet.up.bilinear,
                   channels_div_factor=self.config.hrposenet.up.ch_div_factor,
                   use_double_conv=self.config.hrposenet.up.use_double_conv,
                   skip_double_conv=self.config.hrposenet.up.skip_double_conv),
                Up(in_ch=self.config.hrposenet.up.in_ch,
                   out_ch=self.config.hrposenet.up.out_ch,
                   use_conv_trans2d=self.config.hrposenet.up.use_convt2d,
                   bilinear=self.config.hrposenet.up.bilinear,
                   channels_div_factor=self.config.hrposenet.up.ch_div_factor,
                   as_last_layer=True,
                   use_double_conv=self.config.hrposenet.up.use_double_conv,
                   skip_double_conv=self.config.hrposenet.up.skip_double_conv)
            )
        elif self.use_correctors:
            self.head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.head_corrector = nn.Identity()

        self.init_weights(pretrained=self.config.hrposenet.pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)

        if not self.config.hrposenet.use_up_module and out.shape[2:] != x.shape[2:]:
            out = resize(
                input=out,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        out = self.head_corrector(out)
        return out

    def calculate_loss(self, pred, target):
        return self.loss_function(pred, target)

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = pred.sigmoid() if apply_sigmoid else pred
        return weight_factor * loss_function(pred.sigmoid(), target)


if __name__ == '__main__':
    m = HRPoseNetwork(OmegaConf.load('../../src/position_maps/config/model/model.yaml'), None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
