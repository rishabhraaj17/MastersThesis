from typing import Tuple, Optional, Callable

import torch
from mmpose.models import TopdownHeatmapSimpleHead, HourglassNet, MobileNetV2, ResNet, ResNeSt, ResNetV1d, ResNeXt, \
    SEResNet, ShuffleNetV2
from mmseg.ops import resize
from omegaconf import OmegaConf, DictConfig
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
}

# look out for sync norm
BACKBONE = {
    'hourglass': {
        'net': HourglassNet(
            downsample_times=5,
            num_stacks=2,
            stage_channels=[256, 256, 384, 384, 384, 512],
            stage_blocks=[2, 2, 2, 2, 2, 4],
            feat_channel=256
        ),
        'head': {
            'in_channels': 256
        }
    },
    'hourglass_s1d5': {
        'net': HourglassNet(
            downsample_times=5,
            num_stacks=1,
            stage_channels=[256, 256, 384, 384, 384, 512],
            stage_blocks=[2, 2, 2, 2, 2, 4],
            feat_channel=256
        ),
        'head': {
            'in_channels': 256
        }
    },
    'hourglass_s1d5_medium': {
        'net': HourglassNet(
            downsample_times=5,
            num_stacks=1,
            stage_channels=[256, 256, 384, 384, 384, 512],
            stage_blocks=[2, 1, 2, 1, 1, 3],
            feat_channel=256
        ),
        'head': {
            'in_channels': 256
        }
    },
    'hourglass_s2d5_medium': {
        'net': HourglassNet(
            downsample_times=5,
            num_stacks=2,
            stage_channels=[256, 256, 384, 384, 384, 512],
            stage_blocks=[2, 1, 2, 1, 1, 3],
            feat_channel=256
        ),
        'head': {
            'in_channels': 256
        }
    },
    'hourglass_s2d2_small': {
        'net': HourglassNet(
            downsample_times=2,
            num_stacks=2,
            stage_channels=[256, 384, 512],
            stage_blocks=[2, 2, 4],
            feat_channel=256
        ),
        'head': {
            'in_channels': 256
        }
    },
    'hourglass_s1d2_small': {
        'net': HourglassNet(
            downsample_times=2,
            num_stacks=1,
            stage_channels=[256, 384, 512],
            stage_blocks=[2, 2, 4],
            feat_channel=256
        ),
        'head': {
            'in_channels': 256
        }
    },
    'mobile_net': {
        'net': MobileNetV2(
            widen_factor=1,
            out_indices=(7,)
        ),
        'head': {
            'in_channels': 1280
        }
    },
    'resnet50': {
        'net': ResNet(
            depth=50,
            in_channels=3,
        ),
        'head': {
            'in_channels': 2048
        }
    },
    'resnet152': {
        'net': ResNet(
            depth=152,
            in_channels=3,
        ),
        'head': {
            'in_channels': 2048
        }
    },
    'resnet101': {
        'net': ResNet(
            depth=101,
            in_channels=3,
        ),
        'head': {
            'in_channels': 2048
        }
    },
    'resnest': {
        'net': ResNeSt(
            depth=200,
        ),
        'head': {
            'in_channels': 2048
        }
    },
    'resnetv1d101': {
        'net': ResNetV1d(
            depth=101,
            in_channels=3,
        ),
        'head': {
            'in_channels': 2048
        }
    },
    'resnext101': {
        'net': ResNeXt(
            depth=101,
        ),
        'head': {
            'in_channels': 2048
        }
    },
    'seresnet152': {
        'net': SEResNet(
            depth=152,
        ),
        'head': {
            'in_channels': 2048
        }
    },
    'shufflenetv2': {
        'net': ShuffleNetV2(
            widen_factor=1.0,
        ),
        'head': {
            'in_channels': 1024
        }
    },
}


class TopDownPoseNetwork(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(TopDownPoseNetwork, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.align_corners = self.config.topdownposenet.align_corners
        self.use_correctors = self.config.topdownposenet.use_correctors

        norm_cfg = dict(type=self.config.topdownposenet.norm.type,
                        requires_grad=self.config.topdownposenet.norm.requires_grad)
        self.backbone = BACKBONE[self.config.topdownposenet.backbone]['net']
        self.head = TopdownHeatmapSimpleHead(
            in_channels=BACKBONE[self.config.topdownposenet.backbone]['head']['in_channels'],
            out_channels=1,
            num_deconv_layers=HEAD_CONFIG[self.config.topdownposenet.head_conf]['num_deconv_layers'],
            num_deconv_filters=HEAD_CONFIG[self.config.topdownposenet.head_conf]['num_deconv_filters'],
            num_deconv_kernels=HEAD_CONFIG[self.config.topdownposenet.head_conf]['num_deconv_kernels'],
            extra=HEAD_CONFIG[self.config.topdownposenet.head_conf]['extra'],
            align_corners=self.align_corners,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
        )

        if self.config.topdownposenet.use_up_module:
            self.head_corrector = nn.Sequential(
                Up(in_ch=self.config.topdownposenet.up.in_ch,
                   out_ch=self.config.topdownposenet.up.out_ch,
                   use_conv_trans2d=self.config.topdownposenet.up.use_convt2d,
                   bilinear=self.config.topdownposenet.up.bilinear,
                   channels_div_factor=self.config.topdownposenet.up.ch_div_factor,
                   use_double_conv=self.config.topdownposenet.up.use_double_conv,
                   skip_double_conv=self.config.topdownposenet.up.skip_double_conv),
                Up(in_ch=self.config.topdownposenet.up.in_ch,
                   out_ch=self.config.topdownposenet.up.out_ch,
                   use_conv_trans2d=self.config.topdownposenet.up.use_convt2d,
                   bilinear=self.config.topdownposenet.up.bilinear,
                   channels_div_factor=self.config.topdownposenet.up.ch_div_factor,
                   as_last_layer=True,
                   use_double_conv=self.config.topdownposenet.up.use_double_conv,
                   skip_double_conv=self.config.topdownposenet.up.skip_double_conv)
            )
        elif self.use_correctors:
            self.head_corrector = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.head_corrector = nn.Identity()

        self.init_weights(pretrained=self.config.topdownposenet.pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)

        if not self.config.topdownposenet.use_up_module and out.shape[2:] != x.shape[2:]:
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
    m = TopDownPoseNetwork(OmegaConf.load('../../src/position_maps/config/model/model.yaml'), None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
