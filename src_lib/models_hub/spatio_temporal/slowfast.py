from typing import Tuple, List, Optional, Callable

import pytorchvideo.models
import torch
from mmaction.models import ResNet3dSlowFast
from mmpose.models import TopdownHeatmapSimpleHead
from mmseg.models import ASPPHead, DepthwiseSeparableASPPHead, FCNHead
from mmseg.ops import resize
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub import Base, weights_init
from src_lib.models_hub.spatio_temporal.utils import PackPathway

HEAD_CONFIG = {
    'zero': {
        'num_deconv_layers': 0,
        'num_deconv_filters': (256, 256, 256),
        'num_deconv_kernels': (4, 4, 4),
        'extra': dict(final_conv_kernel=1, )
    },
    'three_four': {
        'num_deconv_layers': 3,
        'num_deconv_filters': (256, 256, 256),
        'num_deconv_kernels': (4, 4, 4),
        'extra': dict(final_conv_kernel=1, )
    },
    'three_two': {
        'num_deconv_layers': 3,
        'num_deconv_filters': (256, 256, 256),
        'num_deconv_kernels': (2, 2, 2),
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
    'three_two_small': {
        'num_deconv_layers': 3,
        'num_deconv_filters': (64, 64, 64),
        'num_deconv_kernels': (2, 2, 2),
        'extra': dict(final_conv_kernel=1, )
    },
    'two_two_small': {
        'num_deconv_layers': 2,
        'num_deconv_filters': (64, 64),
        'num_deconv_kernels': (2, 2),
        'extra': dict(final_conv_kernel=1, )
    },
}


class SlowFast(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(SlowFast, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.align_corners = self.config.slow_fast.align_corners
        self.with_aux_head = self.config.slow_fast.with_aux_head
        self.with_deconv_head = self.config.slow_fast.with_deconv_head

        norm_cfg = dict(type=self.config.slow_fast.norm.type,
                        requires_grad=self.config.slow_fast.norm.requires_grad)

        self.backbone = ResNet3dSlowFast(
            pretrained=self.config.slow_fast.pretrained,
            resample_rate=8,
            speed_ratio=8,
            channel_ratio=8,
            slow_pathway=dict(
                type='resnet3d',
                depth=50,
                pretrained=None,
                lateral=True,
                conv1_kernel=(1, 7, 7),
                dilations=(1, 1, 1, 1),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1)),
            fast_pathway=dict(
                type='resnet3d',
                depth=50,
                pretrained=None,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1)
        )
        # it gives two tensors out -> slow pathway and fast pathway
        # either merge them and process or process independently
        # we will merge for now
        self.channel_corrector = nn.Conv3d(2048, 256, 1, 1)
        if self.config.slow_fast.aspp_head:
            self.head = ASPPHead(
                in_channels=2304,
                in_index=0,
                channels=512,
                dilations=(1, 12, 24, 36),
                dropout_ratio=0.1,
                num_classes=self.config.slow_fast.head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        else:
            self.head = DepthwiseSeparableASPPHead(
                in_channels=2304,
                in_index=0,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=0,  # 512,
                c1_channels=48,
                dropout_ratio=0.1,
                num_classes=self.config.slow_fast.head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        if self.with_aux_head:
            self.aux_head = FCNHead(
                in_channels=2304,
                in_index=0,
                channels=512,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=self.config.slow_fast.aux_head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        if self.with_deconv_head:
            self.deconv_head = TopdownHeatmapSimpleHead(
                in_channels=2304,  # 2048,
                out_channels=1,
                in_index=0,  # 3,
                num_deconv_layers=HEAD_CONFIG[self.config.slow_fast.head_conf]['num_deconv_layers'],
                num_deconv_filters=HEAD_CONFIG[self.config.slow_fast.head_conf]['num_deconv_filters'],
                num_deconv_kernels=HEAD_CONFIG[self.config.slow_fast.head_conf]['num_deconv_kernels'],
                extra=HEAD_CONFIG[self.config.slow_fast.head_conf]['extra'],
                align_corners=self.align_corners,
                loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
            )

            self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.head.init_weights()
        self.channel_corrector.apply(weights_init)
        if self.with_aux_head:
            if isinstance(self.aux_head, nn.ModuleList):
                for a_head in self.aux_head:
                    a_head.init_weights()
            else:
                self.aux_head.init_weights()
        if self.with_deconv_head:
            if isinstance(self.deconv_head, nn.ModuleList):
                for d_head in self.deconv_head:
                    d_head.init_weights()
            else:
                self.deconv_head.init_weights()

    def forward(self, x):
        b_slow_out, b_fast_out = self.backbone(x)
        # reduce channel
        b_slow_out = self.channel_corrector(b_slow_out)
        # cat both of them
        out = torch.cat((b_slow_out, b_fast_out), dim=2)
        # make 2D compatible
        out = out.view(out.shape[0], -1, *out.shape[-2:])

        feats = [out]
        out1 = self.head(feats)
        out1 = resize(
            input=out1,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.with_aux_head:
            out2 = self.aux_head(feats)
            out2 = resize(
                input=out2,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        if self.with_deconv_head:
            out3 = self.deconv_head(feats)

        if self.with_aux_head and self.with_deconv_head:
            return out1, out2, out3
        if self.with_deconv_head:
            return out1, out3
        if self.with_aux_head:
            return out1, out2
        return [out1]
        print()


if __name__ == '__main__':
    # net = ResNet3d(depth=18, pretrained=None)
    net = pytorchvideo.models.create_slowfast()

    # inp = torch.rand((2, 3, 2, 720, 480))
    packer = PackPathway()
    inp = packer(torch.rand((2, 3, 32, 240, 240)))
    o = net(inp)
    print()
