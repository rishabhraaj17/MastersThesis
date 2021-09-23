from typing import Tuple, Optional, Callable, List

import torch
from mmpose.models import TopdownHeatmapSimpleHead
from mmseg.models import ResNetV1c, DepthwiseSeparableASPPHead, FCNHead, ASPPHead, UNet
from mmseg.ops import resize
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large

from src.position_maps.utils import ImagePadder
from src_lib.models_hub.base import Base, BaseDDP, weights_init, BaseGAN
from src_lib.models_hub.utils import Up, UpProject

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
}


class DeepLabV3(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(DeepLabV3, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
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
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(DeepLabV3Plus, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.align_corners = self.config.deep_lab_v3_plus.align_corners
        self.with_aux_head = self.config.deep_lab_v3_plus.with_aux_head
        self.with_deconv_head = self.config.deep_lab_v3_plus.with_deconv_head
        self.use_correctors = self.config.deep_lab_v3_plus.use_correctors

        norm_cfg = dict(type=self.config.deep_lab_v3_plus.norm.type,
                        requires_grad=self.config.deep_lab_v3_plus.norm.requires_grad)

        # proper way, but mmsegentation has a bug now
        # init_cfg = None
        # if self.config.deep_lab_v3_plus.pretrained is not None:
        #     init_cfg = dict(type='Pretrained', checkpoint=self.config.deep_lab_v3_plus.pretrained)

        self.backbone = ResNetV1c(
            depth=self.config.deep_lab_v3_plus.resnet_depth,
            in_channels=self.config.deep_lab_v3_plus.in_channels,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=None,
            pretrained=self.config.deep_lab_v3_plus.pretrained
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
        if self.with_deconv_head:
            self.deconv_head = TopdownHeatmapSimpleHead(
                in_channels=1024,  # 2048,
                out_channels=1,
                in_index=2,  # 3,
                num_deconv_layers=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_layers'],
                num_deconv_filters=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_filters'],
                num_deconv_kernels=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_kernels'],
                extra=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['extra'],
                align_corners=self.align_corners,
                loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
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

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.head.init_weights()
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
        feats = self.backbone(x)
        out1 = self.head(feats)
        if self.with_aux_head:
            out2 = self.aux_head(feats)
        if self.with_deconv_head:
            out3 = self.deconv_head(list(feats))

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

        if self.with_aux_head and self.with_deconv_head:
            return out1, out2, out3
        if self.with_deconv_head:
            return out1, out3
        if self.with_aux_head:
            return out1, out2
        return [out1]

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


class UNetDeepLabV3Plus(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(UNetDeepLabV3Plus, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
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
            norm_eval=False,
            init_cfg=None,
            pretrained=self.config.deep_lab_v3_plus.pretrained
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

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
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


class DeepLabV3DDP(BaseDDP):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(DeepLabV3DDP, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
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


class DeepLabV3PlusDDP(BaseDDP):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(DeepLabV3PlusDDP, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.align_corners = self.config.deep_lab_v3_plus.align_corners
        self.with_aux_head = self.config.deep_lab_v3_plus.with_aux_head
        self.with_deconv_head = self.config.deep_lab_v3_plus.with_deconv_head
        self.use_correctors = self.config.deep_lab_v3_plus.use_correctors

        if self.config.deep_lab_v3_plus.load_pretrained_manually:
            self.config.deep_lab_v3_plus.pretrained = None

        norm_cfg = dict(type='SyncBN',
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
            contract_dilation=True,
            init_cfg=None,
            pretrained=self.config.deep_lab_v3_plus.pretrained
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
                align_corners=self.align_corners,
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
        if self.with_deconv_head:
            self.deconv_head = TopdownHeatmapSimpleHead(
                in_channels=1024,  # 2048,
                out_channels=1,
                in_index=2,  # 3,
                num_deconv_layers=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_layers'],
                num_deconv_filters=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_filters'],
                num_deconv_kernels=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_kernels'],
                extra=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['extra'],
                align_corners=self.align_corners,
                loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
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

        self.loss_reducer = getattr(torch.Tensor, self.config.loss.reduction)
        self.additional_loss_weights = self.config.loss.gaussian_weight
        self.additional_loss_activation = self.config.loss.apply_sigmoid

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.config.deep_lab_v3_plus.load_pretrained_manually:
            print(f'Loading pretrained model locally from {self.config.deep_lab_v3_plus.local_pretrained_path}')
            state_dict = torch.load(self.config.deep_lab_v3_plus.local_pretrained_path,
                                    map_location=self.config.device)
            missing_keys = self.backbone.load_state_dict(state_dict=state_dict['state_dict'],
                                                         strict=self.config.deep_lab_v3_plus.load_strictly)
        self.head.init_weights()
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
        feats = self.backbone(x)
        out1 = self.head(feats)
        if self.with_aux_head:
            out2 = self.aux_head(feats)
        if self.with_deconv_head:
            out3 = self.deconv_head(list(feats))

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

        if self.with_aux_head and self.with_deconv_head:
            return out1, out2, out3
        if self.with_deconv_head:
            return out1, out3
        if self.with_aux_head:
            return out1, out2
        return [out1]

    def _one_step(self, batch):
        frames, heat_masks, _, _, _, meta = batch

        padder = ImagePadder(frames.shape[-2:], factor=self.config.preproccesing.pad_factor)
        frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]

        out = self(frames)

        loss1 = self.loss_reducer(self.calculate_loss(out, heat_masks))
        loss2 = self.loss_reducer(self.calculate_additional_losses(
            out, heat_masks, self.additional_loss_weights, self.additional_loss_activation))
        return loss1, loss2

    def training_step(self, batch, batch_idx):
        bfl, gaussian_loss = self._one_step(batch)
        loss = bfl + gaussian_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_bfl', bfl, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_gl', gaussian_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bfl, gaussian_loss = self._one_step(batch)
        loss = bfl + gaussian_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_bfl', bfl, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_gl', gaussian_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


class DeepLabV3PlusTemporal2DDDP(DeepLabV3PlusDDP):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(DeepLabV3PlusTemporal2DDDP, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        norm_cfg = dict(type='SyncBN',
                        requires_grad=self.config.deep_lab_v3_plus.norm.requires_grad)
        self.backbone = ResNetV1c(
            in_channels=self.config.deep_lab_v3_plus.in_channels * self.config.video_based.frames_per_clip,
            depth=self.config.deep_lab_v3_plus.resnet_depth,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=None,
            pretrained=self.config.deep_lab_v3_plus.pretrained
        )
        
    def _one_step(self, batch):
        frames, heat_masks, _, _, _, meta = batch

        padder = ImagePadder(frames.shape[-2:], factor=self.config.preproccesing.pad_factor)
        frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]

        out = self(frames)

        heat_masks = heat_masks[:, self.config.video_based.gt_idx, None, ...]

        loss1 = self.loss_reducer(self.calculate_loss(out, heat_masks))
        loss2 = self.loss_reducer(self.calculate_additional_losses(
            out, heat_masks, self.additional_loss_weights, self.additional_loss_activation))
        return loss1, loss2


class DeepLabV3PlusTemporal2D(DeepLabV3PlusTemporal2DDDP):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(DeepLabV3PlusTemporal2D, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        norm_cfg = dict(type='BN', requires_grad=self.config.deep_lab_v3_plus.norm.requires_grad)
        self.backbone = ResNetV1c(
            in_channels=self.config.deep_lab_v3_plus.in_channels * self.config.video_based.frames_per_clip,
            depth=self.config.deep_lab_v3_plus.resnet_depth,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=None,
            pretrained=self.config.deep_lab_v3_plus.pretrained
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
                align_corners=self.align_corners,
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
        if self.with_deconv_head:
            self.deconv_head = TopdownHeatmapSimpleHead(
                in_channels=1024,  # 2048,
                out_channels=1,
                in_index=2,  # 3,
                num_deconv_layers=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_layers'],
                num_deconv_filters=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_filters'],
                num_deconv_kernels=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['num_deconv_kernels'],
                extra=HEAD_CONFIG[self.config.deep_lab_v3_plus.head_conf]['extra'],
                align_corners=self.align_corners,
                loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
            )


class DeepLabV3PlusSmall(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(DeepLabV3PlusSmall, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.align_corners = self.config.deep_lab_v3_plus.small.align_corners
        self.with_aux_head = self.config.deep_lab_v3_plus.small.with_aux_head
        self.with_deconv_head = self.config.deep_lab_v3_plus.small.with_deconv_head
        self.use_correctors = False  # self.config.deep_lab_v3_plus.use_correctors

        norm_cfg = dict(type=self.config.deep_lab_v3_plus.norm.type,
                        requires_grad=self.config.deep_lab_v3_plus.norm.requires_grad)

        self.backbone = ResNetV1c(
            depth=self.config.deep_lab_v3_plus.small.resnet_depth,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=None,
            pretrained=self.config.deep_lab_v3_plus.small.pretrained
        )
        if self.config.deep_lab_v3_plus.small.aspp_head:
            self.head = ASPPHead(
                in_channels=512,
                in_index=3,
                channels=128,
                dilations=(1, 12, 24, 36),
                dropout_ratio=0.1,
                num_classes=self.config.deep_lab_v3_plus.head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        else:
            self.head = DepthwiseSeparableASPPHead(
                in_channels=512,
                in_index=3,
                channels=128,
                dilations=(1, 12, 24, 36),
                c1_in_channels=64,
                c1_channels=12,
                dropout_ratio=0.1,
                num_classes=self.config.deep_lab_v3_plus.head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        if self.with_aux_head:
            self.aux_head = FCNHead(
                in_channels=256,
                in_index=2,
                channels=64,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=self.config.deep_lab_v3_plus.aux_head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        if self.with_deconv_head:
            self.deconv_head = TopdownHeatmapSimpleHead(
                in_channels=256,  # 2048,
                out_channels=1,
                in_index=2,  # 3,
                num_deconv_layers=HEAD_CONFIG[self.config.deep_lab_v3_plus.small.head_conf]['num_deconv_layers'],
                num_deconv_filters=HEAD_CONFIG[self.config.deep_lab_v3_plus.small.head_conf]['num_deconv_filters'],
                num_deconv_kernels=HEAD_CONFIG[self.config.deep_lab_v3_plus.small.head_conf]['num_deconv_kernels'],
                extra=HEAD_CONFIG[self.config.deep_lab_v3_plus.small.head_conf]['extra'],
                align_corners=self.align_corners,
                loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
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

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.head.init_weights()
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
        feats = self.backbone(x)
        out1 = self.head(feats)
        if self.with_aux_head:
            out2 = self.aux_head(feats)
        if self.with_deconv_head:
            out3 = self.deconv_head(list(feats))

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

        if self.with_aux_head and self.with_deconv_head:
            return out1, out2, out3
        if self.with_deconv_head:
            return out1, out3
        if self.with_aux_head:
            return out1, out2
        return [out1]

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


class DeepLabV3Discriminator(nn.Module):
    def __init__(self, config: DictConfig):
        super(DeepLabV3Discriminator, self).__init__()
        self.config = config

        self.with_aux_head = self.config.deep_lab_v3_gan.with_aux_head
        self.with_deconv_head = self.config.deep_lab_v3_gan.with_deconv_head

        self.discriminator = ResNetV1c(
            depth=18,
            in_channels=1,
            out_indices=(3,),
            norm_eval=False,
            norm_cfg=dict(type='BN'),
        )
        self.discriminator_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.init_weights()

    def init_weights(self):
        self.discriminator.init_weights()
        self.discriminator_fc.apply(weights_init)

    def forward(self, x):
        if self.with_aux_head and self.with_deconv_head:
            out1, out2, out3 = x
            d_out = self.discriminator(torch.cat((out1, out2, out3)))
        elif self.with_aux_head:
            out1, out2 = x
            d_out = self.discriminator(torch.cat((out1, out2)))
        elif self.with_deconv_head:
            out1, out3 = x
            d_out = self.discriminator(torch.cat((out1, out3)))
        else:
            d_out = self.discriminator(x)

        d_out = d_out[-1]
        d_out = self.discriminator_fc(d_out)

        return d_out


class DeepLabV3Generator(nn.Module):
    def __init__(self, config: DictConfig):
        super(DeepLabV3Generator, self).__init__()
        self.config = config

        self.align_corners = self.config.deep_lab_v3_gan.align_corners
        self.with_aux_head = self.config.deep_lab_v3_gan.with_aux_head
        self.with_deconv_head = self.config.deep_lab_v3_gan.with_deconv_head

        norm_cfg = dict(type=self.config.deep_lab_v3_gan.norm.type,
                        requires_grad=self.config.deep_lab_v3_gan.norm.requires_grad)

        self.backbone = ResNetV1c(
            depth=self.config.deep_lab_v3_gan.resnet_depth,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=None,
            pretrained=self.config.deep_lab_v3_gan.pretrained
        )
        if self.config.deep_lab_v3_gan.aspp_head:
            self.head = ASPPHead(
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                dropout_ratio=0.1,
                num_classes=self.config.deep_lab_v3_gan.head.out_ch,
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
                num_classes=self.config.deep_lab_v3_gan.head.out_ch,
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
                num_classes=self.config.deep_lab_v3_gan.aux_head.out_ch,
                norm_cfg=norm_cfg,
                align_corners=self.align_corners
            )
        if self.with_deconv_head:
            self.deconv_head = TopdownHeatmapSimpleHead(
                in_channels=1024,  # 2048,
                out_channels=1,
                in_index=2,  # 3,
                num_deconv_layers=HEAD_CONFIG[self.config.deep_lab_v3_gan.head_conf]['num_deconv_layers'],
                num_deconv_filters=HEAD_CONFIG[self.config.deep_lab_v3_gan.head_conf]['num_deconv_filters'],
                num_deconv_kernels=HEAD_CONFIG[self.config.deep_lab_v3_gan.head_conf]['num_deconv_kernels'],
                extra=HEAD_CONFIG[self.config.deep_lab_v3_gan.head_conf]['extra'],
                align_corners=self.align_corners,
                loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
            )

            self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.head.init_weights()
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
        feats = self.backbone(x)

        # add noise
        feats = [f + torch.randn_like(f) for f in feats]
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


class DeepLabV3GAN(BaseGAN):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, desc_loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(DeepLabV3GAN, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.config = config
        self.with_aux_head = self.config.deep_lab_v3_gan.with_aux_head
        self.with_deconv_head = self.config.deep_lab_v3_gan.with_deconv_head

        self.generator = DeepLabV3Generator(config=config)
        self.discriminator = DeepLabV3Discriminator(config=config)

        self.loss_reducer = getattr(torch.Tensor, self.config.loss.reduction)
        self.additional_loss_weights = self.config.loss.gaussian_weight
        self.additional_loss_activation = self.config.loss.apply_sigmoid

        self.desc_loss_function = desc_loss_function

    def forward_gen_desc(self, x):
        gen_out = self.generator(x)

        des_out = self.discriminator(gen_out)

        if self.with_aux_head and self.with_deconv_head:
            out1, out2, out3 = gen_out
            return (out1, out2, out3), des_out
        if self.with_deconv_head:
            out1, out3 = gen_out
            return (out1, out3), des_out
        if self.with_aux_head:
            out1, out2 = gen_out
            return (out1, out2), des_out
        return [gen_out], des_out

    def forward(self, x):
        gen_out = self.generator(x)
        return [gen_out]

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    def calculate_additional_losses(self, pred, target, weights, apply_sigmoid):
        losses = []
        for loss_fn, weight, use_sigmoid in zip(self.additional_loss_functions, weights, apply_sigmoid):
            pred = [p.sigmoid() if use_sigmoid else p for p in pred]
            losses.append(torch.stack([weight * loss_fn(p, target) for p in pred]))
        return torch.stack(losses)

    def configure_optimizers(self):
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.deep_lab_v3_gan.discriminator.lr,
                                    weight_decay=self.config.deep_lab_v3_gan.discriminator.weight_decay,
                                    amsgrad=self.config.deep_lab_v3_gan.discriminator.weight_decay)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.config.deep_lab_v3_gan.generator.lr,
                                   weight_decay=self.config.deep_lab_v3_gan.generator.weight_decay,
                                   amsgrad=self.config.deep_lab_v3_gan.generator.weight_decay)
        return [opt_disc, opt_gen], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        frames, heat_masks, _, _, _, meta = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step((frames, heat_masks))

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step((frames, heat_masks))

        return result

    def _disc_step(self, x):
        disc_loss = self._get_disc_loss(x)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_step(self, x):
        gen_loss = self._get_gen_loss(x)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def _get_disc_loss(self, x):
        frames, heat_masks = x

        # Train with real
        real_pred = self.discriminator(heat_masks)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.desc_loss_function(real_pred, real_gt)

        # Train with fake
        fake_pred = self.generator(frames)  # with no grad?
        fake_pred = [self.discriminator(f) for f in fake_pred]
        fake_gt = torch.ones_like(fake_pred[0])
        fake_loss = [self.discriminator_criterion(f, fake_gt) for f in fake_pred]
        fake_loss = self.loss_reducer(torch.stack(fake_loss))

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _get_gen_loss(self, x):
        frames, heat_masks = x

        padder = ImagePadder(frames.shape[-2:], factor=self.config.preproccesing.pad_factor)
        frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]

        out = self.generator(frames)

        loss1 = self.loss_reducer(self.calculate_loss(out, heat_masks))
        loss2 = self.loss_reducer(self.calculate_additional_losses(
            out, heat_masks, self.additional_loss_weights, self.additional_loss_activation))

        gen_loss = loss1 + loss2

        return gen_loss


if __name__ == '__main__':
    m = DeepLabV3GAN(OmegaConf.load('../../src/position_maps/config/model/model.yaml'), None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
