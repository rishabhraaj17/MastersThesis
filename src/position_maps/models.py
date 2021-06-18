from typing import Union, List, Callable, Optional, Tuple

import hydra
import torch
from kornia.losses import BinaryFocalLossWithLogits
from mmcv.ops import DeformConv2d
from mmdet.core import multi_apply
from mmdet.models import HourglassNet
from omegaconf import DictConfig
# from pl_bolts.models.vision import UNet  # has some matplotlib issue
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss, Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from average_image.utils import SDDMeta
from baselinev2.stochastic.model_modules import BaselineGenerator
from log import get_logger
from hourglass import PoseNet

logger = get_logger(__name__)


def post_process_multi_apply(x):
    out = [torch.stack(multi_apply_feats) for multi_apply_feats in zip(*x)]
    return out


class UNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

    Args:
        num_classes: Number of output classes required
        desired_output_shape: Out shape of model
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        num_additional_double_conv_layers: Number of layers before U-net starts (default 0)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
            self,
            config: DictConfig,
            num_classes: int,
            desired_output_shape: Tuple[int, int] = None,
            input_channels: int = 3,
            num_layers: int = 5,
            num_additional_double_conv_layers: int = 0,
            features_start: int = 64,
            bilinear: bool = False
    ):

        if num_layers < 1:
            raise ValueError(f'num_layers = {num_layers}, expected: num_layers > 0')

        super().__init__()
        self.num_layers = num_layers
        self.num_additional_double_conv_layers = num_additional_double_conv_layers
        self.desired_output_shape = desired_output_shape  # unused
        self.config = config
        self.sdd_meta = SDDMeta(self.config.root + 'H_SDD.txt')

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(self.num_additional_double_conv_layers):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Extra DownScale
        for layer in self.layers[1:self.num_additional_double_conv_layers + 1]:
            xi.append(layer(xi[-1]))
        # Down path
        for layer in self.layers[self.num_additional_double_conv_layers + 1: self.num_layers +
                                                                             self.num_additional_double_conv_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers + self.num_additional_double_conv_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        up_scaled = xi[-1]
        if self.desired_output_shape is not None:
            up_scaled = F.interpolate(up_scaled, size=self.desired_output_shape)
        out = self.layers[-1](up_scaled)
        return out


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Destination(nn.Module):
    def __init__(self):
        super(Destination, self).__init__()


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PositionMapUNetBase(LightningModule):
    def __init__(self, config: 'DictConfig', train_dataset: 'Dataset', val_dataset: 'Dataset',
                 desired_output_shape: Tuple[int, int] = None, loss_function: 'nn.Module' = None,
                 collate_fn: Optional[Callable] = None):
        super(PositionMapUNetBase, self).__init__()
        self.config = config
        # fixme: update u_net to network for next trainings
        self.u_net = UNet(num_classes=self.config.unet.num_classes,
                          input_channels=self.config.unet.input_channels,
                          num_layers=self.config.unet.num_layers,
                          num_additional_double_conv_layers=self.config.unet.num_additional_double_conv_layers,
                          features_start=self.config.unet.features_start,
                          bilinear=self.config.unet.bilinear,
                          desired_output_shape=desired_output_shape,
                          config=self.config)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss_function = loss_function
        self.collate_fn = collate_fn
        self.desired_output_shape = desired_output_shape

        self.save_hyperparameters(self.config)

        self.init_weights()

    def forward(self, x):
        return self.u_net(x)

    def _one_step(self, batch):
        return NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay,
                               amsgrad=self.config.amsgrad)
        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(opt,
                                               patience=self.config.patience,
                                               verbose=self.config.verbose,
                                               factor=self.config.factor,
                                               min_lr=self.config.min_lr),
                'monitor': self.config.monitor,
                'interval': self.config.interval,
                'frequency': self.config.frequency
            }]
        return [opt], schedulers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.config.batch_size * self.config.val_batch_size_factor,
            shuffle=False, num_workers=self.config.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last)

    def init_weights(self):
        def init_kaiming(m):
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                m.bias.data.fill_(0.01)

        def init_xavier(m):
            if type(m) == [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_kaiming)


class PositionMapUNetHeatmapRegression(PositionMapUNetBase):
    def __init__(self, config: 'DictConfig', train_dataset: 'Dataset', val_dataset: 'Dataset',
                 desired_output_shape: Tuple[int, int] = None, loss_function: 'nn.Module' = MSELoss(),
                 collate_fn: Optional[Callable] = None):
        super(PositionMapUNetHeatmapRegression, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_function,
            collate_fn=collate_fn, desired_output_shape=desired_output_shape)

    def _one_step(self, batch):
        frames, heat_masks, _, _, _, _ = batch
        out = self(frames)
        loss = self.loss_function(out, heat_masks)
        return loss


class PositionMapUNetPositionMapSegmentation(PositionMapUNetBase):
    def __init__(self, config: 'DictConfig', train_dataset: 'Dataset', val_dataset: 'Dataset',
                 desired_output_shape: Tuple[int, int] = None,
                 loss_function: 'nn.Module' = BinaryFocalLossWithLogits(alpha=0.8, reduction='mean'),
                 collate_fn: Optional[Callable] = None):
        super(PositionMapUNetPositionMapSegmentation, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_function,
            collate_fn=collate_fn, desired_output_shape=desired_output_shape)

    def _one_step(self, batch):
        frames, _, position_map, _, _, _ = batch
        out = self(frames)
        loss = self.loss_function(out, position_map.long().squeeze(dim=1))
        return loss


class PositionMapUNetClassMapSegmentation(PositionMapUNetBase):
    def __init__(self, config: 'DictConfig', train_dataset: 'Dataset', val_dataset: 'Dataset',
                 desired_output_shape: Tuple[int, int] = None,
                 loss_function: 'nn.Module' = BinaryFocalLossWithLogits(alpha=0.8, reduction='mean'),
                 collate_fn: Optional[Callable] = None):
        super(PositionMapUNetClassMapSegmentation, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_function,
            collate_fn=collate_fn, desired_output_shape=desired_output_shape)

    def _one_step(self, batch):
        frames, _, _, _, class_maps, _ = batch
        out = self(frames)
        loss = self.loss_function(out, class_maps.long().squeeze(dim=1))
        return loss


class PositionMapUNetHeatmapSegmentation(PositionMapUNetBase):
    def __init__(self, config: 'DictConfig', train_dataset: 'Dataset', val_dataset: 'Dataset',
                 desired_output_shape: Tuple[int, int] = None,
                 loss_function: 'nn.Module' = BinaryFocalLossWithLogits(alpha=0.8, reduction='mean'),
                 collate_fn: Optional[Callable] = None):
        super(PositionMapUNetHeatmapSegmentation, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_function,
            collate_fn=collate_fn, desired_output_shape=desired_output_shape)

    def _one_step(self, batch):
        frames, heat_masks, _, _, _, _ = batch
        out = self(frames)
        loss = self.loss_function(out, heat_masks)
        return loss


class PositionMapStackedHourGlass(PositionMapUNetBase):
    def __init__(self, config: 'DictConfig', train_dataset: 'Dataset', val_dataset: 'Dataset',
                 desired_output_shape: Tuple[int, int] = None,
                 loss_function: 'nn.Module' = BinaryFocalLossWithLogits(alpha=0.8, reduction='mean'),
                 collate_fn: Optional[Callable] = None):
        super(PositionMapStackedHourGlass, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_function,
            collate_fn=collate_fn, desired_output_shape=desired_output_shape)
        self.network = PoseNet(num_stack=self.config.stacked_hourglass.num_stacks,
                               input_channels=self.config.stacked_hourglass.input_channels,
                               num_classes=self.config.stacked_hourglass.num_classes,
                               loss_fn=self.loss_function,
                               bn=self.config.stacked_hourglass.batch_norm,
                               increase=self.config.stacked_hourglass.increase)
        self.u_net = self.network

    def _one_step(self, batch):
        frames, heat_masks, _, _, _, _ = batch
        out = self(frames)
        loss = self.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
        return loss.mean()


class TrajectoryModel(LightningModule):
    def __init__(self, config: 'DictConfig'):
        super(TrajectoryModel, self).__init__()
        self.config = config

        net_params = self.config.trajectory_baseline.generator
        self.net = BaselineGenerator(embedding_dim_scalars=net_params.embedding_dim_scalars,
                                     encoder_h_g_scalar=net_params.encoder_h_g_scalar,
                                     decoder_h_g_scalar=net_params.decoder_h_g_scalar,
                                     pred_len=net_params.pred_len,
                                     noise_scalar=net_params.noise_scalar,
                                     mlp_vec=net_params.mlp_vec,
                                     mlp_scalar=net_params.mlp_scalar,
                                     POV=net_params.POV,
                                     noise_type=net_params.noise_type,
                                     social_attention=net_params.social_attention,
                                     social_dim_scalar=net_params.social_dim_scalar)

    def forward(self, x):
        # input batch["in_dxdy", "in_xy"]
        return self.net(x)  # {"out_xy": out_xy, "out_dxdy": out_dxdy}


class PositionMapWithTrajectories(PositionMapUNetBase):
    def __init__(self, config: 'DictConfig', position_map_model: 'Module', trajectory_model: 'Module',
                 train_dataset: 'Dataset', val_dataset: 'Dataset',
                 desired_output_shape: Tuple[int, int] = None, loss_function: 'nn.Module' = None,
                 collate_fn: Optional[Callable] = None):
        super(PositionMapWithTrajectories, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_function,
            collate_fn=collate_fn, desired_output_shape=desired_output_shape)

        self.position_map_model = position_map_model
        self.trajectory_model = trajectory_model

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.desired_output_shape = desired_output_shape
        self.loss_function = loss_function
        self.collate_fn = collate_fn

        self.first_iter = True

    def _one_step(self, batch):
        frames, heat_masks, position_map, distribution_map, class_maps, meta = batch
        out = self(frames)
        loss = self.loss_function(out, heat_masks)
        return NotImplementedError  # loss

    def freeze_position_map_model(self):
        self.position_map_model.freeze()

    def unfreeze_position_map_model(self):
        self.position_map_model.unfreeze()

    def freeze_trajectory_model(self):
        self.trajectory_model.freeze()

    def unfreeze_trajectory_model(self):
        self.trajectory_model.unfreeze()


class DoubleDeformableConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, config: DictConfig, in_ch: int, out_ch: int, last_layer: bool = False,
                 use_conv_deform_conv=False,
                 deform_groups: int = 1):
        super().__init__()
        self.config = config
        self.first_layer = nn.Conv2d(in_ch, out_ch, kernel_size=self.config.hourglass.deform.kernel,
                                     padding=self.config.hourglass.deform.padding) \
            if use_conv_deform_conv else DeformConv2d(in_ch, out_ch, kernel_size=self.config.hourglass.deform.kernel,
                                                      padding=self.config.hourglass.deform.padding)
        self.post_first_layer = nn.Sequential(
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.net = DeformConv2d(out_ch, out_ch, kernel_size=self.config.hourglass.deform.kernel,
                                padding=self.config.hourglass.deform.padding, deform_groups=deform_groups)
        self.post_net = nn.Sequential() if last_layer else nn.Sequential(nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.use_conv_deform_conv = use_conv_deform_conv

    def forward(self, x, offsets):
        out = self.first_layer(x) if self.use_conv_deform_conv else self.first_layer(x, offsets)
        out = self.post_first_layer(out)
        out = self.net(out, offsets)
        return self.post_net(out)


class DeformableConvUp(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, config: DictConfig, in_ch: int, out_ch: int, bilinear: bool = False, last_layer: bool = False,
                 use_conv_deform_conv=False):
        super().__init__()
        self.config = config
        self.upsample = None
        kernel_size = self.config.hourglass.upsample_params.kernel_bilinear \
            if bilinear else self.config.hourglass.upsample_params.kernel
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=self.config.hourglass.upsample_params.factor,
                            mode=self.config.hourglass.upsample_params.mode,
                            align_corners=self.config.hourglass.upsample_params.align_corners),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=kernel_size),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                               kernel_size=kernel_size,
                                               stride=self.config.hourglass.upsample_params.stride)

        self.conv = DoubleDeformableConv(config, in_ch // 2, out_ch, last_layer=last_layer,
                                         use_conv_deform_conv=use_conv_deform_conv,
                                         deform_groups=self.config.hourglass.deform.groups)
        offset_out_channel = self.config.hourglass.deform.groups * 2 \
                             * self.config.hourglass.deform.kernel \
                             * self.config.hourglass.deform.kernel
        self.conv_offset = nn.ConvTranspose2d(in_ch, offset_out_channel,
                                              kernel_size=self.config.hourglass.upsample_params.offset_kernel,
                                              stride=self.config.hourglass.upsample_params.offset_stride,
                                              padding=self.config.hourglass.upsample_params.offset_padding,
                                              bias=False)

    def forward(self, x_in):
        x = self.upsample(x_in)
        offsets = self.conv_offset(x_in)
        return self.conv(x, offsets)


class HourGlassNetwork(LightningModule):
    def __init__(self, config: DictConfig):
        super(HourGlassNetwork, self).__init__()
        self.config = config
        self.hour_glass = HourglassNet(downsample_times=self.config.hourglass.downsample_times,
                                       num_stacks=self.config.hourglass.num_stacks,
                                       stage_channels=self.config.hourglass.stage_channels,
                                       stage_blocks=self.config.hourglass.stage_blocks,
                                       feat_channel=self.config.hourglass.feat_channel,
                                       norm_cfg=dict(type=self.config.hourglass.norm_cfg_type,
                                                     requires_grad=self.config.hourglass.norm_cfg_requires_grad),
                                       pretrained=self.config.hourglass.pretrained,
                                       init_cfg=self.config.hourglass.init_cfg)
        self.hour_glass.init_weights()

    def forward(self, x):
        return self.hour_glass(x)


class PositionMapHead(LightningModule):
    def __init__(self, config: DictConfig):
        super(PositionMapHead, self).__init__()
        self.config = config

        layers = []
        feats = self.config.hourglass.feat_channel
        for idx in range(self.config.hourglass.head.num_layers):
            if idx == self.config.hourglass.head.num_layers - 1:
                layers.append(DeformableConvUp(self.config, feats, feats // 2, self.config.hourglass.head.bilinear,
                                               self.config.hourglass.head.enable_last_layer_activation,
                                               use_conv_deform_conv=self.config.hourglass.head.use_conv_deform_conv))
            else:
                layers.append(DeformableConvUp(self.config, feats, feats // 2, self.config.hourglass.head.bilinear,
                                               use_conv_deform_conv=self.config.hourglass.head.use_conv_deform_conv))
            feats //= 2

        self.module = nn.Sequential(*layers)
        # self.module.init_weights()

    def forward(self, x):
        return multi_apply(self.forward_single, x)

    def forward_single(self, x):
        return self.module(x)


class HourGlassPositionMapNetwork(LightningModule):
    def __init__(self, config: 'DictConfig', backbone: 'nn.Module', head: 'nn.Module',
                 train_dataset: 'Dataset', val_dataset: 'Dataset',
                 desired_output_shape: Tuple[int, int] = None, loss_function: 'nn.Module' = None,
                 collate_fn: Optional[Callable] = None):
        super(HourGlassPositionMapNetwork, self).__init__()
        self.config = config
        self.backbone = backbone
        self.head = head
        self.last_conv = DeformConv2d(
            in_channels=self.config.hourglass.feat_channel // (2 ** self.config.hourglass.head.num_layers),
            out_channels=self.config.hourglass.last_conv.out_channels,
            kernel_size=self.config.hourglass.last_conv.kernel,
            stride=self.config.hourglass.last_conv.stride,
            padding=self.config.hourglass.last_conv.padding)

        offset_out_channel = self.config.hourglass.deform.groups * 2 \
                             * self.config.hourglass.last_conv.kernel \
                             * self.config.hourglass.last_conv.kernel
        self.conv_offset = nn.Conv2d(
            self.config.hourglass.feat_channel // (2 ** self.config.hourglass.head.num_layers),
            offset_out_channel,
            kernel_size=self.config.hourglass.last_conv.offset_kernel,
            stride=self.config.hourglass.last_conv.offset_stride,
            padding=self.config.hourglass.last_conv.offset_padding,
            bias=False)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss_function = loss_function
        self.collate_fn = collate_fn
        self.desired_output_shape = desired_output_shape

        self.save_hyperparameters(self.config)

    @classmethod
    def from_config(cls, config: DictConfig, train_dataset: Dataset = None, val_dataset: Dataset = None,
                    desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                    collate_fn: Optional[Callable] = None):
        return HourGlassPositionMapNetwork(
            config=config,
            backbone=HourGlassNetwork(config=config),
            head=PositionMapHead(config=config),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            desired_output_shape=desired_output_shape,
            loss_function=loss_function,
            collate_fn=collate_fn)

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)

        out = post_process_multi_apply(out)
        if self.desired_output_shape is not None:
            out = [F.interpolate(o, size=self.desired_output_shape) for o in out]

        return self.forward_last(out)

    def forward_last(self, x):
        return multi_apply(self.forward_last_single, x)

    def forward_last_single(self, x):
        offset = self.conv_offset(x)
        return self.last_conv(x, offset)

    def calc_loss(self, predictions, heatmaps):
        combined_loss = [self.loss_fn(pred, heatmaps) for pred in predictions]
        combined_loss = torch.stack(combined_loss, dim=0)
        return combined_loss

    def _one_step(self, batch):
        frames, heat_masks, _, _, _, meta = batch
        out = self(frames)
        loss = self.calc_loss(out, heat_masks)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay,
                               amsgrad=self.config.amsgrad)
        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(opt,
                                               patience=self.config.patience,
                                               verbose=self.config.verbose,
                                               factor=self.config.factor,
                                               min_lr=self.config.min_lr),
                'monitor': self.config.monitor,
                'interval': self.config.interval,
                'frequency': self.config.frequency
            }]
        return [opt], schedulers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.config.batch_size * self.config.val_batch_size_factor,
            shuffle=False, num_workers=self.config.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last)


@hydra.main(config_path="config", config_name="config")
def verify_nets(cfg):
    model = HourGlassPositionMapNetwork.from_config(config=cfg, desired_output_shape=(720 // 3, 360 // 3)).cuda()
    inp = torch.randn((2, 3, 720 // 2, 360 // 2)).cuda()
    o = model(inp)
    o = post_process_multi_apply(o)
    print()


if __name__ == '__main__':
    verify_nets()
