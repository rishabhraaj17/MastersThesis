from typing import Union, List, Callable, Optional, Tuple

import torch
from kornia.losses import BinaryFocalLossWithLogits
from omegaconf import DictConfig
# from pl_bolts.models.vision import UNet  # has some matplotlib issue
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from average_image.constants import SDDVideoDatasets
from average_image.utils import SDDMeta
from log import get_logger
from hourglass import PoseNet

logger = get_logger(__name__)


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
