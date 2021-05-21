from typing import Union, List, Callable, Optional

import torch
from omegaconf import DictConfig
from pl_bolts.models.vision import UNet
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from log import get_logger

logger = get_logger(__name__)


class PositionMapUNet(LightningModule):
    def __init__(self, config: 'DictConfig', train_dataset: 'Dataset', val_dataset: 'Dataset',
                 loss_function: 'nn.Module' = None, collate_fn: Optional[Callable] = None):
        super(PositionMapUNet, self).__init__()
        self.config = config
        self.u_net = UNet(num_classes=self.config.unet.num_classes,
                          input_channels=self.config.unet.input_channels,
                          num_layers=self.config.unet.num_layers,
                          features_start=self.config.unet.features_start,
                          bilinear=self.config.unet.bilinear)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss_function = loss_function
        self.collate_fn = collate_fn

        self.save_hyperparameters(self.config)

        self.init_weights()

    def forward(self, x):
        return self.u_net(x)

    def _one_step(self, batch):
        frames, heat_masks, meta = batch
        out = self(frames)
        loss = self.loss_function(out, heat_masks)
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
