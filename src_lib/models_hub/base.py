from typing import Tuple, Optional, Callable, List, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader


def weights_init(m, init_type='normal'):
    if init_type == 'normal':
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == nn.Conv3d:
            nn.init.xavier_normal(m.weight.data)
        elif type(m) == nn.Linear:
            nn.init.kaiming_normal(m.weight.data)
        elif type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    else:
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == nn.Conv3d:
            nn.init.xavier_uniform(m.weight.data)
        elif type(m) == nn.Linear:
            nn.init.kaiming_uniform(m.weight.data)
        elif type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Base(LightningModule):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(Base, self).__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss_function = loss_function
        self.additional_loss_functions = additional_loss_functions

        self.collate_fn = collate_fn
        self.desired_output_shape = desired_output_shape

        self.save_hyperparameters(self.config)

    def forward(self, x):
        return NotImplementedError

    def _one_step(self, batch):
        return NotImplementedError

    def calculate_loss(self, pred, target):
        return NotImplementedError

    def calculate_additional_losses(self, pred, target, weights, apply_sigmoid):
        losses = []
        for loss_fn, weight, use_sigmoid in zip(self.additional_loss_functions, weights, apply_sigmoid):
            pred = [p.sigmoid() if use_sigmoid else p for p in pred]
            losses.append(torch.stack([weight * loss_fn(p, target) for p in pred]))
        return torch.stack(losses)

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


class BaseDDP(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(BaseDDP, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )

    def training_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=False))

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.config.batch_size * self.config.val_batch_size_factor,
            shuffle=False, num_workers=self.config.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            sampler=torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False))


class BaseGAN(LightningModule):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(BaseGAN, self).__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss_function = loss_function
        self.additional_loss_functions = additional_loss_functions

        self.collate_fn = collate_fn
        self.desired_output_shape = desired_output_shape

        self.save_hyperparameters(self.config)

    def forward(self, x):
        return NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss = self._one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx):
        loss = self._one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return NotImplemented

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