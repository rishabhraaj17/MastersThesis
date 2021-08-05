from typing import Tuple, List, Optional, Callable, Union

import torch
from mmcls.models import ResNet_CIFAR, GlobalAveragePooling, LinearClsHead
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from src_lib.models_hub import Base


class CropClassifier(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(CropClassifier, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.backbone = ResNet_CIFAR(
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'
        )
        self.neck = GlobalAveragePooling()
        self.head = nn.Sequential(nn.Linear(512, 64), nn.ReLU6(), nn.Linear(64, 1))

    @classmethod
    def from_config(cls, config: DictConfig, train_dataset: Dataset = None, val_dataset: Dataset = None,
                    desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                    additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        return CropClassifier(config=config, train_dataset=train_dataset, val_dataset=val_dataset,
                              desired_output_shape=desired_output_shape, loss_function=loss_function,
                              additional_loss_functions=additional_loss_functions, collate_fn=collate_fn)

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)
        out = self.head(out)
        return out
    
    def _one_step(self, batch):
        crops, labels = batch
        # for offline mode
        # crops, labels = crops.view(-1, *crops.shape[2:]), labels.view(-1, 1)
        labels = labels.view(-1, 1)
        out = self(crops)
        
        loss = self.calculate_loss(out, labels)
        return loss
    
    def calculate_loss(self, pred, target):
        return self.loss_function(pred, target)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), 
                               lr=self.config.crop_classifier.lr,
                               weight_decay=self.config.crop_classifier.weight_decay,
                               amsgrad=self.config.crop_classifier.amsgrad)
        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(opt,
                                               patience=self.config.crop_classifier.patience,
                                               verbose=self.config.crop_classifier.verbose,
                                               factor=self.config.crop_classifier.factor,
                                               min_lr=self.config.crop_classifier.min_lr),
                'monitor': self.config.crop_classifier.monitor,
                'interval': self.config.crop_classifier.interval,
                'frequency': self.config.crop_classifier.frequency
            }]
        return [opt], schedulers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.config.crop_classifier.batch_size,
            shuffle=self.config.crop_classifier.shuffle, num_workers=self.config.crop_classifier.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.crop_classifier.pin_memory,
            drop_last=self.config.crop_classifier.drop_last)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.crop_classifier.batch_size * self.config.crop_classifier.val_batch_size_factor,
            shuffle=False, num_workers=self.config.crop_classifier.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.crop_classifier.pin_memory,
            drop_last=self.config.crop_classifier.drop_last)


class CropClassifierDDP(CropClassifier):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(CropClassifierDDP, self).__init__(
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


if __name__ == '__main__':
    m = CropClassifier.from_config({})
    inp = torch.randn((2, 3, 64, 64))
    o = m(inp)
    print()
