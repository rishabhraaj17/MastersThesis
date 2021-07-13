from typing import Tuple, List, Optional, Callable

from mmcls.models import ResNet_CIFAR, GlobalAveragePooling, LinearClsHead
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

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
        self.head = LinearClsHead(
            num_classes=1,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        )

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