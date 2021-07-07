from typing import Tuple, List, Optional, Callable

import torch
from mmseg.models import MobileNetV3, LRASPPHead
from mmseg.ops import resize
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub import Base

MODELS_CONFIG = OmegaConf.load('../../src/position_maps/config/model/model.yaml')
NORM_CONF = dict(type=MODELS_CONFIG.mm_segmentator.norm.type,
                 requires_grad=MODELS_CONFIG.mm_segmentator.norm.requires_grad)
OUT_CLASSES = 1

BACKBONE = {
    'mobile_net_v3_large': MobileNetV3(
        arch='large',
        norm_cfg=NORM_CONF,
        out_indices=(1, 3, 16),
        pretrained=MODELS_CONFIG.mm_segmentator.pretrained,
        init_cfg=None
    )
}

HEAD = {
    'lraspp_large': LRASPPHead(
        in_channels=(16, 24, 960),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=OUT_CLASSES,
        norm_cfg=NORM_CONF,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
}

AUX_HEAD = {

}

SECONDARY_AUX_HEAD = {

}


class MMSegmentator(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(MMSegmentator, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.align_corners = self.config.mm_segmentator.align_corners
        self.with_aux_head = self.config.mm_segmentator.with_aux_head
        self.with_secondary_aux_head = self.config.mm_segmentator.with_secondary_aux_head

        self.backbone = BACKBONE[self.config.mm_segmentator.backbone]
        self.head = HEAD[self.config.mm_segmentator.head]

        if self.with_aux_head:
            self.aux_head = AUX_HEAD[self.config.mm_segmentator.aux_head]
        if self.with_secondary_aux_head:
            self.secondary_aux_head = SECONDARY_AUX_HEAD[self.config.mm_segmentator.secondary_aux_head]

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
        if self.with_secondary_aux_head:
            if isinstance(self.secondary_aux_head, nn.ModuleList):
                for d_head in self.secondary_aux_head:
                    d_head.init_weights()
            else:
                self.secondary_aux_head.init_weights()

    def forward(self, x):
        feats = self.backbone(x)

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

        if self.with_secondary_aux_head:
            out3 = self.secondary_aux_head(list(feats))

        if self.with_aux_head and self.with_secondary_aux_head:
            return out1, out2, out3
        if self.with_secondary_aux_head:
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


if __name__ == '__main__':
    m = MMSegmentator(MODELS_CONFIG, None, None)
    inp = torch.randn((2, 3, 480, 480))
    o = m(inp)
    print()
