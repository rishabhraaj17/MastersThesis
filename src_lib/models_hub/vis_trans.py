from typing import Tuple, Callable, Optional, List

import hydra
import torch
from mmseg.models import VisionTransformer
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub.base import Base
from src_lib.models_hub.trans_unet import Conv2dReLU


class CustomList(list):
    def __init__(self):
        super(CustomList, self).__init__()

    def cpu(self):
        return [s.cpu() for s in self][0]  # for batch-size 1

    def sigmoid(self):
        return [s.sigmoid() for s in self]

    def view(self, *args):
        o = CustomList()
        for s in self:
            o.append(s.view(*args))
        return o


class VITSegmentationHead(nn.Module):
    def __init__(self, config: DictConfig):
        super(VITSegmentationHead, self).__init__()
        self.config = config
        in_channels = self.config.vit.seg_head.in_channels
        out_channels = self.config.vit.seg_head.out_channels
        kernel_size = self.config.vit.seg_head.kernel_size
        stride = self.config.vit.seg_head.stride
        padding = self.config.vit.seg_head.padding
        use_bn = self.config.vit.seg_head.use_bn

        self.net = nn.Sequential(Conv2dReLU(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, padding=padding, stride=stride,
                                            use_batchnorm=use_bn),
                                 nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, padding=padding, stride=stride))

    def forward(self, x):
        return self.net(x)


# not working!
class VisionTransformerSegmentation(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(VisionTransformerSegmentation, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, 
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.backbone = VisionTransformer(
            img_size=tuple(self.config.vit.img_size), patch_size=self.config.vit.patch_size,
            in_channels=self.config.vit.in_channels,
            embed_dim=self.config.vit.embed_dim, depth=self.config.vit.depth, num_heads=self.config.vit.num_heads,
            mlp_ratio=self.config.vit.mlp_ratio, out_indices=self.config.vit.out_indices,
            qkv_bias=self.config.vit.qkv_bias,
            qk_scale=self.config.vit.qk_scale, drop_rate=self.config.vit.drop_rate,
            attn_drop_rate=self.config.vit.attn_drop_rate, drop_path_rate=self.config.vit.drop_path_rate,
            norm_cfg=dict(type=self.config.vit.norm_cfg.type, eps=self.config.vit.norm_cfg.eps,
                          requires_grad=self.config.vit.norm_cfg.requires_grad),
            act_cfg=dict(type=self.config.vit.act_cfg.type), norm_eval=self.config.vit.norm_eval,
            final_norm=self.config.vit.final_norm, out_shape=self.config.vit.out_shape,
            with_cls_token=self.config.vit.with_cls_token, interpolate_mode=self.config.vit.interpolate_mode,
            with_cp=self.config.vit.with_cp
        )
        self.head = VITSegmentationHead(config=self.config)

    @classmethod
    def from_config(cls, config: DictConfig, train_dataset: Dataset = None, val_dataset: Dataset = None,
                    desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                    collate_fn: Optional[Callable] = None):
        return VisionTransformerSegmentation(config=config, train_dataset=train_dataset, val_dataset=val_dataset,
                                             desired_output_shape=desired_output_shape, loss_function=loss_function,
                                             collate_fn=collate_fn)

    def forward(self, x):
        backbone_out = self.backbone(x)

        out = CustomList()  # []
        for o in backbone_out:
            o = o.contiguous().view(*x.shape)
            o = self.head(o)
            out.append(o)

        return out

    def calculate_loss(self, pred, target):
        return torch.stack([self.loss_function(p, target) for p in pred])

    @staticmethod
    def calculate_additional_loss(loss_function, pred, target, apply_sigmoid=True, weight_factor=1.0):
        pred = [p.sigmoid() if apply_sigmoid else p for p in pred]
        return torch.stack([weight_factor * loss_function(p, target) for p in pred])


@hydra.main(config_path="../../src/position_maps/config", config_name="config")
def verify_vit(cfg):
    inp = torch.randn((2, 3, 128, 128))
    m = VisionTransformerSegmentation.from_config(cfg)
    o = m(inp)
    print()


if __name__ == '__main__':
    verify_vit()
