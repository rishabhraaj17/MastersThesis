from typing import Tuple, Callable, Optional

from mmseg.models import VisionTransformer
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from src.models_hub.base import Base


class VisionTransformerSegmentation(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(VisionTransformerSegmentation, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )
        self.backbone = VisionTransformer(
            img_size=self.config.vit.img_size, patch_size=self.config.vit.patch_size,
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

    def forward(self, x):
        return self.backbone(x)

    def calculate_loss(self, pred, target):
        return self.loss_function(pred, target)
