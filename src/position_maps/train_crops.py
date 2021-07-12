import os
import warnings
from typing import Optional, Callable, List, Tuple

import hydra
import numpy as np
import torch
from mmcls.models import ResNet_CIFAR, GlobalAveragePooling, LinearClsHead
from omegaconf import ListConfig, DictConfig
from pytorch_lightning import seed_everything
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader, Dataset
from tqdm import tqdm

from log import get_logger
from src.position_maps.location_utils import locations_from_heatmaps, get_adjusted_object_locations
from src_lib.models_hub import Base
from train import setup_single_video_dataset, setup_multiple_datasets, build_model, build_loss
from utils import heat_map_collate_fn, ImagePadder

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


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


@hydra.main(config_path="config", config_name="config")
def train_crop_classifier_v0(cfg):
    logger.info(f'Patch Classifier')
    logger.info(f'Setting up DataLoader and Model...')

    # adjust config here
    cfg.device = 'cpu'  # 'cuda:0'
    cfg.single_video_mode.enabled = True  # for now we work on single video
    cfg.preproccesing.pad_factor = 8
    cfg.frame_rate = 20
    cfg.video_based.enabled = False

    if cfg.single_video_mode.enabled:
        # config adapt
        cfg.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
        cfg.single_video_mode.video_numbers_to_use = [[4]]
        cfg.desired_pixel_to_meter_ratio_rgb = 0.07
        cfg.desired_pixel_to_meter_ratio = 0.07

        train_dataset, val_dataset, target_max_shape = setup_single_video_dataset(cfg, use_common_transforms=False)
    else:
        train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    # loss config params are ok!
    loss_fn, gaussian_loss_fn = build_loss(cfg)

    # position map model config
    cfg.model = 'DeepLabV3Plus'
    position_model = build_model(cfg, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_fn,
                                 additional_loss_functions=gaussian_loss_fn, collate_fn=heat_map_collate_fn,
                                 desired_output_shape=target_max_shape)

    if cfg.crop_classifier.use_pretrained.enabled:
        checkpoint_path = f'{cfg.crop_classifier.use_pretrained.checkpoint.root}' \
                          f'{cfg.crop_classifier.use_pretrained.checkpoint.path}' \
                          f'{cfg.crop_classifier.use_pretrained.checkpoint.version}/checkpoints/'
        checkpoint_files = os.listdir(checkpoint_path)

        epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
        epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
        checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

        checkpoint_file = checkpoint_path + checkpoint_files[-cfg.crop_classifier.use_pretrained.checkpoint.top_k]

        logger.info(f'Loading weights from: {checkpoint_file}')
        load_dict = torch.load(checkpoint_file, map_location=cfg.device)

        position_model.load_state_dict(load_dict['state_dict'], strict=False)

    position_model.to(cfg.device)
    position_model.eval()

    # classifier model
    model = CropClassifier.from_config(config=cfg)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.crop_classifier.lr,
                           weight_decay=cfg.crop_classifier.weight_decay, amsgrad=cfg.crop_classifier.amsgrad)
    sch = ReduceLROnPlateau(model,
                            patience=cfg.crop_classifier.patience,
                            verbose=cfg.crop_classifier.verbose,
                            factor=cfg.crop_classifier.factor,
                            min_lr=cfg.crop_classifier.min_lr)

    if isinstance(cfg.crop_classifier.subset_indices, (list, ListConfig)):
        indices = list(cfg.crop_classifier.subset_indices)
    else:
        indices = np.random.choice(len(train_dataset), cfg.crop_classifier.subset_indices, replace=False)
    train_subset = Subset(dataset=train_dataset, indices=indices)

    train_loader = DataLoader(
        train_subset, batch_size=cfg.crop_classifier.batch_size, shuffle=cfg.crop_classifier.shuffle,
        num_workers=cfg.crop_classifier.num_workers, collate_fn=heat_map_collate_fn,
        pin_memory=cfg.crop_classifier.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.crop_classifier.batch_size * 2, shuffle=cfg.crop_classifier.shuffle,
        num_workers=cfg.crop_classifier.num_workers, collate_fn=heat_map_collate_fn,
        pin_memory=cfg.crop_classifier.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    # training + logic

    for epoch in range(cfg.interplay_v0.num_epochs):
        opt.zero_grad()
        model.train()

        train_loss = []
        for t_idx, data in enumerate(tqdm(train_loader)):
            # position_model_opt.zero_grad()

            frames, heat_masks, _, _, _, meta = data

            padder = ImagePadder(frames.shape[-2:], factor=cfg.preproccesing.pad_factor)
            frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]
            frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

            pred_position_maps = position_model(frames)

            noisy_gt_agents_count = [m['bbox_centers'].shape[0] for m in meta]
            frame_numbers = [m['item'] for m in meta]

            pred_object_locations = locations_from_heatmaps(
                frames, cfg.interplay_v0.objectness.kernel,
                cfg.interplay_v0.objectness.loc_cutoff,
                cfg.interplay_v0.objectness.marker_size, pred_position_maps,
                vis_on=False)
            selected_head = pred_position_maps[cfg.interplay_v0.objectness.index_select]
            pred_object_locations_scaled, heat_maps_gt_scaled = get_adjusted_object_locations(
                pred_object_locations[cfg.interplay_v0.objectness.index_select],
                selected_head, meta)
            # get tp crops
            # get tn crops
            # train


if __name__ == '__main__':
    train_crop_classifier_v0()
