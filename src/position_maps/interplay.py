import os
import warnings

import hydra
import numpy as np
import torch
from omegaconf import ListConfig
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader

from log import get_logger
from models import TrajectoryModel
from patch_utils import quick_viz
from train import setup_single_video_dataset, setup_multiple_datasets, build_model, build_loss
from utils import heat_map_collate_fn

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


class Track(object):
    def __init__(self, idx: int, frames: np.ndarray, locations: np.ndarray, inactive: int = 0):
        super(Track, self).__init__()
        self.idx = idx
        self.frames = frames
        self.locations = locations
        self.inactive = inactive

    def __eq__(self, other):
        return self.idx == other.idx

    def __repr__(self):
        return f"Track ID: {self.idx}" \
               f"\n{'Active' if self.inactive == 0 else ('Inactive since'+ str(self.inactive) + 'frames')}" \
               f"\nFrames: {self.frames}" \
               f"\nTrack Positions: {self.locations}"


@hydra.main(config_path="config", config_name="config")
def interplay_v0(cfg):
    logger.info(f'Setting up DataLoader and Model...')

    # adjust config here
    cfg.device = 'cuda:0'
    cfg.single_video_mode.enabled = True  # for now we work on single video

    if cfg.single_video_mode.enabled:
        # config adapt
        cfg.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
        cfg.single_video_mode.video_numbers_to_use = [[4]]
        cfg.desired_pixel_to_meter_ratio_rgb = 0.25
        cfg.desired_pixel_to_meter_ratio = 0.25

        train_dataset, val_dataset, target_max_shape = setup_single_video_dataset(cfg)
    else:
        train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    # loss config params are ok!
    loss_fn, gaussian_loss_fn = build_loss(cfg)

    # position map model config
    cfg.model = 'DeepLabV3Plus'
    position_model = build_model(cfg, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_fn,
                                 additional_loss_functions=gaussian_loss_fn, collate_fn=heat_map_collate_fn,
                                 desired_output_shape=target_max_shape)

    if cfg.interplay_v0.use_pretrained.enabled:
        checkpoint_path = f'{cfg.interplay_v0.use_pretrained.checkpoint.root}' \
                          f'{cfg.interplay_v0.use_pretrained.checkpoint.path}' \
                          f'{cfg.interplay_v0.use_pretrained.checkpoint.version}/checkpoints/'
        checkpoint_files = os.listdir(checkpoint_path)

        epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
        epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
        checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

        checkpoint_file = checkpoint_path + checkpoint_files[-cfg.interplay_v0.use_pretrained.checkpoint.top_k]

        logger.info(f'Loading weights from: {checkpoint_file}')
        load_dict = torch.load(checkpoint_file, map_location=cfg.device)

        position_model.load_state_dict(load_dict['state_dict'], strict=False)

    position_model.to(cfg.device)

    if cfg.interplay_v0.train_position_model:
        position_model.train()
    else:
        position_model.eval()

    position_model_opt = torch.optim.Adam(position_model.parameters(), lr=cfg.lr,
                                          weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)
    position_model_sch = ReduceLROnPlateau(position_model_opt,
                                           patience=cfg.patience,
                                           verbose=cfg.verbose,
                                           factor=cfg.factor,
                                           min_lr=cfg.min_lr)

    tp_model = TrajectoryModel(cfg)
    tp_model_opt = torch.optim.Adam(tp_model.parameters(), lr=cfg.tp_module.optimizer.lr,
                                    weight_decay=cfg.tp_module.optimizer.weight_decay,
                                    amsgrad=cfg.tp_module.optimizer.amsgrad)
    tp_model_sch = ReduceLROnPlateau(tp_model_opt,
                                     patience=cfg.tp_module.scheduler.patience,
                                     verbose=cfg.tp_module.scheduler.verbose,
                                     factor=cfg.tp_module.scheduler.factor,
                                     min_lr=cfg.tp_module.scheduler.min_lr)

    if isinstance(cfg.interplay_v0.subset_indices, (list, ListConfig)):
        indices = list(cfg.interplay_v0.subset_indices)
    else:
        indices = np.random.choice(len(train_dataset), cfg.interplay_v0.subset_indices, replace=False)
    train_subset = Subset(dataset=train_dataset, indices=indices)

    train_loader = DataLoader(train_subset, batch_size=cfg.interplay_v0.batch_size, shuffle=cfg.interplay_v0.shuffle,
                              num_workers=cfg.interplay_v0.num_workers, collate_fn=heat_map_collate_fn,
                              pin_memory=cfg.interplay_v0.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    # training + logic

    # fow now
    position_model.freeze()

    active_tracks = ...
    inactive_tracks = ...

    for epoch in range(cfg.interplay_v0.num_epochs):
        tp_model_opt.zero_grad()
        tp_model.train()

        train_loss = []
        for t_idx, data in enumerate(train_loader):
            # position_model_opt.zero_grad()

            frames, heat_masks, _, _, _, meta = data
            frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

            pred_position_maps = position_model(frames)

            noisy_gt_agents_count = ...

            pred_object_locations = ...
            if t_idx == 0:
                # init tracks
                pass
            else:
                # connect objects
                # get distance matrix
                # Hungarian matching
                # Associate tracks - active and inactive
                pass

            # if we have min history tracks
            # prepare for tp model input
            # inputs are
                # trajectories (encoded by lstm/transformer)
                # patch embeddings? - bring other agents active in consideration

            pred_trajectory_out = ...

            # make heatmaps out of pred agent locations + pred_trajectory_out
            # GMM likelihood based loss (or MSE loss(but it will be independent to each agent))

            # [Advanced version]
            # Classify each agent location in pred agent locations + pred trajectory locations
            # If an agent -> prepare labels for trajectory confidence loss at each keyframe

            # Trajectory predicted with high confidence -> no agent found -> penalise

            # backpropogate trajectory loss

            # If a trajectory is predicted with high confidence -> there must be an object
            # augment that location with a gaussian as TP

            # backpropogate position map loss


if __name__ == '__main__':
    interplay_v0()
