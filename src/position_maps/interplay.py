import os
import warnings
from typing import List

import albumentations as A
import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmdet.models.utils.gaussian_target import get_local_maximum
from omegaconf import ListConfig
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader

from log import get_logger
from models import TrajectoryModel
from patch_utils import quick_viz
from train import setup_single_video_dataset, setup_multiple_datasets, build_model, build_loss
from utils import heat_map_collate_fn, ImagePadder

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


class Track(object):
    def __init__(self, idx: int, frames: List[int], locations: List, inactive: int = 0):
        super(Track, self).__init__()
        self.idx = idx
        self.frames = frames
        self.locations = locations
        self.inactive = inactive

    def __eq__(self, other):
        return self.idx == other.idx

    def __repr__(self):
        return f"Track ID: {self.idx}" \
               f"\n{'Active' if self.inactive == 0 else ('Inactive since' + str(self.inactive) + 'frames')}" \
               f"\nFrames: {self.frames}" \
               f"\nTrack Positions: {self.locations}"


class Tracks(object):
    def __init__(self, tracks: List[Track]):
        self.tracks = tracks

    @classmethod
    def init_with_empty_tracks(cls):
        return Tracks([])


def locations_from_heatmaps(frames, kernel, loc_cutoff, marker_size, out, vis_on=False):
    out = [o.sigmoid() for o in out]
    pruned_locations = []
    loc_maxima_per_output = [get_local_maximum(o, kernel) for o in out]
    for li, loc_max_out in enumerate(loc_maxima_per_output):
        temp_locations = []
        for out_img_idx in range(loc_max_out.shape[0]):
            h_loc, w_loc = torch.where(loc_max_out[out_img_idx].squeeze(0) > loc_cutoff)
            loc = torch.stack((w_loc, h_loc)).t()

            temp_locations.append(loc)

            # viz
            if vis_on:
                plt.imshow(frames[out_img_idx].cpu().permute(1, 2, 0))
                plt.plot(w_loc, h_loc, 'o', markerfacecolor='r', markeredgecolor='k', markersize=marker_size)

                plt.title(f'Out - {li} - {out_img_idx}')
                plt.tight_layout()
                plt.show()

        pruned_locations.append(temp_locations)
    return pruned_locations


def get_position_correction_transform(new_shape):
    h, w = new_shape
    transform = A.Compose(
        [A.Resize(height=h, width=w)],
        keypoint_params=A.KeypointParams(format='xy')
    )
    return transform


def get_adjusted_object_locations(locations, heat_masks, meta):
    adjusted_locations, scaled_images = [], []
    for blobs, m, mask in zip(locations, meta, heat_masks):
        original_shape = m['original_shape']
        transform = get_position_correction_transform(original_shape)
        out = transform(image=mask.squeeze(0).numpy(), keypoints=blobs.numpy())
        adjusted_locations.append(out['keypoints'])
        scaled_images.append(out['image'])

    masks = np.stack(scaled_images)

    return adjusted_locations, masks


@hydra.main(config_path="config", config_name="config")
def interplay_v0(cfg):
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
    tp_model.to(cfg.device)
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
        # indices = np.random.choice(len(train_dataset), cfg.interplay_v0.subset_indices, replace=False)
        indices = np.arange(start=0, stop=cfg.interplay_v0.subset_indices)
    train_subset = Subset(dataset=train_dataset, indices=indices)

    train_loader = DataLoader(train_subset, batch_size=cfg.interplay_v0.batch_size, shuffle=cfg.interplay_v0.shuffle,
                              num_workers=cfg.interplay_v0.num_workers, collate_fn=heat_map_collate_fn,
                              pin_memory=cfg.interplay_v0.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    # training + logic

    # fow now
    position_model.freeze()

    track_ids_used = []
    current_track = 0
    active_tracks = Tracks.init_with_empty_tracks()
    inactive_tracks = Tracks.init_with_empty_tracks()

    for epoch in range(cfg.interplay_v0.num_epochs):
        tp_model_opt.zero_grad()
        tp_model.train()

        train_loss = []
        for t_idx, data in enumerate(train_loader):
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
                vis_on=True)
            selected_head = pred_position_maps[cfg.interplay_v0.objectness.index_select]
            pred_object_locations_scaled, heat_maps_gt_scaled = get_adjusted_object_locations(
                pred_object_locations[cfg.interplay_v0.objectness.index_select],
                selected_head, meta)
            if t_idx == 0:
                # init tracks
                for agent_pred_loc in pred_object_locations_scaled[0]:
                    track = Track(idx=current_track, frames=[frame_numbers[0]], locations=[agent_pred_loc])

                    active_tracks.tracks.append(track)
                    track_ids_used.append(current_track)
                    current_track += 1

                for b_idx in range(1, len(pred_object_locations_scaled)):
                    # get last frame locations
                    last_frame_locations = [t.locations[-1] for t in active_tracks.tracks]
                    # get current frame locations
                    current_frame_locations = [loc for loc in pred_object_locations_scaled[b_idx]]

                    last_frame_locations = np.stack(last_frame_locations)
                    current_frame_locations = np.stack(current_frame_locations)

                    distance_matrix = np.zeros((last_frame_locations.shape[0], current_frame_locations.shape[0]))
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
