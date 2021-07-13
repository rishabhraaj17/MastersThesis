import copy
import os
import warnings
from typing import List

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
import motmetrics as mm
from omegaconf import ListConfig
from pytorch_lightning import seed_everything
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from baselinev2.plot_utils import add_features_to_axis
from log import get_logger
from models import TrajectoryModel
from src.position_maps.location_utils import locations_from_heatmaps, get_adjusted_object_locations
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

    first_frame = None

    track_ids_used = []
    current_track = 0

    # # using MOTAccumulator to verify ease - matches gt with hypothesis - maybe use in some other scenario
    # track_accumulator = mm.MOTAccumulator(auto_id=True)

    for epoch in range(cfg.interplay_v0.num_epochs):
        active_tracks = Tracks.init_with_empty_tracks()
        inactive_tracks = Tracks.init_with_empty_tracks()

        tp_model_opt.zero_grad()
        tp_model.train()

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
            if t_idx == 0:
                # store first frame to viz
                first_frame = interpolate(frames[0, None, ...], size=meta[0]['original_shape'])

                # init tracks
                for agent_pred_loc in pred_object_locations_scaled[0]:
                    agent_pred_loc = list(agent_pred_loc)
                    track = Track(idx=current_track, frames=[frame_numbers[0]], locations=[agent_pred_loc])

                    active_tracks.tracks.append(track)
                    track_ids_used.append(current_track)
                    current_track += 1

                construct_tracks(active_tracks, frame_numbers, inactive_tracks, pred_object_locations_scaled,
                                 batch_start_idx=1)
            else:
                # get distance matrix
                # connect objects
                # Hungarian matching
                # Associate tracks - active and inactive
                construct_tracks(active_tracks, frame_numbers, inactive_tracks, pred_object_locations_scaled)

            viz_tracks(active_tracks, first_frame)
            # if we have min history tracks
            # prepare for tp model input

            trajectory_xy, trajectory_dxdy = [], []
            if all([len(t.locations) >= cfg.interplay_v0.min_history + cfg.interplay_v0.batch_size
                    for t in active_tracks.tracks]):
                # pad trajectories if they are less than expected
                length_per_trajectory = [len(t.locations) for t in active_tracks.tracks]
                for lpt, t in zip(length_per_trajectory, active_tracks.tracks):
                    if lpt < cfg.interplay_v0.in_trajectory_length + cfg.interplay_v0.batch_size:
                        to_pad_location = [list(t.locations[0])]
                        pad_count = cfg.interplay_v0.in_trajectory_length - lpt
                        to_pad_location = to_pad_location * pad_count
                        traj = to_pad_location + t.locations

                        trajectory_xy.append(traj)
                    else:
                        trajectory_xy.append(t.locations)

                # prepare trajectories
                for t_xy in trajectory_xy:
                    temp_dxdy = []
                    for xy in range(1, len(t_xy)):
                        temp_dxdy.append((np.array(t_xy[xy]) - np.array(t_xy[xy - 1])).tolist())
                    trajectory_dxdy.append(temp_dxdy)

                trajectory_xy = torch.from_numpy(np.stack(trajectory_xy))
                trajectory_dxdy = torch.from_numpy(np.stack(trajectory_dxdy))

                # old batching when it was min-length * batch_size - takes all available history
                # good to use all history if available! :)
                # in_trajectory_xy = trajectory_xy[:, :-cfg.interplay_v0.batch_size, ...]
                # in_trajectory_dxdy = trajectory_dxdy[:, :-cfg.interplay_v0.batch_size, ...]
                #
                # out_trajectory_xy = trajectory_xy[:, -cfg.interplay_v0.batch_size:, ...]
                # out_trajectory_dxdy = trajectory_dxdy[:, -cfg.interplay_v0.batch_size:, ...]

                # fixed last min_history is taken
                in_trajectory_xy = trajectory_xy[:,
                                   -(cfg.interplay_v0.min_history + cfg.interplay_v0.batch_size):
                                   -cfg.interplay_v0.batch_size, ...]
                in_trajectory_dxdy = trajectory_dxdy[:,
                                   -(cfg.interplay_v0.min_history + cfg.interplay_v0.batch_size):
                                   -cfg.interplay_v0.batch_size, ...]

                out_trajectory_xy = trajectory_xy[:, -cfg.interplay_v0.batch_size:, ...]
                out_trajectory_dxdy = trajectory_dxdy[:, -cfg.interplay_v0.batch_size:, ...]

                # vis trajectory division
                viz_divided_trajectories_together(first_frame, in_trajectory_xy, out_trajectory_xy)
                print()
            print()
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


def viz_divided_trajectories_together(first_frame, in_trajectory_xy, out_trajectory_xy, show=True):
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    ax.imshow(first_frame.squeeze().permute(1, 2, 0))
    for in_t_xy in in_trajectory_xy:
        add_features_to_axis(ax, in_t_xy, marker_size=3, marker_color='r')
    for out_t_xy in out_trajectory_xy:
        add_features_to_axis(ax, out_t_xy, marker_size=3, marker_color='g')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def construct_tracks(active_tracks, frame_numbers, inactive_tracks, pred_object_locations_scaled, batch_start_idx=0):
    for b_idx in range(batch_start_idx, len(pred_object_locations_scaled)):
        # get last frame locations
        last_frame_locations = [t.locations[-1] for t in active_tracks.tracks]
        # get current frame locations
        current_frame_locations = [loc for loc in pred_object_locations_scaled[b_idx]]

        last_frame_locations = np.stack(last_frame_locations)
        current_frame_locations = np.stack(current_frame_locations)

        # try setting max dist to a reasonable number so that matches are reasonable within a distance
        distance_matrix = mm.distances.norm2squared_matrix(last_frame_locations, current_frame_locations)

        agent_associations = mm.lap.lsa_solve_scipy(distance_matrix)
        match_rows, match_cols = agent_associations
        # track_accumulator.update(np.arange(last_frame_locations.shape[0]),
        #                          np.arange(current_frame_locations.shape[0]), distance_matrix)

        # Hungarian
        # match_rows, match_cols = linear_sum_assignment(distance_matrix)

        rows_to_columns_association = {r: c for r, c in zip(match_rows, match_cols)}
        # track_ids to associations
        last_frame_track_id_to_association = {}
        for m_r in match_rows:
            last_frame_track_id_to_association[active_tracks.tracks[m_r].idx] = m_r

        # filter active tracks and extend tracks
        currently_active_tracks = []
        for track in active_tracks.tracks:
            if track.idx in last_frame_track_id_to_association.keys():
                track.frames.append(frame_numbers[b_idx])

                loc_idx = rows_to_columns_association[last_frame_track_id_to_association[track.idx]]
                loc = current_frame_locations[loc_idx]
                track.locations.append(loc.tolist())

                currently_active_tracks.append(track)
            else:
                track.inactive += 1
                inactive_tracks.tracks.append(track)

        active_tracks.tracks = copy.deepcopy(currently_active_tracks)


def viz_tracks(active_tracks, first_frame, show=True):
    active_tracks_to_vis = np.stack([t.locations for t in active_tracks.tracks])
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    ax.imshow(first_frame.squeeze().permute(1, 2, 0))
    for a_t in active_tracks_to_vis:
        add_features_to_axis(ax=ax, features=a_t, marker_size=1, marker_color='g')
        # add_line_to_axis(ax=ax, features=a_t)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    interplay_v0()