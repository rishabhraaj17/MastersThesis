import copy
import os
import warnings
from pathlib import Path

import hydra
import motmetrics as mm
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import ListConfig
from pytorch_lightning import seed_everything
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from baselinev2.exceptions import TimeoutException
from baselinev2.plot_utils import add_features_to_axis, add_line_to_axis
from interplay_utils import setup_multiple_frame_only_datasets_core, frames_only_collate_fn, Track, Tracks
from location_utils import locations_from_heatmaps, get_adjusted_object_locations, \
    prune_locations_proximity_based, ExtractedLocations, Locations, Location
from log import get_logger
from models import TrajectoryModel
from src.position_maps.analysis import VIDEO_TO_PRUNE_RADIUS_MAPPING
from trajectory_utils import viz_raw_tracks_from_active_inactive
from train import setup_single_video_dataset, setup_multiple_datasets, build_model, build_loss
from utils import heat_map_collate_fn, ImagePadder, get_scaled_shapes_with_pad_values

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


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


# does not start a new track :(
def construct_tracks_v0(active_tracks, frame_numbers, inactive_tracks, pred_object_locations_scaled, batch_start_idx=0):
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


# active_tracks -> all tracks - add new tracks each frame
# active_tracks_selective -> all tracks - add only tracks that makes sense each frame
def construct_tracks(active_tracks, frame_numbers, inactive_tracks, pred_object_locations_scaled,
                     active_tracks_selective, inactive_tracks_selective, do_selective_track_association,
                     track_ids_used, current_track, track_ids_used_selective, current_track_selective,
                     init_track_each_frame=True, batch_start_idx=0):
    current_track_local = current_track
    current_track_selective_local = current_track_selective

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

        # find agents in the current frame (potential new agents in the scene)
        unmatched_agents_current_frame = np.setdiff1d(np.arange(current_frame_locations.shape[0]), match_cols)

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

        # add new potential agents as new track
        if init_track_each_frame:
            for u in unmatched_agents_current_frame:
                init_track = Track(idx=current_track_local, frames=[frame_numbers[b_idx]],
                                   locations=[current_frame_locations[u].tolist()])

                currently_active_tracks.append(init_track)
                track_ids_used.append(current_track_local)
                current_track_local += 1

        # update active tracks
        active_tracks.tracks = copy.deepcopy(currently_active_tracks)

    return current_track_local, current_track_selective_local


def viz_tracks(active_tracks, first_frame, show=True, use_lines=False):
    # active_tracks_to_vis = np.stack([t.locations for t in active_tracks.tracks])
    active_tracks_to_vis = [t.locations for t in active_tracks.tracks]
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    if isinstance(first_frame, torch.Tensor):
        first_frame = first_frame.squeeze().permute(1, 2, 0)
    ax.imshow(first_frame)
    for a_t in active_tracks_to_vis:
        a_t = np.stack(a_t)
        if use_lines:
            add_line_to_axis(ax=ax, features=a_t, marker_size=1, marker_color='g')
        else:
            add_features_to_axis(ax=ax, features=a_t, marker_size=1, marker_color='g')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def setup_frame_only_dataset(cfg):
    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.single_video_mode.video_classes_to_use,
        video_numbers=cfg.single_video_mode.video_numbers_to_use,
        desired_ratio=cfg.desired_pixel_to_meter_ratio_rgb)
    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.single_video_mode.video_classes_to_use,
        video_numbers=cfg.single_video_mode.video_numbers_to_use,
        desired_ratio=cfg.desired_pixel_to_meter_ratio)
    train_dataset = setup_multiple_frame_only_datasets_core(
        cfg=cfg, video_classes_to_use=cfg.single_video_mode.video_classes_to_use,
        video_numbers_to_use=cfg.single_video_mode.video_numbers_to_use,
        num_videos=-1, multiple_videos=False, df=df, df_target=df_target, use_common_transforms=False)
    return train_dataset


def setup_frame_only_dataset_flexible(cfg, video_class, video_number):
    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=video_class,
        video_numbers=video_number,
        desired_ratio=cfg.desired_pixel_to_meter_ratio_rgb)
    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=video_class,
        video_numbers=video_number,
        desired_ratio=cfg.desired_pixel_to_meter_ratio)
    train_dataset = setup_multiple_frame_only_datasets_core(
        cfg=cfg, video_classes_to_use=video_class,
        video_numbers_to_use=video_number,
        num_videos=-1, multiple_videos=False, df=df, df_target=df_target, use_common_transforms=False)
    return train_dataset


@hydra.main(config_path="config", config_name="config")
def extract_trajectories(cfg):
    init_track_each_frame = True

    do_selective_track_association = False  # we don't need it
    offline = False

    logger.info(f'Extract Trajectories...')
    logger.info(f'Setting up DataLoader and Model...')

    # adjust config here
    cfg.device = 'cpu'  # 'cuda:0'
    cfg.single_video_mode.enabled = True  # for now we work on single video
    cfg.preproccesing.pad_factor = 8
    cfg.frame_rate = 30.
    cfg.video_based.enabled = False

    if cfg.single_video_mode.enabled:
        # config adapt
        cfg.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
        cfg.single_video_mode.video_numbers_to_use = [[2]]
        cfg.desired_pixel_to_meter_ratio_rgb = 0.07
        cfg.desired_pixel_to_meter_ratio = 0.07

        train_dataset = setup_frame_only_dataset(cfg)
        val_dataset = None

        # train_dataset, val_dataset, target_max_shape = setup_single_video_dataset(cfg,
        #                                                                           use_common_transforms=False,
        #                                                                           without_split=True)
    else:
        train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    if not offline:
        # loss config params are ok!
        loss_fn, gaussian_loss_fn = build_loss(cfg)

        # position map model config
        cfg.model = 'DeepLabV3Plus'
        position_model = build_model(cfg, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_fn,
                                     additional_loss_functions=gaussian_loss_fn, collate_fn=heat_map_collate_fn,
                                     desired_output_shape=None)

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
        position_model.eval()
    else:
        load_path = os.path.join(os.getcwd(),
                                 f'HeatMapPredictions'
                                 f'/{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).name}'
                                 f'/{cfg.single_video_mode.video_numbers_to_use[0][0]}/predictions.pt')
        position_model = torch.load(load_path)  # mock the model
        out_head_0, out_head_1, out_head_2 = \
            position_model['out_head_0'], position_model['out_head_1'], position_model['out_head_2']
        frames_sequence = position_model['frames_sequence']

    train_loader = DataLoader(train_dataset, batch_size=cfg.interplay_v0.batch_size, shuffle=False,
                              num_workers=cfg.interplay_v0.num_workers, collate_fn=frames_only_collate_fn,
                              pin_memory=False, drop_last=cfg.interplay_v0.drop_last)

    # training + logic
    track_ids_used = []
    current_track = 0

    track_ids_used_selective = []
    current_track_selective = 0

    active_tracks_selective = Tracks.init_with_empty_tracks()
    inactive_tracks_selective = Tracks.init_with_empty_tracks()

    active_tracks = Tracks.init_with_empty_tracks()
    inactive_tracks = Tracks.init_with_empty_tracks()

    # first_frame = None
    train_loss = []
    pred_t_idx = 0
    for t_idx, data in enumerate(tqdm(train_loader)):
        frames, meta = data

        padder = ImagePadder(frames.shape[-2:], factor=cfg.preproccesing.pad_factor)
        frames = padder.pad(frames)[0]
        frames = frames.to(cfg.device)

        if not offline:
            with torch.no_grad():
                pred_position_maps = position_model(frames)
        else:
            pred_position_maps = [
                out_head_0[pred_t_idx: pred_t_idx + cfg.interplay_v0.batch_size, ...],
                out_head_1[pred_t_idx: pred_t_idx + cfg.interplay_v0.batch_size, ...],
                out_head_2[pred_t_idx: pred_t_idx + cfg.interplay_v0.batch_size, ...],
            ]
            frames_seq_stored = frames_sequence[pred_t_idx: pred_t_idx + cfg.interplay_v0.batch_size]

        frame_numbers = [m['item'] for m in meta]

        pred_object_locations = locations_from_heatmaps(
            frames, cfg.interplay_v0.objectness.kernel,
            cfg.interplay_v0.objectness.loc_cutoff,
            cfg.interplay_v0.objectness.marker_size, pred_position_maps,
            vis_on=False)

        # filter out overlapping locations
        selected_locations_pre_pruning = pred_object_locations[cfg.interplay_v0.objectness.index_select]
        selected_locations = []
        for s_loc in selected_locations_pre_pruning:
            pruned_locations, pruned_locations_idx = prune_locations_proximity_based(
                s_loc.numpy(), cfg.eval.objectness.prune_radius)
            selected_locations.append(torch.from_numpy(pruned_locations))

        selected_head = pred_position_maps[cfg.interplay_v0.objectness.index_select]
        pred_object_locations_scaled, heat_maps_gt_scaled = get_adjusted_object_locations(
            selected_locations, selected_head, meta)

        if t_idx == 0:
            # store first frame to viz
            # first_frame = interpolate(frames[0, None, ...], size=meta[0]['original_shape'])

            # init tracks
            for agent_pred_loc in pred_object_locations_scaled[0]:
                agent_pred_loc = list(agent_pred_loc)
                track = Track(idx=current_track, frames=[frame_numbers[0]], locations=[agent_pred_loc])

                active_tracks.tracks.append(track)
                track_ids_used.append(current_track)
                current_track += 1

                if do_selective_track_association:
                    active_tracks_selective.tracks.append(
                        Track(idx=current_track_selective, frames=[frame_numbers[0]], locations=[agent_pred_loc]))
                    track_ids_used_selective.append(current_track_selective)
                    current_track_selective += 1

            current_track, current_track_selective = construct_tracks(
                active_tracks=active_tracks, frame_numbers=frame_numbers, inactive_tracks=inactive_tracks,
                pred_object_locations_scaled=pred_object_locations_scaled,
                active_tracks_selective=active_tracks_selective, inactive_tracks_selective=inactive_tracks_selective,
                do_selective_track_association=do_selective_track_association, track_ids_used=track_ids_used,
                current_track=current_track, track_ids_used_selective=track_ids_used_selective,
                current_track_selective=current_track_selective, init_track_each_frame=init_track_each_frame,
                batch_start_idx=1)
        else:
            current_track, current_track_selective = construct_tracks(
                active_tracks=active_tracks, frame_numbers=frame_numbers, inactive_tracks=inactive_tracks,
                pred_object_locations_scaled=pred_object_locations_scaled,
                active_tracks_selective=active_tracks_selective, inactive_tracks_selective=inactive_tracks_selective,
                do_selective_track_association=do_selective_track_association, track_ids_used=track_ids_used,
                current_track=current_track, track_ids_used_selective=track_ids_used_selective,
                current_track_selective=current_track_selective, init_track_each_frame=init_track_each_frame,
                batch_start_idx=0)

        viz_tracks(active_tracks, interpolate(frames[0, None, ...], size=meta[0]['original_shape']), show=False,
                   use_lines=True)

        if do_selective_track_association:
            viz_tracks(active_tracks_selective,
                       interpolate(frames[0, None, ...], size=meta[0]['original_shape']), show=False)

        pred_t_idx += cfg.interplay_v0.batch_size
    # save extracted trajectories
    if do_selective_track_association:
        save_dict = {
            'track_ids_used': track_ids_used,
            'active': active_tracks,
            'inactive': inactive_tracks,
            'track_ids_used_selective': track_ids_used_selective,
            'active_selective': active_tracks_selective,
            'inactive_selective': inactive_tracks_selective,
        }
        filename = 'extracted_trajectories_with_selective.pt'
    else:
        save_dict = {
            'track_ids_used': track_ids_used,
            'active': active_tracks,
            'inactive': inactive_tracks,
        }
        filename = 'extracted_trajectories.pt'
    save_path = os.path.join(os.getcwd(),
                             f'ExtractedTrajectories'
                             f'/{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).name}'
                             f'/{cfg.single_video_mode.video_numbers_to_use[0][0]}/')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path + filename)
    logger.info(f"Saved trajectories at {save_path}{filename}")


@hydra.main(config_path="config", config_name="config")
def extract_trajectories_resumable(cfg):
    init_track_each_frame = True

    do_selective_track_association = False  # we don't need it
    offline = False

    logger.info(f'Resuming Extract Trajectories...')
    logger.info(f'Setting up DataLoader and Model...')

    # adjust config here
    cfg.device = 'cpu'  # 'cuda:0'
    cfg.single_video_mode.enabled = True  # for now we work on single video
    cfg.preproccesing.pad_factor = 8
    cfg.frame_rate = 30.
    cfg.video_based.enabled = False

    if cfg.single_video_mode.enabled:
        # config adapt
        cfg.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
        cfg.single_video_mode.video_numbers_to_use = [[4]]
        cfg.desired_pixel_to_meter_ratio_rgb = 0.07
        cfg.desired_pixel_to_meter_ratio = 0.07

        train_dataset, val_dataset, target_max_shape = setup_single_video_dataset(cfg,
                                                                                  use_common_transforms=False,
                                                                                  without_split=False)
    else:
        train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    if not offline:
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
        position_model.eval()
    else:
        load_path = os.path.join(os.getcwd(),
                                 f'HeatMapPredictions'
                                 f'/{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).name}'
                                 f'/{cfg.single_video_mode.video_numbers_to_use[0][0]}/predictions.pt')
        position_model = torch.load(load_path)  # mock the model
        out_head_0, out_head_1, out_head_2 = \
            position_model['out_head_0'], position_model['out_head_1'], position_model['out_head_2']
        frames_sequence = position_model['frames_sequence']

    train_loader = DataLoader(val_dataset, batch_size=cfg.interplay_v0.batch_size, shuffle=cfg.interplay_v0.shuffle,
                              num_workers=cfg.interplay_v0.num_workers, collate_fn=heat_map_collate_fn,
                              pin_memory=cfg.interplay_v0.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    # training + logic
    track_ids_used_selective = []
    current_track_selective = 0

    active_tracks_selective = Tracks.init_with_empty_tracks()
    inactive_tracks_selective = Tracks.init_with_empty_tracks()

    load_pth = os.path.join(os.getcwd(),
                            f'ExtractedTrajectories'
                            f'/{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).name}'
                            f'/{cfg.single_video_mode.video_numbers_to_use[0][0]}/extracted_trajectories.pt')
    loaded_tracks = torch.load(load_pth)

    track_ids_used = loaded_tracks['track_ids_used']
    logger.info(f'Resuming Extract Trajectories from {track_ids_used[-1]}')

    current_track = track_ids_used[-1] + 1

    active_tracks = loaded_tracks['active']
    inactive_tracks = loaded_tracks['inactive']

    # first_frame = None
    train_loss = []
    pred_t_idx = 0
    for t_idx, data in enumerate(tqdm(train_loader)):
        frames, heat_masks, _, _, _, meta = data

        padder = ImagePadder(frames.shape[-2:], factor=cfg.preproccesing.pad_factor)
        frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]
        frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

        if not offline:
            with torch.no_grad():
                pred_position_maps = position_model(frames)
        else:
            pred_position_maps = [
                out_head_0[pred_t_idx: pred_t_idx + cfg.interplay_v0.batch_size, ...],
                out_head_1[pred_t_idx: pred_t_idx + cfg.interplay_v0.batch_size, ...],
                out_head_2[pred_t_idx: pred_t_idx + cfg.interplay_v0.batch_size, ...],
            ]
            frames_seq_stored = frames_sequence[pred_t_idx: pred_t_idx + cfg.interplay_v0.batch_size]

        noisy_gt_agents_count = [m['bbox_centers'].shape[0] for m in meta]
        frame_numbers = [m['item'] for m in meta]

        pred_object_locations = locations_from_heatmaps(
            frames, cfg.interplay_v0.objectness.kernel,
            cfg.interplay_v0.objectness.loc_cutoff,
            cfg.interplay_v0.objectness.marker_size, pred_position_maps,
            vis_on=False)

        # filter out overlapping locations
        selected_locations_pre_pruning = pred_object_locations[cfg.interplay_v0.objectness.index_select]
        selected_locations = []
        for s_loc in selected_locations_pre_pruning:
            pruned_locations, pruned_locations_idx = prune_locations_proximity_based(
                s_loc.numpy(), cfg.eval.objectness.prune_radius)
            selected_locations.append(torch.from_numpy(pruned_locations))

        selected_head = pred_position_maps[cfg.interplay_v0.objectness.index_select]
        pred_object_locations_scaled, heat_maps_gt_scaled = get_adjusted_object_locations(
            selected_locations, selected_head, meta)

        current_track, current_track_selective = construct_tracks(
            active_tracks=active_tracks, frame_numbers=frame_numbers, inactive_tracks=inactive_tracks,
            pred_object_locations_scaled=pred_object_locations_scaled,
            active_tracks_selective=active_tracks_selective, inactive_tracks_selective=inactive_tracks_selective,
            do_selective_track_association=do_selective_track_association, track_ids_used=track_ids_used,
            current_track=current_track, track_ids_used_selective=track_ids_used_selective,
            current_track_selective=current_track_selective, init_track_each_frame=init_track_each_frame,
            batch_start_idx=0)

        viz_tracks(active_tracks, interpolate(frames[0, None, ...], size=meta[0]['original_shape']), show=False)

        if do_selective_track_association:
            viz_tracks(active_tracks_selective,
                       interpolate(frames[0, None, ...], size=meta[0]['original_shape']), show=False)

        pred_t_idx += cfg.interplay_v0.batch_size
    # save extracted trajectories
    if do_selective_track_association:
        save_dict = {
            'track_ids_used': track_ids_used,
            'active': active_tracks,
            'inactive': inactive_tracks,
            'track_ids_used_selective': track_ids_used_selective,
            'active_selective': active_tracks_selective,
            'inactive_selective': inactive_tracks_selective,
        }
        filename = 'extracted_trajectories_with_selective_val_extended.pt'
    else:
        save_dict = {
            'track_ids_used': track_ids_used,
            'active': active_tracks,
            'inactive': inactive_tracks,
        }
        filename = 'extracted_trajectories_val_extended.pt'
    save_path = os.path.join(os.getcwd(),
                             f'ExtractedTrajectories'
                             f'/{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).name}'
                             f'/{cfg.single_video_mode.video_numbers_to_use[0][0]}/')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path + filename)
    logger.info(f"Saved Predictions at {save_path}{filename}")


def construct_tracks_from_locations(
        active_tracks, frame_number, inactive_tracks, current_frame_locations,
        track_ids_used, current_track, init_track_each_frame=True, max_distance=float('inf')):
    current_track_local = current_track

    # get last frame locations
    last_frame_locations = np.stack([t.locations[-1] for t in active_tracks.tracks])

    # try setting max dist to a reasonable number so that matches are reasonable within a distance
    distance_matrix = mm.distances.norm2squared_matrix(
        last_frame_locations, current_frame_locations, max_d2=max_distance)

    agent_associations = mm.lap.lsa_solve_scipy(distance_matrix)
    match_rows, match_cols = agent_associations

    # find agents in the current frame (potential new agents in the scene)
    unmatched_agents_current_frame = np.setdiff1d(np.arange(current_frame_locations.shape[0]), match_cols)

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
            track.frames.append(frame_number)

            loc_idx = rows_to_columns_association[last_frame_track_id_to_association[track.idx]]
            loc = current_frame_locations[loc_idx]
            track.locations.append(loc.tolist())

            currently_active_tracks.append(track)
        else:
            track.inactive += 1
            inactive_tracks.tracks.append(track)

    # add new potential agents as new track
    if init_track_each_frame:
        for u in unmatched_agents_current_frame:
            init_track = Track(idx=current_track_local, frames=[frame_number],
                               locations=[current_frame_locations[u].tolist()])

            currently_active_tracks.append(init_track)
            track_ids_used.append(current_track_local)
            current_track_local += 1

    # update active tracks
    active_tracks.tracks = copy.deepcopy(currently_active_tracks)

    return current_track_local


def extract_trajectories_from_locations_core(cfg, enable_forward_pass, out_head, init_track_each_frame,
                                             location_version_to_use='pruned_scaled', max_distance=float('inf')):
    if cfg.single_video_mode.enabled:
        train_dataset = setup_frame_only_dataset(cfg)
        val_dataset = None
    else:
        return NotImplemented
        # train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    if enable_forward_pass:
        # position map model config
        cfg.model = 'DeepLabV3Plus'
        position_model = build_model(cfg, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=None,
                                     additional_loss_functions=None, collate_fn=heat_map_collate_fn,
                                     desired_output_shape=None)

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
        position_model.eval()

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                              num_workers=cfg.interplay_v0.num_workers, collate_fn=frames_only_collate_fn,
                              pin_memory=False, drop_last=cfg.interplay_v0.drop_last)

    locations = out_head

    track_ids_used = []
    current_track = 0

    active_tracks = Tracks.init_with_empty_tracks()
    inactive_tracks = Tracks.init_with_empty_tracks()

    for t_idx, (data, location) in enumerate(
            tqdm(zip(train_loader, locations.locations), total=len(locations.locations))):
        frames, meta = data

        padder = ImagePadder(frames.shape[-2:], factor=cfg.preproccesing.pad_factor)
        frames = padder.pad(frames)[0]
        frames = frames.to(cfg.device)

        if enable_forward_pass:
            with torch.no_grad():
                pred_position_maps = position_model(frames)

        if location_version_to_use == 'default':
            locations_to_use = location.locations
        elif location_version_to_use == 'pruned':
            locations_to_use = location.pruned_locations
        elif location_version_to_use == 'pruned_scaled':
            locations_to_use = location.scaled_locations
        else:
            raise NotImplementedError

        if t_idx == 0:
            # init tracks
            for agent_pred_loc in locations_to_use:
                agent_pred_loc = list(agent_pred_loc)
                track = Track(idx=current_track, frames=[location.frame_number], locations=[agent_pred_loc])

                active_tracks.tracks.append(track)
                track_ids_used.append(current_track)
                current_track += 1
        else:
            current_track = construct_tracks_from_locations(
                active_tracks=active_tracks, frame_number=location.frame_number, inactive_tracks=inactive_tracks,
                current_frame_locations=locations_to_use,
                track_ids_used=track_ids_used,
                current_track=current_track, init_track_each_frame=init_track_each_frame,
                max_distance=max_distance)

        viz_tracks(active_tracks, interpolate(frames[0, None, ...], size=meta[0]['original_shape']), show=True,
                   use_lines=False)
    return active_tracks, inactive_tracks, track_ids_used


def extract_trajectories_from_locations_core_minimal(
        locations, init_track_each_frame, video_path, prune_radius, padded_shape, original_shape,
        location_version_to_use='pruned_scaled', max_distance=float('inf')):
    track_ids_used = []
    current_track = 0

    active_tracks = Tracks.init_with_empty_tracks()
    inactive_tracks = Tracks.init_with_empty_tracks()

    for t_idx, location in enumerate(tqdm(locations.locations)):
        if location_version_to_use == 'default':
            locations_to_use = location.locations
        elif location_version_to_use == 'pruned':
            locations_to_use = location.pruned_locations
        elif location_version_to_use == 'pruned_scaled':
            locations_to_use = location.scaled_locations
        elif location_version_to_use == 'runtime_pruned_scaled':
            # filter out overlapping locations
            try:
                pruned_locations, pruned_locations_idx = prune_locations_proximity_based(
                        location.locations, prune_radius)
                pruned_locations = torch.from_numpy(pruned_locations)
            except TimeoutException:
                pruned_locations = torch.from_numpy(location.locations)

            fake_padded_heatmaps = torch.zeros(size=(1, 1, padded_shape[0], padded_shape[1]))
            pred_object_locations_scaled, _ = get_adjusted_object_locations(
                [pruned_locations], fake_padded_heatmaps, [{'original_shape': original_shape}])
            locations_to_use = np.stack(pred_object_locations_scaled).squeeze() \
                if len(pred_object_locations_scaled) != 0 else np.zeros((0, 2))
        else:
            raise NotImplementedError

        if t_idx == 0:
            # init tracks
            current_track = init_tracks_from_empty(active_tracks, current_track, location, locations_to_use,
                                                   track_ids_used)
        else:
            if len(active_tracks.tracks) == 0 and len(locations_to_use) != 0:
                current_track = init_tracks_from_empty(active_tracks, current_track, location, locations_to_use,
                                                       track_ids_used)
            elif len(active_tracks.tracks) == 0 and len(locations_to_use) == 0:
                continue
            else:
                if locations_to_use.ndim == 1:
                    locations_to_use = locations_to_use[None, ...]
                current_track = construct_tracks_from_locations(
                    active_tracks=active_tracks, frame_number=location.frame_number, inactive_tracks=inactive_tracks,
                    current_frame_locations=locations_to_use,
                    track_ids_used=track_ids_used,
                    current_track=current_track, init_track_each_frame=init_track_each_frame,
                    max_distance=max_distance)

        # viz_tracks(active_tracks, extract_frame_from_video(video_path, location.frame_number), show=True,
        #            use_lines=False)
    # viz_raw_tracks_from_active_inactive(active_tracks, inactive_tracks, 'deathCircle', 3, use_lines=True,
    #                                     marker_size=2, plot_with_last_frame=True)
    return active_tracks, inactive_tracks, track_ids_used


def init_tracks_from_empty(active_tracks, current_track, location, locations_to_use, track_ids_used):
    if locations_to_use.ndim == 1:
        locations_to_use = locations_to_use[None, :]
    for agent_pred_loc in locations_to_use:
        agent_pred_loc = list(agent_pred_loc)
        track = Track(idx=current_track, frames=[location.frame_number], locations=[agent_pred_loc])

        active_tracks.tracks.append(track)
        track_ids_used.append(current_track)
        current_track += 1
    return current_track


@hydra.main(config_path="config", config_name="config")
def extract_trajectories_from_locations(cfg):
    use_minimal_version = True

    location_version_to_use = 'runtime_pruned_scaled'  # 'pruned_scaled' 'runtime_pruned_scaled'
    head_to_use = 0
    # 50 - as small we go more trajectories but shorter trajectories
    max_matching_euclidean_distance = 500.  # 1000 ~ 500. > 200. looks good

    prune_radius = 40  # dc3

    init_track_each_frame = True
    enable_forward_pass = False

    logger.info(f'Extract trajectories from locations...')

    # adjust config here
    cfg.device = 'cpu'  # 'cuda:0'
    cfg.single_video_mode.enabled = True  # for now we work on single video
    cfg.preproccesing.pad_factor = 8
    cfg.frame_rate = 30.
    cfg.video_based.enabled = False

    cfg.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
    cfg.single_video_mode.video_numbers_to_use = [[3]]
    cfg.desired_pixel_to_meter_ratio_rgb = 0.07
    cfg.desired_pixel_to_meter_ratio = 0.07

    logger.info(f'Dataset: {cfg.single_video_mode.video_classes_to_use} | {cfg.single_video_mode.video_numbers_to_use}')
    # Positions are on original scale

    # load_locations
    load_path = os.path.join(os.getcwd(),
                             f'ExtractedLocations'
                             f'/{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).name}'
                             f'/{cfg.single_video_mode.video_numbers_to_use[0][0]}/extracted_locations.pt')
    extracted_locations: ExtractedLocations = torch.load(load_path)['locations']  # mock the model
    out_head_0, out_head_1, out_head_2 = extracted_locations.head0, extracted_locations.head1, extracted_locations.head2

    uq, uc = np.unique([i.frame_number for i in out_head_0.locations], return_counts=True)
    uc_gt = uc > 1
    repeating_frames = uq[uc_gt]

    if head_to_use == 0:
        out_head = out_head_0
    elif head_to_use == 1:
        out_head = out_head_1
    elif head_to_use == 2:
        out_head = out_head_2
    else:
        raise NotImplementedError

    if use_minimal_version:
        video_path = f"{cfg.root}videos/" \
                     f"{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).value}/" \
                     f"video{cfg.single_video_mode.video_numbers_to_use[0][0]}/video.mov"
        active_tracks, inactive_tracks, track_ids_used = extract_trajectories_from_locations_core_minimal(
            locations=out_head, init_track_each_frame=init_track_each_frame,
            video_path=video_path, prune_radius=prune_radius, padded_shape=extracted_locations.padded_shape,
            original_shape=extracted_locations.scaled_shape,
            location_version_to_use=location_version_to_use, max_distance=max_matching_euclidean_distance)
    else:
        logger.info(f'Setting up DataLoader and Model...')
        active_tracks, inactive_tracks, track_ids_used = extract_trajectories_from_locations_core(
            cfg, enable_forward_pass, out_head=out_head, init_track_each_frame=init_track_each_frame,
            location_version_to_use=location_version_to_use, max_distance=max_matching_euclidean_distance)

    # save extracted trajectories
    save_dict = {
        'track_ids_used': track_ids_used,
        'active': active_tracks,
        'inactive': inactive_tracks,
        'repeating_frames': repeating_frames
    }
    filename = 'extracted_trajectories.pt'
    save_path = os.path.join(os.getcwd(),
                             f'ExtractedTrajectories'
                             f'/{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).name}'
                             f'/{cfg.single_video_mode.video_numbers_to_use[0][0]}/')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path + filename)
    logger.info(f"Saved trajectories at {save_path}{filename}")


@hydra.main(config_path="config", config_name="config")
def extract_trajectories_from_locations_multiple_videos(cfg):
    use_minimal_version = True

    location_version_to_use = 'runtime_pruned_scaled'  # 'pruned_scaled' 'runtime_pruned_scaled'
    # head_to_use = 0
    # 50 - as small we go more trajectories but shorter trajectories
    max_matching_euclidean_distance = 500.  # 1000 ~ 500. > 200. looks good

    # prune_radius = 40  # dc3

    init_track_each_frame = True
    enable_forward_pass = False

    logger.info(f'Extract trajectories from locations...')

    # adjust config here
    cfg.device = 'cpu'  # 'cuda:0'
    cfg.single_video_mode.enabled = True  # for now we work on single video
    cfg.preproccesing.pad_factor = 8
    cfg.frame_rate = 30.
    cfg.video_based.enabled = False

    cfg.desired_pixel_to_meter_ratio_rgb = 0.07
    cfg.desired_pixel_to_meter_ratio = 0.07

    video_classes_to_use = [
        SDDVideoClasses.GATES,
        SDDVideoClasses.HYANG,
        SDDVideoClasses.LITTLE,
        SDDVideoClasses.NEXUS,
        SDDVideoClasses.QUAD,
        SDDVideoClasses.BOOKSTORE,
        SDDVideoClasses.COUPA]
    video_numbers_to_use = [
        [i for i in range(9)],
        [i for i in range(15)],
        [i for i in range(4)],
        [i for i in range(12) if i not in [3, 4, 5]],
        [i for i in range(4)],
        [i for i in range(7)],
        [i for i in range(4)], ]

    for v_idx, v_clz in enumerate(video_classes_to_use):
        for v_num in video_numbers_to_use[v_idx]:
            logger.info(f'Dataset: {v_clz.name} - {v_num}')
            # load_locations
            load_path = os.path.join(os.getcwd(),
                                     f'ExtractedLocations'
                                     f'/{v_clz.name}'
                                     f'/{v_num}/extracted_locations.pt')
            extracted_locations: ExtractedLocations = torch.load(load_path)['locations']  # mock the model
            out_head_0, out_head_1, out_head_2 = \
                extracted_locations.head0, extracted_locations.head1, extracted_locations.head2

            uq, uc = np.unique([i.frame_number for i in out_head_0.locations], return_counts=True)
            uc_gt = uc > 1
            repeating_frames = uq[uc_gt]

            # if head_to_use == 0:
            #     out_head = out_head_0
            # elif head_to_use == 1:
            #     out_head = out_head_1
            # elif head_to_use == 2:
            #     out_head = out_head_2
            # else:
            #     raise NotImplementedError
            
            if VIDEO_TO_PRUNE_RADIUS_MAPPING[v_clz][v_num]['head'] == 0:
                out_head = out_head_0
            elif VIDEO_TO_PRUNE_RADIUS_MAPPING[v_clz][v_num]['head'] == 1:
                out_head = out_head_1
            elif VIDEO_TO_PRUNE_RADIUS_MAPPING[v_clz][v_num]['head'] == 2:
                out_head = out_head_2
            else:
                raise NotImplementedError

            prune_radius = VIDEO_TO_PRUNE_RADIUS_MAPPING[v_clz][v_num]['radius']
            logger.info(f"Using head: {VIDEO_TO_PRUNE_RADIUS_MAPPING[v_clz][v_num]['head']}, "
                        f"with radius: {prune_radius}")

            if use_minimal_version:
                video_path = f"{cfg.root}videos/" \
                             f"{v_clz.value}/" \
                             f"video{v_num}/video.mov"
                active_tracks, inactive_tracks, track_ids_used = extract_trajectories_from_locations_core_minimal(
                    locations=out_head, init_track_each_frame=init_track_each_frame,
                    video_path=video_path, prune_radius=prune_radius, padded_shape=extracted_locations.padded_shape,
                    original_shape=extracted_locations.scaled_shape,
                    location_version_to_use=location_version_to_use, max_distance=max_matching_euclidean_distance)
            else:
                logger.info(f'Setting up DataLoader and Model...')
                active_tracks, inactive_tracks, track_ids_used = extract_trajectories_from_locations_core(
                    cfg, enable_forward_pass, out_head=out_head, init_track_each_frame=init_track_each_frame,
                    location_version_to_use=location_version_to_use, max_distance=max_matching_euclidean_distance)

            # save extracted trajectories
            save_dict = {
                'track_ids_used': track_ids_used,
                'active': active_tracks,
                'inactive': inactive_tracks,
                'repeating_frames': repeating_frames
            }
            filename = 'extracted_trajectories.pt'
            save_path = os.path.join(os.getcwd(),
                                     f'ExtractedTrajectories'
                                     f'/{v_clz.name}'
                                     f'/{v_num}/')
            Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(save_dict, save_path + filename)
            logger.info(f"Saved trajectories at {save_path}{filename}")


if __name__ == '__main__':
    # interplay_v0()
    # extract_trajectories()
    # extract_trajectories_from_locations()
    extract_trajectories_from_locations_multiple_videos()
    # extract_trajectories_resumable()
