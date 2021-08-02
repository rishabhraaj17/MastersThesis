import copy
import os
import warnings
from pathlib import Path

import hydra
import motmetrics as mm
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.exceptions import TimeoutException
from baselinev2.plot_utils import add_line_to_axis, add_features_to_axis
from baselinev2.stochastic.model import BaselineGAN
from log import get_logger
from src.position_maps.interplay_utils import setup_multiple_frame_only_datasets_core, Tracks, Track
from src.position_maps.location_utils import ExtractedLocations, prune_locations_proximity_based, \
    get_adjusted_object_locations, Locations, Location
from src.position_maps.segmentation_utils import dump_image_mapping, dump_class_mapping
from src.position_maps.trajectory_utils import plot_trajectory_with_one_frame
from src.position_maps.utils import get_scaled_shapes_with_pad_values, ImagePadder
from src_lib.datasets.extracted_dataset import extracted_collate, get_train_and_val_datasets, get_test_datasets
from src_lib.datasets.trajectory_stgcnn import seq_collate_with_dataset_idx_dict
from src_lib.models_hub import DeepLabV3Plus

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


def viz_tracks(active_tracks, first_frame, show=True, use_lines=False):
    active_tracks_to_vis = [t.locations for t in active_tracks.tracks]
    last_locations = [t.locations[-1] for t in active_tracks.tracks]
    track_ids = [t.idx for t in active_tracks.tracks]
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    if isinstance(first_frame, torch.Tensor):
        first_frame = first_frame.squeeze().permute(1, 2, 0)
    ax.imshow(first_frame)
    for a_t, last_loc, t_idx in zip(active_tracks_to_vis, last_locations, track_ids):
        a_t = np.stack(a_t)
        if use_lines:
            add_line_to_axis(ax=ax, features=a_t, marker_size=1, marker_color='g')
        else:
            add_features_to_axis(ax=ax, features=a_t, marker_size=1, marker_color='g')
        ax.annotate(t_idx, (last_loc[0], last_loc[1]), color='w', weight='bold', fontsize=6, ha='center', va='center')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def init_tracks_from_empty(active_tracks, current_track, location, locations_to_use, track_ids_used):
    for agent_pred_loc in locations_to_use:
        agent_pred_loc = list(agent_pred_loc)
        track = Track(idx=current_track, frames=[location.frame_number], locations=[agent_pred_loc])

        active_tracks.tracks.append(track)
        track_ids_used.append(current_track)
        current_track += 1
    return current_track


def get_valid_locations_from_segmentation_maps(cfg, vid_clz, vid_num):
    seg_root = os.path.split(os.path.split(cfg.root)[0])[0]
    video_mappings = dump_image_mapping(os.path.join(seg_root, f"SDD_SEG_MAPS/"))

    instance_mask = torchvision.io.read_image(
        os.path.join(
            seg_root,
            f"SDD_SEG_MAPS/{video_mappings[vid_clz.value][vid_num][0]}/GLAY/"
            f"{video_mappings[vid_clz.value][vid_num][1]}"))
    instance_mask = instance_mask.permute(1, 2, 0).numpy()

    instance_class_mappings = dump_class_mapping(os.path.join(seg_root, f"SDD_SEG_MAPS/"))

    valid_classes = [v for k, v in instance_class_mappings.items()
                     if k in ['foot_path', 'street', 'grass_path', 'parking']]
    valid_x_axis_locs, valid_y_axis_locs = [], []
    for v in valid_classes:
        y_points, x_points, z_points = np.where(instance_mask == v)
        valid_x_axis_locs.append(x_points)
        valid_y_axis_locs.append(y_points)

        # verify_map = np.zeros_like(instance_mask)
        # for l in np.stack((valid_x_axis_locs[-1], valid_y_axis_locs[-1]), axis=-1):
        #     verify_map[l[1], l[0]] = 255
        # plt.imshow(verify_map, cmap='gray')
        # plt.show()

    valid_x_axis_locs = np.concatenate(valid_x_axis_locs)
    valid_y_axis_locs = np.concatenate(valid_y_axis_locs)
    valid_locs = np.stack((valid_x_axis_locs, valid_y_axis_locs), axis=-1)

    # verify
    # verify_map = np.zeros_like(instance_mask)
    # for l in valid_locs:
    #     verify_map[l[1], l[0]] = 255
    return valid_locs


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


def setup_trajectory_dataset(cfg):
    logger.info(f"Setting up trajectory dataset")
    # train_dataset, val_dataset = setup_dataset(cfg)
    train_dataset, val_dataset = get_train_and_val_datasets(
        video_classes=[getattr(SDDVideoClasses, v_c) for v_c in cfg.tp_module.datasets.train.video_classes],
        video_numbers=cfg.tp_module.datasets.train.video_numbers,
        meta_label=[getattr(SDDVideoDatasets, v_c) for v_c in cfg.tp_module.datasets.train.video_classes],
        val_video_classes=[getattr(SDDVideoClasses, v_c) for v_c in cfg.tp_module.datasets.val.video_classes],
        val_video_numbers=cfg.tp_module.datasets.val.video_numbers,
        val_meta_label=[getattr(SDDVideoDatasets, v_c) for v_c in cfg.tp_module.datasets.val.video_classes],
        get_generated=cfg.tp_module.datasets.use_generated,
        meta_path='../../../Datasets/SDD/H_SDD.txt',
        root='../../../Datasets/SDD/pm_extracted_annotations/'
        if cfg.tp_module.datasets.use_generated else '../../../Datasets/SDD_Features/'
    )
    test_dataset = get_test_datasets(
        video_classes=[getattr(SDDVideoClasses, v_c) for v_c in cfg.tp_module.datasets.test.video_classes],
        video_numbers=cfg.tp_module.datasets.test.video_numbers,
        meta_label=[getattr(SDDVideoDatasets, v_c) for v_c in cfg.tp_module.datasets.test.video_classes],
        get_generated=cfg.tp_module.datasets.use_generated,
        meta_path='../../../Datasets/SDD/H_SDD.txt',
        root='../../../Datasets/SDD/pm_extracted_annotations/'
        if cfg.tp_module.datasets.use_generated else '../../../Datasets/SDD_Features/'
    )
    trajectory_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    return trajectory_dataset


def get_trajectory_loader(cfg, trajectory_dataset):
    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate
    # trajectory_train_loader = DataLoader(
    #     train_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    # trajectory_val_loader = DataLoader(
    #     val_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    trajectory_loader = DataLoader(
        trajectory_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False, collate_fn=collate_fn)
    return trajectory_loader


def setup_trajectory_model():
    # trajectory_model = BaselineGAN(OmegaConf.merge(
    #     OmegaConf.load('../../baselinev2/stochastic/config/model/model.yaml'),
    #     OmegaConf.load('../../baselinev2/stochastic/config/training/training.yaml'),
    #     OmegaConf.load('../../baselinev2/stochastic/config/eval/eval.yaml'),
    # ))
    logger.info(f"Setting up trajectory model")
    version = 12

    base_path = f'../../../baselinev2/stochastic/logs/lightning_logs/version_{version}'
    checkpoint_path = os.path.join(base_path, 'checkpoints/')
    checkpoint_files = os.listdir(checkpoint_path)
    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]
    checkpoint_file = checkpoint_path + checkpoint_files[-1]
    hparam_path = os.path.join(base_path, 'hparams.yaml')

    trajectory_model = BaselineGAN.load_from_checkpoint(
        checkpoint_path=checkpoint_file, hparams_file=hparam_path, map_location='cuda:0')

    return trajectory_model


def get_video_dataset(cfg):
    logger.info(f"Setting up video dataset")
    video_train_dataset = setup_frame_only_dataset_flexible(
        cfg=cfg, video_class=cfg.interplay.video_class, video_number=cfg.interplay.video_number)
    video_train_loader = DataLoader(
        video_train_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False, collate_fn=None)
    return video_train_loader


def get_video_model(cfg):
    video_model = DeepLabV3Plus(config=cfg, train_dataset=None, val_dataset=None)

    if cfg.warm_restart.wandb.enabled:
        version_name = f"{cfg.warm_restart.wandb.checkpoint.run_name}".split('-')[-1]
        checkpoint_root_path = f'{cfg.warm_restart.wandb.checkpoint.root}' \
                               f'{cfg.warm_restart.wandb.checkpoint.run_name}' \
                               f'{cfg.warm_restart.wandb.checkpoint.tail_path}' \
                               f'{cfg.warm_restart.wandb.checkpoint.project_name}/' \
                               f'{version_name}/checkpoints/'
    else:
        checkpoint_root_path = f'{cfg.warm_restart.checkpoint.root}{cfg.warm_restart.checkpoint.path}' \
                               f'{cfg.warm_restart.checkpoint.version}/checkpoints/'

    checkpoint_files = os.listdir(checkpoint_root_path)
    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

    model_path = checkpoint_root_path + checkpoint_files[-cfg.warm_restart.checkpoint.top_k]

    logger.info(f'Loading weights manually as custom load is {cfg.warm_restart.custom_load}')
    load_dict = torch.load(model_path, map_location=cfg.device)
    video_model.load_state_dict(load_dict['state_dict'])
    video_model.to(cfg.device)
    video_model.eval()

    return video_model


def construct_tracks_from_locations(
        active_tracks, frame_number, inactive_tracks, current_frame_locations,
        track_ids_used, current_track, init_track_each_frame=True, max_distance=float('inf'),
        trajectory_model=None, current_frame=None, valid_locs=np.zeros((0, 2))):
    current_track_local = current_track
    recently_killed_tracks = Tracks.init_with_empty_tracks()

    # get last frame locations
    last_frame_locations = np.stack([t.locations[-1] for t in active_tracks.tracks])

    # try setting max dist to a reasonable number so that matches are reasonable within a distance
    distance_matrix = mm.distances.norm2squared_matrix(
        last_frame_locations, current_frame_locations, max_d2=max_distance)

    agent_associations = mm.lap.lsa_solve_scipy(distance_matrix)
    match_rows, match_cols = agent_associations

    # find agents in the current frame (potential new agents in the scene)
    unmatched_agents_current_frame = np.setdiff1d(np.arange(current_frame_locations.shape[0]), match_cols)
    unmatched_agents_last_frame = np.setdiff1d(np.arange(last_frame_locations.shape[0]), match_rows)
    unmatched_tracks = [active_tracks.tracks[t] for t in unmatched_agents_last_frame]

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
            recently_killed_tracks.tracks.append(track)

    active_tracks.tracks = copy.deepcopy(currently_active_tracks)
    if len(recently_killed_tracks.tracks) > 0:
        # viz_tracks(active_tracks, current_frame.squeeze().permute(1, 2, 0).numpy(), show=True,
        #            use_lines=False)
        for t in recently_killed_tracks.tracks:
            in_xy = np.array(t.locations)
            if in_xy.shape[0] > 1:
                in_dxdy = np.diff(in_xy, axis=0)
                batch = {
                    'in_xy': torch.tensor(in_xy).unsqueeze(1).float(),
                    'in_dxdy': torch.tensor(in_dxdy).unsqueeze(1).float()
                }
                out = trajectory_model.test(batch)
                # plot_trajectory_with_one_frame(
                #     frame=current_frame.squeeze().permute(1, 2, 0).numpy(),
                #     last_frame=None,
                #     trajectory=out['out_xy'].squeeze(1),
                #     obs_trajectory=in_xy,
                #     frame_number=frame_number,
                #     track_id=t.idx,
                #     active_tracks=active_tracks,
                #     current_frame_locations=current_frame_locations,
                #     last_frame_locations=last_frame_locations
                # )
                # if an agent was there in the previous ts and in current frame is not under invalid regions, extend
                # using the prediction
                # since it was there in the last frame we just extend it and
                # add it back to the actives if it was not in the
                # in valid region
                valid_locations_to_use_idx = np.all(
                    np.equal(np.round(out['out_xy'][0].numpy()).astype(np.int), valid_locs), axis=-1).any()
                if valid_locations_to_use_idx.item():
                    t.frames.append(frame_number)
                    t.locations.append(out['out_xy'].squeeze(1)[0].tolist())
                    t.inactive -= 1

                    currently_active_tracks.append(t)
                    # remove from inactives
                    inactive_tracks.tracks.remove(t)
                # after some viz
                # we should train model on 1frame jump (maybe instead 8/12 use 24/48?) - ongoing
                # if the predictions go in the bad direction, error accumulates -> filter with invalid region
                # for new agents not matched we start a new track, so we only need to worry about killed tracks

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

    plot_trajectory_with_one_frame(
        frame=current_frame.squeeze().permute(1, 2, 0).numpy(),
        last_frame=None,
        trajectory=np.zeros((0, 2)),
        obs_trajectory=None,
        frame_number=frame_number,
        track_id=0,
        active_tracks=active_tracks,
        current_frame_locations=current_frame_locations,
        last_frame_locations=last_frame_locations,
        plot_first_and_last=False
    )

    return current_track_local


@hydra.main(config_path="config", config_name="config")
def baseline_interplay(cfg):
    use_model = False
    location_version_to_use = 'runtime_pruned_scaled'  # 'pruned_scaled' 'runtime_pruned_scaled'
    head_to_use = 0
    # 50 - as small we go more trajectories but shorter trajectories
    max_distance = 500.  # 1000 ~ 500. > 200. looks good
    prune_radius = 10  # dc3

    trajectory_dataset = setup_trajectory_dataset(cfg)

    # we are interested in extending a trajectory
    # where ever a trajectory stops we want the model to predict further
    # trajectory_loader = get_trajectory_loader(cfg, trajectory_dataset)

    trajectory_model = setup_trajectory_model()

    video_train_loader = get_video_dataset(cfg)

    # we don't need a video model right now
    # video_model = get_video_model(cfg)

    load_path = os.path.join('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/src/position_maps/logs',
                             f'ExtractedLocations'
                             f'/{getattr(SDDVideoClasses, cfg.interplay.video_class[0]).name}'
                             f'/{cfg.interplay.video_number[0][0]}/extracted_locations.pt')
    extracted_locations: ExtractedLocations = torch.load(load_path)['locations']  # mock the model
    out_head_0, out_head_1, out_head_2 = extracted_locations.head0, extracted_locations.head1, extracted_locations.head2

    padded_shape = extracted_locations.padded_shape
    original_shape = extracted_locations.scaled_shape

    uq, uc = np.unique([i.frame_number for i in out_head_0.locations], return_counts=True)
    uc_gt = uc > 1
    repeating_frames = uq[uc_gt]

    if head_to_use == 0:
        locations = out_head_0
    elif head_to_use == 1:
        locations = out_head_1
    elif head_to_use == 2:
        locations = out_head_2
    else:
        raise NotImplementedError

    valid_locs = get_valid_locations_from_segmentation_maps(
        cfg, getattr(SDDVideoClasses, cfg.interplay.video_class[0]), cfg.interplay.video_number[0][0])

    track_ids_used = []
    current_track = 0

    active_tracks = Tracks.init_with_empty_tracks()
    inactive_tracks = Tracks.init_with_empty_tracks()

    for t_idx, (location, video_data) in \
            enumerate(tqdm(zip(locations.locations, video_train_loader), total=len(locations.locations))):
        frame, meta = video_data
        frame_number = meta['item'].item()

        if location_version_to_use == 'default':
            frame = frame.squeeze(0)
            padder = ImagePadder(frame.shape[-2:], factor=cfg.preproccesing.pad_factor)
            frame = padder.pad(frame)[0]
            locations_to_use = location.locations
        elif location_version_to_use == 'pruned':
            frame = frame.squeeze(0)
            padder = ImagePadder(frame.shape[-2:], factor=cfg.preproccesing.pad_factor)
            frame = padder.pad(frame)[0]
            locations_to_use = location.pruned_locations
        elif location_version_to_use == 'pruned_scaled':
            frame = interpolate(frame.squeeze(0), size=original_shape)
            locations_to_use = location.scaled_locations
        elif location_version_to_use == 'runtime_pruned_scaled':
            frame = interpolate(frame.squeeze(0), size=original_shape)
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
                    current_track=current_track, init_track_each_frame=True,
                    max_distance=max_distance,
                    trajectory_model=trajectory_model,
                    current_frame=frame,
                    valid_locs=valid_locs)
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
                             f'/{getattr(SDDVideoClasses, cfg.interplay.video_class[0]).name}'
                             f'/{cfg.interplay.video_number[0][0]}/')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path + filename)
    logger.info(f"Saved trajectories at {save_path}{filename}")


if __name__ == '__main__':
    baseline_interplay()
