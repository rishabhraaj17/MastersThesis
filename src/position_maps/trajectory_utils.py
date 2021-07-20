import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, patches
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.dataset import ConcatenateDataset
from baselinev2.plot_utils import add_line_to_axis, add_features_to_axis, plot_trajectory_alongside_frame
from src.position_maps.interplay_utils import Track, Tracks
from src_lib.datasets.opentraj_based import get_multiple_gt_dataset
from src_lib.datasets.trajectory_stgcnn import STGCNNTrajectoryDataset, seq_collate, seq_collate_dict, \
    seq_collate_with_graphs, seq_collate_with_graphs_dict, seq_collate_with_dataset_idx_dict

VIDEO_CLASS = SDDVideoClasses.DEATH_CIRCLE
VIDEO_NUMBER = 2

FILENAME = 'extracted_trajectories.pt'
BASE_PATH = os.path.join(os.getcwd(), f'logs/ExtractedTrajectories/{VIDEO_CLASS.name}/{VIDEO_NUMBER}/')
LOAD_PATH = f"{BASE_PATH}{FILENAME}"
TRAJECTORIES_LOAD_PATH = os.path.join(os.getcwd(), f'logs/Trajectories/{VIDEO_CLASS.name}/{VIDEO_NUMBER}/')


def get_total_tracks():
    tracks = torch.load(LOAD_PATH)
    active_tracks, inactive_tracks = tracks['active'], tracks['inactive']
    total_tracks = active_tracks.tracks + inactive_tracks.tracks
    return total_tracks


def split_tracks_into_lists(min_track_length, total_tracks, duplicate_frames_to_filter=(0,),
                            filter_nth_frame_from_middle=None):
    for d_frame in duplicate_frames_to_filter:
        total_tracks = filter_out_nth_frame(total_tracks, n=d_frame)

    # if filter_nth_frame_from_middle is not None:
    #     total_tracks = filter_out_nth_frame_from_middle(total_tracks, n=filter_nth_frame_from_middle)

    frame_id, track_id, x, y = [], [], [], []
    for track in total_tracks:
        if len(track.frames) < min_track_length:
            continue
        for f_id, loc in zip(track.frames, track.locations):
            _x, _y = loc

            frame_id.append(f_id)
            track_id.append(track.idx)
            x.append(_x)
            y.append(_y)
    return frame_id, track_id, x, y


def filter_out_nth_frame(total_tracks, n=0):
    # filter out extra nth frame
    tracks_not_starting_on_frame_n, tracks_starting_on_frame_n = [], []
    for track in total_tracks:
        # if n in track.frames:
        if n == track.frames[0]:
            tracks_starting_on_frame_n.append(track)
        else:
            tracks_not_starting_on_frame_n.append(track)
    tracks_starting_on_frame_n_filtered = []
    for t in tracks_starting_on_frame_n:
        track_temp = Track(
            idx=t.idx,
            frames=t.frames[1:],
            locations=t.locations[1:],
            inactive=t.inactive
        )
        tracks_starting_on_frame_n_filtered.append(track_temp)
    total_tracks = tracks_starting_on_frame_n_filtered + tracks_not_starting_on_frame_n
    return total_tracks


# not of use
def filter_out_nth_frame_from_middle(total_tracks, n):
    # filter out extra nth frame
    tracks_not_starting_on_frame_n, tracks_starting_on_frame_n = [], []
    for track in total_tracks:
        if n in track.frames:
            tracks_starting_on_frame_n.append(track)
        else:
            tracks_not_starting_on_frame_n.append(track)
    tracks_starting_on_frame_n_filtered = []
    for t in tracks_starting_on_frame_n:
        track_temp = Track(
            idx=t.idx,
            frames=t.frames[1:],
            locations=t.locations[1:],
            inactive=t.inactive
        )
        tracks_starting_on_frame_n_filtered.append(track_temp)
    total_tracks = tracks_starting_on_frame_n_filtered + tracks_not_starting_on_frame_n
    return total_tracks


def get_dataframe_from_lists(frame_id, track_id, x, y):
    df_dict = {
        'frame': frame_id,
        'track': track_id,
        'x': x,
        'y': y
    }
    df = pd.DataFrame(df_dict)
    df = df.sort_values(by=['frame']).reset_index()
    df = df.drop(columns=['index'])
    return df


def dump_tracks_to_file(min_track_length: int = 20, duplicate_frames_to_filter=(0,), filter_nth_frame_from_middle=None):
    total_tracks: Sequence[Track] = get_total_tracks()

    # lists for frame_id, track_id, x, y
    frame_id, track_id, x, y = split_tracks_into_lists(min_track_length, total_tracks,
                                                       duplicate_frames_to_filter=duplicate_frames_to_filter,
                                                       filter_nth_frame_from_middle=filter_nth_frame_from_middle)

    df = get_dataframe_from_lists(frame_id, track_id, x, y)

    Path(TRAJECTORIES_LOAD_PATH).mkdir(parents=True, exist_ok=True)
    with open(f"{TRAJECTORIES_LOAD_PATH}trajectories.txt", 'a') as f:
        df_to_dump = df.to_string(header=False, index=False)
        f.write(df_to_dump)
    print(f"Dumping Trajectories to {TRAJECTORIES_LOAD_PATH}trajectories.txt")


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
        pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def get_single_dataset(cfg, video_class, v_num, split_dataset):
    load_path = f"{cfg.root}{video_class}/{v_num}/"
    dataset = STGCNNTrajectoryDataset(
        load_path, obs_len=cfg.obs_len, pred_len=cfg.pred_len, skip=cfg.skip,
        delim=cfg.delim, video_class=video_class, video_number=v_num, construct_graph=cfg.construct_graph)

    if not split_dataset:
        return dataset

    val_dataset_len = round(len(dataset) * cfg.val_ratio)
    train_indices = torch.arange(start=0, end=len(dataset) - val_dataset_len)
    val_indices = torch.arange(start=len(dataset) - val_dataset_len, end=len(dataset))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_multiple_datasets(cfg, split_dataset=True, with_dataset_idx=True):
    conf = cfg.tp_module.datasets
    video_classes = conf.video_classes
    video_numbers = conf.video_numbers

    train_datasets, val_datasets = [], []
    for v_idx, video_class in enumerate(tqdm(video_classes)):
        for v_num in video_numbers[v_idx]:
            if split_dataset:
                t_dset, v_dset = get_single_dataset(conf, video_class, v_num, split_dataset)
                train_datasets.append(t_dset)
                val_datasets.append(v_dset)
            else:
                dset = get_single_dataset(cfg, video_class, v_num, split_dataset)
                train_datasets.append(dset)

    if split_dataset:
        return (ConcatenateDataset(train_datasets), ConcatenateDataset(val_datasets)) \
            if with_dataset_idx else (ConcatDataset(train_datasets), ConcatDataset(val_datasets))
    return ConcatenateDataset(train_datasets) if with_dataset_idx else ConcatDataset(train_datasets)


def plot_trajectory_with_initial_and_last_frame(frame, last_frame, trajectory, frame_number, track_id,
                                                epoch='', additional_text='', return_figure_only=False, save_path=None,
                                                use_lines=False):
    fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(18, 10))
    img_axis, last_image_axis, trajectory_axis = ax

    img_axis.imshow(frame)
    last_image_axis.imshow(last_frame)

    if use_lines:
        add_line_to_axis(ax=img_axis, features=trajectory)
        add_line_to_axis(ax=last_image_axis, features=trajectory)
        add_line_to_axis(ax=trajectory_axis, features=trajectory)
    else:
        add_features_to_axis(ax=img_axis, features=trajectory)
        add_features_to_axis(ax=last_image_axis, features=trajectory)
        add_features_to_axis(ax=trajectory_axis, features=trajectory)

    img_axis.set_title('Trajectory on initial frame')
    last_image_axis.set_title('Trajectory on last frame')
    trajectory_axis.set_title('Trajectories')

    fig.suptitle(f'Frame: {frame_number} | Track Id: {track_id}\n{additional_text}')

    legends_dict = {'b': 'Observed - [0 - 7]', 'r': 'True - [8 - 19]', 'g': 'Predicted - [8 - 19]'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.tight_layout()

    if return_figure_only:
        plt.close()
        return fig

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path + f"frame_{epoch}_{frame_number}_track_{track_id}.png")
        plt.close()
    else:
        plt.show()

    return fig


def plot_trajectory_with_one_frame(frame, last_frame, trajectory, frame_number, track_id,
                                   epoch='', additional_text='', return_figure_only=False, save_path=None,
                                   use_lines=False, plot_first_and_last=True, marker_size=8):
    if last_frame is not None:
        fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(16, 10))
        img_axis, last_image_axis = ax

        last_image_axis.imshow(last_frame)
        last_image_axis.set_title('Trajectory on last frame')

        if use_lines:
            add_line_to_axis(ax=last_image_axis, features=trajectory, marker_size=marker_size)
        else:
            add_features_to_axis(ax=last_image_axis, features=trajectory, marker_size=marker_size)

        if plot_first_and_last:
            add_features_to_axis(last_image_axis, np.stack([trajectory[0]]),
                                 marker_color='aqua', marker_size=marker_size + 1)
            add_features_to_axis(last_image_axis, np.stack([trajectory[-1]]),
                                 marker_color='r', marker_size=marker_size + 1)
    else:
        fig, ax = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(8, 10))
        img_axis = ax

    img_axis.imshow(frame)

    if use_lines:
        add_line_to_axis(ax=img_axis, features=trajectory, marker_size=marker_size)
    else:
        add_features_to_axis(ax=img_axis, features=trajectory, marker_size=marker_size)

    if plot_first_and_last:
        add_features_to_axis(img_axis, np.stack([trajectory[0]]), marker_color='aqua', marker_size=marker_size+1)
        add_features_to_axis(img_axis, np.stack([trajectory[-1]]), marker_color='r', marker_size=marker_size+1)

    img_axis.set_title('Trajectory on initial frame')

    fig.suptitle(f'Frame: {frame_number} | Track Id: {track_id}\n{additional_text}')

    legends_dict = {'b': 'Observed - [0 - 7]', 'r': 'True - [8 - 19]', 'g': 'Predicted - [8 - 19]'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.tight_layout()

    if return_figure_only:
        plt.close()
        return fig

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path + f"frame_{epoch}_{frame_number}_track_{track_id}.png")
        plt.close()
    else:
        plt.show()

    return fig


def viz_raw_tracks():
    root = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/'
    video_path = f"{root}videos/{VIDEO_CLASS.value}/video{VIDEO_NUMBER}/video.mov"
    total_tracks: Sequence[Track] = get_total_tracks()
    for tr in total_tracks:
        first_frame = tr.frames[0]
        last_frame = tr.frames[-1]

        plot_trajectory_with_one_frame(
            extract_frame_from_video(video_path, first_frame),
            None,  # extract_frame_from_video(video_path, last_frame),
            np.stack(tr.locations),
            frame_number=f"{first_frame}-{last_frame}", track_id=tr.idx, use_lines=False)
        print()


def viz_raw_tracks_from_active_inactive(active_tracks, inactive_tracks, video_class, video_number, use_lines=False,
                                        plot_first_and_last=True, marker_size=8, plot_with_last_frame=False):
    root = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/'
    video_path = f"{root}videos/{video_class}/video{video_number}/video.mov"
    total_tracks = active_tracks.tracks + inactive_tracks.tracks
    total_tracks = sorted(total_tracks, reverse=True)
    # sort total tracks by length
    for tr in tqdm(total_tracks):
        first_frame = tr.frames[0]
        last_frame = tr.frames[-1]

        plot_trajectory_with_one_frame(
            extract_frame_from_video(video_path, first_frame),
            extract_frame_from_video(video_path, last_frame) if plot_with_last_frame else None,
            np.stack(tr.locations),
            frame_number=f"{first_frame}-{last_frame}", track_id=tr.idx, use_lines=use_lines,
            plot_first_and_last=plot_first_and_last, marker_size=marker_size)
        
        
def viz_dataset_trajectories():
    cfg = OmegaConf.load('config/training/training.yaml')

    if cfg.tp_module.datasets.use_generated:
        train_dataset, val_dataset = get_multiple_datasets(cfg=cfg, split_dataset=True, with_dataset_idx=True)
    else:
        train_dataset, val_dataset = get_multiple_gt_dataset(cfg=cfg, split_dataset=True, with_dataset_idx=True)

    loader = DataLoader(train_dataset, batch_size=1, collate_fn=seq_collate_with_dataset_idx_dict)
    for batch in tqdm(loader):
        target = pred = batch['gt_xy']

        dataset_idx = batch['dataset_idx'].item()
        seq_start_end = batch['seq_start_end']
        frame_nums = batch['in_frames']
        track_lists = batch['in_tracks']

        random_trajectory_idx = np.random.choice(frame_nums.shape[1], 1, replace=False).item()

        obs_trajectory = batch['in_xy'][:, random_trajectory_idx, ...]
        gt_trajectory = target[:, random_trajectory_idx, ...]
        pred_trajectory = pred[:, random_trajectory_idx, ...]

        frame_num = int(frame_nums[:, random_trajectory_idx, ...][0].item())
        track_num = int(track_lists[:, random_trajectory_idx, ...][0].item())

        current_dataset = loader.dataset.datasets[dataset_idx].dataset

        video_path = f"{cfg.root}videos/{getattr(SDDVideoClasses, current_dataset.video_class).value}" \
                     f"/video{current_dataset.video_number}/video.mov"
        frame = extract_frame_from_video(video_path, frame_num)

        plot_trajectory_alongside_frame(
            frame, obs_trajectory, gt_trajectory, pred_trajectory, frame_num, track_id=track_num)


if __name__ == '__main__':
    viz_dataset_trajectories()
    # viz_raw_tracks()
    # dump_tracks_to_file(min_track_length=0, duplicate_frames_to_filter=(0,), filter_nth_frame_from_middle=None)
