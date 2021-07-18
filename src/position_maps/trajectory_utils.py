import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from baselinev2.nn.dataset import ConcatenateDataset
from interplay import Track, Tracks
from src_lib.datasets.trajectory_stgcnn import STGCNNTrajectoryDataset, seq_collate, seq_collate_dict, \
    seq_collate_with_graphs, seq_collate_with_graphs_dict

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


def split_tracks_into_lists(min_track_length, total_tracks, duplicate_frames_to_filter=(0,)):
    for d_frame in duplicate_frames_to_filter:
        total_tracks = filter_out_nth_frame(total_tracks, n=d_frame)

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


def dump_tracks_to_file(min_track_length: int = 20, duplicate_frames_to_filter=(0,)):
    total_tracks: Sequence[Track] = get_total_tracks()

    # lists for frame_id, track_id, x, y
    frame_id, track_id, x, y = split_tracks_into_lists(min_track_length, total_tracks,
                                                       duplicate_frames_to_filter=duplicate_frames_to_filter)

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


def get_multiple_datasets(cfg, split_dataset=True):
    conf = cfg.tp_module.datasets
    video_classes = conf.video_classes
    video_numbers = conf.video_numbers

    train_datasets, val_datasets = [], []
    for v_idx, video_class in enumerate(video_classes):
        for v_num in video_numbers[v_idx]:
            if split_dataset:
                t_dset, v_dset = get_single_dataset(conf, video_class, v_num, split_dataset)
                train_datasets.append(t_dset)
                val_datasets.append(v_dset)
            else:
                dset = get_single_dataset(cfg, video_class, v_num, split_dataset)
                train_datasets.append(dset)

    if split_dataset:
        return ConcatenateDataset(train_datasets)
    return ConcatenateDataset(train_datasets), ConcatenateDataset(val_datasets)


if __name__ == '__main__':
    train_d, val_d = get_multiple_datasets(OmegaConf.load('config/training/training.yaml'))
    loader = DataLoader(train_d, batch_size=4, collate_fn=seq_collate_dict)
    for data in tqdm(loader):
        in_frames, gt_frames = data['in_frames'], data['gt_frames']

    # dump_tracks_to_file(min_track_length=0)
