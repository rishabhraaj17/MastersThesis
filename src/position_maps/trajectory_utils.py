import os
import subprocess
import time
from contextlib import contextmanager
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from interplay import Track, Tracks
from src_lib.datasets.trajectory_stgcnn import STGCNNTrajectoryDataset, seq_collate, seq_collate_dict, \
    seq_collate_with_graphs, seq_collate_with_graphs_dict

VIDEO_CLASS = SDDVideoClasses.DEATH_CIRCLE
VIDEO_NUMBER = 4

FILENAME = 'extracted_trajectories.pt'
BASE_PATH = os.path.join(os.getcwd(), f'logs/ExtractedTrajectories/{VIDEO_CLASS.name}/{VIDEO_NUMBER}/')
LOAD_PATH = f"{BASE_PATH}{FILENAME}"
TRAJECTORIES_LOAD_PATH = os.path.join(os.getcwd(), f'logs/Trajectories/{VIDEO_CLASS.name}/{VIDEO_NUMBER}/')


def get_total_tracks():
    tracks = torch.load(LOAD_PATH)
    active_tracks, inactive_tracks = tracks['active'], tracks['inactive']
    total_tracks = active_tracks.tracks + inactive_tracks.tracks
    return total_tracks


def split_tracks_into_lists(min_track_length, total_tracks):
    total_tracks = filter_out_0th_frame(total_tracks)

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


def filter_out_0th_frame(total_tracks):
    # filter out extra 0th frame
    tracks_not_starting_on_frame0, tracks_starting_on_frame0 = [], []
    for track in total_tracks:
        if 0 in track.frames:
            tracks_starting_on_frame0.append(track)
        else:
            tracks_not_starting_on_frame0.append(track)
    tracks_starting_on_frame0_filtered = []
    for t in tracks_starting_on_frame0:
        track_temp = Track(
            idx=t.idx,
            frames=t.frames[1:],
            locations=t.locations[1:],
            inactive=t.inactive
        )
        tracks_starting_on_frame0_filtered.append(track_temp)
    total_tracks = tracks_starting_on_frame0_filtered + tracks_not_starting_on_frame0
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


def dump_tracks_to_file(min_track_length: int = 20):
    total_tracks: Sequence[Track] = get_total_tracks()

    # lists for frame_id, track_id, x, y
    frame_id, track_id, x, y = split_tracks_into_lists(min_track_length, total_tracks)

    df = get_dataframe_from_lists(frame_id, track_id, x, y)

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
    loss = loss**2
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
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


if __name__ == '__main__':
    dataset = STGCNNTrajectoryDataset(TRAJECTORIES_LOAD_PATH, obs_len=8, pred_len=12, skip=1, delim='space',
                                      video_class=VIDEO_CLASS, video_number=VIDEO_NUMBER, construct_graph=False)
    loader = DataLoader(dataset, batch_size=4, collate_fn=seq_collate_dict)
    for data in tqdm(loader):
        in_frames, gt_frames = data['in_frames'], data['gt_frames']
    # dump_tracks_to_file(min_track_length=0)
