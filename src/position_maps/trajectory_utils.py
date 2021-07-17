import os
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from average_image.constants import SDDVideoClasses
from interplay import Track, Tracks
from src_lib.datasets.trajectory_stgcnn import STGCNNTrajectoryDataset

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


if __name__ == '__main__':
    dataset = STGCNNTrajectoryDataset(TRAJECTORIES_LOAD_PATH, obs_len=8, pred_len=12, skip=1, delim='space')
    loader = DataLoader(dataset, batch_size=1)
    for data in loader:
        print()
    # dump_tracks_to_file(min_track_length=0)
