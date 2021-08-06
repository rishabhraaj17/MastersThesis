import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy
import scipy.interpolate as interp
import torch
from matplotlib import pyplot as plt, patches
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.dataset import ConcatenateDataset
from baselinev2.plot_utils import add_line_to_axis, add_features_to_axis, plot_trajectory_alongside_frame
from src.position_maps.interplay_utils import Track, Tracks
from src_lib.datasets.dataset_utils import adjust_dataframe_framerate
from src_lib.datasets.extracted_dataset import get_train_and_val_datasets, extracted_collate
from src_lib.datasets.opentraj_based import get_multiple_gt_dataset
from src_lib.datasets.trajectory_stgcnn import STGCNNTrajectoryDataset, seq_collate, seq_collate_dict, \
    seq_collate_with_graphs, seq_collate_with_graphs_dict, seq_collate_with_dataset_idx_dict, SmoothTrajectoryDataset, \
    TrajectoryDatasetFromFile

VIDEO_CLASS = SDDVideoClasses.GATES
VIDEO_NUMBER = 0

FILENAME = 'extracted_trajectories.pt'
BASE_PATH = os.path.join(os.getcwd(), f'logs/ExtractedTrajectories/{VIDEO_CLASS.name}/{VIDEO_NUMBER}/')
LOAD_PATH = f"{BASE_PATH}{FILENAME}"
TRAJECTORIES_LOAD_PATH = os.path.join(os.getcwd(), f'logs/Trajectories/{VIDEO_CLASS.name}/{VIDEO_NUMBER}/')


def get_total_tracks(path=None):
    path = path if path is not None else LOAD_PATH
    tracks = torch.load(path)
    active_tracks, inactive_tracks = tracks['active'], tracks['inactive']
    total_tracks = active_tracks.tracks + inactive_tracks.tracks
    if 'repeating_frames' in tracks.keys():
        return total_tracks, tracks['repeating_frames']
    return total_tracks, None


def split_tracks_into_lists(min_track_length, total_tracks, duplicate_frames_to_filter=(0,),
                            filter_nth_frame_from_middle=None):
    for d_frame in duplicate_frames_to_filter:
        total_tracks = filter_out_nth_frame(total_tracks, n=d_frame)

    if filter_nth_frame_from_middle is not None:
        for f in filter_nth_frame_from_middle:
            total_tracks = filter_out_nth_frame_from_middle(total_tracks, n=f)

    frame_id, track_id, x, y = [], [], [], []
    for track in tqdm(total_tracks):
        if len(track.frames) < min_track_length:
            continue
        for f_id, loc in zip(track.frames, track.locations):
            if len(loc) == 0:
                print(f'Had to skip empty track at frame {f_id}')
                continue

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
    tracks_not_having_frame_n, tracks_having_frame_n = [], []
    for track in total_tracks:
        if n in track.frames:
            tracks_having_frame_n.append(track)
        else:
            tracks_not_having_frame_n.append(track)
    tracks_having_frame_n_filtered = []
    for t in tracks_having_frame_n:
        offending_idx = t.frames.index(n)
        track_temp = Track(
            idx=t.idx,
            frames=t.frames[:offending_idx] + t.frames[offending_idx + 1:],
            locations=t.locations[:offending_idx] + t.locations[offending_idx + 1:],
            inactive=t.inactive
        )
        tracks_having_frame_n_filtered.append(track_temp)
    total_tracks = tracks_having_frame_n_filtered + tracks_not_having_frame_n
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


def dump_tracks_to_file(min_track_length: int = 20, duplicate_frames_to_filter=(0,), filter_nth_frame_from_middle=None,
                        dump_as_csv_only=False):
    print(f"Brewing Trajectory file for {VIDEO_CLASS} - {VIDEO_NUMBER}")
    # total_tracks: Sequence[Track] = get_total_tracks()
    total_tracks, repeating_frames = get_total_tracks()

    if filter_nth_frame_from_middle is None:
        filter_nth_frame_from_middle = repeating_frames.tolist()[1:]

    # lists for frame_id, track_id, x, y
    frame_id, track_id, x, y = split_tracks_into_lists(min_track_length, total_tracks,
                                                       duplicate_frames_to_filter=duplicate_frames_to_filter,
                                                       filter_nth_frame_from_middle=filter_nth_frame_from_middle)

    df: pd.DataFrame = get_dataframe_from_lists(frame_id, track_id, x, y)

    Path(TRAJECTORIES_LOAD_PATH).mkdir(parents=True, exist_ok=True)
    if dump_as_csv_only:
        df.to_csv(f"{TRAJECTORIES_LOAD_PATH}trajectories.csv", index=False)
        print(f"Dumping Trajectories to {TRAJECTORIES_LOAD_PATH}trajectories.csv")
        # return None

    with open(f"{TRAJECTORIES_LOAD_PATH}trajectories.txt", 'w') as f:
        df_to_dump = df.to_string(header=False, index=False)
        f.write(df_to_dump)
    print(f"Dumping Trajectories to {TRAJECTORIES_LOAD_PATH}trajectories.txt")


def dump_tracks_to_file_multiple(
        min_track_length: int = 20, duplicate_frames_to_filter=(0,), filter_nth_frame_from_middle=None,
        dump_as_csv_only=False):
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
            print(f'Extracting for {v_clz.name} - {v_num}')

            path = os.path.join(
                os.getcwd(), f'logs/ExtractedTrajectories/{v_clz.name}/{v_num}/extracted_trajectories.pt')
            total_tracks, repeating_frames = get_total_tracks(path=path)

            if filter_nth_frame_from_middle is None:
                filter_nth_frame_from_middle = repeating_frames.tolist()[1:]

            # lists for frame_id, track_id, x, y
            frame_id, track_id, x, y = split_tracks_into_lists(min_track_length, total_tracks,
                                                               duplicate_frames_to_filter=duplicate_frames_to_filter,
                                                               filter_nth_frame_from_middle=filter_nth_frame_from_middle)

            df: pd.DataFrame = get_dataframe_from_lists(frame_id, track_id, x, y)

            save_path = os.path.join(os.getcwd(), f'logs/Trajectories/{v_clz.name}/{v_num}/')
            Path(save_path).mkdir(parents=True, exist_ok=True)
            if dump_as_csv_only:
                df.to_csv(f"{save_path}trajectories.csv", index=False)
                print(f"Dumping Trajectories to {save_path}trajectories.csv")
                # return None

            with open(f"{save_path}trajectories.txt", 'w') as f:
                df_to_dump = df.to_string(header=False, index=False)
                f.write(df_to_dump)
            print(f"Dumping Trajectories to {save_path}trajectories.txt")


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


def get_single_dataset(cfg, video_class, v_num, split_dataset, smooth_trajectories=False, smoother=lambda x: x,
                       threshold=1):
    load_path = f"{cfg.root}{video_class}/{v_num}/"
    dataset = STGCNNTrajectoryDataset(
        load_path, obs_len=cfg.obs_len, pred_len=cfg.pred_len, skip=cfg.skip,
        delim=cfg.delim, video_class=video_class, video_number=v_num, construct_graph=cfg.construct_graph)

    if smooth_trajectories:
        dataset = SmoothTrajectoryDataset(base_dataset=dataset, smoother=smoother, threshold=threshold)

    if not split_dataset:
        return dataset

    val_dataset_len = round(len(dataset) * cfg.val_ratio)
    train_indices = torch.arange(start=0, end=len(dataset) - val_dataset_len)
    val_indices = torch.arange(start=len(dataset) - val_dataset_len, end=len(dataset))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_single_generated_dataset_from_tempfile(cfg, video_class, video_number, split_dataset,
                                               smooth_trajectories=False, smoother=lambda x: x, threshold=1,
                                               frame_rate=30., time_step=0.4):
    filename = cfg.unsupervised_csv
    dataset_folder = 'filtered_generated_annotations_augmented'

    if cfg.unsupervised_root == 'generated_annotations':
        dataset_folder = 'generated_annotations'
    elif cfg.unsupervised_root == 'filtered_generated_annotations':
        dataset_folder = 'filtered_generated_annotations'
    elif cfg.unsupervised_root == 'filtered_generated_annotations_augmented':
        dataset_folder = 'filtered_generated_annotations_augmented'
    elif cfg.unsupervised_root == 'pm_extracted_annotations':
        dataset_folder = 'pm_extracted_annotations'
        filename = 'trajectories.csv'
    elif cfg.unsupervised_root == 'classic_nn_extracted_annotations':
        dataset_folder = 'classic_nn_extracted_annotations'

    load_path = f"{cfg.root}{dataset_folder}/" \
                f"{getattr(SDDVideoClasses, video_class).value}/video{video_number}/{filename}"

    data_df = pd.read_csv(load_path)

    # check out for classic_nn_extracted_annotations
    if cfg.unsupervised_root in \
            ['generated_annotations', 'filtered_generated_annotations', 'filtered_generated_annotations_augmented']:
        data_df = data_df[['frame_number', 'track_id', 'center_x', 'center_y']]
        data_df = data_df.rename(columns={"frame_number": "frame"})

    data_df = adjust_dataframe_framerate(data_df, for_gt=False, frame_rate=frame_rate, time_step=time_step)

    temp_file = tempfile.NamedTemporaryFile(suffix='.txt')
    data_df.to_csv(temp_file, header=False, index=False, sep=' ')
    dataset = TrajectoryDatasetFromFile(
        temp_file, obs_len=cfg.obs_len, pred_len=cfg.pred_len, skip=cfg.skip, min_ped=cfg.min_ped,
        delim=cfg.delim, video_class=video_class, video_number=video_number, construct_graph=cfg.construct_graph)

    if smooth_trajectories:
        dataset = SmoothTrajectoryDataset(base_dataset=dataset, smoother=smoother, threshold=threshold)

    if not split_dataset:
        return dataset

    val_dataset_len = round(len(dataset) * cfg.val_ratio)
    train_indices = torch.arange(start=0, end=len(dataset) - val_dataset_len)
    val_indices = torch.arange(start=len(dataset) - val_dataset_len, end=len(dataset))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_multiple_datasets(cfg, split_dataset=True, with_dataset_idx=True,
                          smooth_trajectories=False, smoother=lambda x: x, threshold=1,
                          from_temp_file=True, frame_rate=30., time_step=0.4):
    conf = cfg.tp_module.datasets
    video_classes = conf.video_classes
    video_numbers = conf.video_numbers

    train_datasets, val_datasets = [], []
    for v_idx, video_class in enumerate(tqdm(video_classes)):
        for v_num in video_numbers[v_idx]:
            if split_dataset:
                if from_temp_file:
                    t_dset, v_dset = get_single_generated_dataset_from_tempfile(conf, video_class, v_num, split_dataset,
                                                                                smooth_trajectories=smooth_trajectories,
                                                                                smoother=smoother, threshold=threshold,
                                                                                frame_rate=frame_rate, time_step=time_step)
                else:
                    t_dset, v_dset = get_single_dataset(conf, video_class, v_num, split_dataset,
                                                        smooth_trajectories=smooth_trajectories,
                                                        smoother=smoother, threshold=threshold)
                train_datasets.append(t_dset)
                val_datasets.append(v_dset)
            else:
                if from_temp_file:
                    dset = get_single_generated_dataset_from_tempfile(cfg, video_class, v_num, split_dataset,
                                                                      smooth_trajectories=smooth_trajectories,
                                                                      smoother=smoother, threshold=threshold,
                                                                      frame_rate=frame_rate, time_step=time_step)
                else:
                    dset = get_single_dataset(cfg, video_class, v_num, split_dataset,
                                              smooth_trajectories=smooth_trajectories,
                                              smoother=smoother, threshold=threshold)
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


def plot_trajectory_with_one_frame(frame, last_frame, trajectory, frame_number, track_id, obs_trajectory=None,
                                   active_tracks=None, current_frame_locations=None, last_frame_locations=None,
                                   epoch='', additional_text='', return_figure_only=False, save_path=None,
                                   use_lines=False, plot_first_and_last=True, marker_size=8):
    if last_frame is not None:
        fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(16, 10))
        img_axis, last_image_axis = ax

        last_image_axis.imshow(last_frame)
        last_image_axis.set_title('Trajectory on last frame')

        if use_lines:
            add_line_to_axis(ax=last_image_axis, features=trajectory, marker_size=marker_size)
            if obs_trajectory is not None:
                add_line_to_axis(
                    ax=last_image_axis, features=obs_trajectory, marker_size=marker_size, marker_color='g')
        else:
            add_features_to_axis(ax=last_image_axis, features=trajectory, marker_size=marker_size)
            if obs_trajectory is not None:
                add_features_to_axis(
                    ax=last_image_axis, features=obs_trajectory, marker_size=marker_size, marker_color='g')

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
        if obs_trajectory is not None:
            add_line_to_axis(
                ax=img_axis, features=obs_trajectory, marker_size=marker_size, marker_color='g')
    else:
        add_features_to_axis(ax=img_axis, features=trajectory, marker_size=marker_size)
        if obs_trajectory is not None:
            add_line_to_axis(
                ax=img_axis, features=obs_trajectory, marker_size=marker_size, marker_color='g')

    if plot_first_and_last:
        add_features_to_axis(img_axis, np.stack([trajectory[0]]), marker_color='aqua', marker_size=marker_size + 1)
        add_features_to_axis(img_axis, np.stack([trajectory[-1]]), marker_color='r', marker_size=marker_size + 1)

    if active_tracks is not None:
        active_tracks_to_vis = [t.locations for t in active_tracks.tracks]
        last_locations = [t.locations[-1] for t in active_tracks.tracks]
        track_ids = [t.idx for t in active_tracks.tracks]

        for a_t, last_loc, t_idx in zip(active_tracks_to_vis, last_locations, track_ids):
            a_t = np.stack(a_t)
            if use_lines:
                add_line_to_axis(ax=img_axis, features=a_t, marker_size=1, marker_color='g')
            else:
                add_features_to_axis(ax=img_axis, features=a_t, marker_size=1, marker_color='g')
            img_axis.annotate(t_idx, (last_loc[0], last_loc[1]), color='w', weight='bold', fontsize=6, ha='center',
                              va='center')

    if current_frame_locations is not None:
        add_features_to_axis(img_axis, current_frame_locations, marker_shape='*')
    if last_frame_locations is not None:
        add_features_to_axis(img_axis, last_frame_locations, marker_shape='+')

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


# smooth trajectories
# 1 - rank 2
def interpolate_polyline(polyline, num_points):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


def simple_scipy_splines(polyline, num_points):
    # throws value error for stationary points
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


# 2
def approximate_b_spline_path(x: list, y: list, n_path_points: int,
                              degree: int = 3) -> tuple:
    """
    approximate points with a B-Spline path
    :param x: x position list of approximated points
    :param y: y position list of approximated points
    :param n_path_points: number of path points
    :param degree: (Optional) B Spline curve degree
    :return: x and y position list of the result path
    """
    t = range(len(x))
    x_tup = interp.splrep(t, x, k=degree)
    y_tup = interp.splrep(t, y, k=degree)

    x_list = list(x_tup)
    x_list[1] = x + np.arange(x.shape[0])  # [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    y_list[1] = y + np.arange(y.shape[0])  # [0.0, 0.0, 0.0, 0.0]

    ipl_t = np.linspace(0.0, len(x) - 1, n_path_points)
    rx = interp.splev(ipl_t, x_list)
    ry = interp.splev(ipl_t, y_list)

    return rx, ry


def interpolate_b_spline_path(x: list, y: list, n_path_points: int,
                              degree: int = 3) -> tuple:
    """
    interpolate points with a B-Spline path
    :param x: x positions of interpolated points
    :param y: y positions of interpolated points
    :param n_path_points: number of path points
    :param degree: B-Spline degree
    :return: x and y position list of the result path
    """
    # seems to be doing nothing
    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = interp.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = interp.make_interp_spline(ipl_t, y, k=degree)

    travel = np.linspace(0.0, len(x) - 1, n_path_points)
    return spl_i_x(travel), spl_i_y(travel)


# 3 - rank 1
def calc_bezier_path(control_points, n_points=100):
    """
    Compute bezier path (trajectory) given control points.
    :param control_points: (numpy array)
    :param n_points: (int) number of points in the trajectory
    :return: (numpy array)
    """
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def bernstein_poly(n, i, t):
    """
    Bernstein polynom.
    :param n: (int) polynom degree
    :param i: (int)
    :param t: (float)
    :return: (float)
    """
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)


def bezier(t, control_points):
    """
    Return one point on the bezier curve.
    :param t: (float) number in [0, 1]
    :param control_points: (numpy array)
    :return: (numpy array) Coordinates of the point
    """
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)


def bezier_derivatives_control_points(control_points, n_derivatives):
    """
    Compute control points of the successive derivatives of a given bezier curve.
    A derivative of a bezier curve is a bezier curve.
    See https://pomax.github.io/bezierinfo/#derivatives
    for detailed explanations
    :param control_points: (numpy array)
    :param n_derivatives: (int)
    e.g., n_derivatives=2 -> compute control points for first and second derivatives
    :return: ([numpy array])
    """
    w = {0: control_points}
    for i in range(n_derivatives):
        n = len(w[i])
        w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j])
                             for j in range(n - 1)])
    return w


def curvature(dx, dy, ddx, ddy):
    """
    Compute curvature at one point given first and second derivatives.
    :param dx: (float) First derivative along x axis
    :param dy: (float)
    :param ddx: (float) Second derivative along x axis
    :param ddy: (float)
    :return: (float)
    """
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)


def bezier_smoother(trajectory, num_points, threshold=1):
    distance = np.diff(trajectory, axis=-1)
    distance = np.hypot(distance[:, 0], distance[:, 1]).sum(axis=-1)

    out_trajectories = np.zeros_like(trajectory)
    # having length greater than threshold
    valid_trajectories_idx = np.where(distance > threshold)
    invalid_trajectories_idx = np.setdiff1d(np.arange(distance.shape[0]), valid_trajectories_idx)

    out_trajectories[invalid_trajectories_idx] = trajectory[invalid_trajectories_idx]
    trajectories_to_smooth = trajectory[valid_trajectories_idx]

    smoothed_trajectories = np.transpose(
        np.stack([calc_bezier_path(t.T, num_points) for t in trajectories_to_smooth]), (0, 2, 1))

    out_trajectories[valid_trajectories_idx] = smoothed_trajectories

    return out_trajectories


def splrep_smoother(trajectory, num_points, threshold=1):
    distance = np.diff(trajectory, axis=-1)
    distance = np.hypot(distance[:, 0], distance[:, 1]).sum(axis=-1)

    out_trajectories = np.zeros_like(trajectory)
    # having length greater than threshold
    valid_trajectories_idx = np.where(distance > threshold)
    invalid_trajectories_idx = np.setdiff1d(np.arange(distance.shape[0]), valid_trajectories_idx)

    out_trajectories[invalid_trajectories_idx] = trajectory[invalid_trajectories_idx]
    trajectories_to_smooth = trajectory[valid_trajectories_idx]

    smoothed_trajectories = np.transpose(
        np.stack([interpolate_polyline(t.T, num_points) for t in trajectories_to_smooth]), (0, 2, 1))

    out_trajectories[valid_trajectories_idx] = smoothed_trajectories

    return out_trajectories


def viz_dataset_trajectories():
    cfg = OmegaConf.load('config/training/training.yaml')

    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
        if cfg.tp_module.datasets.use_generated:
            train_dataset, val_dataset = get_multiple_datasets(
                cfg=cfg, split_dataset=True, with_dataset_idx=True,
                smooth_trajectories=cfg.tp_module.smooth_trajectories.enabled,
                smoother=bezier_smoother if cfg.tp_module.smooth_trajectories.smoother == 'bezier' else splrep_smoother,
                threshold=cfg.tp_module.smooth_trajectories.min_length,
                from_temp_file=cfg.tp_module.datasets.from_temp_file,
                frame_rate=cfg.tp_module.datasets.frame_rate,
                time_step=cfg.tp_module.datasets.time_step
            )
        else:
            train_dataset, val_dataset = get_multiple_gt_dataset(
                cfg=cfg, split_dataset=True, with_dataset_idx=True,
                smooth_trajectories=cfg.tp_module.smooth_trajectories.enabled,
                smoother=bezier_smoother if cfg.tp_module.smooth_trajectories.smoother == 'bezier' else splrep_smoother,
                threshold=cfg.tp_module.smooth_trajectories.min_length,
                frame_rate=cfg.tp_module.datasets.frame_rate,
                time_step=cfg.tp_module.datasets.time_step
            )
    else:
        collate_fn = extracted_collate
        train_dataset, val_dataset = get_train_and_val_datasets(
            video_classes=[getattr(SDDVideoClasses, v_c) for v_c in cfg.tp_module.datasets.train.video_classes],
            video_numbers=cfg.tp_module.datasets.train.video_numbers,
            meta_label=[getattr(SDDVideoDatasets, v_c) for v_c in cfg.tp_module.datasets.train.video_classes],
            val_video_classes=[getattr(SDDVideoClasses, v_c) for v_c in cfg.tp_module.datasets.val.video_classes],
            val_video_numbers=cfg.tp_module.datasets.val.video_numbers,
            val_meta_label=[getattr(SDDVideoDatasets, v_c) for v_c in cfg.tp_module.datasets.val.video_classes],
            get_generated=cfg.tp_module.datasets.use_generated
        )

    loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
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

        current_dataset = loader.dataset.datasets[dataset_idx].dataset \
            if cfg.tp_module.datasets.use_standard_dataset else loader.dataset.datasets[dataset_idx]
        if isinstance(current_dataset, SmoothTrajectoryDataset):
            current_dataset = current_dataset.base_dataset

        if cfg.tp_module.datasets.use_standard_dataset:
            video_path = f"{cfg.root}videos/{getattr(SDDVideoClasses, current_dataset.video_class).value}" \
                         f"/video{current_dataset.video_number}/video.mov"
        else:
            video_path = f"{cfg.root}videos/{current_dataset.video_class.value}" \
                         f"/video{current_dataset.video_number}/video.mov"

        frame = extract_frame_from_video(video_path, frame_num)

        plot_trajectory_alongside_frame(
            frame, obs_trajectory, gt_trajectory, pred_trajectory, frame_num, track_id=track_num)

        # verify_rel_velocities(batch, frame, frame_num, obs_trajectory, pred_trajectory, random_trajectory_idx,
        #                       track_num)


def verify_rel_velocities(batch, frame, frame_num, obs_trajectory,
                          pred_trajectory, random_trajectory_idx, track_num, show=False):
    # viz vel constructed trajectory
    first_pos = obs_trajectory[0, None, :]
    rel_traj = torch.cat((torch.tensor([0., 0.])[None, :],
                          batch['in_dxdy'][:, random_trajectory_idx, ...],
                          batch['gt_dxdy'][:, random_trajectory_idx, ...]), dim=0)
    traj_constructed = rel_traj.cumsum(dim=0) + first_pos
    obs_constructed = traj_constructed[:8, :]
    pred_constructed = traj_constructed[8:, :]
    if show:
        plot_trajectory_alongside_frame(
            frame, obs_constructed, pred_constructed, pred_trajectory, frame_num, track_id=track_num)


if __name__ == '__main__':
    viz_dataset_trajectories()
    # viz_raw_tracks()

    # frame_count = 12490
    # step = 999
    # filter_middle = [i for i in range(step, frame_count, step)]
    # dump_tracks_to_file(min_track_length=0, duplicate_frames_to_filter=(0,), filter_nth_frame_from_middle=None,
    #                     dump_as_csv_only=True)
    # dump_tracks_to_file_multiple(
    #     min_track_length=0, duplicate_frames_to_filter=(0,), filter_nth_frame_from_middle=None,
    #     dump_as_csv_only=True)
