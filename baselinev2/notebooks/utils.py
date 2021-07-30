from typing import Optional

import numpy as np

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import GENERATED_DATASET_ROOT, DATASET_META, SAVE_BASE_PATH
from baselinev2.constants import NetworkMode
from log import get_logger, initialize_logging

initialize_logging()
logger = get_logger('baselinev2.notebooks.utils')


def read_tracks_and_relative_distances(path_to_dataset, split: NetworkMode, mmap_mode: Optional[str] = 'r+'):
    tracks = np.load(f'{path_to_dataset}{split.value}_tracks.npy', allow_pickle=True, mmap_mode=mmap_mode)
    relative_distances = np.load(f'{path_to_dataset}{split.value}_distances.npy', allow_pickle=True,
                                 mmap_mode=mmap_mode)

    return tracks, relative_distances


def get_tracks_and_relative_distances_for_video_sequence(video_class: SDDVideoClasses, video_number: int,
                                                         split: NetworkMode, meta_label: SDDVideoDatasets,
                                                         root: str = SAVE_BASE_PATH, generated: bool = False,
                                                         mmap_mode: Optional[str] = 'r+', use_all_splits=False,
                                                         for_phase2=False):
    ratio = float(DATASET_META.get_meta(meta_label, video_number)[0]['Ratio'].to_numpy()[0])
    if use_all_splits:
        if for_phase2:
            path_to_dataset = f'{root}{video_class.value}/video{video_number}/v0/'
        else:
            path_to_dataset = f'{root}{video_class.value}{video_number}/splits_v3/' if generated else \
                f'{root}{video_class.value}/video{video_number}/splits_v1/'
        tracks_train, relative_distances_train = read_tracks_and_relative_distances(
            path_to_dataset=path_to_dataset, split=NetworkMode.TRAIN, mmap_mode=mmap_mode)
        tracks_val, relative_distances_val = read_tracks_and_relative_distances(
            path_to_dataset=path_to_dataset, split=NetworkMode.VALIDATION, mmap_mode=mmap_mode)
        tracks_test, relative_distances_test = read_tracks_and_relative_distances(
            path_to_dataset=path_to_dataset, split=NetworkMode.TEST, mmap_mode=mmap_mode)
        if tracks_train.size == 0:
            tracks_train = np.zeros((0, 20, 4)) if for_phase2 \
                else (np.zeros((0, 20, 14)) if generated else np.zeros((0, 20, 8)))
        if tracks_val.size == 0:
            tracks_val = np.zeros((0, 20, 4)) if for_phase2 \
                else (np.zeros((0, 20, 14)) if generated else np.zeros((0, 20, 8)))
        if tracks_test.size == 0:
            tracks_test = np.zeros((0, 20, 4)) if for_phase2 \
                else (np.zeros((0, 20, 14)) if generated else np.zeros((0, 20, 8)))

        if relative_distances_train.size == 0:
            relative_distances_train = np.zeros((0, 19, 2))
        if relative_distances_val.size == 0:
            relative_distances_val = np.zeros((0, 19, 2))
        if relative_distances_test.size == 0:
            relative_distances_test = np.zeros((0, 19, 2))
        tracks = np.concatenate((tracks_train, tracks_val, tracks_test), axis=0)
        relative_distances = np.concatenate(
            (relative_distances_train, relative_distances_val, relative_distances_test), axis=0)
    else:
        if for_phase2:
            path_to_dataset = f'{root}{video_class.value}/video{video_number}/v0/'
        else:
            path_to_dataset = f'{root}{video_class.value}{video_number}/splits_v3/' if generated else \
                f'{root}{video_class.value}/video{video_number}/splits_v1/'
        tracks, relative_distances = read_tracks_and_relative_distances(
            path_to_dataset=path_to_dataset, split=split, mmap_mode=mmap_mode)
    return (tracks, relative_distances), ratio


def get_trajectory_splits(video_class: SDDVideoClasses, video_number: int,
                          split: NetworkMode, meta_label: SDDVideoDatasets,
                          root: str = SAVE_BASE_PATH, generated: bool = False,
                          mmap_mode: Optional[str] = 'r+', use_all_splits=False, for_phase2=False):
    observation_length = 8
    (tracks, relative_distances), ratio = get_tracks_and_relative_distances_for_video_sequence(
        video_class, video_number, split, meta_label, root, generated, mmap_mode, use_all_splits=use_all_splits,
        for_phase2=for_phase2)
    if generated:
        observed_trajectory = tracks[..., :observation_length, 6:8]
        prediction_trajectory = tracks[..., observation_length:, 6:8]

        observed_relative_distances = relative_distances[..., :observation_length - 1, :]
        prediction_relative_distances = relative_distances[..., observation_length - 1:, :]
    elif for_phase2:
        observed_trajectory = tracks[..., :observation_length, -2:]
        prediction_trajectory = tracks[..., observation_length:, -2:]

        observed_relative_distances = relative_distances[..., :observation_length - 1, :]
        prediction_relative_distances = relative_distances[..., observation_length - 1:, :]
    else:
        observed_trajectory = tracks[..., :observation_length, -2:]
        prediction_trajectory = tracks[..., observation_length:, -2:]

        observed_relative_distances = relative_distances[..., :observation_length - 1, :]
        prediction_relative_distances = relative_distances[..., observation_length - 1:, :]

    return observed_trajectory, prediction_trajectory, observed_relative_distances, prediction_relative_distances, ratio


def get_trajectory_length(trajectory, use_l2=False):
    length = []
    for idx in range(trajectory.shape[1] - 1):
        if use_l2:
            part_length = np.linalg.norm((trajectory[:, idx + 1, ...] - trajectory[:, idx, ...]), ord=2, axis=-1)
        else:
            part_length = trajectory[:, idx + 1, ...] - trajectory[:, idx, ...]
        length.append(part_length)
    length = np.stack(length, axis=-1)
    return length, length.sum(axis=-1)


if __name__ == '__main__':
    v_clz = SDDVideoClasses.LITTLE
    v_clz_meta = SDDVideoDatasets.LITTLE
    v_number = 3
    split = NetworkMode.TRAIN
    mem_mode = None

    generated_dataset = False

    root_path = GENERATED_DATASET_ROOT if generated_dataset else SAVE_BASE_PATH

    obs_trajectory, pred_trajectory, obs_relative_distances, pred_relative_distances, ratio = \
        get_trajectory_splits(video_class=v_clz, video_number=v_number, split=split, root=root_path,
                              meta_label=v_clz_meta, mmap_mode=mem_mode, generated=generated_dataset)

    get_trajectory_length(pred_trajectory)
