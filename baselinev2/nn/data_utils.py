from enum import Enum
from pathlib import Path
from typing import Dict, List

import cv2 as cv
import numpy as np
import pandas as pd
import skimage
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost
from average_image.constants import SDDVideoClasses
from baselinev2.config import SPLIT_ANNOTATION_SAVE_PATH, TRAIN_SPLIT_PERCENTAGE, VALIDATION_SPLIT_PERCENTAGE, \
    TEST_SPLIT_PERCENTAGE, VIDEO_SAVE_PATH, SDD_VIDEO_CLASSES_RESUME_LIST, SDD_PER_CLASS_VIDEOS_RESUME_LIST, \
    SDD_ANNOTATIONS_ROOT_PATH, SAVE_BASE_PATH, BUNDLED_ANNOTATIONS_VIDEO_CLASSES_LIST, \
    BUNDLED_ANNOTATIONS_PER_CLASSES_VIDEO_LIST, ROOT_PATH, version
from baselinev2.exceptions import InvalidFrameException
from baselinev2.plot_utils import plot_for_video_image_and_box, plot_trajectory_with_relative_data
from baselinev2.structures import TracksDataset, SingleTrack
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.nn.data_utils')


class DataSplitPath(Enum):
    TRAIN_CSV = f'{SPLIT_ANNOTATION_SAVE_PATH}train.csv'
    VALIDATION_CSV = f'{SPLIT_ANNOTATION_SAVE_PATH}val.csv'
    TEST_CSV = f'{SPLIT_ANNOTATION_SAVE_PATH}test.csv'
    TRAIN = f'{SPLIT_ANNOTATION_SAVE_PATH}train.pt'
    VALIDATION = f'{SPLIT_ANNOTATION_SAVE_PATH}val.pt'
    TEST = f'{SPLIT_ANNOTATION_SAVE_PATH}test.pt'


def get_frames_count(video_path):
    video = cv.VideoCapture(video_path)
    count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video.release()
    return count


def extract_frame_from_video(video_path, frame_number):
    video = cv.VideoCapture(video_path)
    count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    if 0 <= frame_number <= count:
        # video.set(cv.CAP_PROP_POS_FRAMES, frame_number - 1)
        video.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        res, frame = video.read()
        video.release()
        return frame if res else None
    else:
        logger.error('Frame Number out of range!')
        video.release()
        raise InvalidFrameException()


def sort_annotations_by_frame_numbers(annotation_path):
    annotations = pd.read_csv(annotation_path, index_col='Unnamed: 0')
    annotations = annotations.sort_values(by=['frame']).reset_index()
    annotations = annotations.drop(columns=['index'])
    return annotations


def sort_generated_annotations_by_frame_numbers(annotation_path):
    annotations = pd.read_csv(annotation_path)
    annotations = annotations.sort_values(by=['frame_number']).reset_index()
    annotations = annotations.drop(columns=['index'])
    return annotations


def sort_annotations_by_track_ids(annotation_path):
    annotations = pd.read_csv(annotation_path, index_col='Unnamed: 0')
    annotations = annotations.sort_values(by=['track_id']).reset_index()
    annotations = annotations.drop(columns=['index'])
    return annotations


def split_annotations(annotation_path):
    annotations = sort_annotations_by_frame_numbers(annotation_path)
    train_set, test_set = train_test_split(annotations, train_size=TRAIN_SPLIT_PERCENTAGE,
                                           test_size=VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE,
                                           shuffle=False, stratify=None)

    train_set, test_set = adjust_splits_for_frame(larger_set=train_set, smaller_set=test_set)

    test_set, val_set = train_test_split(test_set, train_size=0.3,
                                         test_size=0.7, shuffle=False, stratify=None)

    val_set, test_set = adjust_splits_for_frame(larger_set=test_set, smaller_set=val_set)
    return train_set, val_set, test_set


def split_annotations_from_df(annotations, is_generated=False):
    train_set, test_set = train_test_split(annotations, train_size=TRAIN_SPLIT_PERCENTAGE,
                                           test_size=VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE,
                                           shuffle=False, stratify=None)

    train_set, test_set = adjust_splits_for_frame(larger_set=train_set, smaller_set=test_set,
                                                  generated_tracks=is_generated)

    test_set, val_set = train_test_split(test_set, train_size=0.3,
                                         test_size=0.7, shuffle=False, stratify=None)

    val_set, test_set = adjust_splits_for_frame(larger_set=test_set, smaller_set=val_set,
                                                generated_tracks=is_generated)
    return train_set, val_set, test_set


def split_generated_annotations(annotation_path):
    annotations = sort_generated_annotations_by_frame_numbers(annotation_path)
    train_set, test_set = train_test_split(annotations, train_size=TRAIN_SPLIT_PERCENTAGE,
                                           test_size=VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE,
                                           shuffle=False, stratify=None)

    train_set, test_set = adjust_splits_for_frame(larger_set=train_set, smaller_set=test_set, generated_tracks=True)

    test_set, val_set = train_test_split(test_set, train_size=0.3,
                                         test_size=0.7, shuffle=False, stratify=None)

    val_set, test_set = adjust_splits_for_frame(larger_set=test_set, smaller_set=val_set, generated_tracks=True)
    return train_set, val_set, test_set


def split_annotations_by_tracks(annotation_path):
    annotations = sort_annotations_by_track_ids(annotation_path)
    train_set, test_set = train_test_split(annotations, train_size=TRAIN_SPLIT_PERCENTAGE,
                                           test_size=VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE,
                                           shuffle=False, stratify=None)

    train_set, test_set = adjust_splits_for_track(larger_set=train_set, smaller_set=test_set)

    test_set, val_set = train_test_split(test_set, train_size=0.3,
                                         test_size=0.7, shuffle=False, stratify=None)

    val_set, test_set = adjust_splits_for_track(larger_set=test_set, smaller_set=val_set)
    return train_set, val_set, test_set


def adjust_splits_for_frame(larger_set, smaller_set, generated_tracks=False):
    if generated_tracks:
        frame_label = 'frame_number'
    else:
        frame_label = 'frame'
    last_frame_in_larger = larger_set.iloc[-1][frame_label]
    first_frame_in_smaller = smaller_set.iloc[0][frame_label]

    if last_frame_in_larger == first_frame_in_smaller:
        same_frame_larger = larger_set[larger_set[frame_label] == last_frame_in_larger]
        same_frame_smaller = smaller_set[smaller_set[frame_label] == first_frame_in_smaller]

        if len(same_frame_larger) >= len(same_frame_smaller):
            larger_set = pd.concat([larger_set, same_frame_smaller])
            smaller_set = smaller_set.drop(same_frame_smaller.index)
        else:
            smaller_set = pd.concat([same_frame_larger, smaller_set])
            larger_set = larger_set.drop(same_frame_larger.index)

    return larger_set, smaller_set


def adjust_splits_for_track(larger_set, smaller_set):
    last_frame_in_larger = larger_set.iloc[-1]['track_id']
    first_frame_in_smaller = smaller_set.iloc[0]['track_id']

    if last_frame_in_larger == first_frame_in_smaller:
        same_frame_larger = larger_set[larger_set['track_id'] == last_frame_in_larger]
        same_frame_smaller = smaller_set[smaller_set['track_id'] == first_frame_in_smaller]

        if len(same_frame_larger) >= len(same_frame_smaller):
            larger_set = pd.concat([larger_set, same_frame_smaller])
            smaller_set = smaller_set.drop(same_frame_smaller.index)
        else:
            smaller_set = pd.concat([same_frame_larger, smaller_set])
            larger_set = larger_set.drop(same_frame_larger.index)

    return larger_set, smaller_set


def split_annotations_and_save_as_csv(annotation_path, path_to_save):
    train_set, val_set, test_set = split_annotations(annotation_path)
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    train_set.to_csv(path_to_save + 'train.csv', index=False)
    val_set.to_csv(path_to_save + 'val.csv', index=False)
    test_set.to_csv(path_to_save + 'test.csv', index=False)
    logger.info(f'Saved Splits at {path_to_save}')


def split_annotations_and_save_as_track_datasets(annotation_path, path_to_save, by_track=False):
    if by_track:
        train_set, val_set, test_set = split_annotations_by_tracks(annotation_path)
    else:
        train_set, val_set, test_set = split_annotations(annotation_path)

    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    logger.info('Processing Train set')
    train_dataset = turn_splits_into_trajectory_dataset(train_set, dataframe_mode=True)
    logger.info('Processing Validation set')
    val_dataset = turn_splits_into_trajectory_dataset(val_set, dataframe_mode=True)
    logger.info('Processing Test set')
    test_dataset = turn_splits_into_trajectory_dataset(test_set, dataframe_mode=True)

    torch.save(train_dataset, path_to_save + 'train.pt')
    torch.save(val_dataset, path_to_save + 'val.pt')
    torch.save(test_dataset, path_to_save + 'test.pt')

    logger.info(f'Saved track datasets at {path_to_save}')


def split_annotations_and_save_as_track_datasets_by_length(annotation_path, path_to_save, length=20,
                                                           by_track=False):
    if by_track:
        train_set, val_set, test_set = split_annotations_by_tracks(annotation_path)
    else:
        train_set, val_set, test_set = split_annotations(annotation_path)

    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    logger.info('Processing Train set')
    train_dataset = turn_splits_into_trajectory_dataset(train_set, dataframe_mode=True)
    logger.info('Processing Validation set')
    val_dataset = turn_splits_into_trajectory_dataset(val_set, dataframe_mode=True)
    logger.info('Processing Test set')
    test_dataset = turn_splits_into_trajectory_dataset(test_set, dataframe_mode=True)

    logger.info(f'Processing datasets by length: {length}')
    logger.info('Train')
    train_dataset = make_trainable_trajectory_dataset_by_length(train_dataset, length=length)
    logger.info('Validation')
    val_dataset = make_trainable_trajectory_dataset_by_length(val_dataset, length=length)
    logger.info('Test')
    test_dataset = make_trainable_trajectory_dataset_by_length(test_dataset, length=length)

    torch.save(train_dataset, path_to_save + 'train.pt')
    torch.save(val_dataset, path_to_save + 'val.pt')
    torch.save(test_dataset, path_to_save + 'test.pt')

    logger.info(f'Saved track datasets at {path_to_save}')


def verify_annotations_processing(video_path, df, plot_scale_factor=1, desired_fps=5):
    Path(VIDEO_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))

    if w < h:
        original_dims = (
            h / 100 * plot_scale_factor, w / 100 * plot_scale_factor)
        out = cv.VideoWriter(VIDEO_SAVE_PATH + 'proof_out.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (h, w))

        w, h = h, w
    else:
        original_dims = (
            w / 100 * plot_scale_factor, h / 100 * plot_scale_factor)
        out = cv.VideoWriter(VIDEO_SAVE_PATH + 'proof_out.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (w, h))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if ret:
            annotation = get_frame_annotations_and_skip_lost(df, frame_idx)
            # annotation = get_frame_annotations(df, frame_idx)
            boxes = annotation[:, 1:5]

            fig = plot_for_video_image_and_box(gt_rgb=frame, gt_annotations=boxes, frame_number=frame_idx,
                                               original_dims=original_dims, return_figure_only=True)
            canvas = FigureCanvas(fig)
            canvas.draw()

            buf = canvas.buffer_rgba()
            out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
            if out_frame.shape[0] != h or out_frame.shape[1] != w:
                out_frame = skimage.transform.resize(out_frame, (h, w))
                out_frame = (out_frame * 255).astype(np.uint8)
            out.write(out_frame)
            frame_idx += 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()

    cv.destroyAllWindows()


def turn_splits_into_trajectory_dataset(split_path, num_frames_in_jump=12, time_between_frames=0.4,
                                        dataframe_mode=False):
    if dataframe_mode:
        split: pd.DataFrame = split_path
    else:
        split: pd.DataFrame = pd.read_csv(split_path.value)

    dataset: Dict[int, TracksDataset] = {}
    unique_tracks = split.track_id.unique()
    for t_id in tqdm(unique_tracks):
        track_df = split[split.track_id == t_id]
        # unique_frames_in_track = track_df.frame.unique()

        tracks_list: List[SingleTrack] = []
        # for start_frame in unique_frames_in_track:
        #     track_df_with_jumps: pd.DataFrame = track_df.iloc[
        #                                         track_df[track_df.frame == start_frame].index[0]::num_frames_in_jump]
        #     # track_df_with_jumps: pd.DataFrame = track_df.iloc[start_frame::num_frames_in_jump]
        for start_frame in range(len(track_df)):
            track_df_with_jumps: pd.DataFrame = track_df.iloc[start_frame::num_frames_in_jump]
            track: SingleTrack = SingleTrack(
                data=track_df_with_jumps.to_numpy(),
                frames=track_df_with_jumps.frame.to_numpy(),
                valid=True if len(track_df_with_jumps) >= 20 else False
            )
            tracks_list.append(track)

        dataset.update({t_id: TracksDataset(
            track_id=t_id,
            tracks=tracks_list,
            columns=split.columns.to_numpy().tolist()
        )})
    return dataset


def make_trainable_trajectory_dataset(dataset: Dict[int, TracksDataset]):
    final_tracks = []
    for track_id, track_obj in tqdm(dataset.items()):
        for track in track_obj.tracks:
            final_tracks.append(np.hstack((track.data[:, 0:6], track.data[:, 10:])))

    return final_tracks


def make_trainable_trajectory_dataset_by_length(dataset: Dict[int, TracksDataset], length=20, save_path=None,
                                                numpy_save=False):
    final_tracks, relative_distances = [], []
    for track_id, track_obj in tqdm(dataset.items()):
        for track in track_obj.tracks:
            if len(track.data) > length:
                # split array into parts of max length
                splits, remaining_part = array_split_by_length(track.data, length=length)
                for split in splits:
                    relative_distances.append(get_relative_distances(split))
                    final_tracks.append(np.hstack((split[:, 0:6], split[:, 10:])))

    final_tracks = np.stack(final_tracks) if len(final_tracks) != 0 else np.zeros((0, 12))
    relative_distances = np.stack(relative_distances) if len(relative_distances) != 0 else np.zeros((0, 2))

    save_dict = {'tracks': final_tracks, 'distances': relative_distances}
    if save_path is not None:
        if numpy_save:
            with open(save_path, 'wb') as f:
                np.savez(f, tracks=final_tracks, distances=relative_distances, allow_pickle=True)
        else:
            torch.save(save_dict, save_path)
    return save_dict


def turn_splits_into_trajectory_dataset_each_track(split_path, num_frames_in_jump=12, time_between_frames=0.4,
                                                   dataframe_mode=False):
    if dataframe_mode:
        split: pd.DataFrame = split_path
    else:
        split: pd.DataFrame = pd.read_csv(split_path.value)

    unique_tracks = split.track_id.unique()
    final_tracks_save, relative_distances_save = [], []
    for t_id in tqdm(unique_tracks):
        track_df = split[split.track_id == t_id]
        # unique_frames_in_track = track_df.frame.unique()

        tracks_list: List[SingleTrack] = []
        # for start_frame in unique_frames_in_track:
        #     track_df_with_jumps: pd.DataFrame = track_df.iloc[
        #                                         track_df[track_df.frame == start_frame].index[0]::num_frames_in_jump]
        #     # track_df_with_jumps: pd.DataFrame = track_df.iloc[start_frame::num_frames_in_jump]
        for start_frame in range(len(track_df)):
            track_df_with_jumps: pd.DataFrame = track_df.iloc[start_frame::num_frames_in_jump]
            track: SingleTrack = SingleTrack(
                data=track_df_with_jumps.to_numpy(),
                frames=track_df_with_jumps.frame.to_numpy(),
                valid=True if len(track_df_with_jumps) >= 20 else False
            )
            tracks_list.append(track)

        tracks, distances = make_trainable_trajectory_dataset_by_length_each_track(track_obj=tracks_list)
        if tracks.size != 0:
            final_tracks_save.append(tracks)
        if distances.size != 0:
            relative_distances_save.append(distances)

    final_tracks_save = np.concatenate(final_tracks_save) if len(final_tracks_save) != 0 else np.zeros((0, 12))
    relative_distances_save = np.concatenate(relative_distances_save) if len(relative_distances_save) != 0 \
        else np.zeros((0, 2))

    return {'tracks': final_tracks_save, 'distances': relative_distances_save}


def turn_splits_into_trajectory_dataset_each_generated_track(split_path, num_frames_in_jump=12, time_between_frames=0.4,
                                                             dataframe_mode=False):
    if dataframe_mode:
        split: pd.DataFrame = split_path
    else:
        split: pd.DataFrame = pd.read_csv(split_path.value)

    unique_tracks = split.track_id.unique()
    final_tracks_save, relative_distances_save = [], []
    for t_id in tqdm(unique_tracks):
        track_df = split[split.track_id == t_id]
        # unique_frames_in_track = track_df.frame.unique()

        tracks_list: List[SingleTrack] = []
        # for start_frame in unique_frames_in_track:
        #     track_df_with_jumps: pd.DataFrame = track_df.iloc[
        #                                         track_df[track_df.frame == start_frame].index[0]::num_frames_in_jump]
        #     # track_df_with_jumps: pd.DataFrame = track_df.iloc[start_frame::num_frames_in_jump]
        for start_frame in range(len(track_df)):
            track_df_with_jumps: pd.DataFrame = track_df.iloc[start_frame::num_frames_in_jump]
            track: SingleTrack = SingleTrack(
                data=track_df_with_jumps.to_numpy(),
                frames=track_df_with_jumps.frame_number.to_numpy(),
                valid=True if len(track_df_with_jumps) >= 20 else False
            )
            tracks_list.append(track)

        tracks, distances = make_trainable_trajectory_dataset_by_length_each_generated_track(track_obj=tracks_list)
        if tracks.size != 0:
            final_tracks_save.append(tracks)
        if distances.size != 0:
            relative_distances_save.append(distances)

    final_tracks_save = np.concatenate(final_tracks_save) if len(final_tracks_save) != 0 else np.zeros((0, 12))
    relative_distances_save = np.concatenate(relative_distances_save) if len(relative_distances_save) != 0 \
        else np.zeros((0, 2))

    return {'tracks': final_tracks_save, 'distances': relative_distances_save}


def make_trainable_trajectory_dataset_by_length_each_track(track_obj, length=20, save_path=None,
                                                           numpy_save=False):
    final_tracks, relative_distances = [], []
    for track in track_obj:
        if len(track.data) > length:
            # split array into parts of max length
            splits, remaining_part = array_split_by_length(track.data, length=length)
            for split in splits:
                relative_distances.append(get_relative_distances(split))
                final_tracks.append(np.hstack((split[:, 0:6], split[:, 10:])).astype(np.float32))

    final_tracks = np.stack(final_tracks) if len(final_tracks) != 0 else np.zeros((0, 12))
    relative_distances = np.stack(relative_distances) if len(relative_distances) != 0 else np.zeros((0, 2))

    return final_tracks, relative_distances


def make_trainable_trajectory_dataset_by_length_each_generated_track(track_obj, length=20, save_path=None,
                                                                     numpy_save=False):
    final_tracks, relative_distances = [], []
    for track in track_obj:
        if len(track.data) > length:
            # split array into parts of max length
            splits, remaining_part = array_split_by_length(track.data, length=length)
            for split in splits:
                relative_distances.append(get_relative_distances_generated_track(split))
                final_tracks.append(np.hstack((split[:, 0:6], split[:, 7:])).astype(np.float32))

    final_tracks = np.stack(final_tracks) if len(final_tracks) != 0 else np.zeros((0, 12))
    relative_distances = np.stack(relative_distances) if len(relative_distances) != 0 else np.zeros((0, 2))

    return final_tracks, relative_distances


def split_annotations_and_save_as_track_datasets_by_length_for_one(annotation_path, path_to_save, length=20,
                                                                   by_track=False, save_as_numpy=True, mem_mode=True):
    if by_track:
        train_set, val_set, test_set = split_annotations_by_tracks(annotation_path)
    else:
        train_set, val_set, test_set = split_annotations(annotation_path)

    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    logger.info('Processing Train set')
    train_dataset = turn_splits_into_trajectory_dataset_each_track(train_set, dataframe_mode=True)
    logger.info('Processing Validation set')
    val_dataset = turn_splits_into_trajectory_dataset_each_track(val_set, dataframe_mode=True)
    logger.info('Processing Test set')
    test_dataset = turn_splits_into_trajectory_dataset_each_track(test_set, dataframe_mode=True)

    if save_as_numpy:
        if mem_mode:
            np.save(path_to_save + 'train_tracks.npy', train_dataset['tracks'])
            np.save(path_to_save + 'train_distances.npy', train_dataset['distances'])

            np.save(path_to_save + 'val_tracks.npy', val_dataset['tracks'])
            np.save(path_to_save + 'val_distances.npy', val_dataset['distances'])

            np.save(path_to_save + 'test_tracks.npy', test_dataset['tracks'])
            np.save(path_to_save + 'test_distances.npy', test_dataset['distances'])
        else:
            np.savez(path_to_save + 'train.npz', tracks=train_dataset['tracks'], distances=train_dataset['distances'])
            np.savez(path_to_save + 'val.npz', tracks=val_dataset['tracks'], distances=val_dataset['distances'])
            np.savez(path_to_save + 'test.npz', tracks=test_dataset['tracks'], distances=test_dataset['distances'])
    else:
        torch.save(train_dataset, path_to_save + 'train.pt')
        torch.save(val_dataset, path_to_save + 'val.pt')
        torch.save(test_dataset, path_to_save + 'test.pt')

    logger.info(f'Saved track datasets at {path_to_save}')


def split_annotations_and_save_as_generated_track_datasets_by_length_for_one(annotation_path, path_to_save, length=20,
                                                                             by_track=False, save_as_numpy=True,
                                                                             mem_mode=True):
    if by_track:
        train_set, val_set, test_set = split_annotations_by_tracks(annotation_path)
    else:
        train_set, val_set, test_set = split_generated_annotations(annotation_path)

    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    logger.info('Processing Train set')
    train_dataset = turn_splits_into_trajectory_dataset_each_generated_track(train_set, dataframe_mode=True)
    logger.info('Processing Validation set')
    val_dataset = turn_splits_into_trajectory_dataset_each_generated_track(val_set, dataframe_mode=True)
    logger.info('Processing Test set')
    test_dataset = turn_splits_into_trajectory_dataset_each_generated_track(test_set, dataframe_mode=True)

    if save_as_numpy:
        if mem_mode:
            np.save(path_to_save + 'train_tracks.npy', train_dataset['tracks'])
            np.save(path_to_save + 'train_distances.npy', train_dataset['distances'])

            np.save(path_to_save + 'val_tracks.npy', val_dataset['tracks'])
            np.save(path_to_save + 'val_distances.npy', val_dataset['distances'])

            np.save(path_to_save + 'test_tracks.npy', test_dataset['tracks'])
            np.save(path_to_save + 'test_distances.npy', test_dataset['distances'])
        else:
            np.savez(path_to_save + 'train.npz', tracks=train_dataset['tracks'], distances=train_dataset['distances'])
            np.savez(path_to_save + 'val.npz', tracks=val_dataset['tracks'], distances=val_dataset['distances'])
            np.savez(path_to_save + 'test.npz', tracks=test_dataset['tracks'], distances=test_dataset['distances'])
    else:
        torch.save(train_dataset, path_to_save + 'train.pt')
        torch.save(val_dataset, path_to_save + 'val.pt')
        torch.save(test_dataset, path_to_save + 'test.pt')

    logger.info(f'Saved track datasets at {path_to_save}')


def get_relative_distances(arr, use_l2=False):
    relative_distances = []
    for idx in range(len(arr) - 1):
        if use_l2:
            dist = np.linalg.norm((np.expand_dims(arr[idx + 1, -2:], axis=0).astype(np.float32) -
                                   np.expand_dims(arr[idx, -2:], axis=0).astype(np.float32)),
                                  ord=2, axis=0)
        else:
            dist = (np.expand_dims(arr[idx + 1, -2:], axis=0).astype(np.float32) -
                    np.expand_dims(arr[idx, -2:], axis=0).astype(np.float32)).squeeze()
        relative_distances.append(dist)
    # plot_trajectory_with_relative_data(arr, relative_distances, relative_distances)
    return np.array(relative_distances)


def get_relative_distances_debug(arr, use_l2=False):
    relative_distances = []
    rel_dist_l2, rel_dist_simple = [], []
    for idx in range(len(arr) - 1):
        # if use_l2:
        #     dist = np.linalg.norm((np.expand_dims(arr[idx + 1, -2:], axis=0).astype(np.float32) -
        #                            np.expand_dims(arr[idx, -2:], axis=0).astype(np.float32)),
        #                           ord=2, axis=0)
        # else:
        #     dist = (np.expand_dims(arr[idx + 1, -2:], axis=0).astype(np.float32) -
        #             np.expand_dims(arr[idx, -2:], axis=0).astype(np.float32))
        dist1 = np.linalg.norm((np.expand_dims(arr[idx + 1, -2:], axis=0).astype(np.float32) -
                                np.expand_dims(arr[idx, -2:], axis=0).astype(np.float32)),
                               ord=2, axis=0)
        dist_simple = (np.expand_dims(arr[idx + 1, -2:], axis=0).astype(np.float32) -
                       np.expand_dims(arr[idx, -2:], axis=0).astype(np.float32))
        dist = dist_simple
        rel_dist_l2.append(dist1)
        rel_dist_simple.append(dist_simple)
        relative_distances.append(dist)
    plot_trajectory_with_relative_data(arr, rel_dist_l2, rel_dist_simple)
    return np.array(relative_distances)


def get_relative_distances_generated_track(arr, use_l2=False):
    relative_distances = []
    for idx in range(len(arr) - 1):
        if use_l2:
            dist = np.linalg.norm((np.expand_dims(arr[idx + 1, 7:9], axis=0).astype(np.float32) -
                                   np.expand_dims(arr[idx, 7:9], axis=0).astype(np.float32)),
                                  ord=2, axis=0)
        else:
            dist = (np.expand_dims(arr[idx + 1, 7:9], axis=0).astype(np.float32) -
                    np.expand_dims(arr[idx, 7:9], axis=0).astype(np.float32)).squeeze()
        relative_distances.append(dist)
    # plot_trajectory_with_relative_data(arr, relative_distances, relative_distances, generated=True)
    return np.array(relative_distances)


def array_split_by_length(arr, length=20):
    chunks = len(arr) // length
    compatible_array = arr[:chunks * length, ...]
    remaining_array = arr[chunks * length:, ...]
    splits = np.split(compatible_array, chunks)
    return splits, remaining_array


def generate_annotation_for_all():
    # for idx, video_class in enumerate(SDD_VIDEO_CLASSES_LIST):
    #     for video_number in SDD_PER_CLASS_VIDEOS_LIST[idx]:
    for idx, video_class in enumerate(SDD_VIDEO_CLASSES_RESUME_LIST):
        for video_number in SDD_PER_CLASS_VIDEOS_RESUME_LIST[idx]:
            logger.info(f'Processing for {video_class.value} - {video_number}')
            # for one is mem efficient
            split_annotations_and_save_as_track_datasets_by_length_for_one(
                annotation_path=f'{SDD_ANNOTATIONS_ROOT_PATH}{video_class.value}/video{video_number}/'
                                f'annotation_augmented.csv',
                path_to_save=f'{SAVE_BASE_PATH}{video_class.value}/video{video_number}/splits_v1/',
                by_track=False)
    logger.info('Finished generating all annotations!')


def generate_annotation_for_all_generated_tracks():
    for idx, video_class in enumerate(BUNDLED_ANNOTATIONS_VIDEO_CLASSES_LIST):
        for video_number in BUNDLED_ANNOTATIONS_PER_CLASSES_VIDEO_LIST[idx]:
            logger.info(f'Processing for {video_class.value} - {video_number}')
            # for one is mem efficient
            split_annotations_and_save_as_generated_track_datasets_by_length_for_one(
                annotation_path=f'{ROOT_PATH}Plots/baseline_v2/v{version}/{video_class.value}{video_number}/'
                                f'csv_annotation/generated_annotations.csv',
                path_to_save=f'{ROOT_PATH}Plots/baseline_v2/v{version}/{video_class.value}{video_number}/splits_v1/',
                by_track=False)
    logger.info('Finished generating all annotations!')


def generate_annotation_for_all_filtered_generated_tracks():
    video_clazzes = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
                     SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS,
                     SDDVideoClasses.QUAD]
    video_numbers = [[i for i in range(7)], [i for i in range(4)], [i for i in range(5)], [i for i in range(9)],
                     [i for i in range(15)], [i for i in range(4)], [i for i in range(12)], [i for i in range(4)]]
    for idx, video_class in enumerate(video_clazzes):
        for video_number in video_numbers[idx]:
            logger.info(f'Processing filtered annotation for {video_class.value} - {video_number}')
            # for one is mem efficient
            split_annotations_and_save_as_generated_track_datasets_by_length_for_one(
                annotation_path=f'{ROOT_PATH}Plots/baseline_v2/v{version}/{video_class.value}{video_number}/'
                                f'csv_annotation/filtered_generated_annotations.csv',
                path_to_save=f'{ROOT_PATH}Plots/baseline_v2/v{version}/{video_class.value}{video_number}/splits_v3/',
                by_track=False)
    logger.info('Finished generating all annotations!')


if __name__ == '__main__':
    # generate_annotation_for_all()
    # generate_annotation_for_all_generated_tracks()
    generate_annotation_for_all_filtered_generated_tracks()
    # ff = np.load('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/'
    #              'SDD_Features/nexus/video11/splits/train_distances.npy', allow_pickle=True, mmap_mode='r')
    # print()
    # split_annotations_and_save_as_track_datasets_by_length(annotation_path=ANNOTATION_CSV_PATH,
    #                                                        path_to_save=SPLIT_ANNOTATION_SAVE_PATH,
    #                                                        by_track=False)
    # make_trainable_trajectory_dataset_by_length(torch.load(DataSplitPath.TRAIN.value),
    #                                             save_path=SPLIT_ANNOTATION_SAVE_PATH + 'train.npz',
    #                                             numpy_save=False)

    # turn_splits_into_trajectory_dataset(split_path=DataSplitPath.TRAIN_CSV)
    # split_annotations_and_save_as_csv(annotation_path=ANNOTATION_CSV_PATH, path_to_save=SPLIT_ANNOTATION_SAVE_PATH)
    # verify_annotations_processing(video_path=VIDEO_PATH, df=sort_annotations_by_frame_numbers(ANNOTATION_CSV_PATH))
    # verify_annotations_processing(video_path=VIDEO_PATH, df=pd.read_csv(ANNOTATION_CSV_PATH, index_col='Unnamed: 0'))
    # annot = pd.read_csv(ANNOTATION_TXT_PATH, sep=' ')
    # annot.columns = ANNOTATION_COLUMNS
    # verify_annotations_processing(video_path=VIDEO_PATH, df=annot)
