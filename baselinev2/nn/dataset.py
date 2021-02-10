from enum import Enum
from pathlib import Path
from typing import Dict, List

import cv2 as cv
import numpy as np
import pandas as pd
import skimage
import torch
from torch.utils.data import Dataset, ConcatDataset, ChainDataset
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, get_frame_annotations
from average_image.constants import ANNOTATION_COLUMNS, SDDVideoClasses
from baselinev2.config import VIDEO_PATH, ANNOTATION_CSV_PATH, VIDEO_SAVE_PATH, ANNOTATION_TXT_PATH, \
    TRAIN_SPLIT_PERCENTAGE, VALIDATION_SPLIT_PERCENTAGE, TEST_SPLIT_PERCENTAGE, SPLIT_ANNOTATION_SAVE_PATH, \
    SDD_ANNOTATIONS_ROOT_PATH, SDD_VIDEO_CLASSES_LIST, SDD_PER_CLASS_VIDEOS_LIST, SAVE_BASE_PATH, \
    SDD_VIDEO_CLASSES_RESUME_LIST, SDD_PER_CLASS_VIDEOS_RESUME_LIST
from baselinev2.constants import NetworkMode
from baselinev2.plot_utils import plot_for_video_image_and_box
from baselinev2.structures import TracksDataset, SingleTrack
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.nn.dataset')


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


def sort_annotations_by_frame_numbers(annotation_path):
    annotations = pd.read_csv(annotation_path, index_col='Unnamed: 0')
    annotations = annotations.sort_values(by=['frame']).reset_index()
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


def adjust_splits_for_frame(larger_set, smaller_set):
    last_frame_in_larger = larger_set.iloc[-1]['frame']
    first_frame_in_smaller = smaller_set.iloc[0]['frame']

    if last_frame_in_larger == first_frame_in_smaller:
        same_frame_larger = larger_set[larger_set['frame'] == last_frame_in_larger]
        same_frame_smaller = smaller_set[smaller_set['frame'] == first_frame_in_smaller]

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

    return {'tracks': np.concatenate(final_tracks_save), 'distances': np.concatenate(relative_distances_save)}


def make_trainable_trajectory_dataset_by_length_each_track(track_obj, length=20, save_path=None,
                                                           numpy_save=False):
    final_tracks, relative_distances = [], []
    for track in track_obj:
        if len(track.data) > length:
            # split array into parts of max length
            splits, remaining_part = array_split_by_length(track.data, length=length)
            for split in splits:
                relative_distances.append(get_relative_distances(split))
                final_tracks.append(np.hstack((split[:, 0:6], split[:, 10:])))

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


def get_relative_distances(arr):
    relative_distances = []
    for idx in range(len(arr) - 1):
        dist = np.linalg.norm((np.expand_dims(arr[idx + 1, -2:], axis=0).astype(np.float32) -
                               np.expand_dims(arr[idx, -2:], axis=0).astype(np.float32)),
                              ord=2, axis=0)
        relative_distances.append(dist)
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
                path_to_save=f'{SAVE_BASE_PATH}{video_class.value}/video{video_number}/splits/',
                by_track=False)
    logger.info('Finished generating all annotations!')


class BaselineDataset(Dataset):
    def __init__(self, video_class: SDDVideoClasses, video_number: int, split: NetworkMode, root: str = SAVE_BASE_PATH,
                 observation_length: int = 8, prediction_length: int = 12):
        super(BaselineDataset, self).__init__()
        path_to_dataset = f'{root}{video_class.value}/video{video_number}/splits/{split.value}.pt'
        dataset = torch.load(path_to_dataset)

        self.tracks = dataset['tracks']
        self.relative_distances = dataset['distances']

        self.prediction_length = prediction_length
        self.observation_length = observation_length

    def __len__(self):
        return len(self.relative_distances)

    def __getitem__(self, item):
        tracks, relative_distances = self.tracks[item], self.relative_distances[item]
        in_xy = torch.from_numpy(tracks[..., :self.observation_length, -2:].astype(np.float32))
        gt_xy = torch.from_numpy(tracks[..., self.observation_length:, -2:].astype(np.float32))
        in_velocities = torch.from_numpy(relative_distances[..., :self.observation_length - 1, :].astype(np.float32))
        gt_velocities = torch.from_numpy(relative_distances[..., self.observation_length:, :].astype(np.float32))
        in_track_ids = torch.from_numpy(tracks[..., :self.observation_length, 0].astype(np.float32))
        gt_track_ids = torch.from_numpy(tracks[..., self.observation_length:, 0].astype(np.float32))
        in_frame_numbers = torch.from_numpy(tracks[..., :self.observation_length, 5].astype(np.float32))
        gt_frame_numbers = torch.from_numpy(tracks[..., self.observation_length:, 5].astype(np.float32))

        return in_xy, gt_xy, in_velocities, gt_velocities, in_track_ids, gt_track_ids, in_frame_numbers, \
               gt_frame_numbers


if __name__ == '__main__':
    # d1 = BaselineDataset(SDDVideoClasses.NEXUS, 11, NetworkMode.TRAIN)
    # d2 = BaselineDataset(SDDVideoClasses.NEXUS, 11, NetworkMode.TRAIN)
    # d = ChainDataset([d1, d2])
    # dd = d[0:32]
    # print()
    generate_annotation_for_all()
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
