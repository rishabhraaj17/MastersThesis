from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from baselinev2.nn.data_utils import array_split_by_length
from baselinev2.structures import SingleTrack
from log import get_logger

logger = get_logger(__name__)

FUTURE_TIMESTEPS = 2
IS_GAN = False
EXTRACTION_BASE_PATH = f'../../Datasets/SDD/classic_nn_extended_annotations_new_v1/d{FUTURE_TIMESTEPS}/{"gan" if IS_GAN else "simple"}/'
SAVE_BASE_PATH = f'../../Datasets/SDD/classic_nn_extended_annotations_new_v1_splits/d{FUTURE_TIMESTEPS}/{"gan" if IS_GAN else "simple"}/'

ALL_VIDEO_CLASSES = [
    SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
    SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS,
    SDDVideoClasses.QUAD]
ALL_VIDEO_NUMBERS = [
    [i for i in range(7)], [i for i in range(4)],
    [i for i in range(5)],
    [i for i in range(9)],
    [i for i in range(15)], [i for i in range(4)], [i for i in range(12) if i not in [3, 4, 5]], [i for i in range(4)]]


def adjust_dataframe_framerate(df, for_gt, frame_rate=30., time_step=0.4):
    frame_txt = 'frame_id' if for_gt else 'frame'
    df = df[df[frame_txt] % int(round(frame_rate * time_step)) == 0]
    df[frame_txt] /= int(round(frame_rate * time_step))
    return df


def adjust_splits_for_frame(larger_set, smaller_set):
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


def split_annotations(annotations):
    train_set, test_set = train_test_split(annotations, train_size=0.7,
                                           test_size=0.3,
                                           shuffle=False, stratify=None)

    train_set, test_set = adjust_splits_for_frame(larger_set=train_set, smaller_set=test_set)

    test_set, val_set = train_test_split(test_set, train_size=0.3,
                                         test_size=0.7, shuffle=False, stratify=None)

    val_set, test_set = adjust_splits_for_frame(larger_set=test_set, smaller_set=val_set)
    return train_set, val_set, test_set


def make_trainable_trajectory_dataset_by_length(track_obj, length=20):
    final_tracks, relative_distances = [], []
    for track in track_obj:
        if len(track.data) > length:
            # split array into parts of max length
            splits, remaining_part = array_split_by_length(track.data, length=length)
            for split in splits:
                rel_distances = np.diff(split[:, 2:], axis=0).astype(np.float32)
                relative_distances.append(rel_distances)
                final_tracks.append(split.astype(np.float32))

    final_tracks = np.stack(final_tracks) if len(final_tracks) != 0 else np.zeros((0, 4))
    relative_distances = np.stack(relative_distances) if len(relative_distances) != 0 else np.zeros((0, 2))

    return final_tracks, relative_distances


def turn_splits_into_trajectory_dataset(split: pd.DataFrame, num_frames_in_jump=12, time_between_frames=0.4):
    # unique_tracks = split.track.unique()
    unique_tracks = split.track_id.unique()

    final_tracks_save, relative_distances_save = [], []
    for t_id in tqdm(unique_tracks):
        # track_df = split[split.track == t_id]
        track_df = split[split.track_id == t_id]

        tracks_list: List[SingleTrack] = []
        for start_frame in range(len(track_df)):
            track_df_with_jumps: pd.DataFrame = track_df.iloc[start_frame::num_frames_in_jump]
            track: SingleTrack = SingleTrack(
                data=track_df_with_jumps.to_numpy(),
                frames=track_df_with_jumps.frame.to_numpy(),
                valid=True if len(track_df_with_jumps) >= 20 else False
            )
            tracks_list.append(track)

        tracks, distances = make_trainable_trajectory_dataset_by_length(track_obj=tracks_list)
        if tracks.size != 0:
            final_tracks_save.append(tracks)
        if distances.size != 0:
            relative_distances_save.append(distances)

    final_tracks_save = np.concatenate(final_tracks_save) if len(final_tracks_save) != 0 else np.zeros((0, 4))
    relative_distances_save = np.concatenate(relative_distances_save) if len(relative_distances_save) != 0 \
        else np.zeros((0, 2))

    return {'tracks': final_tracks_save, 'distances': relative_distances_save}


def process_annotation(annotation_path, path_to_save):
    df = pd.read_csv(annotation_path)

    train_set, val_set, test_set = split_annotations(df)

    logger.info('Processing Train set')
    train_dataset = turn_splits_into_trajectory_dataset(train_set, num_frames_in_jump=1)
    logger.info('Processing Validation set')
    val_dataset = turn_splits_into_trajectory_dataset(val_set, num_frames_in_jump=1)
    logger.info('Processing Test set')
    test_dataset = turn_splits_into_trajectory_dataset(test_set, num_frames_in_jump=1)

    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    np.save(path_to_save + 'train_tracks.npy', train_dataset['tracks'])
    np.save(path_to_save + 'train_distances.npy', train_dataset['distances'])

    np.save(path_to_save + 'val_tracks.npy', val_dataset['tracks'])
    np.save(path_to_save + 'val_distances.npy', val_dataset['distances'])

    np.save(path_to_save + 'test_tracks.npy', test_dataset['tracks'])
    np.save(path_to_save + 'test_distances.npy', test_dataset['distances'])

    logger.info(f'Saved track datasets at {path_to_save}')


def generate_annotation_for_all_extracted_tracks(video_classes, video_numbers, annotation_path=EXTRACTION_BASE_PATH,
                                                 path_to_save=SAVE_BASE_PATH):
    for idx, video_class in enumerate(video_classes):
        for video_number in video_numbers[idx]:
            logger.info(f'Processing extracted annotation for {video_class.value} - {video_number}')
            process_annotation(
                annotation_path=f'{annotation_path}{video_class.value}/video{video_number}/annotation.csv',
                path_to_save=f'{path_to_save}{video_class.value}/video{video_number}/v0/')
    logger.info('Finished generating all annotations!')


if __name__ == '__main__':
    future_time_steps = [2, 5]
    model_version = ['simple', 'gan']
    for ft in future_time_steps:
        for mv in model_version:
            annotation_path = f'../../Datasets/SDD/classic_nn_extended_annotations_new_v1/d{ft}/{mv}/'
            path_to_save = f'../../Datasets/SDD/extended_splits/classic_nn_extended_annotations_new_v1/d{ft}/{mv}/'
            generate_annotation_for_all_extracted_tracks(
                video_classes=ALL_VIDEO_CLASSES,
                video_numbers=ALL_VIDEO_NUMBERS,
                annotation_path=annotation_path,
                path_to_save=path_to_save)
