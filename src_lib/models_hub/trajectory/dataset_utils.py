from typing import Sequence

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from average_image.constants import SDDVideoClasses
from log import get_logger

logger = get_logger(__name__)

EXTRACTION_BASE_PATH = '../../../src/position_maps/logs/Trajectories/'
SAVE_BASE_PATH = '../../../Datasets/SDD/pm_extracted_annotations/'

ALL_VIDEO_CLASSES = [
    SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
    SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS,
    SDDVideoClasses.QUAD]
ALL_VIDEO_NUMBERS = [
    [i for i in range(7)], [i for i in range(4)], [i for i in range(5)], [i for i in range(9)],
    [i for i in range(15)], [i for i in range(4)], [i for i in range(12)], [i for i in range(4)]]


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


def process_annotation(annotation_path, path_to_save):
    df = pd.read_csv(annotation_path)

    train_set, val_set, test_set = split_annotations(df)
    print()


def generate_annotation_for_all_extracted_tracks(video_classes, video_numbers):
    for idx, video_class in enumerate(video_classes):
        for video_number in video_numbers[idx]:
            logger.info(f'Processing extracted annotation for {video_class.value} - {video_number}')
            process_annotation(
                annotation_path=f'{EXTRACTION_BASE_PATH}/{video_class.name}/{video_number}/trajectories.csv',
                path_to_save=f'{SAVE_BASE_PATH}/{video_class.value}/video{video_number}/v0/')
    logger.info('Finished generating all annotations!')


if __name__ == '__main__':
    generate_annotation_for_all_extracted_tracks(
        video_classes=[ALL_VIDEO_CLASSES[2]],
        video_numbers=[ALL_VIDEO_NUMBERS[2]])
