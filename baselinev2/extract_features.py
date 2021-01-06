from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses
from average_image.feature_extractor import MOG2
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames

initialize_logging()
logger = get_logger(__name__)

SAVE_BASE_PATH = "../Datasets/SDD_Features/"
# SAVE_BASE_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD_Features/"
BASE_PATH = "../Datasets/SDD/"
# BASE_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/"
VIDEO_LABEL = SDDVideoClasses.LITTLE
VIDEO_NUMBER = 3
SAVE_PATH = f'{SAVE_BASE_PATH}{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/baseline_v2/'
FILE_NAME_STEP_1 = 'features_v0.pt'
LOAD_FILE_STEP_1 = SAVE_PATH + FILE_NAME_STEP_1
TIME_STEPS = 5

ENABLE_OF_OPTIMIZATION = True
ALPHA = 1
TOP_K = 1
WEIGHT_POINTS_INSIDE_BBOX_MORE = True

# -1 for both steps
EXECUTE_STEP = 2


def build_mog2_bg_model(n, frames, kernel, algo):
    out = None
    for frame in range(0, n):
        if out is None:
            out = np.zeros(shape=(0, frames[0].shape[0], frames[0].shape[1]))

        mask = algo.apply(frames[frame])
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        out = np.concatenate((out, np.expand_dims(mask, axis=0)), axis=0)
    return out


def get_mog2_foreground_mask(frames, interest_frame_idx, time_gap_within_frames, total_frames, step, n, kernel,
                             var_threshold):
    selected_past = [(interest_frame_idx - i * time_gap_within_frames) % total_frames for i in range(1, step + 1)]
    selected_future = [(interest_frame_idx + i * time_gap_within_frames) % total_frames for i in range(1, step + 1)]
    selected_frames = selected_past + selected_future
    frames_building_model = [frames[s] for s in selected_frames]

    algo = cv.createBackgroundSubtractorMOG2(history=n, varThreshold=var_threshold)
    _ = build_mog2_bg_model(n, frames_building_model, kernel, algo)

    mask = algo.apply(frames[interest_frame_idx], learningRate=0)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask


def associate_frame_with_ground_truth(frames, frame_numbers):
    return 0


def preprocess_data(save_per_part_path=SAVE_PATH):
    # feature_extractor = MOG2.for_frames()
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, 16)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    save_per_part_path += 'parts/'
    accumulated_features = {}
    for part_idx, data in enumerate(tqdm(data_loader)):
        frames, frame_numbers = data
        frames = frames.squeeze()
        frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        frames_count = frames.shape[0]
        original_shape = new_shape = [frames.shape[1], frames.shape[2]]
        for frame_idx, (frame, frame_number) in enumerate(zip(frames, frame_numbers)):
            if part_idx == 0:
                first_frame_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                            time_gap_within_frames=3,
                                                            total_frames=frames_count, step=step, n=n,
                                                            kernel=kernel, var_threshold=None)
            frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number)
            annotations, bbox_centers = scale_annotations(frame_annotation, original_scale=original_shape,
                                                          new_scale=new_shape, return_track_id=False,
                                                          tracks_with_annotations=True)
        gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
        if save_per_part_path is not None:
            Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
            f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
            torch.save(accumulated_features, save_per_part_path + f_n)

    return accumulated_features


if __name__ == '__main__':
    feats = preprocess_data()
