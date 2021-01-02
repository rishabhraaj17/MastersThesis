import copy
import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
# import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from average_image.feature_extractor import MOG2, FeatureExtractor
from average_image.utils import BasicTestData, BasicTrainData
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames
from baseline.extracted_of_optimization import cost_function
from unsupervised_tp_0.nn_clustering_0 import find_center_point

# ray.init()
initialize_logging()
logger = get_logger(__name__)

SAVE_BASE_PATH = "../Datasets/SDD_Features/"
# SAVE_BASE_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD_Features/"
BASE_PATH = "../Datasets/SDD/"
# BASE_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/"
VIDEO_LABEL = SDDVideoClasses.LITTLE
VIDEO_NUMBER = 3
SAVE_PATH = f'{SAVE_BASE_PATH}{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/'
FILE_NAME_STEP_1 = 'time_distributed_dict_with_gt_bbox_centers_and_bbox_gt_velocity.pt'
LOAD_FILE_STEP_1 = SAVE_PATH + FILE_NAME_STEP_1
TIME_STEPS = 10

ENABLE_OF_OPTIMIZATION = True
ALPHA = 1
TOP_K = 1
WEIGHT_POINTS_INSIDE_BBOX_MORE = True

# -1 for both steps
EXECUTE_STEP = 1


def _process_preloaded_n_frames(n, frames, kernel, algo):
    out = None
    for frame in range(0, n):
        if out is None:
            out = np.zeros(shape=(0, frames[0].shape[0], frames[0].shape[1]))

        mask = algo.apply(frames[frame])
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        out = np.concatenate((out, np.expand_dims(mask, axis=0)), axis=0)
    return out


def bg_sub_for_frame_equal_time_distributed(frames, fr, time_gap_within_frames, total_frames, step, n, kernel,
                                            var_threshold, interest_fr):
    selected_past = [(fr - i * time_gap_within_frames) % total_frames for i in range(1, step + 1)]
    selected_future = [(fr + i * time_gap_within_frames) % total_frames for i in range(1, step + 1)]
    selected_frames = selected_past + selected_future
    frames_building_model = [frames[s] for s in selected_frames]

    algo = cv.createBackgroundSubtractorMOG2(history=n, varThreshold=var_threshold)
    _ = _process_preloaded_n_frames(n, frames_building_model, kernel, algo)

    mask = algo.apply(frames[interest_fr], learningRate=0)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask


def preprocess_data(classic_clustering=False, equal_time_distributed=True, save_per_part_path=SAVE_PATH,
                    num_frames_to_build_bg_sub_model=12):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, 16)
    save_per_part_path += 'parts/'
    remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch = None, None, None
    past_12_frames_optical_flow = []
    gt_velocity_dict, accumulated_features = {}, {}
    for part_idx, data in enumerate(tqdm(data_loader)):
        frames, frame_numbers = data
        frames = frames.squeeze()
        feature_extractor = MOG2.for_frames()
        features_, remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch, \
        past_12_frames_optical_flow, gt_velocity_dict = feature_extractor. \
            keyframe_based_clustering_from_frames_nn(frames=frames, n=30, use_last_n_to_build_model=False,
                                                     frames_to_build_model=num_frames_to_build_bg_sub_model,
                                                     original_shape=sdd_simple.original_shape,
                                                     resized_shape=sdd_simple.new_scale,
                                                     classic_clustering=classic_clustering,
                                                     object_of_interest_only=False,
                                                     var_threshold=None, track_ids=None,
                                                     all_object_of_interest_only=True,
                                                     equal_time_distributed=equal_time_distributed,
                                                     frame_numbers=frame_numbers,
                                                     df=sdd_simple.annotations_df,
                                                     return_normalized=False,
                                                     remaining_frames=remaining_frames,
                                                     remaining_frames_idx=remaining_frames_idx,
                                                     past_12_frames_optical_flow=past_12_frames_optical_flow,
                                                     last_frame_from_last_used_batch=
                                                     last_frame_from_last_used_batch,
                                                     gt_velocity_dict=gt_velocity_dict)
        # plot_extracted_features_and_verify_flow(features_, frames)
        accumulated_features = {**accumulated_features, **features_}
        if save_per_part_path is not None:
            Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
            f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
            torch.save(accumulated_features, save_per_part_path + f_n)

    return accumulated_features


def get_resume_data(part_idx, frames, frame_numbers, last_itr_frames, last_itr_frame_numbers, resume_past_12_of):
    save_per_part_path = SAVE_PATH + 'parts/'
    file_name = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity_optimized_of_{part_idx}.pt'
    # processed_data = torch.load(save_per_part_path + file_name)

    features_, total_frames = None, None
    remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch = None, None, None
    past_12_frames_optical_flow, last12_bg_sub_mask = {}, {}

    remaining_frames_idx = frame_numbers[4:]
    last_frame_from_last_used_batch_idx = frame_numbers[3]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    if part_idx == 0:
        frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        frames_count = frames.shape[0]
        for idx, p_idx in zip(range(0, 4), frame_numbers):
            fg_mask = bg_sub_for_frame_equal_time_distributed(frames=frames, fr=idx,
                                                              time_gap_within_frames=3,
                                                              total_frames=frames_count, step=30 // 2, n=30,
                                                              kernel=kernel, var_threshold=None,
                                                              interest_fr=idx)
            last12_bg_sub_mask.update({p_idx.item(): fg_mask})
        for idx, p_idx in zip(range(1, 4), frame_numbers):
            # fg_mask = bg_sub_for_frame_equal_time_distributed(frames=frames, fr=idx,
            #                                                   time_gap_within_frames=3,
            #                                                   total_frames=frames_count, step=30//2, n=30,
            #                                                   kernel=kernel, var_threshold=None,
            #                                                   interest_fr=idx)
            # last12_bg_sub_mask.update({p_idx.item(): fg_mask})

            previous = cv.cvtColor(frames[idx - 1], cv.COLOR_BGR2GRAY)
            next_frame = cv.cvtColor(frames[idx], cv.COLOR_BGR2GRAY)

            past_flow_per_frame, past_rgb, past_mag, past_ang = FeatureExtractor.get_optical_flow(
                previous_frame=previous,
                next_frame=next_frame,
                all_results_out=True)
            resume_past_12_of.update({f'{p_idx}-{p_idx + 1}': past_flow_per_frame})
        # fg_mask = bg_sub_for_frame_equal_time_distributed(frames=frames, fr=3,
        #                                                   time_gap_within_frames=3,
        #                                                   total_frames=frames_count, step=30 // 2, n=30,
        #                                                   kernel=kernel, var_threshold=None,
        #                                                   interest_fr=3)
        # last12_bg_sub_mask.update({3: fg_mask})
    if part_idx > 0:
        if last_itr_frame_numbers is not None:
            last_round_frames = torch.cat((last_itr_frame_numbers, frame_numbers[:4]))
            total_frames = torch.cat((last_itr_frames[-12:], frames), 0)
            last_frame_from_last_used_batch = (total_frames[15].permute(1, 2, 0) * 255).int().numpy()

        features_idx_ = list(range(last_frame_from_last_used_batch_idx - 15, last_frame_from_last_used_batch_idx + 1))
        past_12_idx = list(range(last_frame_from_last_used_batch_idx - 12, last_frame_from_last_used_batch_idx + 1))

        # total_frames = (total_frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        remaining_frames = (last_itr_frames[-12:] * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        frames_for_of = np.concatenate((remaining_frames, frames), axis=0)
        frames_count = frames_for_of.shape[0]
        for idx, p_idx in zip(range(3, 16), past_12_idx):
            fg_mask = bg_sub_for_frame_equal_time_distributed(frames=frames_for_of, fr=idx,
                                                              time_gap_within_frames=3,
                                                              total_frames=frames_count, step=30 // 2, n=30,
                                                              kernel=kernel, var_threshold=None,
                                                              interest_fr=idx)
            last12_bg_sub_mask.update({p_idx: fg_mask})

        for idx, p_idx in zip(range(4, 17), past_12_idx[:-1]):
            # fg_mask = bg_sub_for_frame_equal_time_distributed(frames=frames_for_of, fr=idx,
            #                                                   time_gap_within_frames=3,
            #                                                   total_frames=frames_count, step=30 // 2, n=30,
            #                                                   kernel=kernel, var_threshold=None,
            #                                                   interest_fr=idx)
            # last12_bg_sub_mask.update({p_idx: fg_mask})

            previous = cv.cvtColor(frames_for_of[idx - 1], cv.COLOR_BGR2GRAY)
            next_frame = cv.cvtColor(frames_for_of[idx], cv.COLOR_BGR2GRAY)

            past_flow_per_frame, past_rgb, past_mag, past_ang = FeatureExtractor.get_optical_flow(
                previous_frame=previous,
                next_frame=next_frame,
                all_results_out=True)
            resume_past_12_of.update({f'{p_idx}-{p_idx + 1}': past_flow_per_frame})

        # fg_mask = bg_sub_for_frame_equal_time_distributed(frames=frames_for_of, fr=17,
        #                                                   time_gap_within_frames=3,
        #                                                   total_frames=frames_count, step=30 // 2, n=30,
        #                                                   kernel=kernel, var_threshold=None,
        #                                                   interest_fr=17)
        # last12_bg_sub_mask.update({past_12_idx[-1]: fg_mask})

    # resume_past_12_of = None
    if len(resume_past_12_of) > 12:
        temp_past_12_frames_optical_flow = {}
        for i in list(resume_past_12_of)[-12:]:
            temp_past_12_frames_optical_flow.update({i: resume_past_12_of[i]})
        resume_past_12_of = temp_past_12_frames_optical_flow

    if len(last12_bg_sub_mask) > 13:
        temp_last12_bg_sub_mask = {}
        for i in list(last12_bg_sub_mask)[-13:]:
            temp_last12_bg_sub_mask.update({i: last12_bg_sub_mask[i]})
        last12_bg_sub_mask = temp_last12_bg_sub_mask

    return features_, remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch, \
           resume_past_12_of, last12_bg_sub_mask, total_frames


def preprocess_optical_flow_optimized_data(classic_clustering=False, equal_time_distributed=True,
                                           save_per_part_path=SAVE_PATH,
                                           num_frames_to_build_bg_sub_model=12,
                                           resume=True, resume_idx=680):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, 16)
    save_per_part_path += 'parts/'
    remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch = None, None, None
    last_optical_flow_map, last12_bg_sub_mask = None, {}
    past_12_frames_optical_flow, last_itr_past_12_frames_optical_flow, last_itr_past_12_bg_sub_mask = {}, {}, {}
    gt_velocity_dict, accumulated_features = {}, {}
    last_itr_frames, last_itr_frame_numbers = None, None
    resume_past_12_of = {}
    if resume and resume_idx is not None:
        file_name = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity_optimized_of_{resume_idx}.pt'
        # accumulated_features = torch.load(save_per_part_path + file_name)
    for part_idx, data in enumerate(tqdm(data_loader)):
        frames, frame_numbers = data
        frames = frames.squeeze()
        feature_extractor = MOG2.for_frames()
        if resume and part_idx <= resume_idx:
            # features_, remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch, \
            # past_12_frames_optical_flow, gt_velocity_dict, last_optical_flow_map, last12_bg_sub_mask = feature_extractor. \
            #     keyframe_based_feature_extraction_optimized_optical_flow_from_frames_nn(
            #     frames=frames, n=30,
            #     use_last_n_to_build_model=False,
            #     frames_to_build_model=num_frames_to_build_bg_sub_model,
            #     original_shape=sdd_simple.original_shape,
            #     resized_shape=sdd_simple.new_scale,
            #     classic_clustering=classic_clustering,
            #     object_of_interest_only=False,
            #     var_threshold=None, track_ids=None,
            #     all_object_of_interest_only=False,
            #     all_object_of_interest_only_with_optimized_of=True,
            #     equal_time_distributed=equal_time_distributed,
            #     frame_numbers=frame_numbers,
            #     df=sdd_simple.annotations_df,
            #     return_normalized=False,
            #     remaining_frames=remaining_frames,
            #     remaining_frames_idx=remaining_frames_idx,
            #     past_12_frames_optical_flow=past_12_frames_optical_flow,
            #     last_frame_from_last_used_batch=
            #     last_frame_from_last_used_batch,
            #     gt_velocity_dict=gt_velocity_dict,
            #     last_optical_flow_map=last_optical_flow_map,
            #     last12_bg_sub_mask=last12_bg_sub_mask,
            #     resume_mode=True)
            last_itr_past_12_frames_optical_flow = copy.deepcopy(resume_past_12_of)
            r_features_, r_remaining_frames, r_remaining_frames_idx, r_last_frame_from_last_used_batch, \
            resume_past_12_of, r_last12_bg_sub_mask, r_total_frames = get_resume_data(
                part_idx, frames, frame_numbers, last_itr_frames, last_itr_frame_numbers,
                resume_past_12_of)
            last_itr_frames, last_itr_frame_numbers = frames.clone(), frame_numbers.clone()
            # else:
            features_, remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch, \
            past_12_frames_optical_flow, gt_velocity_dict, last_optical_flow_map, last12_bg_sub_mask = feature_extractor. \
                keyframe_based_feature_extraction_optimized_optical_flow_from_frames_nn(
                frames=frames, n=30,
                use_last_n_to_build_model=False,
                frames_to_build_model=num_frames_to_build_bg_sub_model,
                original_shape=sdd_simple.original_shape,
                resized_shape=sdd_simple.new_scale,
                classic_clustering=classic_clustering,
                object_of_interest_only=False,
                var_threshold=None, track_ids=None,
                all_object_of_interest_only=False,
                all_object_of_interest_only_with_optimized_of=True,
                equal_time_distributed=equal_time_distributed,
                frame_numbers=frame_numbers,
                df=sdd_simple.annotations_df,
                return_normalized=False,
                remaining_frames=remaining_frames,
                remaining_frames_idx=remaining_frames_idx,
                past_12_frames_optical_flow=past_12_frames_optical_flow,
                last_frame_from_last_used_batch=
                last_frame_from_last_used_batch,
                gt_velocity_dict=gt_velocity_dict,
                last_optical_flow_map=last_optical_flow_map,
                last12_bg_sub_mask=last12_bg_sub_mask,
                resume_mode=True)  # fixme: remove
            # last_itr_past_12_frames_optical_flow = resume_past_12_of
            accumulated_features = {**accumulated_features, **features_}
            last_itr_past_12_bg_sub_mask = copy.deepcopy(r_last12_bg_sub_mask)

        # plot_extracted_features_and_verify_flow(features_, frames)
        # accumulated_features = {**accumulated_features, **features_}
        if (part_idx % 170 == 0) and not part_idx < resume_idx and save_per_part_path is not None:
            Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
            f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity_optimized_of_{part_idx}.pt'
            logger.info(f'Saving : {f_n}')
            torch.save(accumulated_features, save_per_part_path + f_n)

    return accumulated_features


def process_complex_features_rnn(features_dict: dict, test_mode=False, time_steps=TIME_STEPS,
                                 num_frames_to_build_bg_sub_model=12):
    processed_data = {}
    per_frame_data, per_batch_data = [], []
    total_frames = len(features_dict)
    for frame_ in tqdm(range(num_frames_to_build_bg_sub_model,
                             total_frames - num_frames_to_build_bg_sub_model * time_steps)):
        for k in range(1, time_steps + 1):
            pair_0 = features_dict[frame_ + num_frames_to_build_bg_sub_model * (k - 1)]
            pair_1 = features_dict[(frame_ + num_frames_to_build_bg_sub_model * k) % total_frames]
            for i in pair_0:
                for j in pair_1:
                    if i == j:  # and i.track_id == 8:  # bad readability
                        if test_mode:
                            per_frame_data.append(BasicTestData(frame=frame_, track_id=i.track_id,
                                                                pair_0_features=i.features,
                                                                pair_1_features=j.features,
                                                                pair_0_normalize=i.normalize_params,
                                                                pair_1_normalize=j.normalize_params))
                        else:
                            per_frame_data.append(
                                BasicTrainData(frame=frame_ + num_frames_to_build_bg_sub_model * (k - 1),
                                               track_id=i.track_id,
                                               pair_0_features=i.features,
                                               pair_1_features=j.features,
                                               frame_t0=i.frame_number,
                                               frame_t1=j.frame_number,
                                               bbox_center_t0=i.bbox_center,
                                               bbox_center_t1=j.bbox_center,
                                               bbox_t0=i.bbox,
                                               bbox_t1=j.bbox,
                                               track_gt_velocity_t0=i.track_gt_velocity,
                                               track_gt_velocity_t1=j.track_gt_velocity))
                            # per_frame_data.append([i.features, j.features])
            per_batch_data.append(per_frame_data)
            per_frame_data = []
        processed_data.update({frame_: per_batch_data})
        per_batch_data = []
    return processed_data


def extract_trainable_features_rnn(data, return_frame_info=True):
    frame_info, track_id_info = [], []
    x_, y_ = [], []
    bbox_x, bbox_y, bbox_center_x, bbox_center_y, gt_velocity_x, gt_velocity_y = [], [], [], [], [], []

    for key, value in tqdm(data.items()):
        num_frames = len(value)
        t_0 = value[0]
        t_rest = [value[v] for v in range(1, num_frames)]
        for fr in t_0:
            temp_x, temp_y, temp_f_info, temp_track_info = [], [], [], []
            temp_bbox_x, temp_bbox_y, temp_bbox_center_x, temp_bbox_center_y = [], [], [], []
            temp_gt_velocity_x, temp_gt_velocity_y = [], []

            temp_f_info.append(fr.frame)
            temp_track_info.append(fr.track_id)
            temp_x.append(fr.pair_0_features)
            temp_y.append(fr.pair_1_features)
            temp_bbox_center_x.append(fr.bbox_center_t0)
            temp_bbox_center_y.append(fr.bbox_center_t1)
            temp_bbox_x.append(fr.bbox_t0)
            temp_bbox_y.append(fr.bbox_t1)
            temp_gt_velocity_x.append(fr.track_gt_velocity_t0)
            temp_gt_velocity_y.append(fr.track_gt_velocity_t1)

            for t_i in t_rest:
                for fr_other in t_i:
                    if fr == fr_other:
                        temp_f_info.append(fr_other.frame)
                        temp_track_info.append(fr_other.track_id)
                        temp_x.append(fr_other.pair_0_features)
                        temp_y.append(fr_other.pair_1_features)
                        temp_bbox_center_x.append(fr_other.bbox_center_t0)
                        temp_bbox_center_y.append(fr_other.bbox_center_t1)
                        temp_bbox_x.append(fr_other.bbox_t0)
                        temp_bbox_y.append(fr_other.bbox_t1)
                        temp_gt_velocity_x.append(fr_other.track_gt_velocity_t0)
                        temp_gt_velocity_y.append(fr_other.track_gt_velocity_t1)

            frame_info.append(temp_f_info)
            track_id_info.append(temp_track_info)
            x_.append(temp_x)
            y_.append(temp_y)
            bbox_center_x.append(temp_bbox_center_x)
            bbox_center_y.append(temp_bbox_center_y)
            bbox_x.append(temp_bbox_x)
            bbox_y.append(temp_bbox_y)
            gt_velocity_x.append(temp_gt_velocity_x)
            gt_velocity_y.append(temp_gt_velocity_y)

    if return_frame_info:
        return x_, y_, frame_info, track_id_info, bbox_center_x, bbox_center_y, bbox_x, bbox_y, gt_velocity_x, \
               gt_velocity_y
    return x_, y_


def center_based_dataset(features):
    features_x, features_y, frames, track_ids, center_x, center_y, bbox_x, bbox_y, gt_velocity_x, gt_velocity_y \
        = features['x'], features['y'], features['frames'], features['track_ids'], features['bbox_center_x'], \
          features['bbox_center_y'], features['bbox_x'], features['bbox_y'], features['gt_velocity_x'], \
          features['gt_velocity_y']
    skipped_counter = 0
    features_center = []
    for feat1, velocity_x, y_center, y_bbox in tqdm(zip(features_x, gt_velocity_x, center_y, bbox_y),
                                                    total=len(features_x)):
        f1_list = []
        for f1, vx, y_cen, y_box in zip(feat1, velocity_x, y_center, y_bbox):
            center_xy, center_true_uv, center_past_uv = find_center_point(f1)
            if ENABLE_OF_OPTIMIZATION:
                shifted_points = f1[:, :2] + f1[:, 2:4]
                _, weighted_shifted_points_top_k = cost_function(shifted_points, y_cen, top_k=TOP_K, alpha=ALPHA,
                                                                 bbox=y_box,
                                                                 weight_points_inside_bbox_more=
                                                                 WEIGHT_POINTS_INSIDE_BBOX_MORE)
                shifted_points_center = weighted_shifted_points_top_k[0]
            else:
                shifted_points_center = (f1[:, :2] + f1[:, 2:4]).mean(axis=0)
            try:
                f1_list.append(np.vstack((center_xy, center_true_uv, center_past_uv, shifted_points_center, vx)))
            except ValueError:
                skipped_counter += 1
        features_center.append(f1_list)

    save_dict = {'center_based': features_center,
                 'x': features_x,
                 'y': features_y,
                 'frames': frames,
                 'track_ids': track_ids,
                 'bbox_center_x': center_x,
                 'bbox_center_y': center_y,
                 'bbox_x': bbox_x,
                 'bbox_y': bbox_y,
                 'gt_velocity_x': gt_velocity_x,
                 'gt_velocity_y': gt_velocity_y}
    logger.info(f'Skipped: {skipped_counter}')
    return save_dict


def step_one(with_optimized_of=True):
    if with_optimized_of:
        accumulated_features = preprocess_optical_flow_optimized_data()
    else:
        accumulated_features = preprocess_data()
    logger.info(f'Saving the features for video {VIDEO_LABEL.value}, video {VIDEO_NUMBER}')
    if SAVE_PATH:
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    torch.save(accumulated_features, SAVE_PATH + FILE_NAME_STEP_1)


def step_two():
    features = torch.load(LOAD_FILE_STEP_1)

    # Step 2.1
    logger.info(f'Processing the features for video {VIDEO_LABEL.value}, video {VIDEO_NUMBER}, {TIME_STEPS} time-steps')
    feats_data = process_complex_features_rnn(features, time_steps=TIME_STEPS)

    # Step 2.2
    logger.info(f'Extracting the features for video {VIDEO_LABEL.value}, video {VIDEO_NUMBER}, {TIME_STEPS} time-steps')

    x, y, frame_info, track_id_info, bbox_center_x, bbox_center_y, bbox_x, bbox_y, gt_velocity_x, gt_velocity_y = \
        extract_trainable_features_rnn(feats_data, return_frame_info=True)
    features_save_dict = {'x': x, 'y': y, 'frames': frame_info,
                          'track_ids': track_id_info,
                          'bbox_center_x': bbox_center_x, 'bbox_center_y': bbox_center_y,
                          'bbox_x': bbox_x, 'bbox_y': bbox_y, 'gt_velocity_x': gt_velocity_x,
                          'gt_velocity_y': gt_velocity_y}

    # Step 2.3
    logger.info(f'Preparing the features for video {VIDEO_LABEL.value}, video {VIDEO_NUMBER}, {TIME_STEPS} time-steps')
    features = center_based_dataset(features_save_dict)
    file_name = f'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_' \
                f'center_based_gt_velocity_of_optimized_t{TIME_STEPS}.pt'
    if SAVE_PATH:
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    torch.save(features, SAVE_PATH + file_name)
    logger.info(f'Completed the feature extraction for video {VIDEO_LABEL.value}, video {VIDEO_NUMBER}')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if EXECUTE_STEP == 1:
            # STEP 1
            # Extract features from video
            step_one(with_optimized_of=True)
        if EXECUTE_STEP == 2:
            # STEP 2
            # Trainable Format
            step_two()
        if EXECUTE_STEP == -1:
            step_one(with_optimized_of=True)
            step_two()
