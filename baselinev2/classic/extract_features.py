import copy
import math
import os
from pathlib import Path
from typing import Dict, Any

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy
import skimage
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as tvf
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.feature_extractor import FeatureExtractor
from average_image.utils import is_inside_bbox
from baselinev2.config import ROOT_PATH, BATCH_CHECKPOINT, RESUME_MODE, CSV_MODE, EXECUTE_STEP, DATASET_META, \
    META_LABEL, VIDEO_LABEL, VIDEO_NUMBER, SAVE_PATH, BASE_PATH, video_save_path, plot_save_path, features_save_path, \
    version, TIMEOUT_MODE, GENERATE_BUNDLED_ANNOTATIONS, BUNDLED_ANNOTATIONS_VIDEO_CLASSES_LIST, \
    BUNDLED_ANNOTATIONS_PER_CLASSES_VIDEO_LIST, WITH_OBJECT_CLASSIFIER, OC_USE_RESNET, OC_USE_PRETRAINED, \
    OC_SMALLER_RESNET, OC_BATCH_SIZE, OC_NUM_WORKERS, OC_CHECKPOINT_PATH, OC_CHECKPOINT_VERSION, OC_DEVICE, \
    OC_ADDITIONAL_CROP_H, OC_ADDITIONAL_CROP_W
from baselinev2.constants import STEP, ObjectDetectionParameters
from baselinev2.exceptions import TimeoutException
from baselinev2.improve_metrics.crop_utils import show_image_with_crop_boxes
from baselinev2.improve_metrics.model import make_conv_blocks, Activations, make_classifier_block, PersonClassifier, \
    people_collate_fn
from baselinev2.improve_metrics.modules import resnet18, resnet9
from baselinev2.structures import ObjectFeatures, MinimalObjectFeatures, AgentFeatures, FrameFeatures, TrackFeatures, \
    Track
from baselinev2.utils import find_points_inside_box, first_violation_till_now, mean_shift_clustering, prune_clusters, \
    extract_features_inside_circle, features_included_in_live_tracks, get_track_history, get_agent_track_history, \
    get_gt_track_history, get_track_velocity_history, get_agent_track_velocity_history, angle_between, filter_features, \
    append_features, filter_for_one_to_one_matches, get_bbox_center, remove_entries_from_dict, \
    remove_from_dict_except_entries, eval_metrics, is_box_overlapping_live_boxes, extract_features_per_bounding_box, \
    evaluate_shifted_bounding_box, optical_flow_processing, features_filter_append_preprocessing, \
    first_frame_processing_and_gt_association, get_mog2_foreground_mask, filter_tracks_through_all_steps, \
    evaluate_extracted_features, extracted_features_in_csv, get_generated_frame_annotations, \
    mean_shift_clustering_with_timeout
from baselinev2.plot_utils import plot_for_video, plot_for_video_current_frame, plot_for_video_current_frame_single, \
    plot_processing_steps
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames, SimpleVideoDatasetBase

initialize_logging()
logger = get_logger('baselinev2.extract_features')


def preprocess_data_one_shot(save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
                             plot=False,
                             radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
                             plot_scale_factor=1,
                             desired_fps=5, custom_video_shape=True, plot_save_path=None, save_checkpoint=False,
                             begin_track_mode=True, use_circle_to_keep_track_alive=True, iou_threshold=0.5,
                             generic_box_wh=100,
                             extra_radius=50, use_is_box_overlapping_live_boxes=True, detect_shadows=True,
                             premature_kill_save=False, distance_threshold=2, save_every_n_batch_itr=None,
                             drop_last_batch=True):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'
    first_frame_bounding_boxes, first_frame_mask, last_frame, second_last_frame = None, None, None, None
    first_frame_live_tracks, last_frame_live_tracks, last_frame_mask = None, None, None
    current_track_idx, track_ids_used = 0, []
    precision_list, recall_list, matching_boxes_with_iou_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []
    meter_tp_list, meter_fp_list, meter_fn_list = [], [], []
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = [], [], []
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = [], [], []
    selected_track_distances = []
    accumulated_features = {}
    track_based_accumulated_features: Dict[int, TrackFeatures] = {}
    last_frame_gt_tracks = {}

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))  # (1200, 1000))  # (video_shape[0], video_shape[1]))
            # (video_shape[1], video_shape[0]))
            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))  # (1200, 1000))  # (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if part_idx == 0 and frame_idx == 0:
                    # STEP 1: a> Get GT for the first frame
                    first_annotations, first_frame_mask = first_frame_processing_and_gt_association(
                        df, first_frame_mask,
                        frame_idx, frame_number,
                        frames, frames_count,
                        kernel, n, new_shape,
                        original_shape, step,
                        var_threshold, detect_shadows=detect_shadows)

                    # STEP 1: b> Store them for the next ts and update these variables in further iterations
                    first_frame_bounding_boxes = first_annotations[:, :-1]
                    last_frame = frame.copy()
                    second_last_frame = last_frame.copy()
                    # last_frame_live_tracks = [Track(box, idx) for idx, box in enumerate(first_frame_bounding_boxes)]
                    last_frame_live_tracks = [Track(box, idx, gt_t_id) for idx, (box, gt_t_id) in
                                              enumerate(zip(first_frame_bounding_boxes, first_annotations[:, -1]))]
                    last_frame_mask = first_frame_mask.copy()
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    annotations, bbox_centers = scale_annotations(frame_annotation,
                                                                  original_scale=original_shape,
                                                                  new_scale=new_shape, return_track_id=False,
                                                                  tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        if xy.size == 0:
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        if xy_current_frame.size == 0:
                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected in the next time-step
                                current_track_obj_features = ObjectFeatures(idx=current_track_idx,
                                                                            xy=xy_current_frame,
                                                                            past_xy=xy,
                                                                            final_xy=xy_current_frame,
                                                                            flow=xy_displacement,
                                                                            past_flow=past_xy_displacement,
                                                                            past_bbox=box,
                                                                            final_bbox=np.array(shifted_box),
                                                                            is_track_live=False,
                                                                            frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)
                                continue

                        running_tracks.append(Track(bbox=shifted_box, idx=current_track_idx))

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = features_filter_append_preprocessing(
                            overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)

                        if plot:
                            plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                  shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                  xy_cloud_current_frame=xy_current_frame,
                                                  frame_number=frame_number.item(),
                                                  track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                                  selected_current=closest_n_xy_current_frame_pair)
                        # STEP 4g: save the information gathered
                        current_track_obj_features = ObjectFeatures(idx=current_track_idx,
                                                                    xy=xy_current_frame,
                                                                    past_xy=xy,
                                                                    final_xy=final_features_xy,
                                                                    flow=xy_displacement,
                                                                    past_flow=past_xy_displacement,
                                                                    past_bbox=box,
                                                                    final_bbox=np.array(shifted_box),
                                                                    frame_number=frame_number.item())
                        object_features.append(current_track_obj_features)
                        if current_track_idx not in track_based_accumulated_features:
                            track_feats = TrackFeatures(current_track_idx)
                            track_feats.object_features.append(current_track_obj_features)
                            track_based_accumulated_features.update(
                                {current_track_idx: track_feats})
                        else:
                            track_based_accumulated_features[current_track_idx].object_features.append(
                                current_track_obj_features)
                        if current_track_idx not in track_ids_used:
                            track_ids_used.append(current_track_idx)

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    ratio = float(meta_info.flatten()[-1])

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(annotations[:, :-1])
                    a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    try:
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    except IndexError:
                        if a_boxes.ndim < 2:
                            a_boxes = a_boxes.unsqueeze(0)
                        if r_boxes.ndim < 2:
                            r_boxes = r_boxes.unsqueeze(0)
                        logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                        logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                        # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes = torch.randn((0)).numpy()

                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    if r_boxes_np.size != 0:
                        for a_i, a_box in enumerate(a_boxes_np):
                            for r_i, r_box in enumerate(r_boxes_np):
                                dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                       get_bbox_center(r_box).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]

                        gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                        for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                            gt_t_idx = match_rows_tracks_idx[m_c_idx]
                            # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                            track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                            track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                                gt_track_box_mapping[gt_t_idx]
                            try:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                                    last_frame_gt_tracks[gt_t_idx]
                                gt_distance = np.linalg.norm(
                                    (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                                     get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = gt_distance
                            except KeyError:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = [0, 0]

                        last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    if iou_boxes_threshold.size != 0:
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []

                        #  precision/recall
                        tp = 0
                        fp = 0
                        fn = len(a_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    bbox_distance_to_of_centers_iou_based = []
                    bbox_distance_to_of_centers_iou_based_idx = []
                    boxes_distance = []
                    # boxes_distance_for_metric = []
                    r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()

                    # TODO: Replace with Hungarian
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T

                    match_idx = np.where(iou_boxes)
                    matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)

                    for a_box_idx, r_box_idx in zip(*match_idx):
                        dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                               get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                        boxes_distance.append([(a_box_idx, r_box_idx), dist])
                        # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                        matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                        predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                            point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                        )
                        bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                      iou_boxes[a_box_idx, r_box_idx]])
                        bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                                                                          iou_boxes[a_box_idx, r_box_idx]])
                        if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                                if (b == r_boxes[r_box_idx]).all()][0]:
                            selected_track_distances.append(dist)

                    # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    if r_boxes_np.size != 0:
                        a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                            matched_boxes_l2_distance_matrix
                        )

                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = \
                            np.where(predicted_box_center_inside_gt_box_matrix)
                    else:
                        a_matched_boxes_l2_distance_matrix_idx, \
                        r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)

                        meter_precision = meter_tp / (meter_tp + meter_fp)
                        meter_recall = meter_tp / (meter_tp + meter_fn)

                        center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)

                        center_precision = center_tp / (center_tp + center_fp)
                        center_recall = center_tp / (center_tp + center_fn)

                        if len(match_rows) != len(match_cols):
                            logger.info('Matching arrays length not same!')
                        l2_distance_hungarian_tp = len(match_rows)
                        l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                        l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)

                        l2_distance_hungarian_precision = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                        l2_distance_hungarian_recall = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    else:
                        meter_tp = 0
                        meter_fp = 0
                        meter_fn = len(a_boxes)

                        meter_precision = 0
                        meter_recall = 0

                        center_tp = 0
                        center_fp = 0
                        center_fn = len(a_boxes)

                        center_precision = 0
                        center_recall = 0

                        l2_distance_hungarian_tp = 0
                        l2_distance_hungarian_fp = 0
                        l2_distance_hungarian_fn = len(a_boxes)

                        l2_distance_hungarian_precision = 0
                        l2_distance_hungarian_recall = 0

                    meter_tp_list.append(meter_tp)
                    meter_fp_list.append(meter_fp)
                    meter_fn_list.append(meter_fn)

                    center_inside_tp_list.append(center_tp)
                    center_inside_fp_list.append(center_fp)
                    center_inside_fn_list.append(center_fn)

                    l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            annotations, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            # STEP 4h: b> cluster to group points
                            mean_shift, n_clusters = mean_shift_clustering(
                                features_skipped, bin_seeding=False, min_bin_freq=8,
                                cluster_all=True, bandwidth=4, max_iter=100)
                            cluster_centers = mean_shift.cluster_centers

                            # STEP 4h: c> prune cluster centers
                            # combine centers inside radius + eliminate noise
                            final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                cluster_centers, mean_shift, radius + extra_radius,
                                min_points_in_cluster=min_points_in_cluster)

                            if final_cluster_centers.size != 0:
                                t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                # STEP 4h: d> start new potential tracks
                                for cluster_center in final_cluster_centers:
                                    cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                    # t_id = max(track_ids_used) + 1
                                    # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                    t_box = torchvision.ops.box_convert(
                                        torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                        'cxcywh', 'xyxy').int().numpy()
                                    # Note: Do not start track if bbox is out of frame
                                    if use_is_box_overlapping_live_boxes:
                                        if not (np.sign(t_box) < 0).any() and \
                                                not is_box_overlapping_live_boxes(t_box,
                                                                                  [t.bbox for t in running_tracks]):
                                            # NOTE: the second check might result in killing potential tracks!
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                    else:
                                        if not (np.sign(t_box) < 0).any():
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers, num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))

                    # # NMS Debugging
                    # current_track_boxes = [t.bbox for t in running_tracks]
                    # current_track_boxes = torch.tensor(current_track_boxes)
                    # current_track_boxes_iou = torchvision.ops.box_iou(current_track_boxes, current_track_boxes).numpy()
                    # current_match_a, current_match_b = np.where(current_track_boxes_iou)
                    # print()
                    # print(f'Frame: {frame_number}')
                    # print(current_track_boxes_iou)
                    # print()
                    # print(f'{current_match_a} <|> {current_match_b}')

                    if video_mode:
                        # fig = plot_for_video(
                        #     gt_rgb=frame, gt_mask=fg_mask, last_frame_rgb=last_frame,
                        #     last_frame_mask=last_frame_mask, current_frame_rgb=frame,
                        #     current_frame_mask=fg_mask, gt_annotations=annotations[:, :-1],
                        #     last_frame_annotation=[t.bbox for t in last_frame_live_tracks],
                        #     current_frame_annotation=[t.bbox for t in running_tracks],
                        #     new_track_annotation=new_track_boxes,
                        #     frame_number=frame_number,
                        #     additional_text=
                        #     # f'Track Ids Used: {track_ids_used}\n'
                        #     f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        #     f'Track Ids Killed: '
                        #     f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                        #     video_mode=video_mode)

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)
                    else:
                        fig = plot_for_video(
                            gt_rgb=frame, gt_mask=fg_mask, last_frame_rgb=last_frame,
                            last_frame_mask=last_frame_mask, current_frame_rgb=frame,
                            current_frame_mask=fg_mask, gt_annotations=annotations[:, :-1],
                            last_frame_annotation=[t.bbox for t in last_frame_live_tracks],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False,
                            save_path=f'{plot_save_path}zero_shot/plots{min_points_in_cluster}/')

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                        np.array(l2_distance_hungarian_fn_list).sum()
                    batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    logger.info(f'Batch: {part_idx}, '
                                f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
                if save_every_n_batch_itr is not None and frame_number != 0:
                    save_dict = {'frame_number': frame_number,
                                 'part_idx': part_idx,
                                 'second_last_frame': second_last_frame,
                                 'last_frame': last_frame,
                                 'last_frame_mask': last_frame_mask,
                                 'last_frame_live_tracks': last_frame_live_tracks,
                                 'running_tracks': running_tracks,
                                 'track_ids_used': track_ids_used,
                                 'new_track_boxes': new_track_boxes,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'tp_list': tp_list,
                                 'fp_list': fp_list,
                                 'fn_list': fn_list,
                                 'meter_tp_list': meter_tp_list,
                                 'meter_fp_list': meter_fp_list,
                                 'meter_fn_list': meter_fn_list,
                                 'center_inside_tp_list': center_inside_tp_list,
                                 'center_inside_fp_list': center_inside_fp_list,
                                 'center_inside_fn_list': center_inside_fn_list,
                                 'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                 'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                 'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                 'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                 'track_based_accumulated_features': track_based_accumulated_features,
                                 'accumulated_features': accumulated_features}
                    if part_idx % save_every_n_batch_itr == 0 and part_idx != 0:
                        Path(video_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                        f_n = f'features_dict_part{part_idx}.pt'
                        torch.save(save_dict, video_save_path + 'parts/' + f_n)

                        accumulated_features = {}
                        live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                        track_based_accumulated_features = remove_entries_from_dict(live_track_ids,
                                                                                    track_based_accumulated_features)
                    # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
                if save_per_part_path is not None:
                    Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                    f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                    torch.save(accumulated_features, save_per_part_path + f_n)
                # if video_mode:
                #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path).mkdir(parents=True, exist_ok=True)
        f_n = f'accumulated_features_from_finally.pt'
        torch.save(premature_save_dict, features_save_path + f_n)

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


def preprocess_data_zero_shot(save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
                              radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
                              plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
                              save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
                              use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
                              use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
                              distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
                              detect_shadows=True, allow_only_for_first_frame_use_is_box_overlapping_live_boxes=True,
                              filter_switch_boxes_based_on_angle_and_recent_history=True,
                              compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
                              angle_threshold_to_filter=120):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'
    first_frame_bounding_boxes, first_frame_mask, last_frame, second_last_frame = None, None, None, None
    first_frame_live_tracks, last_frame_live_tracks, last_frame_mask = None, None, None
    current_track_idx, track_ids_used = 0, []
    precision_list, recall_list, matching_boxes_with_iou_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []
    meter_tp_list, meter_fp_list, meter_fn_list = [], [], []
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = [], [], []
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = [], [], []
    selected_track_distances = []
    accumulated_features = {}
    track_based_accumulated_features: Dict[int, TrackFeatures] = {}
    last_frame_gt_tracks = {}
    ground_truth_track_histories = []

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))  # (1200, 1000))  # (video_shape[0], video_shape[1]))
            # (video_shape[1], video_shape[0]))
            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))  # (1200, 1000))  # (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if part_idx == 0 and frame_idx == 0:
                    # STEP 1: a> Get GT for the first frame
                    validation_annotations, first_frame_mask = first_frame_processing_and_gt_association(
                        df, first_frame_mask,
                        frame_idx, frame_number,
                        frames, frames_count,
                        kernel, n, new_shape,
                        original_shape, step,
                        var_threshold, detect_shadows=detect_shadows)

                    running_tracks, new_track_boxes = [], []
                    all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                        None, first_frame_mask, radius, running_tracks, plot=False)

                    all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                    features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                    if features_skipped_idx.size != 0:
                        features_skipped = all_cloud[features_skipped_idx]

                        # STEP 4h: b> cluster to group points
                        mean_shift, n_clusters = mean_shift_clustering(
                            features_skipped, bin_seeding=False, min_bin_freq=8,
                            cluster_all=True, bandwidth=4, max_iter=100)
                        cluster_centers = mean_shift.cluster_centers

                        # STEP 4h: c> prune cluster centers
                        # combine centers inside radius + eliminate noise
                        final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                            cluster_centers, mean_shift, radius + extra_radius,
                            min_points_in_cluster=min_points_in_cluster)

                        if final_cluster_centers.size != 0:
                            t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                            # STEP 4h: d> start new potential tracks
                            for cluster_center, cluster_center_idx in \
                                    zip(final_cluster_centers, final_cluster_centers_idx):
                                cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)

                                # flexible_box, points_in_current_cluster = calculate_flexible_bounding_box(
                                #     cluster_center_idx, cluster_center_x, cluster_center_y, mean_shift)

                                # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                t_box = torchvision.ops.box_convert(
                                    torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                    'cxcywh', 'xyxy').int().numpy()
                                # Note: Do not start track if bbox is out of frame
                                if allow_only_for_first_frame_use_is_box_overlapping_live_boxes and \
                                        not (np.sign(t_box) < 0).any() and \
                                        not is_box_overlapping_live_boxes(t_box, [t.bbox for t in running_tracks]):
                                    # NOTE: the second check might result in killing potential tracks!
                                    t_id = max(track_ids_used) + 1 if len(track_ids_used) else 0
                                    running_tracks.append(Track(bbox=t_box, idx=t_id))
                                    track_ids_used.append(t_id)
                                    new_track_boxes.append(t_box)
                                elif use_is_box_overlapping_live_boxes:
                                    if not (np.sign(t_box) < 0).any() and \
                                            not is_box_overlapping_live_boxes(t_box,
                                                                              [t.bbox for t in running_tracks]):
                                        # NOTE: the second check might result in killing potential tracks!
                                        t_id = max(track_ids_used) + 1
                                        running_tracks.append(Track(bbox=t_box, idx=t_id))
                                        track_ids_used.append(t_id)
                                        new_track_boxes.append(t_box)
                                else:
                                    if not (np.sign(t_box) < 0).any():
                                        t_id = max(track_ids_used) + 1
                                        running_tracks.append(Track(bbox=t_box, idx=t_id))
                                        track_ids_used.append(t_id)
                                        new_track_boxes.append(t_box)

                            # plot_features_with_circles(
                            #     all_cloud, features_covered, features_skipped, first_frame_mask, marker_size=8,
                            #     cluster_centers=final_cluster_centers, num_clusters=final_cluster_centers.shape[0],
                            #     frame_number=frame_number, boxes=validation_annotations[:, :-1],
                            #     radius=radius+extra_radius,
                            #     additional_text=
                            #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                            #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))

                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(validation_annotations[:, :-1])
                    iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                    matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                               for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]

                    #  precision/recall
                    tp = len(a_match_idx_threshold)
                    fp = len(r_boxes) - len(a_match_idx_threshold)
                    fn = len(a_boxes) - len(a_match_idx_threshold)

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    last_frame_live_tracks = []

                    fig = plot_for_video(
                        gt_rgb=frame, gt_mask=first_frame_mask, last_frame_rgb=frame,
                        last_frame_mask=first_frame_mask, current_frame_rgb=frame,
                        current_frame_mask=first_frame_mask, gt_annotations=validation_annotations[:, :-1],
                        last_frame_annotation=last_frame_live_tracks,
                        current_frame_annotation=[t.bbox for t in running_tracks],
                        new_track_annotation=new_track_boxes,
                        frame_number=frame_number,
                        additional_text=
                        f'Precision: {precision} | Recall: {recall}\n'
                        f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        f'Track Ids Killed: '
                        f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                        video_mode=False,
                        save_path=f'{plot_save_path}zero_shot/plots{min_points_in_cluster}/')

                    # STEP 1: b> Store them for the next ts and update these variables in further iterations
                    last_frame = frame.copy()
                    second_last_frame = last_frame.copy()
                    last_frame_live_tracks = running_tracks
                    last_frame_mask = first_frame_mask.copy()
                    last_frame_gt_tracks = {a[-1]: a[:-1] for a in validation_annotations}
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    annotations, bbox_centers = scale_annotations(frame_annotation,
                                                                  original_scale=original_shape,
                                                                  new_scale=new_shape, return_track_id=False,
                                                                  tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        current_track_features = track_based_accumulated_features[
                            current_track_idx].object_features[-1] \
                            if current_track_idx in track_based_accumulated_features.keys() else []
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        # if frame_number == 110:
                        #     print()

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # STEP 4c: 1> Switch boxes: Idea 1 - Only keep points inside box
                        shifted_xy = find_points_inside_box(shifted_xy, shifted_box)
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        if xy_current_frame.size == 0 or (filter_switch_boxes_based_on_angle_and_recent_history
                                                          and not isinstance(current_track_features, list)
                                                          and current_track_features.velocity_direction.size != 0
                                                          and len(current_track_features.velocity_direction) >
                                                          min_track_length_to_filter_switch_box
                                                          and first_violation_till_now(
                                    current_track_features.velocity_direction, angle_threshold_to_filter)
                                                          and current_track_features.velocity_direction[-1] >
                                                          angle_threshold_to_filter
                                                          and not np.isnan(
                                    current_track_features.velocity_direction[-1])):

                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
                                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                    current_track_history = get_track_history(current_track_idx,
                                                                              track_based_accumulated_features)
                                    current_track_velocity_history = get_track_velocity_history(
                                        current_track_idx, track_based_accumulated_features)
                                    current_track_velocity_history = np.array(current_track_velocity_history)
                                    current_direction = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        current_direction.append((angle_between(
                                            v1=current_track_history[track_history_idx],
                                            v2=current_track_history[track_history_idx + 1]
                                        )))
                                    # current direction can be removed
                                    current_direction = np.array(current_direction)

                                    current_velocity_direction = []
                                    for track_history_idx in range(len(current_track_velocity_history) - 1):
                                        current_velocity_direction.append(math.degrees(angle_between(
                                            v1=current_track_velocity_history[track_history_idx],
                                            v2=current_track_velocity_history[track_history_idx + 1]
                                        )))
                                    current_velocity_direction = np.array(current_velocity_direction)

                                    # not really required ############################################################
                                    if len(current_track_history) != 0:
                                        current_running_velocity = np.linalg.norm(
                                            np.expand_dims(current_track_history[-1], axis=0) -
                                            np.expand_dims(current_track_history[0], axis=0),
                                            2, axis=0
                                        ) / len(current_track_history) / 30
                                    else:
                                        current_running_velocity = None

                                    current_per_step_distance = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        d = np.linalg.norm(
                                            np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                            np.expand_dims(current_track_history[track_history_idx], axis=0),
                                            2, axis=0
                                        )
                                        current_per_step_distance.append(d)

                                    current_per_step_distance = np.array(current_per_step_distance)
                                    ###################################################################################

                                    # track_sign = []
                                    # for t in range(
                                    #         len(track_based_accumulated_features[8].object_features[-1].track_history)
                                    #         - 1):
                                    #     track_sign.append(np.sign(
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t + 1] -
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t]))

                                    # just use gt
                                    current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                    track_based_accumulated_features)
                                else:
                                    current_track_history, current_gt_track_history = None, None
                                    current_direction, current_velocity_direction = None, None
                                    current_track_velocity_history = None
                                    # not really required ############################################################
                                    current_per_step_distance = None
                                    current_running_velocity = None
                                    ###################################################################################
                                current_track_obj_features = ObjectFeatures(idx=current_track_idx,
                                                                            history=current_track_history,
                                                                            gt_history=current_gt_track_history,
                                                                            track_direction=current_direction,
                                                                            velocity_direction=
                                                                            current_velocity_direction,
                                                                            velocity_history=
                                                                            current_track_velocity_history,
                                                                            xy=xy_current_frame,
                                                                            past_xy=xy,
                                                                            final_xy=xy_current_frame,
                                                                            flow=xy_displacement,
                                                                            past_flow=past_xy_displacement,
                                                                            past_bbox=box,
                                                                            final_bbox=np.array(shifted_box),
                                                                            is_track_live=False,
                                                                            per_step_distance=current_per_step_distance,
                                                                            running_velocity=current_running_velocity,
                                                                            frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)

                                continue

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?
                        final_shifted_box, final_shifted_xy_center = evaluate_shifted_bounding_box(shifted_box,
                                                                                                   final_features_xy,
                                                                                                   shifted_xy)

                        running_tracks.append(Track(bbox=final_shifted_box, idx=current_track_idx))

                        # if not (final_shifted_box == shifted_box).all():
                        #     logger.warn('Final Shifted Box differs from Shifted Box!')

                        if plot:
                            plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                  shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                  xy_cloud_current_frame=xy_current_frame,
                                                  frame_number=frame_number.item(),
                                                  track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                                  selected_current=closest_n_xy_current_frame_pair)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            current_track_history = get_track_history(current_track_idx,
                                                                      track_based_accumulated_features)
                            current_track_velocity_history = get_track_velocity_history(
                                current_track_idx, track_based_accumulated_features)
                            current_track_velocity_history = np.array(current_track_velocity_history)
                            current_direction = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                current_direction.append((angle_between(
                                    v1=current_track_history[track_history_idx],
                                    v2=current_track_history[track_history_idx + 1]
                                )))
                            # current direction can be removed
                            current_direction = np.array(current_direction)

                            current_velocity_direction = []
                            for track_history_idx in range(len(current_track_velocity_history) - 1):
                                current_velocity_direction.append(math.degrees(angle_between(
                                    v1=current_track_velocity_history[track_history_idx],
                                    v2=current_track_velocity_history[track_history_idx + 1]
                                )))
                            current_velocity_direction = np.array(current_velocity_direction)

                            current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                            track_based_accumulated_features)

                            # not really required ############################################################
                            if len(current_track_history) != 0:
                                current_running_velocity = np.linalg.norm(
                                    np.expand_dims(current_track_history[-1], axis=0) -
                                    np.expand_dims(current_track_history[0], axis=0),
                                    2, axis=0
                                ) / len(current_track_history) / 30
                            else:
                                current_running_velocity = None

                            current_per_step_distance = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                d = np.linalg.norm(
                                    np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                    np.expand_dims(current_track_history[track_history_idx], axis=0),
                                    2, axis=0
                                )
                                current_per_step_distance.append(d)

                            current_per_step_distance = np.array(current_per_step_distance)
                            ###################################################################################
                        else:
                            current_track_history, current_gt_track_history = None, None
                            current_direction, current_velocity_direction = None, None
                            current_track_velocity_history = None
                            # not really required ############################################################
                            current_per_step_distance = None
                            current_running_velocity = None
                            ###################################################################################
                        # STEP 4g: save the information gathered
                        current_track_obj_features = ObjectFeatures(idx=current_track_idx,
                                                                    history=current_track_history,
                                                                    gt_history=current_gt_track_history,
                                                                    track_direction=current_direction,
                                                                    velocity_direction=current_velocity_direction,
                                                                    velocity_history=current_track_velocity_history,
                                                                    xy=xy_current_frame,
                                                                    past_xy=xy,
                                                                    final_xy=final_features_xy,
                                                                    flow=xy_displacement,
                                                                    past_flow=past_xy_displacement,
                                                                    past_bbox=box,
                                                                    # final_bbox=np.array(shifted_box),
                                                                    final_bbox=np.array(final_shifted_box),
                                                                    per_step_distance=current_per_step_distance,
                                                                    running_velocity=current_running_velocity,
                                                                    frame_number=frame_number.item())
                        object_features.append(current_track_obj_features)
                        if current_track_idx not in track_based_accumulated_features:
                            track_feats = TrackFeatures(current_track_idx)
                            track_feats.object_features.append(current_track_obj_features)
                            track_based_accumulated_features.update(
                                {current_track_idx: track_feats})
                        else:
                            track_based_accumulated_features[current_track_idx].object_features.append(
                                current_track_obj_features)
                        if current_track_idx not in track_ids_used:
                            track_ids_used.append(current_track_idx)

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    ratio = float(meta_info.flatten()[-1])

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(annotations[:, :-1])
                    a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    try:
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    except IndexError:
                        if a_boxes.ndim < 2:
                            a_boxes = a_boxes.unsqueeze(0)
                        if r_boxes.ndim < 2:
                            r_boxes = r_boxes.unsqueeze(0)
                        logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                        logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                        # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes = torch.randn((0)).numpy()

                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    if r_boxes_np.size != 0:
                        for a_i, a_box in enumerate(a_boxes_np):
                            for r_i, r_box in enumerate(r_boxes_np):
                                dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                       get_bbox_center(r_box).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]

                        gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                        for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                            gt_t_idx = match_rows_tracks_idx[m_c_idx]
                            # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                            track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                            track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                                gt_track_box_mapping[gt_t_idx]
                            try:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                                    last_frame_gt_tracks[gt_t_idx]
                                gt_distance = np.linalg.norm(
                                    (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                                     get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = gt_distance
                            except KeyError:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = [0, 0]

                        last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    # generated_to_gt_track_association check ###################################################
                    # filter_nones = True
                    # plot_gt_ids = []
                    # plot_gt_boxes = []
                    # plot_generated_ids = []
                    # plot_generated_boxes = []
                    #
                    # for k, v in track_based_accumulated_features.items():
                    #     plot_feats = v.object_features[-1]
                    #     if filter_nones and plot_feats.gt_box is not None:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #     if not filter_nones:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #
                    # plot_image_set_of_boxes(frame, plot_gt_boxes, plot_generated_boxes,
                    #                         annotate=[plot_gt_ids, plot_generated_ids])
                    #############################################################################################
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    if iou_boxes_threshold.size != 0:
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []

                        #  precision/recall
                        tp = 0
                        fp = 0
                        fn = len(a_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    bbox_distance_to_of_centers_iou_based = []
                    bbox_distance_to_of_centers_iou_based_idx = []
                    boxes_distance = []
                    # boxes_distance_for_metric = []
                    r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()

                    # TODO: Replace with Hungarian
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T

                    match_idx = np.where(iou_boxes)
                    matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)

                    for a_box_idx, r_box_idx in zip(*match_idx):
                        dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                               get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                        boxes_distance.append([(a_box_idx, r_box_idx), dist])
                        # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                        matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                        predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                            point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                        )
                        bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                      iou_boxes[a_box_idx, r_box_idx]])
                        bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                                                                          iou_boxes[a_box_idx, r_box_idx]])
                        if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                                if (b == r_boxes[r_box_idx]).all()][0]:
                            selected_track_distances.append(dist)

                    # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    if r_boxes_np.size != 0:
                        a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                            matched_boxes_l2_distance_matrix
                        )

                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = \
                            np.where(predicted_box_center_inside_gt_box_matrix)
                    else:
                        a_matched_boxes_l2_distance_matrix_idx, \
                        r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)

                        meter_precision = meter_tp / (meter_tp + meter_fp)
                        meter_recall = meter_tp / (meter_tp + meter_fn)

                        center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)

                        center_precision = center_tp / (center_tp + center_fp)
                        center_recall = center_tp / (center_tp + center_fn)

                        if len(match_rows) != len(match_cols):
                            logger.info('Matching arrays length not same!')
                        l2_distance_hungarian_tp = len(match_rows)
                        l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                        l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)

                        l2_distance_hungarian_precision = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                        l2_distance_hungarian_recall = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    else:
                        meter_tp = 0
                        meter_fp = 0
                        meter_fn = len(a_boxes)

                        meter_precision = 0
                        meter_recall = 0

                        center_tp = 0
                        center_fp = 0
                        center_fn = len(a_boxes)

                        center_precision = 0
                        center_recall = 0

                        l2_distance_hungarian_tp = 0
                        l2_distance_hungarian_fp = 0
                        l2_distance_hungarian_fn = len(a_boxes)

                        l2_distance_hungarian_precision = 0
                        l2_distance_hungarian_recall = 0

                    meter_tp_list.append(meter_tp)
                    meter_fp_list.append(meter_fp)
                    meter_fn_list.append(meter_fn)

                    center_inside_tp_list.append(center_tp)
                    center_inside_fp_list.append(center_fp)
                    center_inside_fn_list.append(center_fn)

                    l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            annotations, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            # STEP 4h: b> cluster to group points
                            mean_shift, n_clusters = mean_shift_clustering(
                                features_skipped, bin_seeding=False, min_bin_freq=8,
                                cluster_all=True, bandwidth=4, max_iter=100)
                            cluster_centers = mean_shift.cluster_centers

                            # STEP 4h: c> prune cluster centers
                            # combine centers inside radius + eliminate noise
                            final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                cluster_centers, mean_shift, radius + extra_radius,
                                min_points_in_cluster=min_points_in_cluster)

                            if final_cluster_centers.size != 0:
                                t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                # STEP 4h: d> start new potential tracks
                                for cluster_center in final_cluster_centers:
                                    cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                    # t_id = max(track_ids_used) + 1
                                    # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                    t_box = torchvision.ops.box_convert(
                                        torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                        'cxcywh', 'xyxy').int().numpy()
                                    # Note: Do not start track if bbox is out of frame
                                    if use_is_box_overlapping_live_boxes:
                                        if not (np.sign(t_box) < 0).any() and \
                                                not is_box_overlapping_live_boxes(t_box,
                                                                                  [t.bbox for t in running_tracks]):
                                            # NOTE: the second check might result in killing potential tracks!
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                    else:
                                        if not (np.sign(t_box) < 0).any():
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers,
                                #     num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    if video_mode:
                        # fig = plot_for_video(
                        #     gt_rgb=frame, gt_mask=fg_mask, last_frame_rgb=last_frame,
                        #     last_frame_mask=last_frame_mask, current_frame_rgb=frame,
                        #     current_frame_mask=fg_mask, gt_annotations=annotations[:, :-1],
                        #     last_frame_annotation=[t.bbox for t in last_frame_live_tracks],
                        #     current_frame_annotation=[t.bbox for t in running_tracks],
                        #     new_track_annotation=new_track_boxes,
                        #     frame_number=frame_number,
                        #     additional_text=
                        #     # f'Track Ids Used: {track_ids_used}\n'
                        #     f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        #     f'Track Ids Killed: '
                        #     f'{np.setdiff1d([t.idx for t in last_frame_live_tracks],
                        #     [t.idx for t in running_tracks])}',
                        #     video_mode=video_mode)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            frame_feats = accumulated_features[frame_number.item()]
                            tracks_histories = []
                            tracks_gt_histories = []
                            for obj_feature in frame_feats.object_features:
                                tracks_histories.extend(obj_feature.track_history)
                                tracks_gt_histories.extend(obj_feature.gt_history)

                            # no need to reverse
                            # tracks_histories.reverse()
                            # tracks_gt_histories.reverse()

                            tracks_histories = np.array(tracks_histories)
                            tracks_gt_histories = np.array(tracks_gt_histories)

                            if tracks_histories.size == 0:
                                tracks_histories = np.zeros(shape=(0, 2))
                            if tracks_gt_histories.size == 0:
                                tracks_gt_histories = np.zeros(shape=(0, 2))
                        else:
                            tracks_histories = np.zeros(shape=(0, 2))
                            tracks_gt_histories = np.zeros(shape=(0, 2))

                        # for gt_annotation_box in annotations[:, :-1]:
                        #     ground_truth_track_histories.append(get_bbox_center(gt_annotation_box).flatten())
                        # ground_truth_track_histories = np.array(ground_truth_track_histories)

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[a_boxes_idx.tolist(), r_boxes_idx],
                            generated_track_histories=tracks_histories,
                            gt_track_histories=tracks_gt_histories,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)

                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=36)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=201)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=212)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=3)
                    else:
                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False, original_dims=original_dims, zero_shot=True)
                        # fig = plot_for_video(
                        #     gt_rgb=frame, gt_mask=fg_mask, last_frame_rgb=last_frame,
                        #     last_frame_mask=last_frame_mask, current_frame_rgb=frame,
                        #     current_frame_mask=fg_mask, gt_annotations=annotations[:, :-1],
                        #     last_frame_annotation=[t.bbox for t in last_frame_live_tracks],
                        #     current_frame_annotation=[t.bbox for t in running_tracks],
                        #     new_track_annotation=new_track_boxes,
                        #     frame_number=frame_number,
                        #     additional_text=
                        #     f'Precision: {precision} | Recall: {recall}\n'
                        #     f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                        #     f'Recall: {l2_distance_hungarian_recall}\n'
                        #     f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                        #     f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        #     f'Track Ids Killed: '
                        #     f'{np.setdiff1d([t.idx for t in last_frame_live_tracks],
                        #     [t.idx for t in running_tracks])}',
                        #     video_mode=False,
                        #     save_path=f'{plot_save_path}zero_shot/plots{min_points_in_cluster}/')

                    # : save stuff and reiterate - moved up
                    # accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                    #                                                                 object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                        np.array(l2_distance_hungarian_fn_list).sum()
                    batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    logger.info(f'Video: {VIDEO_LABEL.value} - {VIDEO_NUMBER} | Batch: {part_idx} | '
                                f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
            if save_every_n_batch_itr is not None:
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0:
                    save_dict = {'frame_number': frame_number,
                                 'part_idx': part_idx,
                                 'current_track_idx': current_track_idx,
                                 'second_last_frame': second_last_frame,
                                 'last_frame': last_frame,
                                 'last_frame_mask': last_frame_mask,
                                 'last_frame_live_tracks': last_frame_live_tracks,
                                 'running_tracks': running_tracks,
                                 'track_ids_used': track_ids_used,
                                 'new_track_boxes': new_track_boxes,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'tp_list': tp_list,
                                 'fp_list': fp_list,
                                 'fn_list': fn_list,
                                 'meter_tp_list': meter_tp_list,
                                 'meter_fp_list': meter_fp_list,
                                 'meter_fn_list': meter_fn_list,
                                 'center_inside_tp_list': center_inside_tp_list,
                                 'center_inside_fp_list': center_inside_fp_list,
                                 'center_inside_fn_list': center_inside_fn_list,
                                 'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                 'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                 'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                 'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                 'track_based_accumulated_features': track_based_accumulated_features,
                                 'accumulated_features': accumulated_features}
                    Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part_{part_idx}.pt'
                    torch.save(save_dict, features_save_path + 'parts/' + f_n)

                    accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_from_dict_except_entries(live_track_ids,
                                                                                       track_based_accumulated_features)
                    logger.info(f'Saved part {part_idx} at frame {frame_number}')
            # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)
            # if video_mode:
            #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'current_track_idx': current_track_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
        # f_n = f'accumulated_features_from_finally.pt'
        f_n = f'features_dict_part_{part_idx}.pt'
        torch.save(premature_save_dict, features_save_path + 'parts/' + f_n)

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


def preprocess_data_zero_shot_minimal(
        save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
        radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
        plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
        save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
        use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
        use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
        distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
        detect_shadows=True, allow_only_for_first_frame_use_is_box_overlapping_live_boxes=True,
        filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
        angle_threshold_to_filter=120):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'
    first_frame_bounding_boxes, first_frame_mask, last_frame, second_last_frame = None, None, None, None
    first_frame_live_tracks, last_frame_live_tracks, last_frame_mask = None, None, None
    current_track_idx, track_ids_used = 0, []
    precision_list, recall_list, matching_boxes_with_iou_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []
    meter_tp_list, meter_fp_list, meter_fn_list = [], [], []
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = [], [], []
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = [], [], []
    selected_track_distances = []
    accumulated_features = {}
    track_based_accumulated_features: Dict[int, TrackFeatures] = {}
    last_frame_gt_tracks = {}
    ground_truth_track_histories = []

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))

            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if part_idx == 0 and frame_idx == 0:
                    # STEP 1: a> Get GT for the first frame
                    validation_annotations, first_frame_mask = first_frame_processing_and_gt_association(
                        df, first_frame_mask,
                        frame_idx, frame_number,
                        frames, frames_count,
                        kernel, n, new_shape,
                        original_shape, step,
                        var_threshold, detect_shadows=detect_shadows)

                    running_tracks, new_track_boxes = [], []
                    all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                        None, first_frame_mask, radius, running_tracks, plot=False)

                    all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                    features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                    if features_skipped_idx.size != 0:
                        features_skipped = all_cloud[features_skipped_idx]

                        # STEP 4h: b> cluster to group points
                        mean_shift, n_clusters = mean_shift_clustering(
                            features_skipped, bin_seeding=False, min_bin_freq=8,
                            cluster_all=True, bandwidth=4, max_iter=100)
                        cluster_centers = mean_shift.cluster_centers

                        # STEP 4h: c> prune cluster centers
                        # combine centers inside radius + eliminate noise
                        final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                            cluster_centers, mean_shift, radius + extra_radius,
                            min_points_in_cluster=min_points_in_cluster)

                        if final_cluster_centers.size != 0:
                            t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                            # STEP 4h: d> start new potential tracks
                            for cluster_center, cluster_center_idx in \
                                    zip(final_cluster_centers, final_cluster_centers_idx):
                                cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)

                                # flexible_box, points_in_current_cluster = calculate_flexible_bounding_box(
                                #     cluster_center_idx, cluster_center_x, cluster_center_y, mean_shift)

                                # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                t_box = torchvision.ops.box_convert(
                                    torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                    'cxcywh', 'xyxy').int().numpy()
                                # Note: Do not start track if bbox is out of frame
                                if allow_only_for_first_frame_use_is_box_overlapping_live_boxes and \
                                        not (np.sign(t_box) < 0).any() and \
                                        not is_box_overlapping_live_boxes(t_box, [t.bbox for t in running_tracks]):
                                    # NOTE: the second check might result in killing potential tracks!
                                    t_id = max(track_ids_used) + 1 if len(track_ids_used) else 0
                                    running_tracks.append(Track(bbox=t_box, idx=t_id))
                                    track_ids_used.append(t_id)
                                    new_track_boxes.append(t_box)
                                elif use_is_box_overlapping_live_boxes:
                                    if not (np.sign(t_box) < 0).any() and \
                                            not is_box_overlapping_live_boxes(t_box,
                                                                              [t.bbox for t in running_tracks]):
                                        # NOTE: the second check might result in killing potential tracks!
                                        t_id = max(track_ids_used) + 1
                                        running_tracks.append(Track(bbox=t_box, idx=t_id))
                                        track_ids_used.append(t_id)
                                        new_track_boxes.append(t_box)
                                else:
                                    if not (np.sign(t_box) < 0).any():
                                        t_id = max(track_ids_used) + 1
                                        running_tracks.append(Track(bbox=t_box, idx=t_id))
                                        track_ids_used.append(t_id)
                                        new_track_boxes.append(t_box)

                            # plot_features_with_circles(
                            #     all_cloud, features_covered, features_skipped, first_frame_mask, marker_size=8,
                            #     cluster_centers=final_cluster_centers, num_clusters=final_cluster_centers.shape[0],
                            #     frame_number=frame_number, boxes=validation_annotations[:, :-1],
                            #     radius=radius+extra_radius,
                            #     additional_text=
                            #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                            #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))

                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(validation_annotations[:, :-1])
                    # Fixme: If no boxes in first frame!!
                    iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                    matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                               for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]

                    #  precision/recall
                    tp = len(a_match_idx_threshold)
                    fp = len(r_boxes) - len(a_match_idx_threshold)
                    fn = len(a_boxes) - len(a_match_idx_threshold)

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    last_frame_live_tracks = []

                    fig = plot_for_video(
                        gt_rgb=frame, gt_mask=first_frame_mask, last_frame_rgb=frame,
                        last_frame_mask=first_frame_mask, current_frame_rgb=frame,
                        current_frame_mask=first_frame_mask, gt_annotations=validation_annotations[:, :-1],
                        last_frame_annotation=last_frame_live_tracks,
                        current_frame_annotation=[t.bbox for t in running_tracks],
                        new_track_annotation=new_track_boxes,
                        frame_number=frame_number,
                        additional_text=
                        f'Precision: {precision} | Recall: {recall}\n'
                        f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        f'Track Ids Killed: '
                        f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                        video_mode=False,
                        save_path=f'{plot_save_path}zero_shot/plots{min_points_in_cluster}/')

                    # STEP 1: b> Store them for the next ts and update these variables in further iterations
                    last_frame = frame.copy()
                    second_last_frame = last_frame.copy()
                    last_frame_live_tracks = running_tracks
                    last_frame_mask = first_frame_mask.copy()
                    last_frame_gt_tracks = {a[-1]: a[:-1] for a in validation_annotations}
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    annotations, bbox_centers = scale_annotations(frame_annotation,
                                                                  original_scale=original_shape,
                                                                  new_scale=new_shape, return_track_id=False,
                                                                  tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        current_track_features = track_based_accumulated_features[
                            current_track_idx].object_features[-1] \
                            if current_track_idx in track_based_accumulated_features.keys() else []
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        # if frame_number == 110:
                        #     print()

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # STEP 4c: 1> Switch boxes: Idea 1 - Only keep points inside box
                        shifted_xy = find_points_inside_box(shifted_xy, shifted_box)
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        if xy_current_frame.size == 0 or (filter_switch_boxes_based_on_angle_and_recent_history
                                                          and not isinstance(current_track_features, list)
                                                          and current_track_features.velocity_direction.size != 0
                                                          and len(current_track_features.velocity_direction) >
                                                          min_track_length_to_filter_switch_box
                                                          and first_violation_till_now(
                                    current_track_features.velocity_direction, angle_threshold_to_filter)
                                                          and current_track_features.velocity_direction[-1] >
                                                          angle_threshold_to_filter
                                                          and not np.isnan(
                                    current_track_features.velocity_direction[-1])):

                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
                                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                    current_track_history = get_track_history(current_track_idx,
                                                                              track_based_accumulated_features)
                                    current_track_velocity_history = get_track_velocity_history(
                                        current_track_idx, track_based_accumulated_features)
                                    current_track_velocity_history = np.array(current_track_velocity_history)
                                    current_direction = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        current_direction.append((angle_between(
                                            v1=current_track_history[track_history_idx],
                                            v2=current_track_history[track_history_idx + 1]
                                        )))
                                    # current direction can be removed
                                    current_direction = np.array(current_direction)

                                    current_velocity_direction = []
                                    for track_history_idx in range(len(current_track_velocity_history) - 1):
                                        current_velocity_direction.append(math.degrees(angle_between(
                                            v1=current_track_velocity_history[track_history_idx],
                                            v2=current_track_velocity_history[track_history_idx + 1]
                                        )))
                                    current_velocity_direction = np.array(current_velocity_direction)

                                    # not really required ############################################################
                                    # if len(current_track_history) != 0:
                                    #     current_running_velocity = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[-1], axis=0) -
                                    #         np.expand_dims(current_track_history[0], axis=0),
                                    #         2, axis=0
                                    #     ) / len(current_track_history) / 30
                                    # else:
                                    #     current_running_velocity = None
                                    #
                                    # current_per_step_distance = []
                                    # for track_history_idx in range(len(current_track_history) - 1):
                                    #     d = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                    #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                                    #         2, axis=0
                                    #     )
                                    #     current_per_step_distance.append(d)
                                    #
                                    # current_per_step_distance = np.array(current_per_step_distance)
                                    ###################################################################################

                                    # track_sign = []
                                    # for t in range(
                                    #         len(track_based_accumulated_features[8].object_features[-1].track_history)
                                    #         - 1):
                                    #     track_sign.append(np.sign(
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t + 1] -
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t]))

                                    # just use gt
                                    current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                    track_based_accumulated_features)
                                else:
                                    current_track_history, current_gt_track_history = None, None
                                    current_direction, current_velocity_direction = None, None
                                    current_track_velocity_history = None
                                    # not really required ############################################################
                                    current_per_step_distance = None
                                    current_running_velocity = None
                                    ###################################################################################
                                current_track_obj_features = MinimalObjectFeatures(
                                    idx=current_track_idx,
                                    history=current_track_history,
                                    gt_history=current_gt_track_history,
                                    track_direction=current_direction,
                                    velocity_direction=
                                    current_velocity_direction,
                                    velocity_history=
                                    current_track_velocity_history,
                                    flow=xy_displacement,
                                    past_flow=past_xy_displacement,
                                    past_bbox=box,
                                    final_bbox=np.array(shifted_box),
                                    is_track_live=False,
                                    frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)

                                continue

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?
                        final_shifted_box, final_shifted_xy_center = evaluate_shifted_bounding_box(shifted_box,
                                                                                                   final_features_xy,
                                                                                                   shifted_xy)

                        running_tracks.append(Track(bbox=final_shifted_box, idx=current_track_idx))

                        # if not (final_shifted_box == shifted_box).all():
                        #     logger.warn('Final Shifted Box differs from Shifted Box!')

                        if plot:
                            plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                  shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                  xy_cloud_current_frame=xy_current_frame,
                                                  frame_number=frame_number.item(),
                                                  track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                                  selected_current=closest_n_xy_current_frame_pair)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            current_track_history = get_track_history(current_track_idx,
                                                                      track_based_accumulated_features)
                            current_track_velocity_history = get_track_velocity_history(
                                current_track_idx, track_based_accumulated_features)
                            current_track_velocity_history = np.array(current_track_velocity_history)
                            current_direction = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                current_direction.append((angle_between(
                                    v1=current_track_history[track_history_idx],
                                    v2=current_track_history[track_history_idx + 1]
                                )))
                            # current direction can be removed
                            current_direction = np.array(current_direction)

                            current_velocity_direction = []
                            for track_history_idx in range(len(current_track_velocity_history) - 1):
                                current_velocity_direction.append(math.degrees(angle_between(
                                    v1=current_track_velocity_history[track_history_idx],
                                    v2=current_track_velocity_history[track_history_idx + 1]
                                )))
                            current_velocity_direction = np.array(current_velocity_direction)

                            current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                            track_based_accumulated_features)

                            # not really required ############################################################
                            # if len(current_track_history) != 0:
                            #     current_running_velocity = np.linalg.norm(
                            #         np.expand_dims(current_track_history[-1], axis=0) -
                            #         np.expand_dims(current_track_history[0], axis=0),
                            #         2, axis=0
                            #     ) / len(current_track_history) / 30
                            # else:
                            #     current_running_velocity = None
                            #
                            # current_per_step_distance = []
                            # for track_history_idx in range(len(current_track_history) - 1):
                            #     d = np.linalg.norm(
                            #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                            #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                            #         2, axis=0
                            #     )
                            #     current_per_step_distance.append(d)
                            #
                            # current_per_step_distance = np.array(current_per_step_distance)
                            ###################################################################################
                        else:
                            current_track_history, current_gt_track_history = None, None
                            current_direction, current_velocity_direction = None, None
                            current_track_velocity_history = None
                            # not really required ############################################################
                            current_per_step_distance = None
                            current_running_velocity = None
                            ###################################################################################
                        # STEP 4g: save the information gathered
                        current_track_obj_features = MinimalObjectFeatures(
                            idx=current_track_idx,
                            history=current_track_history,
                            gt_history=current_gt_track_history,
                            track_direction=current_direction,
                            velocity_direction=current_velocity_direction,
                            velocity_history=current_track_velocity_history,
                            flow=xy_displacement,
                            past_flow=past_xy_displacement,
                            past_bbox=box,
                            final_bbox=np.array(final_shifted_box),
                            frame_number=frame_number.item())
                        object_features.append(current_track_obj_features)
                        if current_track_idx not in track_based_accumulated_features:
                            track_feats = TrackFeatures(current_track_idx)
                            track_feats.object_features.append(current_track_obj_features)
                            track_based_accumulated_features.update(
                                {current_track_idx: track_feats})
                        else:
                            track_based_accumulated_features[current_track_idx].object_features.append(
                                current_track_obj_features)
                        if current_track_idx not in track_ids_used:
                            track_ids_used.append(current_track_idx)

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    ratio = float(meta_info.flatten()[-1])

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(annotations[:, :-1])
                    a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    try:
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    except IndexError:
                        if a_boxes.ndim < 2:
                            a_boxes = a_boxes.unsqueeze(0)
                        if r_boxes.ndim < 2:
                            r_boxes = r_boxes.unsqueeze(0)
                        logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                        logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                        # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes = torch.randn((0)).numpy()

                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    if r_boxes_np.size != 0:
                        for a_i, a_box in enumerate(a_boxes_np):
                            for r_i, r_box in enumerate(r_boxes_np):
                                dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                       get_bbox_center(r_box).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]

                        gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                        for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                            gt_t_idx = match_rows_tracks_idx[m_c_idx]
                            # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                            track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                            track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                                gt_track_box_mapping[gt_t_idx]
                            try:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                                    last_frame_gt_tracks[gt_t_idx]
                                gt_distance = np.linalg.norm(
                                    (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                                     get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = gt_distance
                            except KeyError:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = [0, 0]

                        last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    # generated_to_gt_track_association check ###################################################
                    # filter_nones = True
                    # plot_gt_ids = []
                    # plot_gt_boxes = []
                    # plot_generated_ids = []
                    # plot_generated_boxes = []
                    #
                    # for k, v in track_based_accumulated_features.items():
                    #     plot_feats = v.object_features[-1]
                    #     if filter_nones and plot_feats.gt_box is not None:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #     if not filter_nones:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #
                    # plot_image_set_of_boxes(frame, plot_gt_boxes, plot_generated_boxes,
                    #                         annotate=[plot_gt_ids, plot_generated_ids])
                    #############################################################################################
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    if iou_boxes_threshold.size != 0:
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []

                        #  precision/recall
                        tp = 0
                        fp = 0
                        fn = len(a_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    bbox_distance_to_of_centers_iou_based = []
                    bbox_distance_to_of_centers_iou_based_idx = []
                    boxes_distance = []
                    # boxes_distance_for_metric = []
                    r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()

                    # TODO: Replace with Hungarian
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T

                    match_idx = np.where(iou_boxes)
                    matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)

                    for a_box_idx, r_box_idx in zip(*match_idx):
                        dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                               get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                        boxes_distance.append([(a_box_idx, r_box_idx), dist])
                        # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                        matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                        predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                            point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                        )
                        bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                      iou_boxes[a_box_idx, r_box_idx]])
                        bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                                                                          iou_boxes[a_box_idx, r_box_idx]])
                        if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                                if (b == r_boxes[r_box_idx]).all()][0]:
                            selected_track_distances.append(dist)

                    # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    if r_boxes_np.size != 0:
                        a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                            matched_boxes_l2_distance_matrix
                        )

                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = \
                            np.where(predicted_box_center_inside_gt_box_matrix)
                    else:
                        a_matched_boxes_l2_distance_matrix_idx, \
                        r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)

                        meter_precision = meter_tp / (meter_tp + meter_fp)
                        meter_recall = meter_tp / (meter_tp + meter_fn)

                        center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)

                        center_precision = center_tp / (center_tp + center_fp)
                        center_recall = center_tp / (center_tp + center_fn)

                        if len(match_rows) != len(match_cols):
                            logger.info('Matching arrays length not same!')
                        l2_distance_hungarian_tp = len(match_rows)
                        l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                        l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)

                        l2_distance_hungarian_precision = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                        l2_distance_hungarian_recall = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    else:
                        meter_tp = 0
                        meter_fp = 0
                        meter_fn = len(a_boxes)

                        meter_precision = 0
                        meter_recall = 0

                        center_tp = 0
                        center_fp = 0
                        center_fn = len(a_boxes)

                        center_precision = 0
                        center_recall = 0

                        l2_distance_hungarian_tp = 0
                        l2_distance_hungarian_fp = 0
                        l2_distance_hungarian_fn = len(a_boxes)

                        l2_distance_hungarian_precision = 0
                        l2_distance_hungarian_recall = 0

                    meter_tp_list.append(meter_tp)
                    meter_fp_list.append(meter_fp)
                    meter_fn_list.append(meter_fn)

                    center_inside_tp_list.append(center_tp)
                    center_inside_fp_list.append(center_fp)
                    center_inside_fn_list.append(center_fn)

                    l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    new_track_ids = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            annotations, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            # STEP 4h: b> cluster to group points
                            mean_shift, n_clusters = mean_shift_clustering(
                                features_skipped, bin_seeding=False, min_bin_freq=8,
                                cluster_all=True, bandwidth=4, max_iter=100)
                            cluster_centers = mean_shift.cluster_centers

                            # STEP 4h: c> prune cluster centers
                            # combine centers inside radius + eliminate noise
                            final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                cluster_centers, mean_shift, radius + extra_radius,
                                min_points_in_cluster=min_points_in_cluster)

                            if final_cluster_centers.size != 0:
                                t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                # STEP 4h: d> start new potential tracks
                                for cluster_center in final_cluster_centers:
                                    cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                    # t_id = max(track_ids_used) + 1
                                    # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                    t_box = torchvision.ops.box_convert(
                                        torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                        'cxcywh', 'xyxy').int().numpy()
                                    # Note: Do not start track if bbox is out of frame
                                    if use_is_box_overlapping_live_boxes:
                                        if not (np.sign(t_box) < 0).any() and \
                                                not is_box_overlapping_live_boxes(t_box,
                                                                                  [t.bbox for t in running_tracks]):
                                            # NOTE: the second check might result in killing potential tracks!
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                            new_track_ids.append(t_id)
                                    else:
                                        if not (np.sign(t_box) < 0).any():
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                            new_track_ids.append(t_id)

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers,
                                #     num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))
                    new_track_ids = np.stack(new_track_ids) if len(new_track_ids) > 0 else np.empty(shape=(0,))

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    if video_mode:
                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            frame_feats = accumulated_features[frame_number.item()]
                            tracks_histories = []
                            tracks_gt_histories = []
                            for obj_feature in frame_feats.object_features:
                                tracks_histories.extend(obj_feature.track_history)
                                tracks_gt_histories.extend(obj_feature.gt_history)

                            # no need to reverse
                            # tracks_histories.reverse()
                            # tracks_gt_histories.reverse()

                            tracks_histories = np.array(tracks_histories)
                            tracks_gt_histories = np.array(tracks_gt_histories)

                            if tracks_histories.size == 0:
                                tracks_histories = np.zeros(shape=(0, 2))
                            if tracks_gt_histories.size == 0:
                                tracks_gt_histories = np.zeros(shape=(0, 2))
                        else:
                            tracks_histories = np.zeros(shape=(0, 2))
                            tracks_gt_histories = np.zeros(shape=(0, 2))

                        # for gt_annotation_box in annotations[:, :-1]:
                        #     ground_truth_track_histories.append(get_bbox_center(gt_annotation_box).flatten())
                        # ground_truth_track_histories = np.array(ground_truth_track_histories)

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[a_boxes_idx.tolist(), r_boxes_idx],
                            generated_track_histories=tracks_histories,
                            gt_track_histories=tracks_gt_histories,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)

                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=36)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=201)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=212)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=3)
                    else:
                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False, original_dims=original_dims, zero_shot=True)

                    # : save stuff and reiterate - moved up
                    # accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                    #                                                                 object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                        np.array(l2_distance_hungarian_fn_list).sum()
                    batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    logger.info(f'Video: {VIDEO_LABEL.value} - {VIDEO_NUMBER} | Batch: {part_idx} | '
                                f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
            if save_every_n_batch_itr is not None:
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0:
                    save_dict = {'frame_number': frame_number,
                                 'part_idx': part_idx,
                                 'current_track_idx': current_track_idx,
                                 'second_last_frame': second_last_frame,
                                 'last_frame': last_frame,
                                 'last_frame_mask': last_frame_mask,
                                 'last_frame_live_tracks': last_frame_live_tracks,
                                 'running_tracks': running_tracks,
                                 'track_ids_used': track_ids_used,
                                 'new_track_boxes': new_track_boxes,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'tp_list': tp_list,
                                 'fp_list': fp_list,
                                 'fn_list': fn_list,
                                 'meter_tp_list': meter_tp_list,
                                 'meter_fp_list': meter_fp_list,
                                 'meter_fn_list': meter_fn_list,
                                 'center_inside_tp_list': center_inside_tp_list,
                                 'center_inside_fp_list': center_inside_fp_list,
                                 'center_inside_fn_list': center_inside_fn_list,
                                 'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                 'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                 'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                 'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                 'track_based_accumulated_features': track_based_accumulated_features,
                                 'accumulated_features': accumulated_features}
                    Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part_{part_idx}.pt'
                    torch.save(save_dict, features_save_path + 'parts/' + f_n)

                    accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_from_dict_except_entries(live_track_ids,
                                                                                       track_based_accumulated_features)
                    logger.info(f'Saved part {part_idx} at frame {frame_number}')
            # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)
            # if video_mode:
            #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'current_track_idx': current_track_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
        # f_n = f'accumulated_features_from_finally.pt'
        f_n = f'features_dict_part_{part_idx}.pt'
        torch.save(premature_save_dict, features_save_path + 'parts/' + f_n)

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


def preprocess_data_zero_shot_minimal_with_timeout(
        save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
        radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
        plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
        save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
        use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
        use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
        distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
        detect_shadows=True, allow_only_for_first_frame_use_is_box_overlapping_live_boxes=True,
        filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
        angle_threshold_to_filter=120):
    clustering_failed = False
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'
    first_frame_bounding_boxes, first_frame_mask, last_frame, second_last_frame = None, None, None, None
    first_frame_live_tracks, last_frame_live_tracks, last_frame_mask = None, None, None
    current_track_idx, track_ids_used = 0, []
    precision_list, recall_list, matching_boxes_with_iou_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []
    meter_tp_list, meter_fp_list, meter_fn_list = [], [], []
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = [], [], []
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = [], [], []
    selected_track_distances = []
    accumulated_features = {}
    track_based_accumulated_features: Dict[int, TrackFeatures] = {}
    last_frame_gt_tracks = {}
    ground_truth_track_histories = []

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))

            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if (part_idx == 0 and frame_idx == 0) or clustering_failed:
                    # STEP 1: a> Get GT for the first frame
                    validation_annotations, first_frame_mask = first_frame_processing_and_gt_association(
                        df, first_frame_mask,
                        frame_idx, frame_number,
                        frames, frames_count,
                        kernel, n, new_shape,
                        original_shape, step,
                        var_threshold, detect_shadows=detect_shadows)

                    running_tracks, new_track_boxes = [], []
                    all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                        None, first_frame_mask, radius, running_tracks, plot=False)

                    all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                    features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                    if features_skipped_idx.size != 0:
                        features_skipped = all_cloud[features_skipped_idx]

                        try:
                            # STEP 4h: b> cluster to group points
                            mean_shift, n_clusters = mean_shift_clustering_with_timeout(
                                features_skipped, bin_seeding=False, min_bin_freq=8,
                                cluster_all=True, bandwidth=4, max_iter=100)
                            cluster_centers = mean_shift.cluster_centers

                            # STEP 4h: c> prune cluster centers
                            # combine centers inside radius + eliminate noise
                            final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                cluster_centers, mean_shift, radius + extra_radius,
                                min_points_in_cluster=min_points_in_cluster)

                            if final_cluster_centers.size != 0:
                                t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                # STEP 4h: d> start new potential tracks
                                for cluster_center, cluster_center_idx in \
                                        zip(final_cluster_centers, final_cluster_centers_idx):
                                    cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)

                                    # flexible_box, points_in_current_cluster = calculate_flexible_bounding_box(
                                    #     cluster_center_idx, cluster_center_x, cluster_center_y, mean_shift)

                                    # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                    t_box = torchvision.ops.box_convert(
                                        torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                        'cxcywh', 'xyxy').int().numpy()
                                    # Note: Do not start track if bbox is out of frame
                                    if allow_only_for_first_frame_use_is_box_overlapping_live_boxes and \
                                            not (np.sign(t_box) < 0).any() and \
                                            not is_box_overlapping_live_boxes(t_box, [t.bbox for t in running_tracks]):
                                        # NOTE: the second check might result in killing potential tracks!
                                        t_id = max(track_ids_used) + 1 if len(track_ids_used) else 0
                                        running_tracks.append(Track(bbox=t_box, idx=t_id))
                                        track_ids_used.append(t_id)
                                        new_track_boxes.append(t_box)
                                    elif use_is_box_overlapping_live_boxes:
                                        if not (np.sign(t_box) < 0).any() and \
                                                not is_box_overlapping_live_boxes(t_box,
                                                                                  [t.bbox for t in running_tracks]):
                                            # NOTE: the second check might result in killing potential tracks!
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                    else:
                                        if not (np.sign(t_box) < 0).any():
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)

                            clustering_failed = False
                            logger.info('Clustering worked out! Moving on to phase 2!')
                        except TimeoutException:
                            clustering_failed = True
                            logger.info('Clustering took too much time for first/last frame, trying next now')

                        # plot_features_with_circles(
                        #     all_cloud, features_covered, features_skipped, first_frame_mask, marker_size=8,
                        #     cluster_centers=final_cluster_centers, num_clusters=final_cluster_centers.shape[0],
                        #     frame_number=frame_number, boxes=validation_annotations[:, :-1],
                        #     radius=radius+extra_radius,
                        #     additional_text=
                        #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                        #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    if not clustering_failed:
                        new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(
                            shape=(0,))

                        r_boxes = [b.bbox for b in running_tracks]
                        r_boxes_idx = [b.idx for b in running_tracks]
                        select_track_idx = 4

                        r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                        a_boxes = torch.from_numpy(validation_annotations[:, :-1])
                        # Fixme: If no boxes in first frame!!
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes_threshold = iou_boxes.copy()
                        iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                        # TODO: Replace with Hungarian
                        iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                        iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                        match_idx = np.where(iou_boxes)
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]

                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        tp_list.append(tp)
                        fp_list.append(fp)
                        fn_list.append(fn)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
                        precision_list.append(precision)
                        recall_list.append(recall)
                        matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                        last_frame_live_tracks = []

                        fig = plot_for_video(
                            gt_rgb=frame, gt_mask=first_frame_mask, last_frame_rgb=frame,
                            last_frame_mask=first_frame_mask, current_frame_rgb=frame,
                            current_frame_mask=first_frame_mask, gt_annotations=validation_annotations[:, :-1],
                            last_frame_annotation=last_frame_live_tracks,
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False,
                            save_path=f'{plot_save_path}zero_shot/plots{min_points_in_cluster}/')

                        # STEP 1: b> Store them for the next ts and update these variables in further iterations
                        last_frame = frame.copy()
                        second_last_frame = last_frame.copy()
                        last_frame_live_tracks = running_tracks
                        last_frame_mask = first_frame_mask.copy()
                        last_frame_gt_tracks = {a[-1]: a[:-1] for a in validation_annotations}
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    annotations, bbox_centers = scale_annotations(frame_annotation,
                                                                  original_scale=original_shape,
                                                                  new_scale=new_shape, return_track_id=False,
                                                                  tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        current_track_features = track_based_accumulated_features[
                            current_track_idx].object_features[-1] \
                            if current_track_idx in track_based_accumulated_features.keys() else []
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        # if frame_number == 110:
                        #     print()

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # STEP 4c: 1> Switch boxes: Idea 1 - Only keep points inside box
                        shifted_xy = find_points_inside_box(shifted_xy, shifted_box)
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        if xy_current_frame.size == 0 or (filter_switch_boxes_based_on_angle_and_recent_history
                                                          and not isinstance(current_track_features, list)
                                                          and current_track_features.velocity_direction.size != 0
                                                          and len(current_track_features.velocity_direction) >
                                                          min_track_length_to_filter_switch_box
                                                          and first_violation_till_now(
                                    current_track_features.velocity_direction, angle_threshold_to_filter)
                                                          and current_track_features.velocity_direction[-1] >
                                                          angle_threshold_to_filter
                                                          and not np.isnan(
                                    current_track_features.velocity_direction[-1])):

                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
                                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                    current_track_history = get_track_history(current_track_idx,
                                                                              track_based_accumulated_features)
                                    current_track_velocity_history = get_track_velocity_history(
                                        current_track_idx, track_based_accumulated_features)
                                    current_track_velocity_history = np.array(current_track_velocity_history)
                                    current_direction = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        current_direction.append((angle_between(
                                            v1=current_track_history[track_history_idx],
                                            v2=current_track_history[track_history_idx + 1]
                                        )))
                                    # current direction can be removed
                                    current_direction = np.array(current_direction)

                                    current_velocity_direction = []
                                    for track_history_idx in range(len(current_track_velocity_history) - 1):
                                        current_velocity_direction.append(math.degrees(angle_between(
                                            v1=current_track_velocity_history[track_history_idx],
                                            v2=current_track_velocity_history[track_history_idx + 1]
                                        )))
                                    current_velocity_direction = np.array(current_velocity_direction)

                                    # not really required ############################################################
                                    # if len(current_track_history) != 0:
                                    #     current_running_velocity = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[-1], axis=0) -
                                    #         np.expand_dims(current_track_history[0], axis=0),
                                    #         2, axis=0
                                    #     ) / len(current_track_history) / 30
                                    # else:
                                    #     current_running_velocity = None
                                    #
                                    # current_per_step_distance = []
                                    # for track_history_idx in range(len(current_track_history) - 1):
                                    #     d = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                    #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                                    #         2, axis=0
                                    #     )
                                    #     current_per_step_distance.append(d)
                                    #
                                    # current_per_step_distance = np.array(current_per_step_distance)
                                    ###################################################################################

                                    # track_sign = []
                                    # for t in range(
                                    #         len(track_based_accumulated_features[8].object_features[-1].track_history)
                                    #         - 1):
                                    #     track_sign.append(np.sign(
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t + 1] -
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t]))

                                    # just use gt
                                    current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                    track_based_accumulated_features)
                                else:
                                    current_track_history, current_gt_track_history = None, None
                                    current_direction, current_velocity_direction = None, None
                                    current_track_velocity_history = None
                                    # not really required ############################################################
                                    current_per_step_distance = None
                                    current_running_velocity = None
                                    ###################################################################################
                                current_track_obj_features = MinimalObjectFeatures(
                                    idx=current_track_idx,
                                    history=current_track_history,
                                    gt_history=current_gt_track_history,
                                    track_direction=current_direction,
                                    velocity_direction=
                                    current_velocity_direction,
                                    velocity_history=
                                    current_track_velocity_history,
                                    flow=xy_displacement,
                                    past_flow=past_xy_displacement,
                                    past_bbox=box,
                                    final_bbox=np.array(shifted_box),
                                    is_track_live=False,
                                    frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)

                                continue

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?
                        final_shifted_box, final_shifted_xy_center = evaluate_shifted_bounding_box(shifted_box,
                                                                                                   final_features_xy,
                                                                                                   shifted_xy)

                        running_tracks.append(Track(bbox=final_shifted_box, idx=current_track_idx))

                        # if not (final_shifted_box == shifted_box).all():
                        #     logger.warn('Final Shifted Box differs from Shifted Box!')

                        if plot:
                            plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                  shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                  xy_cloud_current_frame=xy_current_frame,
                                                  frame_number=frame_number.item(),
                                                  track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                                  selected_current=closest_n_xy_current_frame_pair)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            current_track_history = get_track_history(current_track_idx,
                                                                      track_based_accumulated_features)
                            current_track_velocity_history = get_track_velocity_history(
                                current_track_idx, track_based_accumulated_features)
                            current_track_velocity_history = np.array(current_track_velocity_history)
                            current_direction = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                current_direction.append((angle_between(
                                    v1=current_track_history[track_history_idx],
                                    v2=current_track_history[track_history_idx + 1]
                                )))
                            # current direction can be removed
                            current_direction = np.array(current_direction)

                            current_velocity_direction = []
                            for track_history_idx in range(len(current_track_velocity_history) - 1):
                                current_velocity_direction.append(math.degrees(angle_between(
                                    v1=current_track_velocity_history[track_history_idx],
                                    v2=current_track_velocity_history[track_history_idx + 1]
                                )))
                            current_velocity_direction = np.array(current_velocity_direction)

                            current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                            track_based_accumulated_features)

                            # not really required ############################################################
                            # if len(current_track_history) != 0:
                            #     current_running_velocity = np.linalg.norm(
                            #         np.expand_dims(current_track_history[-1], axis=0) -
                            #         np.expand_dims(current_track_history[0], axis=0),
                            #         2, axis=0
                            #     ) / len(current_track_history) / 30
                            # else:
                            #     current_running_velocity = None
                            #
                            # current_per_step_distance = []
                            # for track_history_idx in range(len(current_track_history) - 1):
                            #     d = np.linalg.norm(
                            #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                            #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                            #         2, axis=0
                            #     )
                            #     current_per_step_distance.append(d)
                            #
                            # current_per_step_distance = np.array(current_per_step_distance)
                            ###################################################################################
                        else:
                            current_track_history, current_gt_track_history = None, None
                            current_direction, current_velocity_direction = None, None
                            current_track_velocity_history = None
                            # not really required ############################################################
                            current_per_step_distance = None
                            current_running_velocity = None
                            ###################################################################################
                        # STEP 4g: save the information gathered
                        current_track_obj_features = MinimalObjectFeatures(
                            idx=current_track_idx,
                            history=current_track_history,
                            gt_history=current_gt_track_history,
                            track_direction=current_direction,
                            velocity_direction=current_velocity_direction,
                            velocity_history=current_track_velocity_history,
                            flow=xy_displacement,
                            past_flow=past_xy_displacement,
                            past_bbox=box,
                            final_bbox=np.array(final_shifted_box),
                            frame_number=frame_number.item())
                        object_features.append(current_track_obj_features)
                        if current_track_idx not in track_based_accumulated_features:
                            track_feats = TrackFeatures(current_track_idx)
                            track_feats.object_features.append(current_track_obj_features)
                            track_based_accumulated_features.update(
                                {current_track_idx: track_feats})
                        else:
                            track_based_accumulated_features[current_track_idx].object_features.append(
                                current_track_obj_features)
                        if current_track_idx not in track_ids_used:
                            track_ids_used.append(current_track_idx)

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    try:
                        ratio = float(meta_info.flatten()[-1])
                    except IndexError:
                        ratio = 1.0

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(annotations[:, :-1])
                    a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    try:
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    except IndexError:
                        if a_boxes.ndim < 2:
                            a_boxes = a_boxes.unsqueeze(0)
                        if r_boxes.ndim < 2:
                            r_boxes = r_boxes.unsqueeze(0)
                        logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                        logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                        # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes = torch.randn((0)).numpy()

                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    if r_boxes_np.size != 0:
                        for a_i, a_box in enumerate(a_boxes_np):
                            for r_i, r_box in enumerate(r_boxes_np):
                                dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                       get_bbox_center(r_box).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]

                        gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                        for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                            gt_t_idx = match_rows_tracks_idx[m_c_idx]
                            # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                            track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                            track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                                gt_track_box_mapping[gt_t_idx]
                            try:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                                    last_frame_gt_tracks[gt_t_idx]
                                gt_distance = np.linalg.norm(
                                    (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                                     get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = gt_distance
                            except KeyError:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = [0, 0]

                        last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    # generated_to_gt_track_association check ###################################################
                    # filter_nones = True
                    # plot_gt_ids = []
                    # plot_gt_boxes = []
                    # plot_generated_ids = []
                    # plot_generated_boxes = []
                    #
                    # for k, v in track_based_accumulated_features.items():
                    #     plot_feats = v.object_features[-1]
                    #     if filter_nones and plot_feats.gt_box is not None:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #     if not filter_nones:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #
                    # plot_image_set_of_boxes(frame, plot_gt_boxes, plot_generated_boxes,
                    #                         annotate=[plot_gt_ids, plot_generated_ids])
                    #############################################################################################
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    if iou_boxes_threshold.size != 0:
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []

                        #  precision/recall
                        tp = 0
                        fp = 0
                        fn = len(a_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    bbox_distance_to_of_centers_iou_based = []
                    bbox_distance_to_of_centers_iou_based_idx = []
                    boxes_distance = []
                    # boxes_distance_for_metric = []
                    r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()

                    # TODO: Replace with Hungarian
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T

                    match_idx = np.where(iou_boxes)
                    matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)

                    for a_box_idx, r_box_idx in zip(*match_idx):
                        dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                               get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                        boxes_distance.append([(a_box_idx, r_box_idx), dist])
                        # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                        matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                        predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                            point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                        )
                        bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                      iou_boxes[a_box_idx, r_box_idx]])
                        bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                                                                          iou_boxes[a_box_idx, r_box_idx]])
                        if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                                if (b == r_boxes[r_box_idx]).all()][0]:
                            selected_track_distances.append(dist)

                    # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    if r_boxes_np.size != 0:
                        a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                            matched_boxes_l2_distance_matrix
                        )

                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = \
                            np.where(predicted_box_center_inside_gt_box_matrix)
                    else:
                        a_matched_boxes_l2_distance_matrix_idx, \
                        r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)

                        meter_precision = meter_tp / (meter_tp + meter_fp)
                        meter_recall = meter_tp / (meter_tp + meter_fn)

                        center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)

                        center_precision = center_tp / (center_tp + center_fp)
                        center_recall = center_tp / (center_tp + center_fn)

                        if len(match_rows) != len(match_cols):
                            logger.info('Matching arrays length not same!')
                        l2_distance_hungarian_tp = len(match_rows)
                        l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                        l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)

                        l2_distance_hungarian_precision = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                        l2_distance_hungarian_recall = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    else:
                        meter_tp = 0
                        meter_fp = 0
                        meter_fn = len(a_boxes)

                        meter_precision = 0
                        meter_recall = 0

                        center_tp = 0
                        center_fp = 0
                        center_fn = len(a_boxes)

                        center_precision = 0
                        center_recall = 0

                        l2_distance_hungarian_tp = 0
                        l2_distance_hungarian_fp = 0
                        l2_distance_hungarian_fn = len(a_boxes)

                        l2_distance_hungarian_precision = 0
                        l2_distance_hungarian_recall = 0

                    meter_tp_list.append(meter_tp)
                    meter_fp_list.append(meter_fp)
                    meter_fn_list.append(meter_fn)

                    center_inside_tp_list.append(center_tp)
                    center_inside_fp_list.append(center_fp)
                    center_inside_fn_list.append(center_fn)

                    l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    new_track_ids = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            annotations, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            try:
                                # STEP 4h: b> cluster to group points
                                mean_shift, n_clusters = mean_shift_clustering_with_timeout(
                                    features_skipped, bin_seeding=False, min_bin_freq=8,
                                    cluster_all=True, bandwidth=4, max_iter=100)
                                cluster_centers = mean_shift.cluster_centers

                                # STEP 4h: c> prune cluster centers
                                # combine centers inside radius + eliminate noise
                                final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                    cluster_centers, mean_shift, radius + extra_radius,
                                    min_points_in_cluster=min_points_in_cluster)

                                if final_cluster_centers.size != 0:
                                    t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                    # STEP 4h: d> start new potential tracks
                                    for cluster_center in final_cluster_centers:
                                        cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                        # t_id = max(track_ids_used) + 1
                                        # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                        t_box = torchvision.ops.box_convert(
                                            torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                            'cxcywh', 'xyxy').int().numpy()
                                        # Note: Do not start track if bbox is out of frame
                                        if use_is_box_overlapping_live_boxes:
                                            if not (np.sign(t_box) < 0).any() and \
                                                    not is_box_overlapping_live_boxes(t_box,
                                                                                      [t.bbox for t in running_tracks]):
                                                # NOTE: the second check might result in killing potential tracks!
                                                t_id = max(track_ids_used) + 1
                                                running_tracks.append(Track(bbox=t_box, idx=t_id))
                                                track_ids_used.append(t_id)
                                                new_track_boxes.append(t_box)
                                                new_track_ids.append(t_id)
                                        else:
                                            if not (np.sign(t_box) < 0).any():
                                                t_id = max(track_ids_used) + 1
                                                running_tracks.append(Track(bbox=t_box, idx=t_id))
                                                track_ids_used.append(t_id)
                                                new_track_boxes.append(t_box)
                                                new_track_ids.append(t_id)
                            except TimeoutException:
                                logger.warn('Clustering took too much time, skipping!')

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers,
                                #     num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))
                    new_track_ids = np.stack(new_track_ids) if len(new_track_ids) > 0 else np.empty(shape=(0,))

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    if video_mode:
                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            frame_feats = accumulated_features[frame_number.item()]
                            tracks_histories = []
                            tracks_gt_histories = []
                            for obj_feature in frame_feats.object_features:
                                tracks_histories.extend(obj_feature.track_history)
                                tracks_gt_histories.extend(obj_feature.gt_history)

                            # no need to reverse
                            # tracks_histories.reverse()
                            # tracks_gt_histories.reverse()

                            tracks_histories = np.array(tracks_histories)
                            tracks_gt_histories = np.array(tracks_gt_histories)

                            if tracks_histories.size == 0:
                                tracks_histories = np.zeros(shape=(0, 2))
                            if tracks_gt_histories.size == 0:
                                tracks_gt_histories = np.zeros(shape=(0, 2))
                        else:
                            tracks_histories = np.zeros(shape=(0, 2))
                            tracks_gt_histories = np.zeros(shape=(0, 2))

                        # for gt_annotation_box in annotations[:, :-1]:
                        #     ground_truth_track_histories.append(get_bbox_center(gt_annotation_box).flatten())
                        # ground_truth_track_histories = np.array(ground_truth_track_histories)

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[a_boxes_idx.tolist(), r_boxes_idx],
                            generated_track_histories=tracks_histories,
                            gt_track_histories=tracks_gt_histories,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)

                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=36)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=201)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=212)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=3)
                    else:
                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False, original_dims=original_dims, zero_shot=True)

                    # : save stuff and reiterate - moved up
                    # accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                    #                                                                 object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                        np.array(l2_distance_hungarian_fn_list).sum()
                    batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    logger.info(f'Video: {VIDEO_LABEL.value} - {VIDEO_NUMBER} | Batch: {part_idx} | '
                                f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
            if save_every_n_batch_itr is not None:
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0:
                    save_dict = {'frame_number': frame_number,
                                 'part_idx': part_idx,
                                 'current_track_idx': current_track_idx,
                                 'second_last_frame': second_last_frame,
                                 'last_frame': last_frame,
                                 'last_frame_mask': last_frame_mask,
                                 'last_frame_live_tracks': last_frame_live_tracks,
                                 'running_tracks': running_tracks,
                                 'track_ids_used': track_ids_used,
                                 'new_track_boxes': new_track_boxes,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'tp_list': tp_list,
                                 'fp_list': fp_list,
                                 'fn_list': fn_list,
                                 'meter_tp_list': meter_tp_list,
                                 'meter_fp_list': meter_fp_list,
                                 'meter_fn_list': meter_fn_list,
                                 'center_inside_tp_list': center_inside_tp_list,
                                 'center_inside_fp_list': center_inside_fp_list,
                                 'center_inside_fn_list': center_inside_fn_list,
                                 'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                 'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                 'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                 'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                 'track_based_accumulated_features': track_based_accumulated_features,
                                 'accumulated_features': accumulated_features}
                    Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part_{part_idx}.pt'
                    torch.save(save_dict, features_save_path + 'parts/' + f_n)

                    accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_from_dict_except_entries(live_track_ids,
                                                                                       track_based_accumulated_features)
                    logger.info(f'Saved part {part_idx} at frame {frame_number}')
            # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)
            # if video_mode:
            #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'current_track_idx': current_track_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
        # f_n = f'accumulated_features_from_finally.pt'
        f_n = f'features_dict_part_{part_idx}.pt'
        torch.save(premature_save_dict, features_save_path + 'parts/' + f_n)

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


def preprocess_data_zero_shot_minimal_resumable(
        last_saved_features_dict,
        save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
        radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
        plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
        save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
        use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
        use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
        distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
        detect_shadows=True, allow_only_for_first_frame_use_is_box_overlapping_live_boxes=True,
        filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
        angle_threshold_to_filter=120):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'

    part_idx_yet = last_saved_features_dict['part_idx']
    frame_number_yet = last_saved_features_dict['frame_number'].item()
    last_frame, second_last_frame = last_saved_features_dict['last_frame'], \
                                    last_saved_features_dict['second_last_frame']
    last_frame_live_tracks, last_frame_mask = last_saved_features_dict['last_frame_live_tracks'], \
                                              last_saved_features_dict['last_frame_mask']
    running_tracks = last_saved_features_dict['running_tracks']
    current_track_idx, track_ids_used = last_saved_features_dict['current_track_idx'] \
                                            if 'current_track_idx' in last_saved_features_dict.keys() else 0, \
                                        last_saved_features_dict['track_ids_used']
    new_track_boxes = last_saved_features_dict['new_track_boxes']
    precision_list, recall_list, matching_boxes_with_iou_list = last_saved_features_dict['precision'], \
                                                                last_saved_features_dict['recall'], \
                                                                last_saved_features_dict['matching_boxes_with_iou_list']
    tp_list, fp_list, fn_list = last_saved_features_dict['tp_list'], \
                                last_saved_features_dict['fp_list'], \
                                last_saved_features_dict['fn_list']
    meter_tp_list, meter_fp_list, meter_fn_list = last_saved_features_dict['meter_tp_list'], \
                                                  last_saved_features_dict['meter_fp_list'], \
                                                  last_saved_features_dict['meter_fn_list']
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = \
        last_saved_features_dict['l2_distance_hungarian_tp_list'], \
        last_saved_features_dict['l2_distance_hungarian_fp_list'], \
        last_saved_features_dict['l2_distance_hungarian_fn_list']
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = \
        last_saved_features_dict['center_inside_tp_list'], \
        last_saved_features_dict['center_inside_fp_list'], \
        last_saved_features_dict['center_inside_fn_list']
    selected_track_distances = []
    accumulated_features = {}
    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
    track_based_accumulated_features: Dict[int, TrackFeatures] = remove_from_dict_except_entries(
        live_track_ids, last_saved_features_dict['track_based_accumulated_features'])
    last_frame_gt_tracks = {}
    ground_truth_track_histories = []

    del last_saved_features_dict

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))

            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if part_idx <= part_idx_yet and frame_number <= frame_number_yet:
                    continue
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    annotations, bbox_centers = scale_annotations(frame_annotation,
                                                                  original_scale=original_shape,
                                                                  new_scale=new_shape, return_track_id=False,
                                                                  tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        current_track_features = track_based_accumulated_features[
                            current_track_idx].object_features[-1] \
                            if current_track_idx in track_based_accumulated_features.keys() else []
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        # if frame_number == 110:
                        #     print()

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # STEP 4c: 1> Switch boxes: Idea 1 - Only keep points inside box
                        shifted_xy = find_points_inside_box(shifted_xy, shifted_box)
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        if xy_current_frame.size == 0 or (filter_switch_boxes_based_on_angle_and_recent_history
                                                          and not isinstance(current_track_features, list)
                                                          and current_track_features.velocity_direction.size != 0
                                                          and len(current_track_features.velocity_direction) >
                                                          min_track_length_to_filter_switch_box
                                                          and first_violation_till_now(
                                    current_track_features.velocity_direction, angle_threshold_to_filter)
                                                          and current_track_features.velocity_direction[-1] >
                                                          angle_threshold_to_filter
                                                          and not np.isnan(
                                    current_track_features.velocity_direction[-1])):

                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
                                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                    current_track_history = get_track_history(current_track_idx,
                                                                              track_based_accumulated_features)
                                    current_track_velocity_history = get_track_velocity_history(
                                        current_track_idx, track_based_accumulated_features)
                                    current_track_velocity_history = np.array(current_track_velocity_history)
                                    current_direction = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        current_direction.append((angle_between(
                                            v1=current_track_history[track_history_idx],
                                            v2=current_track_history[track_history_idx + 1]
                                        )))
                                    # current direction can be removed
                                    current_direction = np.array(current_direction)

                                    current_velocity_direction = []
                                    for track_history_idx in range(len(current_track_velocity_history) - 1):
                                        current_velocity_direction.append(math.degrees(angle_between(
                                            v1=current_track_velocity_history[track_history_idx],
                                            v2=current_track_velocity_history[track_history_idx + 1]
                                        )))
                                    current_velocity_direction = np.array(current_velocity_direction)

                                    # not really required ############################################################
                                    # if len(current_track_history) != 0:
                                    #     current_running_velocity = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[-1], axis=0) -
                                    #         np.expand_dims(current_track_history[0], axis=0),
                                    #         2, axis=0
                                    #     ) / len(current_track_history) / 30
                                    # else:
                                    #     current_running_velocity = None
                                    #
                                    # current_per_step_distance = []
                                    # for track_history_idx in range(len(current_track_history) - 1):
                                    #     d = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                    #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                                    #         2, axis=0
                                    #     )
                                    #     current_per_step_distance.append(d)
                                    #
                                    # current_per_step_distance = np.array(current_per_step_distance)
                                    ###################################################################################

                                    # track_sign = []
                                    # for t in range(
                                    #         len(track_based_accumulated_features[8].object_features[-1].track_history)
                                    #         - 1):
                                    #     track_sign.append(np.sign(
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t + 1] -
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t]))

                                    # just use gt
                                    current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                    track_based_accumulated_features)
                                else:
                                    current_track_history, current_gt_track_history = None, None
                                    current_direction, current_velocity_direction = None, None
                                    current_track_velocity_history = None
                                    # not really required ############################################################
                                    current_per_step_distance = None
                                    current_running_velocity = None
                                    ###################################################################################
                                current_track_obj_features = MinimalObjectFeatures(
                                    idx=current_track_idx,
                                    history=current_track_history,
                                    gt_history=current_gt_track_history,
                                    track_direction=current_direction,
                                    velocity_direction=
                                    current_velocity_direction,
                                    velocity_history=
                                    current_track_velocity_history,
                                    flow=xy_displacement,
                                    past_flow=past_xy_displacement,
                                    past_bbox=box,
                                    final_bbox=np.array(shifted_box),
                                    is_track_live=False,
                                    frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)

                                continue

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?
                        final_shifted_box, final_shifted_xy_center = evaluate_shifted_bounding_box(shifted_box,
                                                                                                   final_features_xy,
                                                                                                   shifted_xy)

                        running_tracks.append(Track(bbox=final_shifted_box, idx=current_track_idx))

                        # if not (final_shifted_box == shifted_box).all():
                        #     logger.warn('Final Shifted Box differs from Shifted Box!')

                        if plot:
                            plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                  shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                  xy_cloud_current_frame=xy_current_frame,
                                                  frame_number=frame_number.item(),
                                                  track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                                  selected_current=closest_n_xy_current_frame_pair)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            current_track_history = get_track_history(current_track_idx,
                                                                      track_based_accumulated_features)
                            current_track_velocity_history = get_track_velocity_history(
                                current_track_idx, track_based_accumulated_features)
                            current_track_velocity_history = np.array(current_track_velocity_history)
                            current_direction = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                current_direction.append((angle_between(
                                    v1=current_track_history[track_history_idx],
                                    v2=current_track_history[track_history_idx + 1]
                                )))
                            # current direction can be removed
                            current_direction = np.array(current_direction)

                            current_velocity_direction = []
                            for track_history_idx in range(len(current_track_velocity_history) - 1):
                                current_velocity_direction.append(math.degrees(angle_between(
                                    v1=current_track_velocity_history[track_history_idx],
                                    v2=current_track_velocity_history[track_history_idx + 1]
                                )))
                            current_velocity_direction = np.array(current_velocity_direction)

                            current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                            track_based_accumulated_features)

                            # not really required ############################################################
                            # if len(current_track_history) != 0:
                            #     current_running_velocity = np.linalg.norm(
                            #         np.expand_dims(current_track_history[-1], axis=0) -
                            #         np.expand_dims(current_track_history[0], axis=0),
                            #         2, axis=0
                            #     ) / len(current_track_history) / 30
                            # else:
                            #     current_running_velocity = None
                            #
                            # current_per_step_distance = []
                            # for track_history_idx in range(len(current_track_history) - 1):
                            #     d = np.linalg.norm(
                            #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                            #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                            #         2, axis=0
                            #     )
                            #     current_per_step_distance.append(d)
                            #
                            # current_per_step_distance = np.array(current_per_step_distance)
                            ###################################################################################
                        else:
                            current_track_history, current_gt_track_history = None, None
                            current_direction, current_velocity_direction = None, None
                            current_track_velocity_history = None
                            # not really required ############################################################
                            current_per_step_distance = None
                            current_running_velocity = None
                            ###################################################################################
                        # STEP 4g: save the information gathered
                        current_track_obj_features = MinimalObjectFeatures(
                            idx=current_track_idx,
                            history=current_track_history,
                            gt_history=current_gt_track_history,
                            track_direction=current_direction,
                            velocity_direction=current_velocity_direction,
                            velocity_history=current_track_velocity_history,
                            flow=xy_displacement,
                            past_flow=past_xy_displacement,
                            past_bbox=box,
                            final_bbox=np.array(final_shifted_box),
                            frame_number=frame_number.item())
                        object_features.append(current_track_obj_features)
                        if current_track_idx not in track_based_accumulated_features:
                            track_feats = TrackFeatures(current_track_idx)
                            track_feats.object_features.append(current_track_obj_features)
                            track_based_accumulated_features.update(
                                {current_track_idx: track_feats})
                        else:
                            track_based_accumulated_features[current_track_idx].object_features.append(
                                current_track_obj_features)
                        if current_track_idx not in track_ids_used:
                            track_ids_used.append(current_track_idx)

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    ratio = float(meta_info.flatten()[-1])

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(annotations[:, :-1])
                    a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    try:
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    except IndexError:
                        if a_boxes.ndim < 2:
                            a_boxes = a_boxes.unsqueeze(0)
                        if r_boxes.ndim < 2:
                            r_boxes = r_boxes.unsqueeze(0)
                        logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                        logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                        # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes = torch.randn((0)).numpy()

                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    if r_boxes_np.size != 0:
                        for a_i, a_box in enumerate(a_boxes_np):
                            for r_i, r_box in enumerate(r_boxes_np):
                                dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                       get_bbox_center(r_box).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]

                        gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                        for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                            gt_t_idx = match_rows_tracks_idx[m_c_idx]
                            # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                            track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                            track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                                gt_track_box_mapping[gt_t_idx]
                            try:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                                    last_frame_gt_tracks[gt_t_idx]
                                gt_distance = np.linalg.norm(
                                    (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                                     get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = gt_distance
                            except KeyError:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = [0, 0]

                        last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    # generated_to_gt_track_association check ###################################################
                    # filter_nones = True
                    # plot_gt_ids = []
                    # plot_gt_boxes = []
                    # plot_generated_ids = []
                    # plot_generated_boxes = []
                    #
                    # for k, v in track_based_accumulated_features.items():
                    #     plot_feats = v.object_features[-1]
                    #     if filter_nones and plot_feats.gt_box is not None:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #     if not filter_nones:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #
                    # plot_image_set_of_boxes(frame, plot_gt_boxes, plot_generated_boxes,
                    #                         annotate=[plot_gt_ids, plot_generated_ids])
                    #############################################################################################
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    if iou_boxes_threshold.size != 0:
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []

                        #  precision/recall
                        tp = 0
                        fp = 0
                        fn = len(a_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    bbox_distance_to_of_centers_iou_based = []
                    bbox_distance_to_of_centers_iou_based_idx = []
                    boxes_distance = []
                    # boxes_distance_for_metric = []
                    r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()

                    # TODO: Replace with Hungarian
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T

                    match_idx = np.where(iou_boxes)
                    matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)

                    for a_box_idx, r_box_idx in zip(*match_idx):
                        dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                               get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                        boxes_distance.append([(a_box_idx, r_box_idx), dist])
                        # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                        matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                        predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                            point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                        )
                        bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                      iou_boxes[a_box_idx, r_box_idx]])
                        bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                                                                          iou_boxes[a_box_idx, r_box_idx]])
                        if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                                if (b == r_boxes[r_box_idx]).all()][0]:
                            selected_track_distances.append(dist)

                    # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    if r_boxes_np.size != 0:
                        a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                            matched_boxes_l2_distance_matrix
                        )

                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = \
                            np.where(predicted_box_center_inside_gt_box_matrix)
                    else:
                        a_matched_boxes_l2_distance_matrix_idx, \
                        r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)

                        meter_precision = meter_tp / (meter_tp + meter_fp)
                        meter_recall = meter_tp / (meter_tp + meter_fn)

                        center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)

                        center_precision = center_tp / (center_tp + center_fp)
                        center_recall = center_tp / (center_tp + center_fn)

                        if len(match_rows) != len(match_cols):
                            logger.info('Matching arrays length not same!')
                        l2_distance_hungarian_tp = len(match_rows)
                        l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                        l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)

                        l2_distance_hungarian_precision = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                        l2_distance_hungarian_recall = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    else:
                        meter_tp = 0
                        meter_fp = 0
                        meter_fn = len(a_boxes)

                        meter_precision = 0
                        meter_recall = 0

                        center_tp = 0
                        center_fp = 0
                        center_fn = len(a_boxes)

                        center_precision = 0
                        center_recall = 0

                        l2_distance_hungarian_tp = 0
                        l2_distance_hungarian_fp = 0
                        l2_distance_hungarian_fn = len(a_boxes)

                        l2_distance_hungarian_precision = 0
                        l2_distance_hungarian_recall = 0

                    meter_tp_list.append(meter_tp)
                    meter_fp_list.append(meter_fp)
                    meter_fn_list.append(meter_fn)

                    center_inside_tp_list.append(center_tp)
                    center_inside_fp_list.append(center_fp)
                    center_inside_fn_list.append(center_fn)

                    l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    new_track_ids = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            annotations, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            # STEP 4h: b> cluster to group points
                            mean_shift, n_clusters = mean_shift_clustering(
                                features_skipped, bin_seeding=False, min_bin_freq=8,
                                cluster_all=True, bandwidth=4, max_iter=100)
                            cluster_centers = mean_shift.cluster_centers

                            # STEP 4h: c> prune cluster centers
                            # combine centers inside radius + eliminate noise
                            final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                cluster_centers, mean_shift, radius + extra_radius,
                                min_points_in_cluster=min_points_in_cluster)

                            if final_cluster_centers.size != 0:
                                t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                # STEP 4h: d> start new potential tracks
                                for cluster_center in final_cluster_centers:
                                    cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                    # t_id = max(track_ids_used) + 1
                                    # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                    t_box = torchvision.ops.box_convert(
                                        torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                        'cxcywh', 'xyxy').int().numpy()
                                    # Note: Do not start track if bbox is out of frame
                                    if use_is_box_overlapping_live_boxes:
                                        if not (np.sign(t_box) < 0).any() and \
                                                not is_box_overlapping_live_boxes(t_box,
                                                                                  [t.bbox for t in running_tracks]):
                                            # NOTE: the second check might result in killing potential tracks!
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                            new_track_ids.append(t_id)
                                    else:
                                        if not (np.sign(t_box) < 0).any():
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                            new_track_ids.append(t_id)

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers,
                                #     num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))
                    new_track_ids = np.stack(new_track_ids) if len(new_track_ids) > 0 else np.empty(shape=(0,))

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    if video_mode:
                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            frame_feats = accumulated_features[frame_number.item()]
                            tracks_histories = []
                            tracks_gt_histories = []
                            for obj_feature in frame_feats.object_features:
                                tracks_histories.extend(obj_feature.track_history)
                                tracks_gt_histories.extend(obj_feature.gt_history)

                            # no need to reverse
                            # tracks_histories.reverse()
                            # tracks_gt_histories.reverse()

                            tracks_histories = np.array(tracks_histories)
                            tracks_gt_histories = np.array(tracks_gt_histories)

                            if tracks_histories.size == 0:
                                tracks_histories = np.zeros(shape=(0, 2))
                            if tracks_gt_histories.size == 0:
                                tracks_gt_histories = np.zeros(shape=(0, 2))
                        else:
                            tracks_histories = np.zeros(shape=(0, 2))
                            tracks_gt_histories = np.zeros(shape=(0, 2))

                        # for gt_annotation_box in annotations[:, :-1]:
                        #     ground_truth_track_histories.append(get_bbox_center(gt_annotation_box).flatten())
                        # ground_truth_track_histories = np.array(ground_truth_track_histories)

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[a_boxes_idx.tolist(), r_boxes_idx],
                            generated_track_histories=tracks_histories,
                            gt_track_histories=tracks_gt_histories,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)

                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=36)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=201)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=212)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=3)
                    else:
                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False, original_dims=original_dims, zero_shot=True)

                    # : save stuff and reiterate - moved up
                    # accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                    #                                                                 object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                        np.array(l2_distance_hungarian_fn_list).sum()
                    batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    logger.info(f'Video: {VIDEO_LABEL.value} - {VIDEO_NUMBER} | Batch: {part_idx} | '
                                f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
            if save_every_n_batch_itr is not None:
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0 and part_idx > part_idx_yet:
                    save_dict = {'frame_number': frame_number,
                                 'part_idx': part_idx,
                                 'current_track_idx': current_track_idx,
                                 'second_last_frame': second_last_frame,
                                 'last_frame': last_frame,
                                 'last_frame_mask': last_frame_mask,
                                 'last_frame_live_tracks': last_frame_live_tracks,
                                 'running_tracks': running_tracks,
                                 'track_ids_used': track_ids_used,
                                 'new_track_boxes': new_track_boxes,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'tp_list': tp_list,
                                 'fp_list': fp_list,
                                 'fn_list': fn_list,
                                 'meter_tp_list': meter_tp_list,
                                 'meter_fp_list': meter_fp_list,
                                 'meter_fn_list': meter_fn_list,
                                 'center_inside_tp_list': center_inside_tp_list,
                                 'center_inside_fp_list': center_inside_fp_list,
                                 'center_inside_fn_list': center_inside_fn_list,
                                 'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                 'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                 'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                 'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                 'track_based_accumulated_features': track_based_accumulated_features,
                                 'accumulated_features': accumulated_features}
                    Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part_{part_idx}.pt'
                    torch.save(save_dict, features_save_path + 'parts/' + f_n)
                    logger.info(f"Saved at {features_save_path + 'parts/' + f_n}")

                    accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_from_dict_except_entries(live_track_ids,
                                                                                       track_based_accumulated_features)
                    logger.info(f'Saved part {part_idx} at frame {frame_number}')
            # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)
            # if video_mode:
            #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'current_track_idx': current_track_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
        # f_n = f'accumulated_features_from_finally.pt'
        f_n = f'features_dict_part_{part_idx}.pt'
        torch.save(premature_save_dict, features_save_path + 'parts/' + f_n)
        logger.info(f"Saved at {features_save_path + 'parts/' + f_n}")

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


def preprocess_data_zero_shot_minimal_resumable_with_timeout(
        last_saved_features_dict,
        save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
        radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
        plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
        save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
        use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
        use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
        distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
        detect_shadows=True, allow_only_for_first_frame_use_is_box_overlapping_live_boxes=True,
        filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
        angle_threshold_to_filter=120):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'

    part_idx_yet = last_saved_features_dict['part_idx']
    frame_number_yet = last_saved_features_dict['frame_number'].item()
    last_frame, second_last_frame = last_saved_features_dict['last_frame'], \
                                    last_saved_features_dict['second_last_frame']
    last_frame_live_tracks, last_frame_mask = last_saved_features_dict['last_frame_live_tracks'], \
                                              last_saved_features_dict['last_frame_mask']
    running_tracks = last_saved_features_dict['running_tracks']
    current_track_idx, track_ids_used = last_saved_features_dict['current_track_idx'] \
                                            if 'current_track_idx' in last_saved_features_dict.keys() else 0, \
                                        last_saved_features_dict['track_ids_used']
    new_track_boxes = last_saved_features_dict['new_track_boxes']
    precision_list, recall_list, matching_boxes_with_iou_list = last_saved_features_dict['precision'], \
                                                                last_saved_features_dict['recall'], \
                                                                last_saved_features_dict['matching_boxes_with_iou_list']
    tp_list, fp_list, fn_list = last_saved_features_dict['tp_list'], \
                                last_saved_features_dict['fp_list'], \
                                last_saved_features_dict['fn_list']
    meter_tp_list, meter_fp_list, meter_fn_list = last_saved_features_dict['meter_tp_list'], \
                                                  last_saved_features_dict['meter_fp_list'], \
                                                  last_saved_features_dict['meter_fn_list']
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = \
        last_saved_features_dict['l2_distance_hungarian_tp_list'], \
        last_saved_features_dict['l2_distance_hungarian_fp_list'], \
        last_saved_features_dict['l2_distance_hungarian_fn_list']
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = \
        last_saved_features_dict['center_inside_tp_list'], \
        last_saved_features_dict['center_inside_fp_list'], \
        last_saved_features_dict['center_inside_fn_list']
    selected_track_distances = []
    accumulated_features = {}
    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
    track_based_accumulated_features: Dict[int, TrackFeatures] = remove_from_dict_except_entries(
        live_track_ids, last_saved_features_dict['track_based_accumulated_features'])
    last_frame_gt_tracks = {}
    ground_truth_track_histories = []

    del last_saved_features_dict

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))

            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if part_idx <= part_idx_yet and frame_number <= frame_number_yet:
                    continue
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    annotations, bbox_centers = scale_annotations(frame_annotation,
                                                                  original_scale=original_shape,
                                                                  new_scale=new_shape, return_track_id=False,
                                                                  tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        current_track_features = track_based_accumulated_features[
                            current_track_idx].object_features[-1] \
                            if current_track_idx in track_based_accumulated_features.keys() else []
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        # if frame_number == 110:
                        #     print()

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # STEP 4c: 1> Switch boxes: Idea 1 - Only keep points inside box
                        shifted_xy = find_points_inside_box(shifted_xy, shifted_box)
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        if xy_current_frame.size == 0 or (filter_switch_boxes_based_on_angle_and_recent_history
                                                          and not isinstance(current_track_features, list)
                                                          and current_track_features.velocity_direction.size != 0
                                                          and len(current_track_features.velocity_direction) >
                                                          min_track_length_to_filter_switch_box
                                                          and first_violation_till_now(
                                    current_track_features.velocity_direction, angle_threshold_to_filter)
                                                          and current_track_features.velocity_direction[-1] >
                                                          angle_threshold_to_filter
                                                          and not np.isnan(
                                    current_track_features.velocity_direction[-1])):

                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
                                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                    current_track_history = get_track_history(current_track_idx,
                                                                              track_based_accumulated_features)
                                    current_track_velocity_history = get_track_velocity_history(
                                        current_track_idx, track_based_accumulated_features)
                                    current_track_velocity_history = np.array(current_track_velocity_history)
                                    current_direction = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        current_direction.append((angle_between(
                                            v1=current_track_history[track_history_idx],
                                            v2=current_track_history[track_history_idx + 1]
                                        )))
                                    # current direction can be removed
                                    current_direction = np.array(current_direction)

                                    current_velocity_direction = []
                                    for track_history_idx in range(len(current_track_velocity_history) - 1):
                                        current_velocity_direction.append(math.degrees(angle_between(
                                            v1=current_track_velocity_history[track_history_idx],
                                            v2=current_track_velocity_history[track_history_idx + 1]
                                        )))
                                    current_velocity_direction = np.array(current_velocity_direction)

                                    # not really required ############################################################
                                    # if len(current_track_history) != 0:
                                    #     current_running_velocity = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[-1], axis=0) -
                                    #         np.expand_dims(current_track_history[0], axis=0),
                                    #         2, axis=0
                                    #     ) / len(current_track_history) / 30
                                    # else:
                                    #     current_running_velocity = None
                                    #
                                    # current_per_step_distance = []
                                    # for track_history_idx in range(len(current_track_history) - 1):
                                    #     d = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                    #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                                    #         2, axis=0
                                    #     )
                                    #     current_per_step_distance.append(d)
                                    #
                                    # current_per_step_distance = np.array(current_per_step_distance)
                                    ###################################################################################

                                    # track_sign = []
                                    # for t in range(
                                    #         len(track_based_accumulated_features[8].object_features[-1].track_history)
                                    #         - 1):
                                    #     track_sign.append(np.sign(
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t + 1] -
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t]))

                                    # just use gt
                                    current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                    track_based_accumulated_features)
                                else:
                                    current_track_history, current_gt_track_history = None, None
                                    current_direction, current_velocity_direction = None, None
                                    current_track_velocity_history = None
                                    # not really required ############################################################
                                    current_per_step_distance = None
                                    current_running_velocity = None
                                    ###################################################################################
                                current_track_obj_features = MinimalObjectFeatures(
                                    idx=current_track_idx,
                                    history=current_track_history,
                                    gt_history=current_gt_track_history,
                                    track_direction=current_direction,
                                    velocity_direction=
                                    current_velocity_direction,
                                    velocity_history=
                                    current_track_velocity_history,
                                    flow=xy_displacement,
                                    past_flow=past_xy_displacement,
                                    past_bbox=box,
                                    final_bbox=np.array(shifted_box),
                                    is_track_live=False,
                                    frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)

                                continue

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?
                        final_shifted_box, final_shifted_xy_center = evaluate_shifted_bounding_box(shifted_box,
                                                                                                   final_features_xy,
                                                                                                   shifted_xy)

                        running_tracks.append(Track(bbox=final_shifted_box, idx=current_track_idx))

                        # if not (final_shifted_box == shifted_box).all():
                        #     logger.warn('Final Shifted Box differs from Shifted Box!')

                        if plot:
                            plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                  shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                  xy_cloud_current_frame=xy_current_frame,
                                                  frame_number=frame_number.item(),
                                                  track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                                  selected_current=closest_n_xy_current_frame_pair)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            current_track_history = get_track_history(current_track_idx,
                                                                      track_based_accumulated_features)
                            current_track_velocity_history = get_track_velocity_history(
                                current_track_idx, track_based_accumulated_features)
                            current_track_velocity_history = np.array(current_track_velocity_history)
                            current_direction = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                current_direction.append((angle_between(
                                    v1=current_track_history[track_history_idx],
                                    v2=current_track_history[track_history_idx + 1]
                                )))
                            # current direction can be removed
                            current_direction = np.array(current_direction)

                            current_velocity_direction = []
                            for track_history_idx in range(len(current_track_velocity_history) - 1):
                                current_velocity_direction.append(math.degrees(angle_between(
                                    v1=current_track_velocity_history[track_history_idx],
                                    v2=current_track_velocity_history[track_history_idx + 1]
                                )))
                            current_velocity_direction = np.array(current_velocity_direction)

                            current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                            track_based_accumulated_features)

                            # not really required ############################################################
                            # if len(current_track_history) != 0:
                            #     current_running_velocity = np.linalg.norm(
                            #         np.expand_dims(current_track_history[-1], axis=0) -
                            #         np.expand_dims(current_track_history[0], axis=0),
                            #         2, axis=0
                            #     ) / len(current_track_history) / 30
                            # else:
                            #     current_running_velocity = None
                            #
                            # current_per_step_distance = []
                            # for track_history_idx in range(len(current_track_history) - 1):
                            #     d = np.linalg.norm(
                            #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                            #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                            #         2, axis=0
                            #     )
                            #     current_per_step_distance.append(d)
                            #
                            # current_per_step_distance = np.array(current_per_step_distance)
                            ###################################################################################
                        else:
                            current_track_history, current_gt_track_history = None, None
                            current_direction, current_velocity_direction = None, None
                            current_track_velocity_history = None
                            # not really required ############################################################
                            current_per_step_distance = None
                            current_running_velocity = None
                            ###################################################################################
                        # STEP 4g: save the information gathered
                        current_track_obj_features = MinimalObjectFeatures(
                            idx=current_track_idx,
                            history=current_track_history,
                            gt_history=current_gt_track_history,
                            track_direction=current_direction,
                            velocity_direction=current_velocity_direction,
                            velocity_history=current_track_velocity_history,
                            flow=xy_displacement,
                            past_flow=past_xy_displacement,
                            past_bbox=box,
                            final_bbox=np.array(final_shifted_box),
                            frame_number=frame_number.item())
                        object_features.append(current_track_obj_features)
                        if current_track_idx not in track_based_accumulated_features:
                            track_feats = TrackFeatures(current_track_idx)
                            track_feats.object_features.append(current_track_obj_features)
                            track_based_accumulated_features.update(
                                {current_track_idx: track_feats})
                        else:
                            track_based_accumulated_features[current_track_idx].object_features.append(
                                current_track_obj_features)
                        if current_track_idx not in track_ids_used:
                            track_ids_used.append(current_track_idx)

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    ratio = float(meta_info.flatten()[-1])

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(annotations[:, :-1])
                    a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    try:
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    except IndexError:
                        if a_boxes.ndim < 2:
                            a_boxes = a_boxes.unsqueeze(0)
                        if r_boxes.ndim < 2:
                            r_boxes = r_boxes.unsqueeze(0)
                        logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                        logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                        # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes = torch.randn((0)).numpy()

                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    if r_boxes_np.size != 0:
                        for a_i, a_box in enumerate(a_boxes_np):
                            for r_i, r_box in enumerate(r_boxes_np):
                                dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                       get_bbox_center(r_box).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]

                        gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                        for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                            gt_t_idx = match_rows_tracks_idx[m_c_idx]
                            # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                            track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                            track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                                gt_track_box_mapping[gt_t_idx]
                            try:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                                    last_frame_gt_tracks[gt_t_idx]
                                gt_distance = np.linalg.norm(
                                    (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                                     get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = gt_distance
                            except KeyError:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = [0, 0]

                        last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    # generated_to_gt_track_association check ###################################################
                    # filter_nones = True
                    # plot_gt_ids = []
                    # plot_gt_boxes = []
                    # plot_generated_ids = []
                    # plot_generated_boxes = []
                    #
                    # for k, v in track_based_accumulated_features.items():
                    #     plot_feats = v.object_features[-1]
                    #     if filter_nones and plot_feats.gt_box is not None:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #     if not filter_nones:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #
                    # plot_image_set_of_boxes(frame, plot_gt_boxes, plot_generated_boxes,
                    #                         annotate=[plot_gt_ids, plot_generated_ids])
                    #############################################################################################
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    if iou_boxes_threshold.size != 0:
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []

                        #  precision/recall
                        tp = 0
                        fp = 0
                        fn = len(a_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    bbox_distance_to_of_centers_iou_based = []
                    bbox_distance_to_of_centers_iou_based_idx = []
                    boxes_distance = []
                    # boxes_distance_for_metric = []
                    r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()

                    # TODO: Replace with Hungarian
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T

                    match_idx = np.where(iou_boxes)
                    matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)

                    for a_box_idx, r_box_idx in zip(*match_idx):
                        dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                               get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                        boxes_distance.append([(a_box_idx, r_box_idx), dist])
                        # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                        matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                        predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                            point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                        )
                        bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                      iou_boxes[a_box_idx, r_box_idx]])
                        bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                                                                          iou_boxes[a_box_idx, r_box_idx]])
                        if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                                if (b == r_boxes[r_box_idx]).all()][0]:
                            selected_track_distances.append(dist)

                    # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    if r_boxes_np.size != 0:
                        a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                            matched_boxes_l2_distance_matrix
                        )

                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = \
                            np.where(predicted_box_center_inside_gt_box_matrix)
                    else:
                        a_matched_boxes_l2_distance_matrix_idx, \
                        r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)

                        meter_precision = meter_tp / (meter_tp + meter_fp)
                        meter_recall = meter_tp / (meter_tp + meter_fn)

                        center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)

                        center_precision = center_tp / (center_tp + center_fp)
                        center_recall = center_tp / (center_tp + center_fn)

                        if len(match_rows) != len(match_cols):
                            logger.info('Matching arrays length not same!')
                        l2_distance_hungarian_tp = len(match_rows)
                        l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                        l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)

                        l2_distance_hungarian_precision = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                        l2_distance_hungarian_recall = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    else:
                        meter_tp = 0
                        meter_fp = 0
                        meter_fn = len(a_boxes)

                        meter_precision = 0
                        meter_recall = 0

                        center_tp = 0
                        center_fp = 0
                        center_fn = len(a_boxes)

                        center_precision = 0
                        center_recall = 0

                        l2_distance_hungarian_tp = 0
                        l2_distance_hungarian_fp = 0
                        l2_distance_hungarian_fn = len(a_boxes)

                        l2_distance_hungarian_precision = 0
                        l2_distance_hungarian_recall = 0

                    meter_tp_list.append(meter_tp)
                    meter_fp_list.append(meter_fp)
                    meter_fn_list.append(meter_fn)

                    center_inside_tp_list.append(center_tp)
                    center_inside_fp_list.append(center_fp)
                    center_inside_fn_list.append(center_fn)

                    l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    new_track_ids = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            annotations, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            # STEP 4h: b> cluster to group points
                            try:
                                mean_shift, n_clusters = mean_shift_clustering_with_timeout(
                                    features_skipped, bin_seeding=False, min_bin_freq=8,
                                    cluster_all=True, bandwidth=4, max_iter=100)
                                cluster_centers = mean_shift.cluster_centers

                                # STEP 4h: c> prune cluster centers
                                # combine centers inside radius + eliminate noise
                                final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                    cluster_centers, mean_shift, radius + extra_radius,
                                    min_points_in_cluster=min_points_in_cluster)

                                if final_cluster_centers.size != 0:
                                    t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                    # STEP 4h: d> start new potential tracks
                                    for cluster_center in final_cluster_centers:
                                        cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                        # t_id = max(track_ids_used) + 1
                                        # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                        t_box = torchvision.ops.box_convert(
                                            torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                            'cxcywh', 'xyxy').int().numpy()
                                        # Note: Do not start track if bbox is out of frame
                                        if use_is_box_overlapping_live_boxes:
                                            if not (np.sign(t_box) < 0).any() and \
                                                    not is_box_overlapping_live_boxes(t_box,
                                                                                      [t.bbox for t in running_tracks]):
                                                # NOTE: the second check might result in killing potential tracks!
                                                t_id = max(track_ids_used) + 1
                                                running_tracks.append(Track(bbox=t_box, idx=t_id))
                                                track_ids_used.append(t_id)
                                                new_track_boxes.append(t_box)
                                                new_track_ids.append(t_id)
                                        else:
                                            if not (np.sign(t_box) < 0).any():
                                                t_id = max(track_ids_used) + 1
                                                running_tracks.append(Track(bbox=t_box, idx=t_id))
                                                track_ids_used.append(t_id)
                                                new_track_boxes.append(t_box)
                                                new_track_ids.append(t_id)
                            except TimeoutException:
                                logger.warn('Clustering took too much time, skipping!')

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers,
                                #     num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))
                    new_track_ids = np.stack(new_track_ids) if len(new_track_ids) > 0 else np.empty(shape=(0,))

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    if video_mode:
                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            frame_feats = accumulated_features[frame_number.item()]
                            tracks_histories = []
                            tracks_gt_histories = []
                            for obj_feature in frame_feats.object_features:
                                tracks_histories.extend(obj_feature.track_history)
                                tracks_gt_histories.extend(obj_feature.gt_history)

                            # no need to reverse
                            # tracks_histories.reverse()
                            # tracks_gt_histories.reverse()

                            tracks_histories = np.array(tracks_histories)
                            tracks_gt_histories = np.array(tracks_gt_histories)

                            if tracks_histories.size == 0:
                                tracks_histories = np.zeros(shape=(0, 2))
                            if tracks_gt_histories.size == 0:
                                tracks_gt_histories = np.zeros(shape=(0, 2))
                        else:
                            tracks_histories = np.zeros(shape=(0, 2))
                            tracks_gt_histories = np.zeros(shape=(0, 2))

                        # for gt_annotation_box in annotations[:, :-1]:
                        #     ground_truth_track_histories.append(get_bbox_center(gt_annotation_box).flatten())
                        # ground_truth_track_histories = np.array(ground_truth_track_histories)

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[a_boxes_idx.tolist(), r_boxes_idx],
                            generated_track_histories=tracks_histories,
                            gt_track_histories=tracks_gt_histories,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)

                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=36)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=201)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=212)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=3)
                    else:
                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False, original_dims=original_dims, zero_shot=True)

                    # : save stuff and reiterate - moved up
                    # accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                    #                                                                 object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                        np.array(l2_distance_hungarian_fn_list).sum()
                    batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    logger.info(f'Video: {VIDEO_LABEL.value} - {VIDEO_NUMBER} | Batch: {part_idx} | '
                                f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
            if save_every_n_batch_itr is not None:
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0 and part_idx > part_idx_yet:
                    save_dict = {'frame_number': frame_number,
                                 'part_idx': part_idx,
                                 'current_track_idx': current_track_idx,
                                 'second_last_frame': second_last_frame,
                                 'last_frame': last_frame,
                                 'last_frame_mask': last_frame_mask,
                                 'last_frame_live_tracks': last_frame_live_tracks,
                                 'running_tracks': running_tracks,
                                 'track_ids_used': track_ids_used,
                                 'new_track_boxes': new_track_boxes,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'tp_list': tp_list,
                                 'fp_list': fp_list,
                                 'fn_list': fn_list,
                                 'meter_tp_list': meter_tp_list,
                                 'meter_fp_list': meter_fp_list,
                                 'meter_fn_list': meter_fn_list,
                                 'center_inside_tp_list': center_inside_tp_list,
                                 'center_inside_fp_list': center_inside_fp_list,
                                 'center_inside_fn_list': center_inside_fn_list,
                                 'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                 'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                 'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                 'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                 'track_based_accumulated_features': track_based_accumulated_features,
                                 'accumulated_features': accumulated_features}
                    Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part_{part_idx}.pt'
                    torch.save(save_dict, features_save_path + 'parts/' + f_n)
                    logger.info(f"Saved at {features_save_path + 'parts/' + f_n}")

                    accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_from_dict_except_entries(live_track_ids,
                                                                                       track_based_accumulated_features)
                    logger.info(f'Saved part {part_idx} at frame {frame_number}')
            # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)
            # if video_mode:
            #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'current_track_idx': current_track_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
        # f_n = f'accumulated_features_from_finally.pt'
        f_n = f'features_dict_part_{part_idx}.pt'
        torch.save(premature_save_dict, features_save_path + 'parts/' + f_n)
        logger.info(f"Saved at {features_save_path + 'parts/' + f_n}")

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


def preprocess_data_zero_shot_resumable(
        last_saved_features_dict,
        save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
        radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
        plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
        save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
        use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
        use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
        distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
        detect_shadows=True, allow_only_for_first_frame_use_is_box_overlapping_live_boxes=True,
        filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
        angle_threshold_to_filter=120):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'

    part_idx_yet = last_saved_features_dict['part_idx']
    frame_number_yet = last_saved_features_dict['frame_number'].item()
    last_frame, second_last_frame = last_saved_features_dict['last_frame'], \
                                    last_saved_features_dict['second_last_frame']
    last_frame_live_tracks, last_frame_mask = last_saved_features_dict['last_frame_live_tracks'], \
                                              last_saved_features_dict['last_frame_mask']
    running_tracks = last_saved_features_dict['running_tracks']
    current_track_idx, track_ids_used = last_saved_features_dict['current_track_idx'] \
                                            if 'current_track_idx' in last_saved_features_dict.keys() else 0, \
                                        last_saved_features_dict['track_ids_used']
    new_track_boxes = last_saved_features_dict['new_track_boxes']
    precision_list, recall_list, matching_boxes_with_iou_list = last_saved_features_dict['precision'], \
                                                                last_saved_features_dict['recall'], \
                                                                last_saved_features_dict['matching_boxes_with_iou_list']
    tp_list, fp_list, fn_list = last_saved_features_dict['tp_list'], \
                                last_saved_features_dict['fp_list'], \
                                last_saved_features_dict['fn_list']
    meter_tp_list, meter_fp_list, meter_fn_list = last_saved_features_dict['meter_tp_list'], \
                                                  last_saved_features_dict['meter_fp_list'], \
                                                  last_saved_features_dict['meter_fn_list']
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = \
        last_saved_features_dict['l2_distance_hungarian_tp_list'], \
        last_saved_features_dict['l2_distance_hungarian_fp_list'], \
        last_saved_features_dict['l2_distance_hungarian_fn_list']
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = \
        last_saved_features_dict['center_inside_tp_list'], \
        last_saved_features_dict['center_inside_fp_list'], \
        last_saved_features_dict['center_inside_fn_list']
    selected_track_distances = []
    accumulated_features = {}
    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
    track_based_accumulated_features: Dict[int, TrackFeatures] = remove_from_dict_except_entries(
        live_track_ids, last_saved_features_dict['track_based_accumulated_features'])
    last_frame_gt_tracks = {}
    ground_truth_track_histories = []

    del last_saved_features_dict

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))

            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if part_idx <= part_idx_yet and frame_number <= frame_number_yet:
                    continue
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    annotations, bbox_centers = scale_annotations(frame_annotation,
                                                                  original_scale=original_shape,
                                                                  new_scale=new_shape, return_track_id=False,
                                                                  tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        current_track_features = track_based_accumulated_features[
                            current_track_idx].object_features[-1] \
                            if current_track_idx in track_based_accumulated_features.keys() else []
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        # if frame_number == 110:
                        #     print()

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # STEP 4c: 1> Switch boxes: Idea 1 - Only keep points inside box
                        shifted_xy = find_points_inside_box(shifted_xy, shifted_box)
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        if xy_current_frame.size == 0 or (filter_switch_boxes_based_on_angle_and_recent_history
                                                          and not isinstance(current_track_features, list)
                                                          and current_track_features.velocity_direction.size != 0
                                                          and len(current_track_features.velocity_direction) >
                                                          min_track_length_to_filter_switch_box
                                                          and first_violation_till_now(
                                    current_track_features.velocity_direction, angle_threshold_to_filter)
                                                          and current_track_features.velocity_direction[-1] >
                                                          angle_threshold_to_filter
                                                          and not np.isnan(
                                    current_track_features.velocity_direction[-1])):

                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
                                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                    current_track_history = get_track_history(current_track_idx,
                                                                              track_based_accumulated_features)
                                    current_track_velocity_history = get_track_velocity_history(
                                        current_track_idx, track_based_accumulated_features)
                                    current_track_velocity_history = np.array(current_track_velocity_history)
                                    current_direction = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        current_direction.append((angle_between(
                                            v1=current_track_history[track_history_idx],
                                            v2=current_track_history[track_history_idx + 1]
                                        )))
                                    # current direction can be removed
                                    current_direction = np.array(current_direction)

                                    current_velocity_direction = []
                                    for track_history_idx in range(len(current_track_velocity_history) - 1):
                                        current_velocity_direction.append(math.degrees(angle_between(
                                            v1=current_track_velocity_history[track_history_idx],
                                            v2=current_track_velocity_history[track_history_idx + 1]
                                        )))
                                    current_velocity_direction = np.array(current_velocity_direction)

                                    # not really required ############################################################
                                    # if len(current_track_history) != 0:
                                    #     current_running_velocity = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[-1], axis=0) -
                                    #         np.expand_dims(current_track_history[0], axis=0),
                                    #         2, axis=0
                                    #     ) / len(current_track_history) / 30
                                    # else:
                                    #     current_running_velocity = None
                                    #
                                    # current_per_step_distance = []
                                    # for track_history_idx in range(len(current_track_history) - 1):
                                    #     d = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                    #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                                    #         2, axis=0
                                    #     )
                                    #     current_per_step_distance.append(d)
                                    #
                                    # current_per_step_distance = np.array(current_per_step_distance)
                                    ###################################################################################

                                    # track_sign = []
                                    # for t in range(
                                    #         len(track_based_accumulated_features[8].object_features[-1].track_history)
                                    #         - 1):
                                    #     track_sign.append(np.sign(
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t + 1] -
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t]))

                                    # just use gt
                                    current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                    track_based_accumulated_features)
                                else:
                                    current_track_history, current_gt_track_history = None, None
                                    current_direction, current_velocity_direction = None, None
                                    current_track_velocity_history = None
                                    # not really required ############################################################
                                    # current_per_step_distance = None
                                    # current_running_velocity = None
                                    ###################################################################################
                                current_track_obj_features = ObjectFeatures(idx=current_track_idx,
                                                                            history=current_track_history,
                                                                            gt_history=current_gt_track_history,
                                                                            track_direction=current_direction,
                                                                            velocity_direction=
                                                                            current_velocity_direction,
                                                                            velocity_history=
                                                                            current_track_velocity_history,
                                                                            xy=xy_current_frame,
                                                                            past_xy=xy,
                                                                            final_xy=xy_current_frame,
                                                                            flow=xy_displacement,
                                                                            past_flow=past_xy_displacement,
                                                                            past_bbox=box,
                                                                            final_bbox=np.array(shifted_box),
                                                                            is_track_live=False,
                                                                            per_step_distance=None,
                                                                            running_velocity=None,
                                                                            frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)

                                continue

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?
                        final_shifted_box, final_shifted_xy_center = evaluate_shifted_bounding_box(shifted_box,
                                                                                                   final_features_xy,
                                                                                                   shifted_xy)

                        running_tracks.append(Track(bbox=final_shifted_box, idx=current_track_idx))

                        # if not (final_shifted_box == shifted_box).all():
                        #     logger.warn('Final Shifted Box differs from Shifted Box!')

                        if plot:
                            plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                  shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                  xy_cloud_current_frame=xy_current_frame,
                                                  frame_number=frame_number.item(),
                                                  track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                                  selected_current=closest_n_xy_current_frame_pair)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            current_track_history = get_track_history(current_track_idx,
                                                                      track_based_accumulated_features)
                            current_track_velocity_history = get_track_velocity_history(
                                current_track_idx, track_based_accumulated_features)
                            current_track_velocity_history = np.array(current_track_velocity_history)
                            current_direction = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                current_direction.append((angle_between(
                                    v1=current_track_history[track_history_idx],
                                    v2=current_track_history[track_history_idx + 1]
                                )))
                            # current direction can be removed
                            current_direction = np.array(current_direction)

                            current_velocity_direction = []
                            for track_history_idx in range(len(current_track_velocity_history) - 1):
                                current_velocity_direction.append(math.degrees(angle_between(
                                    v1=current_track_velocity_history[track_history_idx],
                                    v2=current_track_velocity_history[track_history_idx + 1]
                                )))
                            current_velocity_direction = np.array(current_velocity_direction)

                            current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                            track_based_accumulated_features)

                            # not really required ############################################################
                            # if len(current_track_history) != 0:
                            #     current_running_velocity = np.linalg.norm(
                            #         np.expand_dims(current_track_history[-1], axis=0) -
                            #         np.expand_dims(current_track_history[0], axis=0),
                            #         2, axis=0
                            #     ) / len(current_track_history) / 30
                            # else:
                            #     current_running_velocity = None
                            #
                            # current_per_step_distance = []
                            # for track_history_idx in range(len(current_track_history) - 1):
                            #     d = np.linalg.norm(
                            #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                            #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                            #         2, axis=0
                            #     )
                            #     current_per_step_distance.append(d)
                            #
                            # current_per_step_distance = np.array(current_per_step_distance)
                            ###################################################################################
                        else:
                            current_track_history, current_gt_track_history = None, None
                            current_direction, current_velocity_direction = None, None
                            current_track_velocity_history = None
                            # not really required ############################################################
                            # current_per_step_distance = None
                            # current_running_velocity = None
                            ###################################################################################
                        # STEP 4g: save the information gathered
                        current_track_obj_features = ObjectFeatures(idx=current_track_idx,
                                                                    history=current_track_history,
                                                                    gt_history=current_gt_track_history,
                                                                    track_direction=current_direction,
                                                                    velocity_direction=current_velocity_direction,
                                                                    velocity_history=current_track_velocity_history,
                                                                    xy=xy_current_frame,
                                                                    past_xy=xy,
                                                                    final_xy=final_features_xy,
                                                                    flow=xy_displacement,
                                                                    past_flow=past_xy_displacement,
                                                                    past_bbox=box,
                                                                    # final_bbox=np.array(shifted_box),
                                                                    final_bbox=np.array(final_shifted_box),
                                                                    per_step_distance=None,
                                                                    running_velocity=None,
                                                                    frame_number=frame_number.item())
                        object_features.append(current_track_obj_features)
                        if current_track_idx not in track_based_accumulated_features:
                            track_feats = TrackFeatures(current_track_idx)
                            track_feats.object_features.append(current_track_obj_features)
                            track_based_accumulated_features.update(
                                {current_track_idx: track_feats})
                        else:
                            track_based_accumulated_features[current_track_idx].object_features.append(
                                current_track_obj_features)
                        if current_track_idx not in track_ids_used:
                            track_ids_used.append(current_track_idx)

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    ratio = float(meta_info.flatten()[-1])

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(annotations[:, :-1])
                    a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    try:
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    except IndexError:
                        if a_boxes.ndim < 2:
                            a_boxes = a_boxes.unsqueeze(0)
                        if r_boxes.ndim < 2:
                            r_boxes = r_boxes.unsqueeze(0)
                        logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                        logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                        # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes = torch.randn((0)).numpy()

                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    if r_boxes_np.size != 0:
                        for a_i, a_box in enumerate(a_boxes_np):
                            for r_i, r_box in enumerate(r_boxes_np):
                                dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                       get_bbox_center(r_box).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]

                        gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                        for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                            gt_t_idx = match_rows_tracks_idx[m_c_idx]
                            # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                            track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                            track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                                gt_track_box_mapping[gt_t_idx]
                            try:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                                    last_frame_gt_tracks[gt_t_idx]
                                gt_distance = np.linalg.norm(
                                    (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                                     get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = gt_distance
                            except KeyError:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = [0, 0]

                        last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    # generated_to_gt_track_association check ###################################################
                    # filter_nones = True
                    # plot_gt_ids = []
                    # plot_gt_boxes = []
                    # plot_generated_ids = []
                    # plot_generated_boxes = []
                    #
                    # for k, v in track_based_accumulated_features.items():
                    #     plot_feats = v.object_features[-1]
                    #     if filter_nones and plot_feats.gt_box is not None:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #     if not filter_nones:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #
                    # plot_image_set_of_boxes(frame, plot_gt_boxes, plot_generated_boxes,
                    #                         annotate=[plot_gt_ids, plot_generated_ids])
                    #############################################################################################
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    if iou_boxes_threshold.size != 0:
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []

                        #  precision/recall
                        tp = 0
                        fp = 0
                        fn = len(a_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    bbox_distance_to_of_centers_iou_based = []
                    bbox_distance_to_of_centers_iou_based_idx = []
                    boxes_distance = []
                    # boxes_distance_for_metric = []
                    r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()

                    # TODO: Replace with Hungarian
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T

                    match_idx = np.where(iou_boxes)
                    matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)

                    for a_box_idx, r_box_idx in zip(*match_idx):
                        dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                               get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                        boxes_distance.append([(a_box_idx, r_box_idx), dist])
                        # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                        matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                        predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                            point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                        )
                        bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                      iou_boxes[a_box_idx, r_box_idx]])
                        bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                                                                          iou_boxes[a_box_idx, r_box_idx]])
                        if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                                if (b == r_boxes[r_box_idx]).all()][0]:
                            selected_track_distances.append(dist)

                    # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    if r_boxes_np.size != 0:
                        a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                            matched_boxes_l2_distance_matrix
                        )

                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = \
                            np.where(predicted_box_center_inside_gt_box_matrix)
                    else:
                        a_matched_boxes_l2_distance_matrix_idx, \
                        r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)

                        meter_precision = meter_tp / (meter_tp + meter_fp)
                        meter_recall = meter_tp / (meter_tp + meter_fn)

                        center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)

                        center_precision = center_tp / (center_tp + center_fp)
                        center_recall = center_tp / (center_tp + center_fn)

                        if len(match_rows) != len(match_cols):
                            logger.info('Matching arrays length not same!')
                        l2_distance_hungarian_tp = len(match_rows)
                        l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                        l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)

                        l2_distance_hungarian_precision = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                        l2_distance_hungarian_recall = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    else:
                        meter_tp = 0
                        meter_fp = 0
                        meter_fn = len(a_boxes)

                        meter_precision = 0
                        meter_recall = 0

                        center_tp = 0
                        center_fp = 0
                        center_fn = len(a_boxes)

                        center_precision = 0
                        center_recall = 0

                        l2_distance_hungarian_tp = 0
                        l2_distance_hungarian_fp = 0
                        l2_distance_hungarian_fn = len(a_boxes)

                        l2_distance_hungarian_precision = 0
                        l2_distance_hungarian_recall = 0

                    meter_tp_list.append(meter_tp)
                    meter_fp_list.append(meter_fp)
                    meter_fn_list.append(meter_fn)

                    center_inside_tp_list.append(center_tp)
                    center_inside_fp_list.append(center_fp)
                    center_inside_fn_list.append(center_fn)

                    l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    new_track_ids = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            annotations, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            # STEP 4h: b> cluster to group points
                            mean_shift, n_clusters = mean_shift_clustering(
                                features_skipped, bin_seeding=False, min_bin_freq=8,
                                cluster_all=True, bandwidth=4, max_iter=100)
                            cluster_centers = mean_shift.cluster_centers

                            # STEP 4h: c> prune cluster centers
                            # combine centers inside radius + eliminate noise
                            final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                cluster_centers, mean_shift, radius + extra_radius,
                                min_points_in_cluster=min_points_in_cluster)

                            if final_cluster_centers.size != 0:
                                t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                # STEP 4h: d> start new potential tracks
                                for cluster_center in final_cluster_centers:
                                    cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                    # t_id = max(track_ids_used) + 1
                                    # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                    t_box = torchvision.ops.box_convert(
                                        torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                        'cxcywh', 'xyxy').int().numpy()
                                    # Note: Do not start track if bbox is out of frame
                                    if use_is_box_overlapping_live_boxes:
                                        if not (np.sign(t_box) < 0).any() and \
                                                not is_box_overlapping_live_boxes(t_box,
                                                                                  [t.bbox for t in running_tracks]):
                                            # NOTE: the second check might result in killing potential tracks!
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                            new_track_ids.append(t_id)
                                    else:
                                        if not (np.sign(t_box) < 0).any():
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                            new_track_ids.append(t_id)

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers,
                                #     num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))
                    new_track_ids = np.stack(new_track_ids) if len(new_track_ids) > 0 else np.empty(shape=(0,))

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    if video_mode:
                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            frame_feats = accumulated_features[frame_number.item()]
                            tracks_histories = []
                            tracks_gt_histories = []
                            for obj_feature in frame_feats.object_features:
                                tracks_histories.extend(obj_feature.track_history)
                                tracks_gt_histories.extend(obj_feature.gt_history)

                            # no need to reverse
                            # tracks_histories.reverse()
                            # tracks_gt_histories.reverse()

                            tracks_histories = np.array(tracks_histories)
                            tracks_gt_histories = np.array(tracks_gt_histories)

                            if tracks_histories.size == 0:
                                tracks_histories = np.zeros(shape=(0, 2))
                            if tracks_gt_histories.size == 0:
                                tracks_gt_histories = np.zeros(shape=(0, 2))
                        else:
                            tracks_histories = np.zeros(shape=(0, 2))
                            tracks_gt_histories = np.zeros(shape=(0, 2))

                        # for gt_annotation_box in annotations[:, :-1]:
                        #     ground_truth_track_histories.append(get_bbox_center(gt_annotation_box).flatten())
                        # ground_truth_track_histories = np.array(ground_truth_track_histories)

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[a_boxes_idx.tolist(), r_boxes_idx],
                            generated_track_histories=tracks_histories,
                            gt_track_histories=tracks_gt_histories,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)

                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=36)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=201)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=212)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=3)
                    else:
                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False, original_dims=original_dims, zero_shot=True)

                    # : save stuff and reiterate - moved up
                    # accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                    #                                                                 object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                        np.array(l2_distance_hungarian_fn_list).sum()
                    batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    logger.info(f'Video: {VIDEO_LABEL.value} - {VIDEO_NUMBER} | Batch: {part_idx} | '
                                f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
            if save_every_n_batch_itr is not None:
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0 and part_idx > part_idx_yet:
                    save_dict = {'frame_number': frame_number,
                                 'part_idx': part_idx,
                                 'current_track_idx': current_track_idx,
                                 'second_last_frame': second_last_frame,
                                 'last_frame': last_frame,
                                 'last_frame_mask': last_frame_mask,
                                 'last_frame_live_tracks': last_frame_live_tracks,
                                 'running_tracks': running_tracks,
                                 'track_ids_used': track_ids_used,
                                 'new_track_boxes': new_track_boxes,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'tp_list': tp_list,
                                 'fp_list': fp_list,
                                 'fn_list': fn_list,
                                 'meter_tp_list': meter_tp_list,
                                 'meter_fp_list': meter_fp_list,
                                 'meter_fn_list': meter_fn_list,
                                 'center_inside_tp_list': center_inside_tp_list,
                                 'center_inside_fp_list': center_inside_fp_list,
                                 'center_inside_fn_list': center_inside_fn_list,
                                 'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                 'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                 'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                 'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                 'track_based_accumulated_features': track_based_accumulated_features,
                                 'accumulated_features': accumulated_features}
                    Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part_{part_idx}.pt'
                    torch.save(save_dict, features_save_path + 'parts/' + f_n)
                    logger.info(f"Saved at {features_save_path + 'parts/' + f_n}")

                    accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_from_dict_except_entries(live_track_ids,
                                                                                       track_based_accumulated_features)
                    logger.info(f'Saved part {part_idx} at frame {frame_number}')
            # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)
            # if video_mode:
            #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'current_track_idx': current_track_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
        # f_n = f'accumulated_features_from_finally.pt'
        f_n = f'features_dict_part_{part_idx}.pt'
        torch.save(premature_save_dict, features_save_path + 'parts/' + f_n)
        logger.info(f"Saved at {features_save_path + 'parts/' + f_n}")

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


def preprocess_data_zero_shot_custom_video(
        save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
        radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
        plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
        save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
        use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
        use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
        distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
        detect_shadows=True, allow_only_for_first_frame_use_is_box_overlapping_live_boxes=True,
        filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
        angle_threshold_to_filter=120, custom_video=None):
    video_dataset: SimpleVideoDatasetBase = custom_video['dataset'](custom_video['video_path'], custom_video['start'],
                                                                    custom_video['end'],
                                                                    custom_video['pts_unit'])
    data_loader = DataLoader(video_dataset, batch_size, drop_last=drop_last_batch)
    # df = video_dataset.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'
    first_frame_bounding_boxes, first_frame_mask, last_frame, second_last_frame = None, None, None, None
    first_frame_live_tracks, last_frame_live_tracks, last_frame_mask = None, None, None
    current_track_idx, track_ids_used = 0, []
    precision_list, recall_list, matching_boxes_with_iou_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []
    meter_tp_list, meter_fp_list, meter_fn_list = [], [], []
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = [], [], []
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = [], [], []
    selected_track_distances = []
    accumulated_features = {}
    track_based_accumulated_features: Dict[int, TrackFeatures] = {}
    last_frame_gt_tracks = {}
    ground_truth_track_histories = []

    out = None
    frames_shape = video_dataset.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    video_shape = list(video_shape)
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))  # (1200, 1000))  # (video_shape[0], video_shape[1]))
            # (video_shape[1], video_shape[0]))
            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))  # (1200, 1000))  # (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze().numpy().astype(np.uint8)
            # frames = (frames * 255.0).permute(0, 3, 1, 2).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if part_idx == 0 and frame_idx == 0:
                    # STEP 1: a> Get GT for the first frame
                    # validation_annotations, first_frame_mask = first_frame_processing_and_gt_association(
                    #     df, first_frame_mask,
                    #     frame_idx, frame_number,
                    #     frames, frames_count,
                    #     kernel, n, new_shape,
                    #     original_shape, step,
                    #     var_threshold, detect_shadows=detect_shadows)
                    first_frame_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                                time_gap_within_frames=3,
                                                                total_frames=frames_count, step=step, n=n,
                                                                kernel=kernel, var_threshold=var_threshold,
                                                                detect_shadows=detect_shadows)

                    running_tracks, new_track_boxes = [], []
                    all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                        None, first_frame_mask, radius, running_tracks, plot=False)

                    all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                    features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                    if features_skipped_idx.size != 0:
                        features_skipped = all_cloud[features_skipped_idx]

                        # STEP 4h: b> cluster to group points
                        mean_shift, n_clusters = mean_shift_clustering(
                            features_skipped, bin_seeding=False, min_bin_freq=8,
                            cluster_all=True, bandwidth=4, max_iter=100)
                        cluster_centers = mean_shift.cluster_centers

                        # STEP 4h: c> prune cluster centers
                        # combine centers inside radius + eliminate noise
                        final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                            cluster_centers, mean_shift, radius + extra_radius,
                            min_points_in_cluster=min_points_in_cluster)

                        if final_cluster_centers.size != 0:
                            t_w, t_h = generic_box_wh[0], generic_box_wh[1]  # 100, 100
                            # STEP 4h: d> start new potential tracks
                            for cluster_center, cluster_center_idx in \
                                    zip(final_cluster_centers, final_cluster_centers_idx):
                                cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)

                                # flexible_box, points_in_current_cluster = calculate_flexible_bounding_box(
                                #     cluster_center_idx, cluster_center_x, cluster_center_y, mean_shift)

                                # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                t_box = torchvision.ops.box_convert(
                                    torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                    'cxcywh', 'xyxy').int().numpy()
                                # Note: Do not start track if bbox is out of frame
                                if allow_only_for_first_frame_use_is_box_overlapping_live_boxes and \
                                        not (np.sign(t_box) < 0).any() and \
                                        not is_box_overlapping_live_boxes(t_box, [t.bbox for t in running_tracks]):
                                    # NOTE: the second check might result in killing potential tracks!
                                    t_id = max(track_ids_used) + 1 if len(track_ids_used) else 0
                                    running_tracks.append(Track(bbox=t_box, idx=t_id))
                                    track_ids_used.append(t_id)
                                    new_track_boxes.append(t_box)
                                elif use_is_box_overlapping_live_boxes:
                                    if not (np.sign(t_box) < 0).any() and \
                                            not is_box_overlapping_live_boxes(t_box,
                                                                              [t.bbox for t in running_tracks]):
                                        # NOTE: the second check might result in killing potential tracks!
                                        t_id = max(track_ids_used) + 1
                                        running_tracks.append(Track(bbox=t_box, idx=t_id))
                                        track_ids_used.append(t_id)
                                        new_track_boxes.append(t_box)
                                else:
                                    if not (np.sign(t_box) < 0).any():
                                        t_id = max(track_ids_used) + 1
                                        running_tracks.append(Track(bbox=t_box, idx=t_id))
                                        track_ids_used.append(t_id)
                                        new_track_boxes.append(t_box)

                            # plot_features_with_circles(
                            #     all_cloud, features_covered, features_skipped, first_frame_mask, marker_size=8,
                            #     cluster_centers=final_cluster_centers, num_clusters=final_cluster_centers.shape[0],
                            #     frame_number=frame_number, boxes=validation_annotations[:, :-1],
                            #     radius=radius+extra_radius,
                            #     additional_text=
                            #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                            #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))

                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    # a_boxes = torch.from_numpy(validation_annotations[:, :-1])
                    # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    # iou_boxes_threshold = iou_boxes.copy()
                    # iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    # iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    # iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    # match_idx = np.where(iou_boxes)
                    # a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                    # matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                    #                            for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]

                    #  precision/recall
                    # tp = len(a_match_idx_threshold)
                    # fp = len(r_boxes) - len(a_match_idx_threshold)
                    # fn = len(a_boxes) - len(a_match_idx_threshold)
                    #
                    # tp_list.append(tp)
                    # fp_list.append(fp)
                    # fn_list.append(fn)
                    #
                    # precision = tp / (tp + fp)
                    # recall = tp / (tp + fn)
                    # precision_list.append(precision)
                    # recall_list.append(recall)
                    # matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    last_frame_live_tracks = []

                    fig = plot_for_video(
                        gt_rgb=frame, gt_mask=first_frame_mask, last_frame_rgb=frame,
                        last_frame_mask=first_frame_mask, current_frame_rgb=frame,
                        current_frame_mask=first_frame_mask, gt_annotations=[],
                        last_frame_annotation=last_frame_live_tracks,
                        current_frame_annotation=[t.bbox for t in running_tracks],
                        new_track_annotation=new_track_boxes,
                        frame_number=frame_number,
                        additional_text=
                        f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        f'Track Ids Killed: '
                        f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                        video_mode=False,
                        save_path=f'{plot_save_path}zero_shot/plots{min_points_in_cluster}/')

                    # STEP 1: b> Store them for the next ts and update these variables in further iterations
                    last_frame = frame.copy()
                    second_last_frame = last_frame.copy()
                    last_frame_live_tracks = running_tracks
                    last_frame_mask = first_frame_mask.copy()
                    # last_frame_gt_tracks = {a[-1]: a[:-1] for a in validation_annotations}
                    last_frame_gt_tracks = {}
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    # frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    # annotations, bbox_centers = scale_annotations(frame_annotation,
                    #                                               original_scale=original_shape,
                    #                                               new_scale=new_shape, return_track_id=False,
                    #                                               tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        current_track_features = track_based_accumulated_features[
                            current_track_idx].object_features[-1] \
                            if current_track_idx in track_based_accumulated_features.keys() else []
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        # if frame_number == 110:
                        #     print()

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # STEP 4c: 1> Switch boxes: Idea 1 - Only keep points inside box
                        shifted_xy = find_points_inside_box(shifted_xy, shifted_box)
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        if xy_current_frame.size == 0 or (filter_switch_boxes_based_on_angle_and_recent_history
                                                          and not isinstance(current_track_features, list)
                                                          and current_track_features.velocity_direction.size != 0
                                                          and len(current_track_features.velocity_direction) >
                                                          min_track_length_to_filter_switch_box
                                                          and first_violation_till_now(
                                    current_track_features.velocity_direction, angle_threshold_to_filter)
                                                          and current_track_features.velocity_direction[-1] >
                                                          angle_threshold_to_filter
                                                          and not np.isnan(
                                    current_track_features.velocity_direction[-1])):

                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
                                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                    current_track_history = get_track_history(current_track_idx,
                                                                              track_based_accumulated_features)
                                    current_track_velocity_history = get_track_velocity_history(
                                        current_track_idx, track_based_accumulated_features)
                                    current_track_velocity_history = np.array(current_track_velocity_history)
                                    current_direction = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        current_direction.append((angle_between(
                                            v1=current_track_history[track_history_idx],
                                            v2=current_track_history[track_history_idx + 1]
                                        )))
                                    # current direction can be removed
                                    current_direction = np.array(current_direction)

                                    current_velocity_direction = []
                                    for track_history_idx in range(len(current_track_velocity_history) - 1):
                                        current_velocity_direction.append(math.degrees(angle_between(
                                            v1=current_track_velocity_history[track_history_idx],
                                            v2=current_track_velocity_history[track_history_idx + 1]
                                        )))
                                    current_velocity_direction = np.array(current_velocity_direction)

                                    # not really required ############################################################
                                    if len(current_track_history) != 0:
                                        current_running_velocity = np.linalg.norm(
                                            np.expand_dims(current_track_history[-1], axis=0) -
                                            np.expand_dims(current_track_history[0], axis=0),
                                            2, axis=0
                                        ) / len(current_track_history) / 30
                                    else:
                                        current_running_velocity = None

                                    current_per_step_distance = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        d = np.linalg.norm(
                                            np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                            np.expand_dims(current_track_history[track_history_idx], axis=0),
                                            2, axis=0
                                        )
                                        current_per_step_distance.append(d)

                                    current_per_step_distance = np.array(current_per_step_distance)
                                    ###################################################################################

                                    # track_sign = []
                                    # for t in range(
                                    #         len(track_based_accumulated_features[8].object_features[-1].track_history)
                                    #         - 1):
                                    #     track_sign.append(np.sign(
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t + 1] -
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t]))

                                    # just use gt
                                    current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                    track_based_accumulated_features)
                                else:
                                    current_track_history, current_gt_track_history = None, None
                                    current_direction, current_velocity_direction = None, None
                                    current_track_velocity_history = None
                                    # not really required ############################################################
                                    current_per_step_distance = None
                                    current_running_velocity = None
                                    ###################################################################################
                                current_track_obj_features = ObjectFeatures(idx=current_track_idx,
                                                                            history=current_track_history,
                                                                            gt_history=current_gt_track_history,
                                                                            track_direction=current_direction,
                                                                            velocity_direction=
                                                                            current_velocity_direction,
                                                                            velocity_history=
                                                                            current_track_velocity_history,
                                                                            xy=xy_current_frame,
                                                                            past_xy=xy,
                                                                            final_xy=xy_current_frame,
                                                                            flow=xy_displacement,
                                                                            past_flow=past_xy_displacement,
                                                                            past_bbox=box,
                                                                            final_bbox=np.array(shifted_box),
                                                                            is_track_live=False,
                                                                            per_step_distance=current_per_step_distance,
                                                                            running_velocity=current_running_velocity,
                                                                            frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)

                                continue

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?
                        final_shifted_box, final_shifted_xy_center = evaluate_shifted_bounding_box(shifted_box,
                                                                                                   final_features_xy,
                                                                                                   shifted_xy)

                        running_tracks.append(Track(bbox=final_shifted_box, idx=current_track_idx))

                        # if not (final_shifted_box == shifted_box).all():
                        #     logger.warn('Final Shifted Box differs from Shifted Box!')

                        if plot:
                            plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                  shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                  xy_cloud_current_frame=xy_current_frame,
                                                  frame_number=frame_number.item(),
                                                  track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                                  selected_current=closest_n_xy_current_frame_pair)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            current_track_history = get_track_history(current_track_idx,
                                                                      track_based_accumulated_features)
                            current_track_velocity_history = get_track_velocity_history(
                                current_track_idx, track_based_accumulated_features)
                            current_track_velocity_history = np.array(current_track_velocity_history)
                            current_direction = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                current_direction.append((angle_between(
                                    v1=current_track_history[track_history_idx],
                                    v2=current_track_history[track_history_idx + 1]
                                )))
                            # current direction can be removed
                            current_direction = np.array(current_direction)

                            current_velocity_direction = []
                            for track_history_idx in range(len(current_track_velocity_history) - 1):
                                current_velocity_direction.append(math.degrees(angle_between(
                                    v1=current_track_velocity_history[track_history_idx],
                                    v2=current_track_velocity_history[track_history_idx + 1]
                                )))
                            current_velocity_direction = np.array(current_velocity_direction)

                            current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                            track_based_accumulated_features)

                            # not really required ############################################################
                            if len(current_track_history) != 0:
                                current_running_velocity = np.linalg.norm(
                                    np.expand_dims(current_track_history[-1], axis=0) -
                                    np.expand_dims(current_track_history[0], axis=0),
                                    2, axis=0
                                ) / len(current_track_history) / 30
                            else:
                                current_running_velocity = None

                            current_per_step_distance = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                d = np.linalg.norm(
                                    np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                    np.expand_dims(current_track_history[track_history_idx], axis=0),
                                    2, axis=0
                                )
                                current_per_step_distance.append(d)

                            current_per_step_distance = np.array(current_per_step_distance)
                            ###################################################################################
                        else:
                            current_track_history, current_gt_track_history = None, None
                            current_direction, current_velocity_direction = None, None
                            current_track_velocity_history = None
                            # not really required ############################################################
                            current_per_step_distance = None
                            current_running_velocity = None
                            ###################################################################################
                        # STEP 4g: save the information gathered
                        current_track_obj_features = ObjectFeatures(idx=current_track_idx,
                                                                    history=current_track_history,
                                                                    gt_history=current_gt_track_history,
                                                                    track_direction=current_direction,
                                                                    velocity_direction=current_velocity_direction,
                                                                    velocity_history=current_track_velocity_history,
                                                                    xy=xy_current_frame,
                                                                    past_xy=xy,
                                                                    final_xy=final_features_xy,
                                                                    flow=xy_displacement,
                                                                    past_flow=past_xy_displacement,
                                                                    past_bbox=box,
                                                                    # final_bbox=np.array(shifted_box),
                                                                    final_bbox=np.array(final_shifted_box),
                                                                    per_step_distance=current_per_step_distance,
                                                                    running_velocity=current_running_velocity,
                                                                    frame_number=frame_number.item())
                        object_features.append(current_track_obj_features)
                        if current_track_idx not in track_based_accumulated_features:
                            track_feats = TrackFeatures(current_track_idx)
                            track_feats.object_features.append(current_track_obj_features)
                            track_based_accumulated_features.update(
                                {current_track_idx: track_feats})
                        else:
                            track_based_accumulated_features[current_track_idx].object_features.append(
                                current_track_obj_features)
                        if current_track_idx not in track_ids_used:
                            track_ids_used.append(current_track_idx)

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    ratio = float(meta_info.flatten()[-1])

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    # a_boxes = torch.from_numpy(annotations[:, :-1])
                    # a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    # try:
                    #     iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    # except IndexError:
                    #     if a_boxes.ndim < 2:
                    #         a_boxes = a_boxes.unsqueeze(0)
                    #     if r_boxes.ndim < 2:
                    #         r_boxes = r_boxes.unsqueeze(0)
                    #     logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                    #     logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                    #     # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    #     iou_boxes = torch.randn((0)).numpy()

                    # iou_boxes_threshold = iou_boxes.copy()
                    # iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    # a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    r_boxes_np = r_boxes.numpy()
                    # l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    # if r_boxes_np.size != 0:
                    #     for a_i, a_box in enumerate(a_boxes_np):
                    #         for r_i, r_box in enumerate(r_boxes_np):
                    #             dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                    #                                    get_bbox_center(r_box).flatten()), 2) * ratio
                    #             l2_distance_boxes_score_matrix[a_i, r_i] = dist
                    #
                    #     l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                    #     l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                    #     # Hungarian
                    #     # match_rows, match_cols = scipy.optimize.linear_sum_assignment(
                    #     -l2_distance_boxes_score_matrix)
                    #     match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                    #     actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                    #     match_rows = match_rows[actually_matched_mask]
                    #     match_cols = match_cols[actually_matched_mask]
                    #     match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                    #     match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]
                    #
                    #     gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                    #     for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                    #         gt_t_idx = match_rows_tracks_idx[m_c_idx]
                    #         # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                    #         track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                    #         track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                    #             gt_track_box_mapping[gt_t_idx]
                    #         try:
                    #             track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                    #                 last_frame_gt_tracks[gt_t_idx]
                    #             gt_distance = np.linalg.norm(
                    #                 (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                    #                  get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                    #             track_based_accumulated_features[matched_c].object_features[-1]. \
                    #                 gt_past_current_distance = gt_distance
                    #         except KeyError:
                    #             track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                    #             track_based_accumulated_features[matched_c].object_features[-1]. \
                    #                 gt_past_current_distance = [0, 0]
                    #
                    #     last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)
                    #
                    #     matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                    #                               for i, j in zip(match_rows, match_cols)]
                    # else:
                    #     match_rows, match_cols = np.array([]), np.array([])
                    #     match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    # generated_to_gt_track_association check ###################################################
                    # filter_nones = True
                    # plot_gt_ids = []
                    # plot_gt_boxes = []
                    # plot_generated_ids = []
                    # plot_generated_boxes = []
                    #
                    # for k, v in track_based_accumulated_features.items():
                    #     plot_feats = v.object_features[-1]
                    #     if filter_nones and plot_feats.gt_box is not None:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #     if not filter_nones:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #
                    # plot_image_set_of_boxes(frame, plot_gt_boxes, plot_generated_boxes,
                    #                         annotate=[plot_gt_ids, plot_generated_ids])
                    #############################################################################################
                    # iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    # iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    # match_idx = np.where(iou_boxes)
                    # if iou_boxes_threshold.size != 0:
                    #     a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                    #     matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                    #                                for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                    #     #  precision/recall
                    #     tp = len(a_match_idx_threshold)
                    #     fp = len(r_boxes) - len(a_match_idx_threshold)
                    #     fn = len(a_boxes) - len(a_match_idx_threshold)
                    #
                    #     precision = tp / (tp + fp)
                    #     recall = tp / (tp + fn)
                    # else:
                    #     a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []
                    #
                    #     #  precision/recall
                    #     tp = 0
                    #     fp = 0
                    #     fn = len(a_boxes)
                    #
                    #     precision = 0
                    #     recall = 0
                    #
                    # tp_list.append(tp)
                    # fp_list.append(fp)
                    # fn_list.append(fn)
                    #
                    # precision_list.append(precision)
                    # recall_list.append(recall)
                    # matching_boxes_with_iou_list.append(matching_boxes_with_iou)
                    #
                    # bbox_distance_to_of_centers_iou_based = []
                    # bbox_distance_to_of_centers_iou_based_idx = []
                    # boxes_distance = []
                    # # boxes_distance_for_metric = []
                    # r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()
                    #
                    # # TODO: Replace with Hungarian
                    # iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    # iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T
                    #
                    # match_idx = np.where(iou_boxes)
                    # matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    # predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)
                    #
                    # for a_box_idx, r_box_idx in zip(*match_idx):
                    #     dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                    #                            get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                    #     boxes_distance.append([(a_box_idx, r_box_idx), dist])
                    #     # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                    #     matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                    #     predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                    #         point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                    #     )
                    #     bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                    #                                                   iou_boxes[a_box_idx, r_box_idx]])
                    #     bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                    #                                                       iou_boxes[a_box_idx, r_box_idx]])
                    #     if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                    #                             if (b == r_boxes[r_box_idx]).all()][0]:
                    #         selected_track_distances.append(dist)
                    #
                    # # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    # matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    # if r_boxes_np.size != 0:
                    #     a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                    #         matched_boxes_l2_distance_matrix
                    #     )
                    #
                    #     a_predicted_box_center_inside_gt_box_matrix_idx, \
                    #     r_predicted_box_center_inside_gt_box_matrix_idx = \
                    #         np.where(predicted_box_center_inside_gt_box_matrix)
                    # else:
                    #     a_matched_boxes_l2_distance_matrix_idx, \
                    #     r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                    #     a_predicted_box_center_inside_gt_box_matrix_idx, \
                    #     r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])
                    #
                    # if len(match_rows) != 0:
                    #     meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                    #     meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                    #     meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                    #
                    #     meter_precision = meter_tp / (meter_tp + meter_fp)
                    #     meter_recall = meter_tp / (meter_tp + meter_fn)
                    #
                    #     center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                    #     center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                    #     center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                    #
                    #     center_precision = center_tp / (center_tp + center_fp)
                    #     center_recall = center_tp / (center_tp + center_fn)
                    #
                    #     if len(match_rows) != len(match_cols):
                    #         logger.info('Matching arrays length not same!')
                    #     l2_distance_hungarian_tp = len(match_rows)
                    #     l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                    #     l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)
                    #
                    #     l2_distance_hungarian_precision = \
                    #         l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                    #     l2_distance_hungarian_recall = \
                    #         l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    # else:
                    #     meter_tp = 0
                    #     meter_fp = 0
                    #     meter_fn = len(a_boxes)
                    #
                    #     meter_precision = 0
                    #     meter_recall = 0
                    #
                    #     center_tp = 0
                    #     center_fp = 0
                    #     center_fn = len(a_boxes)
                    #
                    #     center_precision = 0
                    #     center_recall = 0
                    #
                    #     l2_distance_hungarian_tp = 0
                    #     l2_distance_hungarian_fp = 0
                    #     l2_distance_hungarian_fn = len(a_boxes)
                    #
                    #     l2_distance_hungarian_precision = 0
                    #     l2_distance_hungarian_recall = 0
                    #
                    # meter_tp_list.append(meter_tp)
                    # meter_fp_list.append(meter_fp)
                    # meter_fn_list.append(meter_fn)
                    #
                    # center_inside_tp_list.append(center_tp)
                    # center_inside_fp_list.append(center_fp)
                    # center_inside_fn_list.append(center_fn)
                    #
                    # l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    # l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    # l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            None, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            # STEP 4h: b> cluster to group points
                            mean_shift, n_clusters = mean_shift_clustering(
                                features_skipped, bin_seeding=False, min_bin_freq=8,
                                cluster_all=True, bandwidth=4, max_iter=100)
                            cluster_centers = mean_shift.cluster_centers

                            # STEP 4h: c> prune cluster centers
                            # combine centers inside radius + eliminate noise
                            final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                cluster_centers, mean_shift, radius + extra_radius,
                                min_points_in_cluster=min_points_in_cluster)

                            if final_cluster_centers.size != 0:
                                t_w, t_h = generic_box_wh[0], generic_box_wh[1]  # 100, 100
                                # STEP 4h: d> start new potential tracks
                                for cluster_center in final_cluster_centers:
                                    cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                    # t_id = max(track_ids_used) + 1
                                    # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                    t_box = torchvision.ops.box_convert(
                                        torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                        'cxcywh', 'xyxy').int().numpy()
                                    # Note: Do not start track if bbox is out of frame
                                    if use_is_box_overlapping_live_boxes:
                                        if not (np.sign(t_box) < 0).any() and \
                                                not is_box_overlapping_live_boxes(t_box,
                                                                                  [t.bbox for t in running_tracks]):
                                            # NOTE: the second check might result in killing potential tracks!
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                    else:
                                        if not (np.sign(t_box) < 0).any():
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers,
                                #     num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    if video_mode:
                        # fig = plot_for_video(
                        #     gt_rgb=frame, gt_mask=fg_mask, last_frame_rgb=last_frame,
                        #     last_frame_mask=last_frame_mask, current_frame_rgb=frame,
                        #     current_frame_mask=fg_mask, gt_annotations=annotations[:, :-1],
                        #     last_frame_annotation=[t.bbox for t in last_frame_live_tracks],
                        #     current_frame_annotation=[t.bbox for t in running_tracks],
                        #     new_track_annotation=new_track_boxes,
                        #     frame_number=frame_number,
                        #     additional_text=
                        #     # f'Track Ids Used: {track_ids_used}\n'
                        #     f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        #     f'Track Ids Killed: '
                        #     f'{np.setdiff1d([t.idx for t in last_frame_live_tracks],
                        #     [t.idx for t in running_tracks])}',
                        #     video_mode=video_mode)

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            frame_feats = accumulated_features[frame_number.item()]
                            tracks_histories = []
                            tracks_gt_histories = []
                            for obj_feature in frame_feats.object_features:
                                tracks_histories.extend(obj_feature.track_history)
                                tracks_gt_histories.extend(obj_feature.gt_history)

                            # no need to reverse
                            # tracks_histories.reverse()
                            # tracks_gt_histories.reverse()

                            tracks_histories = np.array(tracks_histories)
                            tracks_gt_histories = np.array(tracks_gt_histories)

                            if tracks_histories.size == 0:
                                tracks_histories = np.zeros(shape=(0, 2))
                            if tracks_gt_histories.size == 0:
                                tracks_gt_histories = np.zeros(shape=(0, 2))
                        else:
                            tracks_histories = np.zeros(shape=(0, 2))
                            tracks_gt_histories = np.zeros(shape=(0, 2))

                        # for gt_annotation_box in annotations[:, :-1]:
                        #     ground_truth_track_histories.append(get_bbox_center(gt_annotation_box).flatten())
                        # ground_truth_track_histories = np.array(ground_truth_track_histories)

                        fig = plot_for_video_current_frame_single(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=[],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[[], r_boxes_idx],
                            generated_track_histories=tracks_histories,
                            gt_track_histories=tracks_gt_histories,
                            additional_text=
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)

                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=36)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=201)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=212)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=3)
                    else:
                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=[],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False, original_dims=original_dims, zero_shot=True)
                        # fig = plot_for_video(
                        #     gt_rgb=frame, gt_mask=fg_mask, last_frame_rgb=last_frame,
                        #     last_frame_mask=last_frame_mask, current_frame_rgb=frame,
                        #     current_frame_mask=fg_mask, gt_annotations=annotations[:, :-1],
                        #     last_frame_annotation=[t.bbox for t in last_frame_live_tracks],
                        #     current_frame_annotation=[t.bbox for t in running_tracks],
                        #     new_track_annotation=new_track_boxes,
                        #     frame_number=frame_number,
                        #     additional_text=
                        #     f'Precision: {precision} | Recall: {recall}\n'
                        #     f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                        #     f'Recall: {l2_distance_hungarian_recall}\n'
                        #     f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                        #     f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        #     f'Track Ids Killed: '
                        #     f'{np.setdiff1d([t.idx for t in last_frame_live_tracks],
                        #     [t.idx for t in running_tracks])}',
                        #     video_mode=False,
                        #     save_path=f'{plot_save_path}zero_shot/plots{min_points_in_cluster}/')

                    # : save stuff and reiterate - moved up
                    # accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                    #                                                                 object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    # batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                    #     np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                    #     np.array(l2_distance_hungarian_fn_list).sum()
                    # batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    # batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    # logger.info(f'Batch: {part_idx}, '
                    #             f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
            if save_every_n_batch_itr is not None:
                save_dict = {'frame_number': frame_number,
                             'part_idx': part_idx,
                             'second_last_frame': second_last_frame,
                             'last_frame': last_frame,
                             'last_frame_mask': last_frame_mask,
                             'last_frame_live_tracks': last_frame_live_tracks,
                             'running_tracks': running_tracks,
                             'track_ids_used': track_ids_used,
                             'new_track_boxes': new_track_boxes,
                             'precision': precision_list,
                             'recall': recall_list,
                             'tp_list': tp_list,
                             'fp_list': fp_list,
                             'fn_list': fn_list,
                             'meter_tp_list': meter_tp_list,
                             'meter_fp_list': meter_fp_list,
                             'meter_fn_list': meter_fn_list,
                             'center_inside_tp_list': center_inside_tp_list,
                             'center_inside_fp_list': center_inside_fp_list,
                             'center_inside_fn_list': center_inside_fn_list,
                             'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                             'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                             'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                             'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                             'track_based_accumulated_features': track_based_accumulated_features,
                             'accumulated_features': accumulated_features}
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0:
                    Path(video_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part{part_idx}.pt'
                    torch.save(save_dict, video_save_path + 'parts/' + f_n)

                    accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_entries_from_dict(live_track_ids,
                                                                                track_based_accumulated_features)
            # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)
            # if video_mode:
            #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path).mkdir(parents=True, exist_ok=True)
        f_n = f'accumulated_features_from_finally.pt'
        torch.save(premature_save_dict, features_save_path + f_n)

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


def twelve_frames_feature_extraction_zero_shot(frames, n, frames_to_build_model, extracted_features, var_threshold=None,
                                               time_gap_within_frames=3, frame_numbers=None, remaining_frames=None,
                                               remaining_frames_idx=None, past_12_frames_optical_flow=None,
                                               last_frame_from_last_used_batch=None, last12_bg_sub_mask=None,
                                               resume_mode=False, detect_shadows=True, overlap_percent=0.4,
                                               track_based_accumulated_features=None, frame_time_gap=12,
                                               save_path_for_plot=None, df=None, ratio=None,
                                               filter_switch_boxes_based_on_angle_and_recent_history=True,
                                               compute_histories_for_plot=True, video_mode=False,
                                               min_track_length_to_filter_switch_box=20, save_path_for_video=None,
                                               angle_threshold_to_filter=120, original_dims=None
                                               ):
    current_batch_figures, frame_track_ids = [], []
    interest_fr = None
    actual_interest_fr = None

    # cat old frames
    if remaining_frames is not None:
        frames = np.concatenate((remaining_frames, frames), axis=0)
        frame_numbers = torch.cat((remaining_frames_idx, frame_numbers))

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    step = 0
    if n is not None:
        step = n // 2
    else:
        n = frames_to_build_model
    total_frames = frames.shape[0]

    data_all_frames = {}

    for fr, actual_fr in tqdm(zip(range(frames.shape[0]), frame_numbers), total=frames.shape[0]):
        interest_fr = fr % total_frames
        actual_interest_fr = actual_fr

        of_interest_fr = (fr + frames_to_build_model) % total_frames
        actual_of_interest_fr = (actual_fr + frames_to_build_model)

        # do not go in circle for flow estimation
        if of_interest_fr < interest_fr:
            break

        # start at 12th frame and then only consider last 12 frames for velocity estimation
        if actual_interest_fr != 0:
            if interest_fr == 0:
                previous = cv.cvtColor(last_frame_from_last_used_batch.astype(np.uint8), cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor(frames[interest_fr], cv.COLOR_BGR2GRAY)

                past_flow_per_frame, past_rgb, past_mag, past_ang = FeatureExtractor.get_optical_flow(
                    previous_frame=previous,
                    next_frame=next_frame,
                    all_results_out=True)
                past_12_frames_optical_flow.update(
                    {f'{actual_interest_fr.item() - 1}-{actual_interest_fr.item()}': past_flow_per_frame})
            else:
                previous = cv.cvtColor(frames[interest_fr - 1], cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor(frames[interest_fr], cv.COLOR_BGR2GRAY)

                past_flow_per_frame, past_rgb, past_mag, past_ang = FeatureExtractor.get_optical_flow(
                    previous_frame=previous,
                    next_frame=next_frame,
                    all_results_out=True)
                last_frame_from_last_used_batch = frames[interest_fr]
                past_12_frames_optical_flow.update(
                    {f'{actual_interest_fr.item() - 1}-{actual_interest_fr.item()}': past_flow_per_frame})

        if len(past_12_frames_optical_flow) > frame_time_gap:
            temp_past_12_frames_optical_flow = {}
            for i in list(past_12_frames_optical_flow)[-frame_time_gap:]:
                temp_past_12_frames_optical_flow.update({i: past_12_frames_optical_flow[i]})
            past_12_frames_optical_flow = temp_past_12_frames_optical_flow
            temp_past_12_frames_optical_flow = None

        if actual_interest_fr < frame_time_gap:
            continue

        if not resume_mode:
            # flow between consecutive frames
            frames_used_in_of_estimation = list(range(actual_interest_fr, actual_of_interest_fr + 1))

            future_12_frames_optical_flow = {}
            # flow = np.zeros(shape=(frames.shape[1], frames.shape[2], 2))
            for of_i, actual_of_i in zip(range(interest_fr, of_interest_fr),
                                         range(actual_interest_fr, actual_of_interest_fr)):
                previous = cv.cvtColor(frames[of_i], cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor(frames[of_i + 1], cv.COLOR_BGR2GRAY)

                flow_per_frame, rgb, mag, ang = FeatureExtractor.get_optical_flow(previous_frame=previous,
                                                                                  next_frame=next_frame,
                                                                                  all_results_out=True)
                future_12_frames_optical_flow.update({f'{actual_of_i}-{actual_of_i + 1}': flow_per_frame})

            if actual_fr.item() > 11:

                original_shape = new_shape = [frames.shape[1], frames.shape[2]]

                frame_annotation_current = get_frame_annotations_and_skip_lost(df, actual_fr.item())
                current_annotations, _ = scale_annotations(frame_annotation_current,
                                                           original_scale=original_shape,
                                                           new_scale=new_shape, return_track_id=False,
                                                           tracks_with_annotations=True)
                current_gt_boxes = current_annotations[:, :-1]
                current_gt_boxes_idx = current_annotations[:, -1]

                frame_annotation_future = get_frame_annotations_and_skip_lost(df, actual_fr.item() + frame_time_gap)
                future_annotations, _ = scale_annotations(frame_annotation_future,
                                                          original_scale=original_shape,
                                                          new_scale=new_shape, return_track_id=False,
                                                          tracks_with_annotations=True)
                future_gt_boxes = future_annotations[:, :-1]
                future_gt_boxes_idx = future_annotations[:, -1]

                frame_annotation_past = get_frame_annotations_and_skip_lost(df, actual_fr.item() - frame_time_gap)
                past_annotations, _ = scale_annotations(frame_annotation_past,
                                                        original_scale=original_shape,
                                                        new_scale=new_shape, return_track_id=False,
                                                        tracks_with_annotations=True)
                past_gt_boxes = past_annotations[:, :-1]
                past_gt_boxes_idx = past_annotations[:, -1]

                object_features = []
                past_flow_yet = np.zeros(shape=(frames.shape[1], frames.shape[2], 2))
                flow_for_future = np.zeros(shape=(frames.shape[1], frames.shape[2], 2))
                past_flow_yet += np.array(list(past_12_frames_optical_flow.values())).sum(0)
                flow_for_future += np.array(list(future_12_frames_optical_flow.values())).sum(0)
                extracted_feature_actual_fr = extracted_features[actual_fr.item()]
                extracted_feature_actual_fr_object_features = extracted_feature_actual_fr.object_features

                future_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=interest_fr + frame_time_gap,
                                                       time_gap_within_frames=time_gap_within_frames,
                                                       total_frames=total_frames, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                past_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=interest_fr - frame_time_gap,
                                                     time_gap_within_frames=time_gap_within_frames,
                                                     total_frames=total_frames, step=step, n=n,
                                                     kernel=kernel, var_threshold=var_threshold,
                                                     detect_shadows=detect_shadows)

                for running_idx, object_feature in enumerate(
                        extracted_feature_actual_fr_object_features):  # fixme: add gt

                    current_track_features = track_based_accumulated_features[
                        object_feature.idx].object_features[-1] \
                        if object_feature.idx in track_based_accumulated_features.keys() else []

                    activations = object_feature.past_xy
                    box = object_feature.past_bbox
                    activations_future_displacement = flow_for_future[activations[:, 1], activations[:, 0]]
                    activations_past_displacement = past_flow_yet[activations[:, 1], activations[:, 0]]

                    activations_displaced_in_future = activations + activations_future_displacement
                    activations_displaced_in_past = activations - activations_past_displacement

                    shifted_box_in_future, shifted_activation_center_in_future = evaluate_shifted_bounding_box(
                        box, activations_displaced_in_future, activations)
                    shifted_box_in_past, shifted_activation_center_in_past = evaluate_shifted_bounding_box(
                        box, activations_displaced_in_past, activations)

                    activations_future_frame = extract_features_per_bounding_box(shifted_box_in_future, future_mask)
                    activations_past_frame = extract_features_per_bounding_box(shifted_box_in_past, past_mask)

                    # ground-truth association ######################################################################
                    future_gt_track_box_mapping = {a[-1]: a[:-1] for a in future_annotations}
                    past_gt_track_box_mapping = {a[-1]: a[:-1] for a in past_annotations}

                    generated_box_future = np.array([shifted_box_in_future])
                    generated_box_idx_future = [object_feature.idx]
                    generated_box_past = np.array([shifted_box_in_past])
                    generated_box_idx_past = [object_feature.idx]

                    l2_distance_boxes_score_matrix_future = np.zeros(shape=(len(future_gt_boxes),
                                                                            len(generated_box_future)))
                    l2_distance_boxes_score_matrix_past = np.zeros(shape=(len(past_gt_boxes), len(generated_box_past)))

                    for a_i, a_box in enumerate(future_gt_boxes):
                        for r_i, r_box_future in enumerate(generated_box_future):
                            dist_future = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                          get_bbox_center(r_box_future).flatten()), 2) * ratio
                            l2_distance_boxes_score_matrix_future[a_i, r_i] = dist_future

                    for a_i, a_box in enumerate(past_gt_boxes):
                        for r_i, r_box_past in enumerate(generated_box_past):
                            dist_past = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                        get_bbox_center(r_box_past).flatten()), 2) * ratio
                            l2_distance_boxes_score_matrix_past[a_i, r_i] = dist_past

                    l2_distance_boxes_score_matrix_future = 2 - l2_distance_boxes_score_matrix_future
                    l2_distance_boxes_score_matrix_future[l2_distance_boxes_score_matrix_future < 0] = 10

                    l2_distance_boxes_score_matrix_past = 2 - l2_distance_boxes_score_matrix_past
                    l2_distance_boxes_score_matrix_past[l2_distance_boxes_score_matrix_past < 0] = 10
                    # Hungarian
                    match_rows_future, match_cols_future = scipy.optimize.linear_sum_assignment(
                        l2_distance_boxes_score_matrix_future)
                    matching_distribution_future = [[i, j, l2_distance_boxes_score_matrix_future[i, j]] for i, j in zip(
                        match_rows_future, match_cols_future)]
                    actually_matched_mask_future = l2_distance_boxes_score_matrix_future[match_rows_future,
                                                                                         match_cols_future] < 10
                    match_rows_future = match_rows_future[actually_matched_mask_future]
                    match_cols_future = match_cols_future[actually_matched_mask_future]
                    match_rows_tracks_idx_future = [future_gt_boxes_idx[m].item() for m in match_rows_future]
                    match_cols_tracks_idx_future = [generated_box_idx_future[m] for m in match_cols_future]

                    match_rows_past, match_cols_past = scipy.optimize.linear_sum_assignment(
                        l2_distance_boxes_score_matrix_past)
                    matching_distribution_past = [[i, j, l2_distance_boxes_score_matrix_past[i, j]] for i, j in zip(
                        match_rows_past, match_cols_past)]
                    actually_matched_mask_past = l2_distance_boxes_score_matrix_past[match_rows_past,
                                                                                     match_cols_past] < 10
                    match_rows_past = match_rows_past[actually_matched_mask_past]
                    match_cols_past = match_cols_past[actually_matched_mask_past]
                    match_rows_tracks_idx_past = [past_gt_boxes_idx[m].item() for m in match_rows_past]
                    match_cols_tracks_idx_past = [generated_box_idx_past[m] for m in match_cols_past]

                    if len(match_rows_tracks_idx_future) != 0 and \
                            object_feature.gt_track_idx != match_rows_tracks_idx_future[0]:
                        logger.warn(f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                                    f' {match_rows_tracks_idx_future[0]}')
                    if len(match_rows_tracks_idx_past) != 0 and \
                            object_feature.gt_track_idx != match_rows_tracks_idx_past[0]:
                        logger.warn(f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                                    f' {match_rows_tracks_idx_past[0]}')
                    ###################################################################################################

                    if activations_future_frame.size == 0 or (filter_switch_boxes_based_on_angle_and_recent_history
                                                              and not isinstance(current_track_features, list)
                                                              and current_track_features.velocity_direction.size != 0
                                                              and len(current_track_features.velocity_direction) >
                                                              min_track_length_to_filter_switch_box
                                                              and first_violation_till_now(
                                current_track_features.velocity_direction, angle_threshold_to_filter)
                                                              and current_track_features.velocity_direction[-1] >
                                                              angle_threshold_to_filter
                                                              and not np.isnan(
                                current_track_features.velocity_direction[-1])):

                        logger.info(f'Ending Track! No features found in future!')

                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            current_track_history = get_agent_track_history(object_feature.idx,
                                                                            track_based_accumulated_features)
                            current_track_velocity_history = get_agent_track_velocity_history(
                                object_feature.idx, track_based_accumulated_features)
                            current_track_velocity_history = np.array(current_track_velocity_history)
                            current_direction = []
                            for track_history_idx in range(len(current_track_history) - 1):
                                current_direction.append((angle_between(
                                    v1=current_track_history[track_history_idx],
                                    v2=current_track_history[track_history_idx + 1]
                                )))
                            # current direction can be removed
                            current_direction = np.array(current_direction)

                            current_velocity_direction = []
                            for track_history_idx in range(len(current_track_velocity_history) - 1):
                                current_velocity_direction.append(math.degrees(angle_between(
                                    v1=current_track_velocity_history[track_history_idx],
                                    v2=current_track_velocity_history[track_history_idx + 1]
                                )))
                            current_velocity_direction = np.array(current_velocity_direction)

                            # just use gt
                            current_gt_track_history = get_gt_track_history(object_feature.idx,
                                                                            track_based_accumulated_features)
                        else:
                            current_track_history, current_gt_track_history = None, None
                            current_direction, current_velocity_direction = None, None
                            current_track_velocity_history = None

                        current_track_obj_features = AgentFeatures(
                            track_idx=object_feature.idx,
                            activations_t=activations,
                            activations_t_minus_one=activations_displaced_in_past,
                            activations_t_plus_one=activations_displaced_in_future,
                            future_flow=activations_future_displacement,
                            past_flow=activations_past_displacement,
                            bbox_t=box,
                            bbox_t_minus_one=shifted_box_in_past,
                            bbox_t_plus_one=shifted_box_in_future,
                            frame_number=object_feature.frame_number,
                            activations_future_frame=activations_future_frame,
                            activations_past_frame=activations_past_frame,
                            final_features_future_activations=None,
                            is_track_live=False,
                            frame_number_t=actual_fr.item(),
                            frame_number_t_minus_one=actual_fr.item() - frame_time_gap,
                            frame_number_t_plus_one=actual_fr.item() + frame_time_gap,
                            past_frames_used_in_of_estimation=list(past_12_frames_optical_flow.keys()),
                            future_frames_used_in_of_estimation=list(future_12_frames_optical_flow.keys()),
                            gt_box=object_feature.gt_box,
                            gt_track_idx=object_feature.gt_track_idx,
                            gt_past_current_distance=l2_distance_boxes_score_matrix_past[match_rows_past[0],
                                                                                         match_cols_past[0]]
                            if len(match_rows_past) != 0 and len(match_cols_past) != 0 else None,
                            past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]]
                            if len(match_rows_tracks_idx_past) != 0 else None,
                            future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]]
                            if len(match_rows_tracks_idx_future) != 0 else None,
                            gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                                                                             match_cols_future[0]]
                            if len(match_rows_future) != 0 and len(match_cols_future) != 0 else None,
                            past_gt_track_idx=match_rows_tracks_idx_past[0]
                            if len(match_rows_tracks_idx_past) != 0 else None,
                            future_gt_track_idx=match_rows_tracks_idx_future[0]
                            if len(match_rows_tracks_idx_future) != 0 else None,
                            future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0]
                            if len(match_rows_tracks_idx_future) != 0 else None,
                            past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0]
                            if len(match_rows_tracks_idx_past) != 0 else None,
                            frame_by_frame_estimation=False,
                            gt_history=current_gt_track_history, history=current_track_history,
                            track_direction=current_direction, velocity_history=current_track_velocity_history,
                            velocity_direction=current_velocity_direction
                        )
                        object_features.append(current_track_obj_features)
                        if object_feature.idx in track_based_accumulated_features:
                            track_based_accumulated_features[object_feature.idx].object_features.append(
                                current_track_obj_features)

                        continue

                    closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                        features_filter_append_preprocessing(
                            overlap_percent, activations_displaced_in_future, activations_future_frame)

                    filtered_shifted_future_activations = filter_features(
                        activations_displaced_in_future, closest_n_shifted_xy_pair)
                    final_features_future_activations = append_features(
                        filtered_shifted_future_activations, closest_n_xy_current_frame_pair)

                    if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                        current_track_history = get_agent_track_history(object_feature.idx,
                                                                        track_based_accumulated_features)
                        current_track_velocity_history = get_agent_track_velocity_history(
                            object_feature.idx, track_based_accumulated_features)
                        current_track_velocity_history = np.array(current_track_velocity_history)
                        current_direction = []
                        for track_history_idx in range(len(current_track_history) - 1):
                            current_direction.append((angle_between(
                                v1=current_track_history[track_history_idx],
                                v2=current_track_history[track_history_idx + 1]
                            )))
                        # current direction can be removed
                        current_direction = np.array(current_direction)

                        current_velocity_direction = []
                        for track_history_idx in range(len(current_track_velocity_history) - 1):
                            current_velocity_direction.append(math.degrees(angle_between(
                                v1=current_track_velocity_history[track_history_idx],
                                v2=current_track_velocity_history[track_history_idx + 1]
                            )))
                        current_velocity_direction = np.array(current_velocity_direction)

                        current_gt_track_history = get_gt_track_history(object_feature.idx,
                                                                        track_based_accumulated_features)
                    else:
                        current_track_history, current_gt_track_history = None, None
                        current_direction, current_velocity_direction = None, None
                        current_track_velocity_history = None

                    current_track_obj_features = AgentFeatures(
                        track_idx=object_feature.idx,
                        activations_t=activations,
                        activations_t_minus_one=activations_displaced_in_past,
                        activations_t_plus_one=activations_displaced_in_future,
                        future_flow=activations_future_displacement,
                        past_flow=activations_past_displacement,
                        bbox_t=box,
                        bbox_t_minus_one=shifted_box_in_past,
                        bbox_t_plus_one=shifted_box_in_future,
                        frame_number=object_feature.frame_number,
                        activations_future_frame=activations_future_frame,
                        activations_past_frame=activations_past_frame,
                        final_features_future_activations=final_features_future_activations,
                        frame_number_t=actual_fr.item(),
                        frame_number_t_minus_one=actual_fr.item() - frame_time_gap,
                        frame_number_t_plus_one=actual_fr.item() + frame_time_gap,
                        past_frames_used_in_of_estimation=list(past_12_frames_optical_flow.keys()),
                        future_frames_used_in_of_estimation=list(future_12_frames_optical_flow.keys()),
                        gt_box=object_feature.gt_box,
                        gt_track_idx=object_feature.gt_track_idx,
                        gt_past_current_distance=l2_distance_boxes_score_matrix_past[match_rows_past[0],
                                                                                     match_cols_past[0]]
                        if len(match_rows_past) != 0 and len(match_cols_past) != 0 else None,
                        past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]]
                        if len(match_rows_tracks_idx_past) != 0 else None,
                        future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]]
                        if len(match_rows_tracks_idx_future) != 0 else None,
                        gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                                                                         match_cols_future[0]]
                        if len(match_rows_future) != 0 and len(match_cols_future) != 0 else None,
                        past_gt_track_idx=match_rows_tracks_idx_past[0]
                        if len(match_rows_tracks_idx_past) != 0 else None,
                        future_gt_track_idx=match_rows_tracks_idx_future[0]
                        if len(match_rows_tracks_idx_future) != 0 else None,
                        future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0]
                        if len(match_rows_tracks_idx_future) != 0 else None,
                        past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0]
                        if len(match_rows_tracks_idx_past) != 0 else None,
                        frame_by_frame_estimation=False,
                        gt_history=current_gt_track_history, history=current_track_history,
                        track_direction=current_direction, velocity_history=current_track_velocity_history,
                        velocity_direction=current_velocity_direction
                    )

                    object_features.append(current_track_obj_features)
                    if object_feature.idx not in track_based_accumulated_features:
                        track_feats = TrackFeatures(object_feature.idx)
                        track_feats.object_features.append(current_track_obj_features)
                        track_based_accumulated_features.update(
                            {object_feature.idx: track_feats})
                    else:
                        track_based_accumulated_features[object_feature.idx].object_features.append(
                            current_track_obj_features)
                    #
                    # plot_tracks_with_features(frame_t=frames[fr],
                    #                           frame_t_minus_one=frames[fr - frame_time_gap],
                    #                           frame_t_plus_one=frames[fr + frame_time_gap],
                    #                           features_t=activations,
                    #                           features_t_minus_one=activations_displaced_in_past,
                    #                           features_t_plus_one=activations_displaced_in_future,
                    #                           box_t=[box],
                    #                           box_t_minus_one=[shifted_box_in_past],
                    #                           box_t_plus_one=[shifted_box_in_future],
                    #                           frame_number=actual_fr.item(),
                    #                           marker_size=1,
                    #                           track_id=object_feature.idx,
                    #                           file_idx=running_idx,
                    #                           save_path=save_path_for_plot,
                    #                           annotations=[[object_feature.idx], [object_feature.idx],
                    #                                        [object_feature.idx]],
                    #                           additional_text=f'Past: {actual_fr.item() - frame_time_gap} |'
                    #                                           f' Present: {actual_fr.item()} | '
                    #                                           f'Future: {actual_fr.item() + frame_time_gap}')

                data_all_frames.update({actual_fr.item(): FrameFeatures(frame_number=actual_fr.item(),
                                                                        object_features=object_features)})

                frame_feats = data_all_frames[actual_fr.item()]
                frame_boxes = []
                frame_track_ids = []

                for obj_feature in frame_feats.object_features:
                    frame_boxes.append(obj_feature.bbox_t)
                    frame_track_ids.append(obj_feature.track_idx)

                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                    tracks_histories = []
                    tracks_gt_histories = []

                    for obj_feature in frame_feats.object_features:
                        tracks_histories.extend(obj_feature.track_history)
                        tracks_gt_histories.extend(obj_feature.gt_history)

                    tracks_histories = np.array(tracks_histories)
                    tracks_gt_histories = np.array(tracks_gt_histories)

                    if tracks_histories.size == 0:
                        tracks_histories = np.zeros(shape=(0, 2))
                    if tracks_gt_histories.size == 0:
                        tracks_gt_histories = np.zeros(shape=(0, 2))
                else:
                    tracks_histories = np.zeros(shape=(0, 2))
                    tracks_gt_histories = np.zeros(shape=(0, 2))

                fig = plot_for_video_current_frame(
                    gt_rgb=frames[fr], current_frame_rgb=frames[fr],
                    gt_annotations=current_gt_boxes,
                    current_frame_annotation=frame_boxes,
                    new_track_annotation=[],
                    frame_number=actual_fr.item(),
                    box_annotation=[current_gt_boxes_idx.tolist(), frame_track_ids],
                    generated_track_histories=tracks_histories,
                    gt_track_histories=tracks_gt_histories,
                    additional_text='',
                    # f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                    # f'Recall: {l2_distance_hungarian_recall}\n'
                    # f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                    # f'Precision: {precision} | Recall: {recall}\n'
                    # f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                    # f'Track Ids Killed: '
                    # f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                    video_mode=False, original_dims=original_dims, zero_shot=True, return_figure_only=True)

                current_batch_figures.append(fig)

                # data_all_frames.update({actual_fr.item(): FrameFeatures(frame_number=actual_fr.item(),
                #                                                         object_features=object_features)})

    return data_all_frames, frames[interest_fr:, ...], \
           torch.arange(interest_fr + frame_numbers[0], frame_numbers[-1] + 1), \
           last_frame_from_last_used_batch, past_12_frames_optical_flow, last12_bg_sub_mask, \
           track_based_accumulated_features, current_batch_figures, frame_track_ids


def twelve_frame_by_frame_feature_extraction_zero_shot(
        frames, n, frames_to_build_model, extracted_features, df,
        var_threshold=None, time_gap_within_frames=3, frame_numbers=None,
        remaining_frames=None, remaining_frames_idx=None, past_12_frames_optical_flow=None,
        last_frame_from_last_used_batch=None, last12_bg_sub_mask=None,
        resume_mode=False, detect_shadows=True, overlap_percent=0.4, ratio=None,
        track_based_accumulated_features=None, frame_time_gap=12, save_path_for_plot=None,
        filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, video_mode=False, original_dims=None,
        min_track_length_to_filter_switch_box=20, save_path_for_video=None,
        angle_threshold_to_filter=120):
    current_batch_figures, frame_track_ids = [], []
    interest_fr = None
    actual_interest_fr = None

    # cat old frames
    if remaining_frames is not None:
        frames = np.concatenate((remaining_frames, frames), axis=0)
        frame_numbers = torch.cat((remaining_frames_idx, frame_numbers))

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    step = 0
    if n is not None:
        step = n // 2
    else:
        n = frames_to_build_model
    total_frames = frames.shape[0]

    data_all_frames = {}

    for fr, actual_fr in tqdm(zip(range(frames.shape[0]), frame_numbers), total=frames.shape[0]):
        interest_fr = fr % total_frames
        actual_interest_fr = actual_fr

        of_interest_fr = (fr + frames_to_build_model) % total_frames
        actual_of_interest_fr = (actual_fr + frames_to_build_model)

        mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=fr,
                                        time_gap_within_frames=time_gap_within_frames,
                                        total_frames=total_frames, step=step, n=n,
                                        kernel=kernel, var_threshold=var_threshold,
                                        detect_shadows=detect_shadows)

        # do not go in circle for flow estimation
        if of_interest_fr < interest_fr:
            break

        last12_bg_sub_mask.update({actual_interest_fr.item(): mask})

        # start at 12th frame and then only consider last 12 frames for velocity estimation
        if actual_interest_fr != 0:
            if interest_fr == 0:
                previous = cv.cvtColor(last_frame_from_last_used_batch.astype(np.uint8), cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor(frames[interest_fr], cv.COLOR_BGR2GRAY)

                past_flow_per_frame, past_rgb, past_mag, past_ang = FeatureExtractor.get_optical_flow(
                    previous_frame=previous,
                    next_frame=next_frame,
                    all_results_out=True)
                past_12_frames_optical_flow.update(
                    {f'{actual_interest_fr.item() - 1}-{actual_interest_fr.item()}': past_flow_per_frame})
            else:
                previous = cv.cvtColor(frames[interest_fr - 1], cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor(frames[interest_fr], cv.COLOR_BGR2GRAY)

                past_flow_per_frame, past_rgb, past_mag, past_ang = FeatureExtractor.get_optical_flow(
                    previous_frame=previous,
                    next_frame=next_frame,
                    all_results_out=True)
                last_frame_from_last_used_batch = frames[interest_fr]
                past_12_frames_optical_flow.update(
                    {f'{actual_interest_fr.item() - 1}-{actual_interest_fr.item()}': past_flow_per_frame})

        if len(past_12_frames_optical_flow) > frame_time_gap:
            temp_past_12_frames_optical_flow = {}
            for i in list(past_12_frames_optical_flow)[-frame_time_gap:]:
                temp_past_12_frames_optical_flow.update({i: past_12_frames_optical_flow[i]})
            past_12_frames_optical_flow = temp_past_12_frames_optical_flow
            temp_past_12_frames_optical_flow = None

        if len(last12_bg_sub_mask) > 13:  # we need one more for of
            temp_last12_bg_sub_mask = {}
            for i in list(last12_bg_sub_mask)[-13:]:
                temp_last12_bg_sub_mask.update({i: last12_bg_sub_mask[i]})
            last12_bg_sub_mask = temp_last12_bg_sub_mask
            temp_last12_bg_sub_mask = None

        if actual_interest_fr < frame_time_gap:
            continue

        if not resume_mode:
            # flow between consecutive frames
            frames_used_in_of_estimation = list(range(actual_interest_fr, actual_of_interest_fr + 1))

            future12_bg_sub_mask = {}
            future_12_frames_optical_flow = {}
            # flow = np.zeros(shape=(frames.shape[1], frames.shape[2], 2))  # put sum of optimized of - using other var
            for of_i, actual_of_i in zip(range(interest_fr, of_interest_fr),
                                         range(actual_interest_fr, actual_of_interest_fr)):
                future_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=of_i + 1,
                                                       time_gap_within_frames=time_gap_within_frames,
                                                       total_frames=total_frames, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)
                future12_bg_sub_mask.update({actual_of_i + 1: future_mask})

                previous = cv.cvtColor(frames[of_i], cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor(frames[of_i + 1], cv.COLOR_BGR2GRAY)

                flow_per_frame, rgb, mag, ang = FeatureExtractor.get_optical_flow(previous_frame=previous,
                                                                                  next_frame=next_frame,
                                                                                  all_results_out=True)
                future_12_frames_optical_flow.update({f'{actual_of_i}-{actual_of_i + 1}': flow_per_frame})

            if actual_fr.item() > 11:

                original_shape = new_shape = [frames.shape[1], frames.shape[2]]

                frame_annotation_current = get_frame_annotations_and_skip_lost(df, actual_fr.item())
                current_annotations, _ = scale_annotations(frame_annotation_current,
                                                           original_scale=original_shape,
                                                           new_scale=new_shape, return_track_id=False,
                                                           tracks_with_annotations=True)
                current_gt_boxes = current_annotations[:, :-1]
                current_gt_boxes_idx = current_annotations[:, -1]

                frame_annotation_future = get_frame_annotations_and_skip_lost(df, actual_fr.item() + frame_time_gap)
                future_annotations, _ = scale_annotations(frame_annotation_future,
                                                          original_scale=original_shape,
                                                          new_scale=new_shape, return_track_id=False,
                                                          tracks_with_annotations=True)
                future_gt_boxes = future_annotations[:, :-1]
                future_gt_boxes_idx = future_annotations[:, -1]

                frame_annotation_past = get_frame_annotations_and_skip_lost(df, actual_fr.item() - frame_time_gap)
                past_annotations, _ = scale_annotations(frame_annotation_past,
                                                        original_scale=original_shape,
                                                        new_scale=new_shape, return_track_id=False,
                                                        tracks_with_annotations=True)
                past_gt_boxes = past_annotations[:, :-1]
                past_gt_boxes_idx = past_annotations[:, -1]

                l2_distance_boxes_score_matrix_past, l2_distance_boxes_score_matrix_future = None, None
                match_rows_future, match_cols_future, match_rows_past, match_cols_past = None, None, None, None
                past_gt_track_box_mapping, future_gt_track_box_mapping = None, None
                match_rows_tracks_idx_past, match_rows_tracks_idx_future = None, None

                object_features = []
                past_flow_yet = np.zeros(shape=(frames.shape[1], frames.shape[2], 2))
                flow_for_future = np.zeros(shape=(frames.shape[1], frames.shape[2], 2))
                past_flow_yet += np.array(list(past_12_frames_optical_flow.values())).sum(0)
                flow_for_future += np.array(list(future_12_frames_optical_flow.values())).sum(0)
                extracted_feature_actual_fr = extracted_features[actual_fr.item()]
                extracted_feature_actual_fr_object_features = extracted_feature_actual_fr.object_features

                # future_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=interest_fr + frame_time_gap,
                #                                        time_gap_within_frames=time_gap_within_frames,
                #                                        total_frames=total_frames, step=step, n=n,
                #                                        kernel=kernel, var_threshold=var_threshold,
                #                                        detect_shadows=detect_shadows)
                #
                # past_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=interest_fr - frame_time_gap,
                #                                      time_gap_within_frames=time_gap_within_frames,
                #                                      total_frames=total_frames, step=step, n=n,
                #                                      kernel=kernel, var_threshold=var_threshold,
                #                                      detect_shadows=detect_shadows)

                past_12_frames_optical_flow_reversed = {}
                past_12_frames_optical_flow_keys = list(past_12_frames_optical_flow.keys())
                past_12_frames_optical_flow_keys.reverse()
                for k in past_12_frames_optical_flow_keys:
                    past_12_frames_optical_flow_reversed.update({k: past_12_frames_optical_flow[k]})

                last12_bg_sub_mask_reversed = {}
                last12_bg_sub_mask_keys = list(last12_bg_sub_mask.keys())
                last12_bg_sub_mask_keys.reverse()
                for k in last12_bg_sub_mask_keys[1:]:
                    last12_bg_sub_mask_reversed.update({k: last12_bg_sub_mask[k]})

                for r_idx, object_feature in enumerate(extracted_feature_actual_fr_object_features):  # fixme: add gt

                    current_track_features = track_based_accumulated_features[
                        object_feature.idx].object_features[-1] \
                        if object_feature.idx in track_based_accumulated_features.keys() else []

                    activations = object_feature.past_xy
                    box = object_feature.past_bbox

                    # activations_per_frame = object_feature.past_xy
                    box_future_per_frame, box_past_per_frame = box, box
                    activations_future_frame_per_frame, activations_past_frame_per_frame = None, None
                    activations_displaced_in_future_per_frame_past, activations_displaced_in_past_per_frame_past = \
                        object_feature.past_xy, object_feature.past_xy
                    activations_displaced_in_future_per_frame, activations_displaced_in_past_per_frame = \
                        None, None
                    shifted_box_in_future_per_frame, shifted_box_in_past_per_frame = None, None
                    activations_future_displacement_list, activations_past_displacement_list = [], []

                    for running_idx, ((past_idx, past_flow), (future_idx, future_flow),
                                      (past_mask_idx, past_mask_from_dict),
                                      (future_mask_idx, future_mask_from_dict)) in enumerate(zip(
                        past_12_frames_optical_flow_reversed.items(), future_12_frames_optical_flow.items(),
                        last12_bg_sub_mask_reversed.items(), future12_bg_sub_mask.items())):

                        activations_future_displacement_per_frame = future_flow[
                            activations_displaced_in_future_per_frame_past[:, 1],
                            activations_displaced_in_future_per_frame_past[:, 0]]
                        activations_past_displacement_per_frame = past_flow[
                            activations_displaced_in_past_per_frame_past[:, 1],
                            activations_displaced_in_past_per_frame_past[:, 0]]

                        activations_future_displacement_list.append(activations_future_displacement_per_frame)
                        activations_past_displacement_list.append(activations_past_displacement_per_frame)

                        activations_displaced_in_future_per_frame = \
                            activations_displaced_in_future_per_frame_past + activations_future_displacement_per_frame
                        activations_displaced_in_past_per_frame = \
                            activations_displaced_in_past_per_frame_past - activations_past_displacement_per_frame

                        shifted_box_in_future_per_frame, shifted_activation_center_in_future_per_frame = \
                            evaluate_shifted_bounding_box(
                                box_future_per_frame, activations_displaced_in_future_per_frame,
                                activations_displaced_in_future_per_frame_past)
                        shifted_box_in_past_per_frame, shifted_activation_center_in_past_per_frame = \
                            evaluate_shifted_bounding_box(
                                box_past_per_frame, activations_displaced_in_past_per_frame,
                                activations_displaced_in_past_per_frame_past)

                        activations_future_frame_per_frame = extract_features_per_bounding_box(
                            shifted_box_in_future_per_frame, future_mask_from_dict)
                        activations_past_frame_per_frame = extract_features_per_bounding_box(
                            shifted_box_in_past_per_frame, past_mask_from_dict)

                        # ground-truth association ####################################################################
                        future_gt_track_box_mapping = {a[-1]: a[:-1] for a in future_annotations}
                        past_gt_track_box_mapping = {a[-1]: a[:-1] for a in past_annotations}

                        generated_box_future = np.array([shifted_box_in_future_per_frame])
                        generated_box_idx_future = [object_feature.idx]
                        generated_box_past = np.array([shifted_box_in_past_per_frame])
                        generated_box_idx_past = [object_feature.idx]

                        l2_distance_boxes_score_matrix_future = np.zeros(shape=(len(future_gt_boxes),
                                                                                len(generated_box_future)))
                        l2_distance_boxes_score_matrix_past = np.zeros(
                            shape=(len(past_gt_boxes), len(generated_box_past)))

                        for a_i, a_box in enumerate(future_gt_boxes):
                            for r_i, r_box_future in enumerate(generated_box_future):
                                dist_future = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                              get_bbox_center(r_box_future).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix_future[a_i, r_i] = dist_future

                        for a_i, a_box in enumerate(past_gt_boxes):
                            for r_i, r_box_past in enumerate(generated_box_past):
                                dist_past = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                            get_bbox_center(r_box_past).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix_past[a_i, r_i] = dist_past

                        l2_distance_boxes_score_matrix_future = 2 - l2_distance_boxes_score_matrix_future
                        l2_distance_boxes_score_matrix_future[l2_distance_boxes_score_matrix_future < 0] = 10

                        l2_distance_boxes_score_matrix_past = 2 - l2_distance_boxes_score_matrix_past
                        l2_distance_boxes_score_matrix_past[l2_distance_boxes_score_matrix_past < 0] = 10
                        # Hungarian
                        match_rows_future, match_cols_future = scipy.optimize.linear_sum_assignment(
                            l2_distance_boxes_score_matrix_future)
                        matching_distribution_future = [[i, j, l2_distance_boxes_score_matrix_future[i, j]] for i, j in
                                                        zip(
                                                            match_rows_future, match_cols_future)]
                        actually_matched_mask_future = l2_distance_boxes_score_matrix_future[match_rows_future,
                                                                                             match_cols_future] < 10
                        match_rows_future = match_rows_future[actually_matched_mask_future]
                        match_cols_future = match_cols_future[actually_matched_mask_future]
                        match_rows_tracks_idx_future = [future_gt_boxes_idx[m].item() for m in match_rows_future]
                        match_cols_tracks_idx_future = [generated_box_idx_future[m] for m in match_cols_future]

                        match_rows_past, match_cols_past = scipy.optimize.linear_sum_assignment(
                            l2_distance_boxes_score_matrix_past)
                        matching_distribution_past = [[i, j, l2_distance_boxes_score_matrix_past[i, j]] for i, j in zip(
                            match_rows_past, match_cols_past)]
                        actually_matched_mask_past = l2_distance_boxes_score_matrix_past[match_rows_past,
                                                                                         match_cols_past] < 10
                        match_rows_past = match_rows_past[actually_matched_mask_past]
                        match_cols_past = match_cols_past[actually_matched_mask_past]
                        match_rows_tracks_idx_past = [past_gt_boxes_idx[m].item() for m in match_rows_past]
                        match_cols_tracks_idx_past = [generated_box_idx_past[m] for m in match_cols_past]

                        # if object_feature.gt_track_idx != match_rows_tracks_idx_future[0]:
                        #     logger.info(
                        #         f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                        #         f' {match_rows_tracks_idx_future[0]}')
                        # if object_feature.gt_track_idx != match_rows_tracks_idx_past[0]:
                        #     logger.info(
                        #         f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                        #         f' {match_rows_tracks_idx_past[0]}')

                        if len(match_rows_tracks_idx_future) != 0 and \
                                object_feature.gt_track_idx != match_rows_tracks_idx_future[0]:
                            logger.warn(
                                f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                                f' {match_rows_tracks_idx_future[0]}')
                        if len(match_rows_tracks_idx_past) != 0 and \
                                object_feature.gt_track_idx != match_rows_tracks_idx_past[0]:
                            logger.warn(
                                f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                                f' {match_rows_tracks_idx_past[0]}')
                        ##############################################################################################

                        if activations_future_frame_per_frame.size == 0 \
                                or activations_past_frame_per_frame.size == 0 or \
                                (filter_switch_boxes_based_on_angle_and_recent_history
                                 and not isinstance(current_track_features, list)
                                 and current_track_features.velocity_direction.size != 0
                                 and len(current_track_features.velocity_direction) >
                                 min_track_length_to_filter_switch_box
                                 and first_violation_till_now(
                                            current_track_features.velocity_direction, angle_threshold_to_filter)
                                 and current_track_features.velocity_direction[-1] >
                                 angle_threshold_to_filter
                                 and not np.isnan(current_track_features.velocity_direction[-1])):

                            logger.info(f'Ending Track {object_feature.idx}! No features found in '
                                        f'{"past" if activations_past_frame_per_frame.size == 0 else "future"} '
                                        f'at {running_idx} frames apart from frame {actual_fr.item()}')

                            if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                current_track_history = get_agent_track_history(object_feature.idx,
                                                                                track_based_accumulated_features)
                                current_track_velocity_history = get_agent_track_velocity_history(
                                    object_feature.idx, track_based_accumulated_features)
                                current_track_velocity_history = np.array(current_track_velocity_history)
                                current_direction = []
                                for track_history_idx in range(len(current_track_history) - 1):
                                    current_direction.append((angle_between(
                                        v1=current_track_history[track_history_idx],
                                        v2=current_track_history[track_history_idx + 1]
                                    )))
                                # current direction can be removed
                                current_direction = np.array(current_direction)

                                current_velocity_direction = []
                                for track_history_idx in range(len(current_track_velocity_history) - 1):
                                    current_velocity_direction.append(math.degrees(angle_between(
                                        v1=current_track_velocity_history[track_history_idx],
                                        v2=current_track_velocity_history[track_history_idx + 1]
                                    )))
                                current_velocity_direction = np.array(current_velocity_direction)

                                # just use gt
                                current_gt_track_history = get_gt_track_history(object_feature.idx,
                                                                                track_based_accumulated_features)
                            else:
                                current_track_history, current_gt_track_history = None, None
                                current_direction, current_velocity_direction = None, None
                                current_track_velocity_history = None

                            current_track_obj_features = AgentFeatures(
                                track_idx=object_feature.idx,
                                activations_t=activations,
                                activations_t_minus_one=activations_displaced_in_past_per_frame,
                                activations_t_plus_one=activations_displaced_in_future_per_frame,
                                future_flow=activations_future_displacement_per_frame,
                                past_flow=activations_past_displacement_per_frame,
                                bbox_t=box,
                                bbox_t_minus_one=shifted_box_in_past_per_frame,
                                bbox_t_plus_one=shifted_box_in_future_per_frame,
                                frame_number=object_feature.frame_number,
                                activations_future_frame=activations_future_frame_per_frame,
                                activations_past_frame=activations_past_frame_per_frame,
                                final_features_future_activations=None,
                                is_track_live=False,
                                frame_number_t=actual_fr.item(),
                                frame_number_t_minus_one=past_mask_idx,
                                frame_number_t_plus_one=future_mask_idx,
                                past_frames_used_in_of_estimation=running_idx,
                                future_frames_used_in_of_estimation=running_idx,
                                gt_box=object_feature.gt_box,
                                gt_track_idx=object_feature.gt_track_idx,
                                # gt_past_current_distance=l2_distance_boxes_score_matrix_past[match_rows_past[0],
                                #                                                              match_cols_past[0]],
                                # past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]],
                                # future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]],
                                # gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                #                                                                  match_cols_future[0]],
                                # past_gt_track_idx=match_rows_tracks_idx_past[0],
                                # future_gt_track_idx=match_rows_tracks_idx_future[0],
                                # future_box_inconsistent=
                                # object_feature.gt_track_idx == match_rows_tracks_idx_future[0],
                                # past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0],
                                # frame_by_frame_estimation=True,
                                gt_past_current_distance=l2_distance_boxes_score_matrix_past[match_rows_past[0],
                                                                                             match_cols_past[0]]
                                if len(match_rows_past) != 0 and len(match_cols_past) != 0 else None,
                                past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]]
                                if len(match_rows_tracks_idx_past) != 0 else None,
                                future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]]
                                if len(match_rows_tracks_idx_future) != 0 else None,
                                gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                                                                                 match_cols_future[0]]
                                if len(match_rows_future) != 0 and len(match_cols_future) != 0 else None,
                                past_gt_track_idx=match_rows_tracks_idx_past[0]
                                if len(match_rows_tracks_idx_past) != 0 else None,
                                future_gt_track_idx=match_rows_tracks_idx_future[0]
                                if len(match_rows_tracks_idx_future) != 0 else None,
                                future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0]
                                if len(match_rows_tracks_idx_future) != 0 else None,
                                past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0]
                                if len(match_rows_tracks_idx_past) != 0 else None,
                                frame_by_frame_estimation=True,
                                gt_history=current_gt_track_history, history=current_track_history,
                                track_direction=current_direction, velocity_history=current_track_velocity_history,
                                velocity_direction=current_velocity_direction
                            )
                            object_features.append(current_track_obj_features)
                            if object_feature.idx in track_based_accumulated_features:
                                track_based_accumulated_features[object_feature.idx].object_features.append(
                                    current_track_obj_features)

                            continue

                        closest_n_shifted_xy_pair_future_per_frame, closest_n_xy_current_future_frame_pair_per_frame = \
                            features_filter_append_preprocessing(
                                overlap_percent, activations_displaced_in_future_per_frame,
                                activations_future_frame_per_frame)

                        filtered_shifted_future_activations_per_frame = filter_features(
                            activations_displaced_in_future_per_frame, closest_n_shifted_xy_pair_future_per_frame)
                        activations_displaced_in_future_per_frame = append_features(
                            filtered_shifted_future_activations_per_frame,
                            closest_n_xy_current_future_frame_pair_per_frame)

                        closest_n_shifted_xy_pair_past_per_frame, closest_n_xy_current_past_frame_pair_per_frame = \
                            features_filter_append_preprocessing(
                                overlap_percent, activations_displaced_in_past_per_frame,
                                activations_past_frame_per_frame)

                        filtered_shifted_past_activations_per_frame = filter_features(
                            activations_displaced_in_past_per_frame, closest_n_shifted_xy_pair_past_per_frame)
                        activations_displaced_in_past_per_frame = append_features(
                            filtered_shifted_past_activations_per_frame, closest_n_xy_current_past_frame_pair_per_frame)

                        activations_displaced_in_future_per_frame_past = np.round(
                            activations_displaced_in_future_per_frame).astype(np.int)
                        activations_displaced_in_past_per_frame_past = np.round(
                            activations_displaced_in_past_per_frame).astype(np.int)

                        # plot_tracks_with_features(frame_t=frames[fr],
                        #                           frame_t_minus_one=frames[fr - running_idx - 1],
                        #                           frame_t_plus_one=frames[fr + running_idx + 1],
                        #                           features_t=activations,
                        #                           features_t_minus_one=activations_displaced_in_past_per_frame,
                        #                           features_t_plus_one=activations_displaced_in_future_per_frame,
                        #                           box_t=[box],
                        #                           box_t_minus_one=[shifted_box_in_past_per_frame],
                        #                           box_t_plus_one=[shifted_box_in_future_per_frame],
                        #                           frame_number=actual_fr.item(),
                        #                           marker_size=1,
                        #                           track_id=object_feature.idx,
                        #                           file_idx=running_idx,
                        #                           save_path=save_path_for_plot,
                        #                           annotations=[[object_feature.idx], [object_feature.idx],
                        #                                        [object_feature.idx]],
                        #                           additional_text=f'Past: {fr - running_idx - 1} | Present: {fr} | '
                        #                                           f'Future: {fr + running_idx + 1}')

                    if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                        current_track_history = get_agent_track_history(object_feature.idx,
                                                                        track_based_accumulated_features)
                        current_track_velocity_history = get_agent_track_velocity_history(
                            object_feature.idx, track_based_accumulated_features)
                        current_track_velocity_history = np.array(current_track_velocity_history)
                        current_direction = []
                        for track_history_idx in range(len(current_track_history) - 1):
                            current_direction.append((angle_between(
                                v1=current_track_history[track_history_idx],
                                v2=current_track_history[track_history_idx + 1]
                            )))
                        # current direction can be removed
                        current_direction = np.array(current_direction)

                        current_velocity_direction = []
                        for track_history_idx in range(len(current_track_velocity_history) - 1):
                            current_velocity_direction.append(math.degrees(angle_between(
                                v1=current_track_velocity_history[track_history_idx],
                                v2=current_track_velocity_history[track_history_idx + 1]
                            )))
                        current_velocity_direction = np.array(current_velocity_direction)

                        current_gt_track_history = get_gt_track_history(object_feature.idx,
                                                                        track_based_accumulated_features)
                    else:
                        current_track_history, current_gt_track_history = None, None
                        current_direction, current_velocity_direction = None, None
                        current_track_velocity_history = None

                    current_track_obj_features = AgentFeatures(
                        track_idx=object_feature.idx,
                        activations_t=activations,
                        activations_t_minus_one=activations_displaced_in_past_per_frame,
                        activations_t_plus_one=activations_displaced_in_future_per_frame,
                        future_flow=activations_future_displacement_list,
                        past_flow=activations_past_displacement_list,
                        bbox_t=box,
                        bbox_t_minus_one=shifted_box_in_past_per_frame,
                        bbox_t_plus_one=shifted_box_in_future_per_frame,
                        frame_number=object_feature.frame_number,
                        activations_future_frame=activations_future_frame_per_frame,
                        activations_past_frame=activations_past_frame_per_frame,
                        final_features_future_activations=activations_displaced_in_future_per_frame,
                        frame_number_t=actual_fr.item(),
                        frame_number_t_minus_one=actual_fr.item() - frame_time_gap,
                        frame_number_t_plus_one=actual_fr.item() + frame_time_gap,
                        past_frames_used_in_of_estimation=list(past_12_frames_optical_flow.keys()),
                        future_frames_used_in_of_estimation=list(future_12_frames_optical_flow.keys()),
                        gt_box=object_feature.gt_box,
                        gt_track_idx=object_feature.gt_track_idx,
                        # gt_past_current_distance=l2_distance_boxes_score_matrix_past[match_rows_past[0],
                        #                                                              match_cols_past[0]],
                        # past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]],
                        # future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]],
                        # gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                        #                                                                  match_cols_future[0]],
                        # past_gt_track_idx=match_rows_tracks_idx_past[0],
                        # future_gt_track_idx=match_rows_tracks_idx_future[0],
                        # future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0],
                        # past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0],
                        gt_past_current_distance=l2_distance_boxes_score_matrix_past[match_rows_past[0],
                                                                                     match_cols_past[0]]
                        if len(match_rows_past) != 0 and len(match_cols_past) != 0 else None,
                        past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]]
                        if len(match_rows_tracks_idx_past) != 0 else None,
                        future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]]
                        if len(match_rows_tracks_idx_future) != 0 else None,
                        gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                                                                         match_cols_future[0]]
                        if len(match_rows_future) != 0 and len(match_cols_future) != 0 else None,
                        past_gt_track_idx=match_rows_tracks_idx_past[0]
                        if len(match_rows_tracks_idx_past) != 0 else None,
                        future_gt_track_idx=match_rows_tracks_idx_future[0]
                        if len(match_rows_tracks_idx_future) != 0 else None,
                        future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0]
                        if len(match_rows_tracks_idx_future) != 0 else None,
                        past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0]
                        if len(match_rows_tracks_idx_past) != 0 else None,
                        frame_by_frame_estimation=True,
                        gt_history=current_gt_track_history, history=current_track_history,
                        track_direction=current_direction, velocity_history=current_track_velocity_history,
                        velocity_direction=current_velocity_direction
                    )
                    object_features.append(current_track_obj_features)
                    if object_feature.idx not in track_based_accumulated_features:
                        track_feats = TrackFeatures(object_feature.idx)
                        track_feats.object_features.append(current_track_obj_features)
                        track_based_accumulated_features.update(
                            {object_feature.idx: track_feats})
                    else:
                        track_based_accumulated_features[object_feature.idx].object_features.append(
                            current_track_obj_features)

                    # activations_future_displacement = flow_for_future[activations[:, 1], activations[:, 0]]
                    # activations_past_displacement = past_flow_yet[activations[:, 1], activations[:, 0]]
                    #
                    # activations_displaced_in_future = activations + activations_future_displacement
                    # activations_displaced_in_past = activations - activations_past_displacement
                    #
                    # shifted_box_in_future, shifted_activation_center_in_future = evaluate_shifted_bounding_box(
                    #     box, activations_displaced_in_future, activations)
                    # shifted_box_in_past, shifted_activation_center_in_past = evaluate_shifted_bounding_box(
                    #     box, activations_displaced_in_past, activations)
                    #
                    # activations_future_frame = extract_features_per_bounding_box(shifted_box_in_future, future_mask)
                    # activations_past_frame = extract_features_per_bounding_box(shifted_box_in_past, past_mask)
                    #
                    # if activations_future_frame.size == 0:
                    #     current_track_obj_features = AgentFeatures(
                    #         track_idx=object_feature.idx,
                    #         activations_t=activations,
                    #         activations_t_minus_one=activations_displaced_in_past,
                    #         activations_t_plus_one=activations_displaced_in_future,
                    #         future_flow=activations_future_displacement,
                    #         past_flow=activations_past_displacement,
                    #         bbox_t=box,
                    #         bbox_t_minus_one=shifted_box_in_past,
                    #         bbox_t_plus_one=shifted_box_in_future,
                    #         frame_number=object_feature.frame_number,
                    #         activations_future_frame=activations_future_frame,
                    #         activations_past_frame=activations_past_frame,
                    #         final_features_future_activations=None,
                    #         is_track_live=False,
                    #         frame_number_t=actual_fr.item(),
                    #         frame_number_t_minus_one=actual_fr.item() - frame_time_gap,
                    #         frame_number_t_plus_one=actual_fr.item() + frame_time_gap,
                    #         past_frames_used_in_of_estimation=list(past_12_frames_optical_flow.keys()),
                    #         future_frames_used_in_of_estimation=list(future_12_frames_optical_flow.keys()),
                    #         frame_by_frame_estimation=True
                    #     )
                    #     object_features.append(current_track_obj_features)
                    #     if object_feature.idx in track_based_accumulated_features:
                    #         track_based_accumulated_features[object_feature.idx].object_features.append(
                    #             current_track_obj_features)
                    #
                    #     continue
                    #
                    # closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                    #     features_filter_append_preprocessing(
                    #         overlap_percent, activations_displaced_in_future, activations_future_frame)
                    #
                    # filtered_shifted_future_activations = filter_features(
                    #     activations_displaced_in_future, closest_n_shifted_xy_pair)
                    # final_features_future_activations = append_features(
                    #     filtered_shifted_future_activations, closest_n_xy_current_frame_pair)
                    #
                    # current_track_obj_features = AgentFeatures(
                    #     track_idx=object_feature.idx,
                    #     activations_t=activations,
                    #     activations_t_minus_one=activations_displaced_in_past,
                    #     activations_t_plus_one=activations_displaced_in_future,
                    #     future_flow=activations_future_displacement,
                    #     past_flow=activations_past_displacement,
                    #     bbox_t=box,
                    #     bbox_t_minus_one=shifted_box_in_past,
                    #     bbox_t_plus_one=shifted_box_in_future,
                    #     frame_number=object_feature.frame_number,
                    #     activations_future_frame=activations_future_frame,
                    #     activations_past_frame=activations_past_frame,
                    #     final_features_future_activations=final_features_future_activations,
                    #     frame_number_t=actual_fr.item(),
                    #     frame_number_t_minus_one=actual_fr.item() - frame_time_gap,
                    #     frame_number_t_plus_one=actual_fr.item() + frame_time_gap,
                    #     past_frames_used_in_of_estimation=list(past_12_frames_optical_flow.keys()),
                    #     future_frames_used_in_of_estimation=list(future_12_frames_optical_flow.keys()),
                    #     frame_by_frame_estimation=True
                    # )
                    # object_features.append(current_track_obj_features)
                    # if object_feature.idx not in track_based_accumulated_features:
                    #     track_feats = TrackFeatures(object_feature.idx)
                    #     track_feats.object_features.append(current_track_obj_features)
                    #     track_based_accumulated_features.update(
                    #         {object_feature.idx: track_feats})
                    # else:
                    #     track_based_accumulated_features[object_feature.idx].object_features.append(
                    #         current_track_obj_features)

                    # plot_tracks_with_features(frame_t=frames[fr],
                    #                           frame_t_minus_one=frames[fr - frame_time_gap],
                    #                           frame_t_plus_one=frames[fr + frame_time_gap],
                    #                           features_t=activations,
                    #                           features_t_minus_one=activations_displaced_in_past,
                    #                           features_t_plus_one=activations_displaced_in_future,
                    #                           box_t=[box],
                    #                           box_t_minus_one=[shifted_box_in_past],
                    #                           box_t_plus_one=[shifted_box_in_future],
                    #                           frame_number=actual_fr.item(),
                    #                           marker_size=1,
                    #                           track_id=object_feature.idx,
                    #                           file_idx=r_idx,
                    #                           save_path=save_path_for_plot + 'frame12apart',
                    #                           annotations=[[object_feature.idx], [object_feature.idx],
                    #                                        [object_feature.idx]],
                    #                           additional_text=f'Past: {actual_fr.item() - frame_time_gap} |'
                    #                                           f' Present: {actual_fr.item()} | '
                    #                                           f'Future: {actual_fr.item() + frame_time_gap}')

                data_all_frames.update({actual_fr.item(): FrameFeatures(frame_number=actual_fr.item(),
                                                                        object_features=object_features)})

                frame_feats = data_all_frames[actual_fr.item()]
                frame_boxes = []
                frame_track_ids = []

                for obj_feature in frame_feats.object_features:
                    frame_boxes.append(obj_feature.bbox_t)
                    frame_track_ids.append(obj_feature.track_idx)

                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                    tracks_histories = []
                    tracks_gt_histories = []

                    for obj_feature in frame_feats.object_features:
                        tracks_histories.extend(obj_feature.track_history)
                        tracks_gt_histories.extend(obj_feature.gt_history)

                    tracks_histories = np.array(tracks_histories)
                    tracks_gt_histories = np.array(tracks_gt_histories)

                    if tracks_histories.size == 0:
                        tracks_histories = np.zeros(shape=(0, 2))
                    if tracks_gt_histories.size == 0:
                        tracks_gt_histories = np.zeros(shape=(0, 2))
                else:
                    tracks_histories = np.zeros(shape=(0, 2))
                    tracks_gt_histories = np.zeros(shape=(0, 2))

                fig = plot_for_video_current_frame(
                    gt_rgb=frames[fr], current_frame_rgb=frames[fr],
                    gt_annotations=current_gt_boxes,
                    current_frame_annotation=frame_boxes,
                    new_track_annotation=[],
                    frame_number=actual_fr.item(),
                    box_annotation=[current_gt_boxes_idx.tolist(), frame_track_ids],
                    generated_track_histories=tracks_histories,
                    gt_track_histories=tracks_gt_histories,
                    additional_text='',
                    # f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                    # f'Recall: {l2_distance_hungarian_recall}\n'
                    # f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                    # f'Precision: {precision} | Recall: {recall}\n'
                    # f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                    # f'Track Ids Killed: '
                    # f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                    video_mode=False, original_dims=original_dims, zero_shot=True, return_figure_only=True)

                current_batch_figures.append(fig)

                # data_all_frames.update({actual_fr.item(): FrameFeatures(frame_number=actual_fr.item(),
                #                                                         object_features=object_features)})

    return data_all_frames, frames[interest_fr:, ...], \
           torch.arange(interest_fr + frame_numbers[0], frame_numbers[-1] + 1), \
           last_frame_from_last_used_batch, past_12_frames_optical_flow, last12_bg_sub_mask, \
           track_based_accumulated_features, current_batch_figures, frame_track_ids


def preprocess_data_zero_shot_12_frames_apart(
        extracted_features, save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None,
        overlap_percent=0.1, radius=50, min_points_in_cluster=5, video_mode=False,
        save_path_for_video=None, plot_scale_factor=1, desired_fps=5, plot=False,
        custom_video_shape=True, save_path_for_plot=None, save_checkpoint=False,
        begin_track_mode=True, generic_box_wh=100, distance_threshold=2,
        use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
        use_is_box_overlapping_live_boxes=True, premature_kill_save=False, save_path_for_features=None,
        save_every_n_batch_itr=None, num_frames_to_build_bg_sub_model=12, drop_last_batch=True,
        frame_by_frame_estimation=False, filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20, angle_threshold_to_filter=120):
    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
    ratio = float(meta_info.flatten()[-1])

    if frame_by_frame_estimation:
        extraction_method = twelve_frame_by_frame_feature_extraction_zero_shot
    else:
        extraction_method = twelve_frames_feature_extraction_zero_shot
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'
    first_frame_bounding_boxes, first_frame_mask, last_frame, second_last_frame = None, None, None, None
    first_frame_live_tracks, last_frame_live_tracks, last_frame_mask = None, None, None
    current_track_idx, track_ids_used = 0, []
    precision_list, recall_list, matching_boxes_with_iou_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []
    meter_tp_list, meter_fp_list, meter_fn_list = [], [], []
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = [], [], []
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = [], [], []
    selected_track_distances = []
    total_accumulated_features = {}
    track_based_accumulated_features = {}

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(save_path_for_video, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))  # (1200, 1000))  # (video_shape[0], video_shape[1]))
            # (video_shape[1], video_shape[0]))
            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(save_path_for_video, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))  # (1200, 1000))  # (video_shape[0], video_shape[1]))

    remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch = None, None, None
    past_12_frames_optical_flow, last_itr_past_12_frames_optical_flow, last_itr_past_12_bg_sub_mask = {}, {}, {}
    last_optical_flow_map, last12_bg_sub_mask = None, {}
    part_idx = 0

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]

            features_, remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch, \
            past_12_frames_optical_flow, last12_bg_sub_mask, track_based_accumulated_features, figures, \
            last_frame_live_tracks = \
                extraction_method(
                    extracted_features=extracted_features,
                    frames=frames, n=30,
                    frames_to_build_model=num_frames_to_build_bg_sub_model,
                    var_threshold=None, frame_numbers=frame_numbers,
                    remaining_frames=remaining_frames,
                    remaining_frames_idx=remaining_frames_idx,
                    past_12_frames_optical_flow=past_12_frames_optical_flow,
                    last_frame_from_last_used_batch=
                    last_frame_from_last_used_batch,
                    last12_bg_sub_mask=last12_bg_sub_mask,
                    track_based_accumulated_features=track_based_accumulated_features,
                    save_path_for_plot=save_path_for_plot,
                    df=df, ratio=ratio, original_dims=original_dims,
                    filter_switch_boxes_based_on_angle_and_recent_history=
                    filter_switch_boxes_based_on_angle_and_recent_history,
                    compute_histories_for_plot=compute_histories_for_plot,
                    min_track_length_to_filter_switch_box=min_track_length_to_filter_switch_box,
                    angle_threshold_to_filter=angle_threshold_to_filter,
                    resume_mode=False)  # fixme: remove??
            total_accumulated_features = {**total_accumulated_features, **features_}
            if video_mode:
                for fig in figures:
                    canvas = FigureCanvas(fig)
                    canvas.draw()

                    buf = canvas.buffer_rgba()
                    out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                    if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                        out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                        out_frame = (out_frame * 255).astype(np.uint8)
                    # out_frame = out_frame.reshape(1200, 1000, 3)
                    out.write(out_frame)
            if save_every_n_batch_itr is not None:
                save_dict = {
                    'total_accumulated_features': total_accumulated_features,
                    'track_based_accumulated_features': track_based_accumulated_features
                }
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0:
                    Path(save_path_for_features + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part{part_idx}.pt'
                    torch.save(save_dict, save_path_for_features + 'parts/' + f_n)

                    total_accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_entries_from_dict(live_track_ids,
                                                                                track_based_accumulated_features)
    except KeyboardInterrupt:
        save_dict = {
            'total_accumulated_features': total_accumulated_features,
            'track_based_accumulated_features': track_based_accumulated_features
        }
        Path(save_path_for_features).mkdir(parents=True, exist_ok=True)
        f_n = f'features_dict_part{part_idx}_cancelled.pt'
        torch.save(save_dict, save_path_for_features + 'parts/' + f_n)
        out.release()
    finally:
        save_dict = {
            'total_accumulated_features': total_accumulated_features,
            'track_based_accumulated_features': track_based_accumulated_features
        }
        Path(save_path_for_features).mkdir(parents=True, exist_ok=True)
        f_n = f'features_dict_from_finally.pt'
        torch.save(save_dict, save_path_for_features + f_n)
        out.release()

    if video_mode:
        out.release()
    return total_accumulated_features


def combine_features_generate_annotations(files_base_path, files_list, csv_save_path, min_track_length_threshold=5,
                                          do_filter=True, low_memory_mode=False):
    part_features: Dict[Any, Any] = torch.load(files_base_path + files_list[0])
    track_based_accumulated_features: Dict[int, TrackFeatures] = \
        part_features['track_based_accumulated_features']
    frame_based_accumulated_features: Dict[int, FrameFeatures] = \
        part_features['accumulated_features']

    logger.info('Combining Features')
    for p_idx in tqdm(range(1, len(files_list))):
        part_features_temp: Dict[Any, Any] = torch.load(files_base_path + files_list[p_idx])
        track_based_accumulated_features_temp: Dict[int, TrackFeatures] = \
            part_features_temp['track_based_accumulated_features']
        frame_based_accumulated_features_temp: Dict[int, FrameFeatures] = \
            part_features_temp['accumulated_features']

        frame_based_accumulated_features = {**frame_based_accumulated_features,
                                            **frame_based_accumulated_features_temp}

        same_keys = np.intersect1d(list(track_based_accumulated_features.keys()),
                                   list(track_based_accumulated_features_temp.keys())).tolist()
        for s_key in same_keys:
            len_0 = len(track_based_accumulated_features[s_key].object_features)
            len_1 = len(track_based_accumulated_features_temp[s_key].object_features)

            track_based_accumulated_features[s_key].object_features.extend(
                track_based_accumulated_features_temp[s_key].object_features)

            del track_based_accumulated_features_temp[s_key]
            assert s_key not in track_based_accumulated_features_temp.keys() \
                   and s_key in track_based_accumulated_features.keys()
            assert len(track_based_accumulated_features[s_key].object_features) == len_0 + len_1

        track_based_accumulated_features = {**track_based_accumulated_features, **track_based_accumulated_features_temp}
    logger.info('All features combined! Next Step Filtering')
    extracted_features_in_csv(track_based_features=track_based_accumulated_features,
                              frame_based_features=frame_based_accumulated_features,
                              do_filter=do_filter,
                              min_track_length_threshold=min_track_length_threshold,
                              csv_save_path=csv_save_path,
                              low_memory_mode=low_memory_mode)
    logger.info('Annotation extraction completed!')


def combine_features_generate_annotations_v2(files_base_path, files_list, csv_save_path, min_track_length_threshold=5,
                                             do_filter=True, low_memory_mode=False,
                                             csv_filename='generated_annotations.csv'):
    annotation_data = []
    part_features_0: Dict[Any, Any] = torch.load(files_base_path + files_list[0])
    track_based_accumulated_features_0: Dict[int, TrackFeatures] = \
        part_features_0['track_based_accumulated_features']
    frame_based_accumulated_features_0: Dict[int, FrameFeatures] = \
        part_features_0['accumulated_features']

    if len(files_list) == 1:
        annotation_data.extend(extracted_features_in_csv(track_based_features=track_based_accumulated_features_0,
                                                         frame_based_features=frame_based_accumulated_features_0,
                                                         do_filter=do_filter,
                                                         min_track_length_threshold=min_track_length_threshold,
                                                         csv_save_path=csv_save_path,
                                                         low_memory_mode=low_memory_mode,
                                                         return_list=True,
                                                         track_ids_to_skip=[],
                                                         csv_filename=csv_filename))
    else:
        logger.info('Combining Features')
        for p_idx in tqdm(range(1, len(files_list))):
            part_features_temp_1: Dict[Any, Any] = torch.load(files_base_path + files_list[p_idx])
            track_based_accumulated_features_1: Dict[int, TrackFeatures] = \
                part_features_temp_1['track_based_accumulated_features']
            frame_based_accumulated_features_1: Dict[int, FrameFeatures] = \
                part_features_temp_1['accumulated_features']

            same_keys = np.intersect1d(list(track_based_accumulated_features_0.keys()),
                                       list(track_based_accumulated_features_1.keys())).tolist()

            keys_to_skip_during_filtering = []
            for s_key in same_keys:
                len_0 = len(track_based_accumulated_features_0[s_key].object_features)
                len_1 = len(track_based_accumulated_features_1[s_key].object_features)

                if len_0 < min_track_length_threshold < len_0 + len_1:
                    # if track length is lower than threshold now but it lasts longer actually don't filter it
                    keys_to_skip_during_filtering.append(s_key)

                # if track length is longer than threshold, it should be longer in next batch
                # since we keep the live tracks during save, remove the one already seen
                # temp_list_0 = track_based_accumulated_features_0[s_key].object_features
                # temp_list_1 = track_based_accumulated_features_1[s_key].object_features

                track_based_accumulated_features_1[s_key].object_features = \
                    track_based_accumulated_features_1[s_key].object_features[len_0:]

                # if track length is smaller in both dicts, filter it out
                assert len(track_based_accumulated_features_1[s_key].object_features) == len_1 - len_0

            annotation_data.extend(extracted_features_in_csv(track_based_features=track_based_accumulated_features_0,
                                                             frame_based_features=frame_based_accumulated_features_0,
                                                             do_filter=do_filter,
                                                             min_track_length_threshold=min_track_length_threshold,
                                                             csv_save_path=csv_save_path,
                                                             low_memory_mode=low_memory_mode,
                                                             return_list=True,
                                                             track_ids_to_skip=keys_to_skip_during_filtering,
                                                             csv_filename=csv_filename))

            track_based_accumulated_features_0 = copy.deepcopy(track_based_accumulated_features_1)
            frame_based_accumulated_features_0 = copy.deepcopy(frame_based_accumulated_features_1)

    logger.info('Annotation extraction completed!')
    df = pd.DataFrame(data=annotation_data, columns=[
        'track_id', 'x_min', 'y_min', 'x_max', 'y_max', 'frame_number', 'label',
        'center_x', 'center_y', 'gt_x_min', 'gt_y_min', 'gt_x_max', 'gt_y_max',
        'gt_center_x', 'gt_center_y'])

    if csv_save_path is not None:
        df.to_csv(csv_save_path + csv_filename, index=False)
        logger.info('CSV saved!')


def visualize_annotations(annotations_df, batch_size=32, drop_last_batch=True, custom_video_shape=False,
                          plot_scale_factor=1, video_mode=True, save_path_for_video=None, desired_fps=5):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(save_path_for_video, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))
            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(save_path_for_video, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))

    try:
        for p_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]

            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                gt_frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                gt_annotations, gt_bbox_centers = scale_annotations(gt_frame_annotation,
                                                                    original_scale=original_shape,
                                                                    new_scale=new_shape, return_track_id=False,
                                                                    tracks_with_annotations=True)
                gt_boxes = gt_annotations[:, :-1]
                gt_track_idx = gt_annotations[:, -1]

                generated_frame_annotation = get_generated_frame_annotations(annotations_df, frame_number.item())
                generated_boxes = generated_frame_annotation[:, 1:5]
                generated_track_idx = generated_frame_annotation[:, 0]

                if video_mode:
                    fig = plot_for_video_current_frame(
                        gt_rgb=frame, current_frame_rgb=frame,
                        gt_annotations=gt_boxes,
                        current_frame_annotation=generated_boxes,
                        new_track_annotation=[],
                        frame_number=frame_number,
                        box_annotation=[gt_track_idx, generated_track_idx],
                        generated_track_histories=None,
                        gt_track_histories=None,
                        additional_text='',
                        video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                    canvas = FigureCanvas(fig)
                    canvas.draw()

                    buf = canvas.buffer_rgba()
                    out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                    if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                        out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                        out_frame = (out_frame * 255).astype(np.uint8)
                    out.write(out_frame)
                else:
                    fig = plot_for_video_current_frame(
                        gt_rgb=frame, current_frame_rgb=frame,
                        gt_annotations=gt_boxes,
                        current_frame_annotation=generated_boxes,
                        new_track_annotation=[],
                        frame_number=frame_number,
                        additional_text='',
                        video_mode=False, original_dims=original_dims, zero_shot=True)
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
    finally:
        if video_mode:
            out.release()
    logger.info('Finished writing video!')


def preprocess_data_zero_shot_minimal_with_timeout_and_classifier(
        save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
        radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
        plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
        save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
        use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
        use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
        distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
        detect_shadows=True, allow_only_for_first_frame_use_is_box_overlapping_live_boxes=True,
        filter_switch_boxes_based_on_angle_and_recent_history=True,
        compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
        angle_threshold_to_filter=120):
    clustering_failed = False
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    if save_per_part_path is not None:
        save_per_part_path += 'parts/'
    first_frame_bounding_boxes, first_frame_mask, last_frame, second_last_frame = None, None, None, None
    first_frame_live_tracks, last_frame_live_tracks, last_frame_mask = None, None, None
    current_track_idx, track_ids_used = 0, []
    precision_list, recall_list, matching_boxes_with_iou_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []
    meter_tp_list, meter_fp_list, meter_fn_list = [], [], []
    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = [], [], []
    center_inside_tp_list, center_inside_fp_list, center_inside_fn_list = [], [], []
    selected_track_distances = []
    accumulated_features = {}
    track_based_accumulated_features: Dict[int, TrackFeatures] = {}
    last_frame_gt_tracks = {}
    ground_truth_track_histories = []

    # setup classifier
    logger.info(f'Setting up model...')

    if OC_USE_RESNET:
        conv_layers = resnet18(pretrained=OC_USE_PRETRAINED) \
            if not OC_SMALLER_RESNET else resnet9(pretrained=OC_USE_PRETRAINED,
                                                  first_in_channel=3,
                                                  first_stride=2,
                                                  first_padding=1)
    else:
        conv_layers = make_conv_blocks(3, [8, 16, 32, 64], [3, 3, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1],
                                       False, non_lin=Activations.RELU, dropout=0.0)
    classifier_layers = make_classifier_block(512, [256, 32, 8, 1], Activations.RELU)

    model = PersonClassifier(conv_block=conv_layers, classifier_block=classifier_layers,
                             train_dataset=None, val_dataset=None, batch_size=OC_BATCH_SIZE,
                             num_workers=OC_NUM_WORKERS, shuffle=False,
                             pin_memory=False, lr=1e-4, collate_fn=people_collate_fn,
                             hparams={})
    checkpoint_path = f'{OC_CHECKPOINT_PATH}{OC_CHECKPOINT_VERSION}/checkpoints/'
    checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]
    load_dict = torch.load(checkpoint_file)

    model.load_state_dict(load_dict['state_dict'])
    model.to(OC_DEVICE)
    model.eval()

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))

            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))

    try:
        for part_idx, data in enumerate(tqdm(data_loader)):
            frames, frame_numbers = data
            frames = frames.squeeze()
            frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            frames_count = frames.shape[0]
            original_shape = new_shape = [frames.shape[1], frames.shape[2]]
            for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                         total=len(frame_numbers)):
                if (part_idx == 0 and frame_idx == 0) or clustering_failed:
                    # STEP 1: a> Get GT for the first frame
                    validation_annotations, first_frame_mask = first_frame_processing_and_gt_association(
                        df, first_frame_mask,
                        frame_idx, frame_number,
                        frames, frames_count,
                        kernel, n, new_shape,
                        original_shape, step,
                        var_threshold, detect_shadows=detect_shadows)

                    running_tracks, new_track_boxes = [], []
                    all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                        None, first_frame_mask, radius, running_tracks, plot=False)

                    all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                    features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                    if features_skipped_idx.size != 0:
                        features_skipped = all_cloud[features_skipped_idx]

                        try:
                            # STEP 4h: b> cluster to group points
                            mean_shift, n_clusters = mean_shift_clustering_with_timeout(
                                features_skipped, bin_seeding=False, min_bin_freq=8,
                                cluster_all=True, bandwidth=4, max_iter=100)
                            cluster_centers = mean_shift.cluster_centers

                            # STEP 4h: c> prune cluster centers
                            # combine centers inside radius + eliminate noise
                            final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                cluster_centers, mean_shift, radius + extra_radius,
                                min_points_in_cluster=min_points_in_cluster)

                            if final_cluster_centers.size != 0:
                                t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                # STEP 4h: d> start new potential tracks
                                for cluster_center, cluster_center_idx in \
                                        zip(final_cluster_centers, final_cluster_centers_idx):
                                    cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)

                                    # flexible_box, points_in_current_cluster = calculate_flexible_bounding_box(
                                    #     cluster_center_idx, cluster_center_x, cluster_center_y, mean_shift)

                                    # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                    t_box = torchvision.ops.box_convert(
                                        torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                        'cxcywh', 'xyxy').int().numpy()
                                    # Note: Do not start track if bbox is out of frame
                                    if allow_only_for_first_frame_use_is_box_overlapping_live_boxes and \
                                            not (np.sign(t_box) < 0).any() and \
                                            not is_box_overlapping_live_boxes(t_box, [t.bbox for t in running_tracks]):
                                        # NOTE: the second check might result in killing potential tracks!
                                        t_id = max(track_ids_used) + 1 if len(track_ids_used) else 0
                                        running_tracks.append(Track(bbox=t_box, idx=t_id))
                                        track_ids_used.append(t_id)
                                        new_track_boxes.append(t_box)
                                    elif use_is_box_overlapping_live_boxes:
                                        if not (np.sign(t_box) < 0).any() and \
                                                not is_box_overlapping_live_boxes(t_box,
                                                                                  [t.bbox for t in running_tracks]):
                                            # NOTE: the second check might result in killing potential tracks!
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)
                                    else:
                                        if not (np.sign(t_box) < 0).any():
                                            t_id = max(track_ids_used) + 1
                                            running_tracks.append(Track(bbox=t_box, idx=t_id))
                                            track_ids_used.append(t_id)
                                            new_track_boxes.append(t_box)

                            clustering_failed = False
                            logger.info('Clustering worked out! Moving on to phase 2!')
                        except TimeoutException:
                            clustering_failed = True
                            logger.info('Clustering took too much time for first/last frame, trying next now')

                        # plot_features_with_circles(
                        #     all_cloud, features_covered, features_skipped, first_frame_mask, marker_size=8,
                        #     cluster_centers=final_cluster_centers, num_clusters=final_cluster_centers.shape[0],
                        #     frame_number=frame_number, boxes=validation_annotations[:, :-1],
                        #     radius=radius+extra_radius,
                        #     additional_text=
                        #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                        #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    if not clustering_failed:
                        new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(
                            shape=(0,))

                        r_boxes = [b.bbox for b in running_tracks]
                        r_boxes_idx = [b.idx for b in running_tracks]
                        select_track_idx = 4

                        r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                        r_boxes_idx = torch.tensor(r_boxes_idx, dtype=torch.int)

                        # classify patches
                        generated_boxes_xywh = torchvision.ops.box_convert(r_boxes, 'xyxy', 'xywh')
                        generated_boxes_xywh = [torch.tensor((b[1], b[0], b[2] + OC_ADDITIONAL_CROP_H,
                                                              b[3] + OC_ADDITIONAL_CROP_W)) for b in
                                                generated_boxes_xywh]
                        try:
                            generated_boxes_xywh = torch.stack(generated_boxes_xywh)

                            generated_crops = [tvf.crop(torch.from_numpy(frame).permute(2, 0, 1),
                                                        top=b[0], left=b[1], width=b[2], height=b[3])
                                               for b in generated_boxes_xywh]
                            generated_crops_resized = [tvf.resize(c, [generic_box_wh, generic_box_wh])
                                                       for c in generated_crops if c.shape[1] != 0 and c.shape[2] != 0]
                            generated_valid_boxes = [c_i for c_i, c in enumerate(generated_crops)
                                                     if c.shape[1] != 0 and c.shape[2] != 0]
                            generated_boxes_xywh = generated_boxes_xywh[generated_valid_boxes]
                            r_boxes_idx = r_boxes_idx[generated_valid_boxes]
                            r_boxes = r_boxes[generated_valid_boxes]
                            generated_crops_resized = torch.stack(generated_crops_resized)
                            generated_crops_resized = (generated_crops_resized.float() / 255.0).to(OC_DEVICE)

                            # plot
                            if plot:
                                show_image_with_crop_boxes(frame,
                                                           [], generated_boxes_xywh, xywh_mode_v2=False,
                                                           xyxy_mode=False,
                                                           title='xywh')
                                gt_crops_grid = torchvision.utils.make_grid(generated_crops_resized)
                                plt.imshow(gt_crops_grid.cpu().permute(1, 2, 0))
                                plt.show()

                            with torch.no_grad():
                                patch_predictions = model(generated_crops_resized)

                            pred_labels = torch.round(torch.sigmoid(patch_predictions))

                            valid_boxes_idx = (pred_labels > 0.5).squeeze().cpu()

                            if valid_boxes_idx.ndim == 0:
                                if valid_boxes_idx.item():
                                    valid_boxes_idx = 0
                                    valid_boxes = generated_boxes_xywh[valid_boxes_idx]
                                    invalid_boxes = []

                                    valid_track_idx = [r_boxes_idx[valid_boxes_idx]]
                                    invalid_track_idx = []
                                    valid_generated_boxes = np.expand_dims(r_boxes[valid_boxes_idx], 0)
                                else:
                                    valid_boxes_idx = 0
                                    valid_boxes = []
                                    invalid_boxes = generated_boxes_xywh[valid_boxes_idx]

                                    valid_track_idx = []
                                    invalid_track_idx = [r_boxes_idx[valid_boxes_idx]]
                                    valid_generated_boxes = np.array([])

                                # valid_generated_boxes = np.expand_dims(r_boxes[valid_boxes_idx], 0)
                            else:
                                valid_boxes = generated_boxes_xywh[valid_boxes_idx]
                                invalid_boxes = generated_boxes_xywh[~valid_boxes_idx]

                                # plot removed boxes
                                if plot:
                                    show_image_with_crop_boxes(frame,
                                                               invalid_boxes, valid_boxes, xywh_mode_v2=False,
                                                               xyxy_mode=False,
                                                               title='xywh')

                                valid_track_idx = r_boxes_idx[valid_boxes_idx]
                                invalid_track_idx = r_boxes_idx[~valid_boxes_idx]
                                valid_generated_boxes = r_boxes[valid_boxes_idx]
                        except RuntimeError:
                            valid_generated_boxes, valid_track_idx = np.array([]), np.array([])

                        running_tracks = [r for r in running_tracks if r.idx in valid_track_idx]

                        a_boxes = torch.from_numpy(validation_annotations[:, :-1])
                        # Fixme: If no boxes in first frame!!
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes_threshold = iou_boxes.copy()
                        iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                        # TODO: Replace with Hungarian
                        iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                        iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                        match_idx = np.where(iou_boxes)
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]

                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        tp_list.append(tp)
                        fp_list.append(fp)
                        fn_list.append(fn)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
                        precision_list.append(precision)
                        recall_list.append(recall)
                        matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                        last_frame_live_tracks = []

                        fig = plot_for_video(
                            gt_rgb=frame, gt_mask=first_frame_mask, last_frame_rgb=frame,
                            last_frame_mask=first_frame_mask, current_frame_rgb=frame,
                            current_frame_mask=first_frame_mask, gt_annotations=validation_annotations[:, :-1],
                            last_frame_annotation=last_frame_live_tracks,
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False,
                            save_path=f'{plot_save_path}zero_shot/plots{min_points_in_cluster}/')

                        # STEP 1: b> Store them for the next ts and update these variables in further iterations
                        last_frame = frame.copy()
                        second_last_frame = last_frame.copy()
                        last_frame_live_tracks = running_tracks
                        last_frame_mask = first_frame_mask.copy()
                        last_frame_gt_tracks = {a[-1]: a[:-1] for a in validation_annotations}
                else:
                    running_tracks, object_features = [], []
                    # STEP 2: Get the OF for both ((t-1), t) and ((t), (t+1))
                    flow, past_flow = optical_flow_processing(frame, last_frame, second_last_frame)

                    # STEP 3: Get Background Subtracted foreground mask for the current frame
                    fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                       time_gap_within_frames=3,
                                                       total_frames=frames_count, step=step, n=n,
                                                       kernel=kernel, var_threshold=var_threshold,
                                                       detect_shadows=detect_shadows)

                    # Note: Only for validation purposes
                    # just for validation #####################################################################
                    frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                    annotations, bbox_centers = scale_annotations(frame_annotation,
                                                                  original_scale=original_shape,
                                                                  new_scale=new_shape, return_track_id=False,
                                                                  tracks_with_annotations=True)
                    ###########################################################################################

                    # STEP 4: For each live track
                    for b_idx, track in enumerate(last_frame_live_tracks):
                        current_track_idx, box = track.idx, track.bbox
                        current_track_features = track_based_accumulated_features[
                            current_track_idx].object_features[-1] \
                            if current_track_idx in track_based_accumulated_features.keys() else []
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        # if frame_number == 110:
                        #     print()

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
                            continue

                        # STEP 4b: calculate flow for the features
                        xy_displacement = flow[xy[:, 1], xy[:, 0]]
                        past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]

                        # STEP 4c: shift the features and bounding box by the average flow for localization
                        shifted_xy = xy + xy_displacement
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # STEP 4c: 1> Switch boxes: Idea 1 - Only keep points inside box
                        shifted_xy = find_points_inside_box(shifted_xy, shifted_box)
                        shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(box, shifted_xy, xy)
                        # box_center_diff = calculate_difference_between_centers(box, shifted_box)

                        # STEP 4d: extract features for the current track in the next time-step
                        # get activations
                        xy_current_frame = extract_features_per_bounding_box(shifted_box, fg_mask)

                        # check with model
                        shifted_box_xywh = torchvision.ops.box_convert(torch.from_numpy(shifted_box).unsqueeze(0)
                                                                       , 'xyxy', 'xywh')
                        shifted_box_xywh = [torch.tensor((b[1], b[0], b[2] + OC_ADDITIONAL_CROP_H,
                                                          b[3] + OC_ADDITIONAL_CROP_W)) for b in
                                            shifted_box_xywh]
                        shifted_crop = [tvf.crop(torch.from_numpy(frame).permute(2, 0, 1),
                                                 top=b[0], left=b[1], width=b[2], height=b[3])
                                        for b in shifted_box_xywh]
                        valid_shifted_crop = [c_i for c_i, c in enumerate(shifted_crop)
                                              if c.shape[1] != 0 and c.shape[2] != 0]
                        if len(valid_shifted_crop) != 0:
                            shifted_crop_resized = [tvf.resize(c, [generic_box_wh, generic_box_wh])
                                                    for c in shifted_crop if c.shape[1] != 0 and c.shape[2] != 0]
                            shifted_crop_resized = torch.stack(shifted_crop_resized)
                            shifted_crop_resized = (shifted_crop_resized.float() / 255.0).to(OC_DEVICE)

                            with torch.no_grad():
                                shifted_box_patch_prediction = model(shifted_crop_resized)

                            shifted_box_patch_label = torch.round(torch.sigmoid(shifted_box_patch_prediction))

                            if not shifted_box_patch_label.bool().item():
                                logger.info(f'No object according to object classifier!')
                        else:
                            logger.info(
                                f'No object according to object classifier in the shifted box'
                                f' - box out of frame!')
                            shifted_box_patch_label = torch.tensor([1])

                        # or to and for shifted_box_patch_label.bool().item()?
                        if xy_current_frame.size == 0 \
                                or not shifted_box_patch_label.bool().item() \
                                or (filter_switch_boxes_based_on_angle_and_recent_history
                                    and not isinstance(current_track_features, list)
                                    and current_track_features.velocity_direction.size != 0
                                    and len(current_track_features.velocity_direction) >
                                    min_track_length_to_filter_switch_box
                                    and first_violation_till_now(
                                    current_track_features.velocity_direction, angle_threshold_to_filter)
                                    and current_track_features.velocity_direction[-1] >
                                    angle_threshold_to_filter
                                    and not np.isnan(
                                    current_track_features.velocity_direction[-1])):

                            # STEP 4e: a> if no feature detected inside bounding box
                            #  -> put a circle of radius N pixels around the center of the shifted bounding box
                            #  -> if features detected inside this circle
                            #  -> shift the bounding box there then, throw and take 80% of the points
                            #  -> keep the track alive
                            logger.info(f'Trying to revive track: {current_track_idx}!')

                            shifted_box_center = get_bbox_center(shifted_box).flatten()
                            # all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                            #                                                                    shifted_xy_center)
                            all_cloud, features_inside_circle = extract_features_inside_circle(fg_mask, radius,
                                                                                               shifted_box_center)
                            if features_inside_circle.size != 0 and use_circle_to_keep_track_alive:
                                shifted_box, shifted_xy_center = evaluate_shifted_bounding_box(
                                    box=shifted_box, shifted_xy=features_inside_circle, xy=shifted_xy)
                                xy_current_frame = features_inside_circle.copy()
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                            else:
                                # plot_features_with_mask(all_cloud, features_inside_circle, center=shifted_xy_center,
                                #                         radius=radius, mask=fg_mask, box=shifted_box, m_size=1,
                                #                         current_boxes=annotations[:, :-1])
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
                                if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                    current_track_history = get_track_history(current_track_idx,
                                                                              track_based_accumulated_features)
                                    current_track_velocity_history = get_track_velocity_history(
                                        current_track_idx, track_based_accumulated_features)
                                    current_track_velocity_history = np.array(current_track_velocity_history)
                                    current_direction = []
                                    for track_history_idx in range(len(current_track_history) - 1):
                                        current_direction.append((angle_between(
                                            v1=current_track_history[track_history_idx],
                                            v2=current_track_history[track_history_idx + 1]
                                        )))
                                    # current direction can be removed
                                    current_direction = np.array(current_direction)

                                    current_velocity_direction = []
                                    for track_history_idx in range(len(current_track_velocity_history) - 1):
                                        current_velocity_direction.append(math.degrees(angle_between(
                                            v1=current_track_velocity_history[track_history_idx],
                                            v2=current_track_velocity_history[track_history_idx + 1]
                                        )))
                                    current_velocity_direction = np.array(current_velocity_direction)

                                    # not really required ############################################################
                                    # if len(current_track_history) != 0:
                                    #     current_running_velocity = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[-1], axis=0) -
                                    #         np.expand_dims(current_track_history[0], axis=0),
                                    #         2, axis=0
                                    #     ) / len(current_track_history) / 30
                                    # else:
                                    #     current_running_velocity = None
                                    #
                                    # current_per_step_distance = []
                                    # for track_history_idx in range(len(current_track_history) - 1):
                                    #     d = np.linalg.norm(
                                    #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                    #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                                    #         2, axis=0
                                    #     )
                                    #     current_per_step_distance.append(d)
                                    #
                                    # current_per_step_distance = np.array(current_per_step_distance)
                                    ###################################################################################

                                    # track_sign = []
                                    # for t in range(
                                    #         len(track_based_accumulated_features[8].object_features[-1].track_history)
                                    #         - 1):
                                    #     track_sign.append(np.sign(
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t + 1] -
                                    #         track_based_accumulated_features[8].object_features[-1].track_history[t]))

                                    # just use gt
                                    current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                    track_based_accumulated_features)
                                else:
                                    current_track_history, current_gt_track_history = None, None
                                    current_direction, current_velocity_direction = None, None
                                    current_track_velocity_history = None
                                    # not really required ############################################################
                                    current_per_step_distance = None
                                    current_running_velocity = None
                                    ###################################################################################
                                current_track_obj_features = MinimalObjectFeatures(
                                    idx=current_track_idx,
                                    history=current_track_history,
                                    gt_history=current_gt_track_history,
                                    track_direction=current_direction,
                                    velocity_direction=
                                    current_velocity_direction,
                                    velocity_history=
                                    current_track_velocity_history,
                                    flow=xy_displacement,
                                    past_flow=past_xy_displacement,
                                    past_bbox=box,
                                    final_bbox=np.array(shifted_box),
                                    is_track_live=False,
                                    frame_number=frame_number.item())
                                object_features.append(current_track_obj_features)
                                if current_track_idx in track_based_accumulated_features:
                                    track_based_accumulated_features[current_track_idx].object_features.append(
                                        current_track_obj_features)

                                logger.info(f'Killing track: {current_track_idx}!')
                                continue

                        # STEP 4f: compare activations to keep and throw - throw N% and keep N%
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?
                        final_shifted_box, final_shifted_xy_center = evaluate_shifted_bounding_box(shifted_box,
                                                                                                   final_features_xy,
                                                                                                   shifted_xy)

                        # check with model
                        final_shifted_box_xywh = torchvision.ops.box_convert(
                            torch.from_numpy(final_shifted_box).unsqueeze(0), 'xyxy', 'xywh')
                        final_shifted_box_xywh = [torch.tensor((b[1], b[0], b[2] + OC_ADDITIONAL_CROP_H,
                                                                b[3] + OC_ADDITIONAL_CROP_W)) for b in
                                                  final_shifted_box_xywh]
                        final_shifted_crop = [tvf.crop(torch.from_numpy(frame).permute(2, 0, 1),
                                                       top=b[0], left=b[1], width=b[2], height=b[3])
                                              for b in final_shifted_box_xywh]
                        valid_final_shifted_crop = [c_i for c_i, c in enumerate(final_shifted_crop)
                                              if c.shape[1] != 0 and c.shape[2] != 0]
                        if len(valid_final_shifted_crop) != 0:
                            final_shifted_crop_resized = [tvf.resize(c, [generic_box_wh, generic_box_wh])
                                                          for c in final_shifted_crop if
                                                          c.shape[1] != 0 and c.shape[2] != 0]
                            final_shifted_crop_resized = torch.stack(final_shifted_crop_resized)
                            final_shifted_crop_resized = (final_shifted_crop_resized.float() / 255.0).to(OC_DEVICE)

                            with torch.no_grad():
                                final_shifted_box_patch_prediction = model(final_shifted_crop_resized)

                            final_shifted_box_patch_label = torch.round(
                                torch.sigmoid(final_shifted_box_patch_prediction))
                        else:
                            logger.info(
                                f'No object according to object classifier in the final shifted box'
                                f' - box out of frame!')
                            final_shifted_box_patch_label = torch.tensor([1])

                        if final_shifted_box_patch_label.bool().item():
                            running_tracks.append(Track(bbox=final_shifted_box, idx=current_track_idx))

                            # if not (final_shifted_box == shifted_box).all():
                            #     logger.warn('Final Shifted Box differs from Shifted Box!')

                            if plot:
                                plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                                      shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                                      xy_cloud_current_frame=xy_current_frame,
                                                      frame_number=frame_number.item(),
                                                      track_id=current_track_idx,
                                                      selected_past=closest_n_shifted_xy_pair,
                                                      selected_current=closest_n_xy_current_frame_pair)

                            if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                                current_track_history = get_track_history(current_track_idx,
                                                                          track_based_accumulated_features)
                                current_track_velocity_history = get_track_velocity_history(
                                    current_track_idx, track_based_accumulated_features)
                                current_track_velocity_history = np.array(current_track_velocity_history)
                                current_direction = []
                                for track_history_idx in range(len(current_track_history) - 1):
                                    current_direction.append((angle_between(
                                        v1=current_track_history[track_history_idx],
                                        v2=current_track_history[track_history_idx + 1]
                                    )))
                                # current direction can be removed
                                current_direction = np.array(current_direction)

                                current_velocity_direction = []
                                for track_history_idx in range(len(current_track_velocity_history) - 1):
                                    current_velocity_direction.append(math.degrees(angle_between(
                                        v1=current_track_velocity_history[track_history_idx],
                                        v2=current_track_velocity_history[track_history_idx + 1]
                                    )))
                                current_velocity_direction = np.array(current_velocity_direction)

                                current_gt_track_history = get_gt_track_history(current_track_idx,
                                                                                track_based_accumulated_features)

                                # not really required ############################################################
                                # if len(current_track_history) != 0:
                                #     current_running_velocity = np.linalg.norm(
                                #         np.expand_dims(current_track_history[-1], axis=0) -
                                #         np.expand_dims(current_track_history[0], axis=0),
                                #         2, axis=0
                                #     ) / len(current_track_history) / 30
                                # else:
                                #     current_running_velocity = None
                                #
                                # current_per_step_distance = []
                                # for track_history_idx in range(len(current_track_history) - 1):
                                #     d = np.linalg.norm(
                                #         np.expand_dims(current_track_history[track_history_idx + 1], axis=0) -
                                #         np.expand_dims(current_track_history[track_history_idx], axis=0),
                                #         2, axis=0
                                #     )
                                #     current_per_step_distance.append(d)
                                #
                                # current_per_step_distance = np.array(current_per_step_distance)
                                ###################################################################################
                            else:
                                current_track_history, current_gt_track_history = None, None
                                current_direction, current_velocity_direction = None, None
                                current_track_velocity_history = None
                                # not really required ############################################################
                                current_per_step_distance = None
                                current_running_velocity = None
                                ###################################################################################
                            # STEP 4g: save the information gathered
                            current_track_obj_features = MinimalObjectFeatures(
                                idx=current_track_idx,
                                history=current_track_history,
                                gt_history=current_gt_track_history,
                                track_direction=current_direction,
                                velocity_direction=current_velocity_direction,
                                velocity_history=current_track_velocity_history,
                                flow=xy_displacement,
                                past_flow=past_xy_displacement,
                                past_bbox=box,
                                final_bbox=np.array(final_shifted_box),
                                frame_number=frame_number.item())
                            object_features.append(current_track_obj_features)
                            if current_track_idx not in track_based_accumulated_features:
                                track_feats = TrackFeatures(current_track_idx)
                                track_feats.object_features.append(current_track_obj_features)
                                track_based_accumulated_features.update(
                                    {current_track_idx: track_feats})
                            else:
                                track_based_accumulated_features[current_track_idx].object_features.append(
                                    current_track_obj_features)
                            if current_track_idx not in track_ids_used:
                                track_ids_used.append(current_track_idx)
                        else:
                            logger.info(f'No object according to object classifier in final shifted box!')

                    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                    try:
                        ratio = float(meta_info.flatten()[-1])
                    except IndexError:
                        ratio = 1.0

                    # NOTE: running ADE/FDE
                    r_boxes = [b.bbox for b in running_tracks]
                    r_boxes_idx = [b.idx for b in running_tracks]
                    select_track_idx = 4

                    r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                    a_boxes = torch.from_numpy(annotations[:, :-1])
                    a_boxes_idx = torch.from_numpy(annotations[:, -1])
                    try:
                        iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                    except IndexError:
                        if a_boxes.ndim < 2:
                            a_boxes = a_boxes.unsqueeze(0)
                        if r_boxes.ndim < 2:
                            r_boxes = r_boxes.unsqueeze(0)
                        logger.info(f'a_boxes -> ndim: {a_boxes.ndim}, shape: {a_boxes.shape}')
                        logger.info(f'r_boxes -> ndim: {r_boxes.ndim}, shape: {r_boxes.shape}')
                        # iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                        iou_boxes = torch.randn((0)).numpy()

                    iou_boxes_threshold = iou_boxes.copy()
                    iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
                    # TODO: Replace with Hungarian
                    a_boxes_np, r_boxes_np = a_boxes.numpy(), r_boxes.numpy()
                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(a_boxes_np), len(r_boxes_np)))
                    if r_boxes_np.size != 0:
                        for a_i, a_box in enumerate(a_boxes_np):
                            for r_i, r_box in enumerate(r_boxes_np):
                                dist = np.linalg.norm((get_bbox_center(a_box).flatten() -
                                                       get_bbox_center(r_box).flatten()), 2) * ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = 2 - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [a_boxes_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [r_boxes_idx[m] for m in match_cols]

                        gt_track_box_mapping = {a[-1]: a[:-1] for a in annotations}
                        for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                            gt_t_idx = match_rows_tracks_idx[m_c_idx]
                            # gt_box_idx = np.argwhere(a_boxes_idx == gt_t_idx)
                            track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                            track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                                gt_track_box_mapping[gt_t_idx]
                            try:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                                    last_frame_gt_tracks[gt_t_idx]
                                gt_distance = np.linalg.norm(
                                    (get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                                     get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = gt_distance
                            except KeyError:
                                track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                                track_based_accumulated_features[matched_c].object_features[-1]. \
                                    gt_past_current_distance = [0, 0]

                        last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    # generated_to_gt_track_association check ###################################################
                    # filter_nones = True
                    # plot_gt_ids = []
                    # plot_gt_boxes = []
                    # plot_generated_ids = []
                    # plot_generated_boxes = []
                    #
                    # for k, v in track_based_accumulated_features.items():
                    #     plot_feats = v.object_features[-1]
                    #     if filter_nones and plot_feats.gt_box is not None:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #     if not filter_nones:
                    #         plot_gt_ids.append(plot_feats.gt_track_idx)
                    #         plot_gt_boxes.append(plot_feats.gt_box)
                    #         plot_generated_ids.append(plot_feats.idx)
                    #         plot_generated_boxes.append(plot_feats.final_bbox)
                    #
                    # plot_image_set_of_boxes(frame, plot_gt_boxes, plot_generated_boxes,
                    #                         annotate=[plot_gt_ids, plot_generated_ids])
                    #############################################################################################
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold)
                    iou_boxes_threshold = filter_for_one_to_one_matches(iou_boxes_threshold.T).T
                    match_idx = np.where(iou_boxes)
                    if iou_boxes_threshold.size != 0:
                        a_match_idx_threshold, r_match_idx_threshold = np.where(iou_boxes_threshold)
                        matching_boxes_with_iou = [(a, r, iou_boxes_threshold[a, r])
                                                   for a, r in zip(a_match_idx_threshold, r_match_idx_threshold)]
                        #  precision/recall
                        tp = len(a_match_idx_threshold)
                        fp = len(r_boxes) - len(a_match_idx_threshold)
                        fn = len(a_boxes) - len(a_match_idx_threshold)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        a_match_idx_threshold, r_match_idx_threshold, matching_boxes_with_iou = [], [], []

                        #  precision/recall
                        tp = 0
                        fp = 0
                        fn = len(a_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    precision_list.append(precision)
                    recall_list.append(recall)
                    matching_boxes_with_iou_list.append(matching_boxes_with_iou)

                    bbox_distance_to_of_centers_iou_based = []
                    bbox_distance_to_of_centers_iou_based_idx = []
                    boxes_distance = []
                    # boxes_distance_for_metric = []
                    r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()

                    # TODO: Replace with Hungarian
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes)
                    iou_boxes = filter_for_one_to_one_matches(iou_boxes.T).T

                    match_idx = np.where(iou_boxes)
                    matched_boxes_l2_distance_matrix = np.zeros_like(iou_boxes)
                    predicted_box_center_inside_gt_box_matrix = np.zeros_like(iou_boxes)

                    for a_box_idx, r_box_idx in zip(*match_idx):
                        dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                               get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                        boxes_distance.append([(a_box_idx, r_box_idx), dist])
                        # boxes_distance_for_metric.append([a_box_idx, r_box_idx, dist])
                        matched_boxes_l2_distance_matrix[a_box_idx, r_box_idx] = dist
                        predicted_box_center_inside_gt_box_matrix[a_box_idx, r_box_idx] = is_inside_bbox(
                            point=get_bbox_center(r_boxes[r_box_idx]).flatten(), bbox=a_boxes[a_box_idx]
                        )
                        bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                      iou_boxes[a_box_idx, r_box_idx]])
                        bbox_distance_to_of_centers_iou_based_idx.append([a_box_idx, r_box_idx, dist,
                                                                          iou_boxes[a_box_idx, r_box_idx]])
                        if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                                if (b == r_boxes[r_box_idx]).all()][0]:
                            selected_track_distances.append(dist)

                    # boxes_distance_for_metric = np.array(boxes_distance_for_metric)
                    matched_boxes_l2_distance_matrix[matched_boxes_l2_distance_matrix > distance_threshold] = 0
                    if r_boxes_np.size != 0:
                        a_matched_boxes_l2_distance_matrix_idx, r_matched_boxes_l2_distance_matrix_idx = np.where(
                            matched_boxes_l2_distance_matrix
                        )

                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = \
                            np.where(predicted_box_center_inside_gt_box_matrix)
                    else:
                        a_matched_boxes_l2_distance_matrix_idx, \
                        r_matched_boxes_l2_distance_matrix_idx = np.array([]), np.array([])
                        a_predicted_box_center_inside_gt_box_matrix_idx, \
                        r_predicted_box_center_inside_gt_box_matrix_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        meter_tp = len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fp = len(r_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)
                        meter_fn = len(a_boxes) - len(a_matched_boxes_l2_distance_matrix_idx)

                        meter_precision = meter_tp / (meter_tp + meter_fp)
                        meter_recall = meter_tp / (meter_tp + meter_fn)

                        center_tp = len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fp = len(r_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)
                        center_fn = len(a_boxes) - len(a_predicted_box_center_inside_gt_box_matrix_idx)

                        center_precision = center_tp / (center_tp + center_fp)
                        center_recall = center_tp / (center_tp + center_fn)

                        if len(match_rows) != len(match_cols):
                            logger.info('Matching arrays length not same!')
                        l2_distance_hungarian_tp = len(match_rows)
                        l2_distance_hungarian_fp = len(r_boxes) - len(match_rows)
                        l2_distance_hungarian_fn = len(a_boxes) - len(match_rows)

                        l2_distance_hungarian_precision = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                        l2_distance_hungarian_recall = \
                            l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
                    else:
                        meter_tp = 0
                        meter_fp = 0
                        meter_fn = len(a_boxes)

                        meter_precision = 0
                        meter_recall = 0

                        center_tp = 0
                        center_fp = 0
                        center_fn = len(a_boxes)

                        center_precision = 0
                        center_recall = 0

                        l2_distance_hungarian_tp = 0
                        l2_distance_hungarian_fp = 0
                        l2_distance_hungarian_fn = len(a_boxes)

                        l2_distance_hungarian_precision = 0
                        l2_distance_hungarian_recall = 0

                    meter_tp_list.append(meter_tp)
                    meter_fp_list.append(meter_fp)
                    meter_fn_list.append(meter_fn)

                    center_inside_tp_list.append(center_tp)
                    center_inside_fp_list.append(center_fp)
                    center_inside_fn_list.append(center_fn)

                    l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
                    l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
                    l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

                    # plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                    #                         save_path=f'{plot_save_path}zero_shot/iou_distance{min_points_in_cluster}/')

                    # STEP 4h: begin tracks
                    new_track_boxes = []
                    new_track_ids = []
                    if begin_track_mode:
                        # STEP 4h: a> Get the features already covered and not covered in live tracks
                        all_cloud, feature_idx_covered, features_covered = features_included_in_live_tracks(
                            annotations, fg_mask, radius, running_tracks)

                        all_indexes = np.arange(start=0, stop=all_cloud.shape[0])
                        features_skipped_idx = np.setdiff1d(all_indexes, feature_idx_covered)

                        if features_skipped_idx.size != 0:
                            features_skipped = all_cloud[features_skipped_idx]

                            try:
                                # STEP 4h: b> cluster to group points
                                mean_shift, n_clusters = mean_shift_clustering_with_timeout(
                                    features_skipped, bin_seeding=False, min_bin_freq=8,
                                    cluster_all=True, bandwidth=4, max_iter=100)
                                cluster_centers = mean_shift.cluster_centers

                                # STEP 4h: c> prune cluster centers
                                # combine centers inside radius + eliminate noise
                                final_cluster_centers, final_cluster_centers_idx = prune_clusters(
                                    cluster_centers, mean_shift, radius + extra_radius,
                                    min_points_in_cluster=min_points_in_cluster)

                                if final_cluster_centers.size != 0:
                                    t_w, t_h = generic_box_wh, generic_box_wh  # 100, 100
                                    # STEP 4h: d> start new potential tracks
                                    for cluster_center in final_cluster_centers:
                                        cluster_center_x, cluster_center_y = np.round(cluster_center).astype(np.int)
                                        # t_id = max(track_ids_used) + 1
                                        # t_box = centroids_to_min_max([cluster_center_x, cluster_center_y, t_w, t_h])
                                        t_box = torchvision.ops.box_convert(
                                            torch.tensor([cluster_center_x, cluster_center_y, t_w, t_h]),
                                            'cxcywh', 'xyxy').int().numpy()

                                        # check if there's object in there
                                        t_box_xywh = torchvision.ops.box_convert(
                                            torch.from_numpy(t_box).unsqueeze(0), 'xyxy', 'xywh')
                                        t_box_xywh = [torch.tensor((b[1], b[0], b[2] + OC_ADDITIONAL_CROP_H,
                                                                    b[3] + OC_ADDITIONAL_CROP_W)) for b in
                                                      t_box_xywh]
                                        t_box_crop = [tvf.crop(torch.from_numpy(frame).permute(2, 0, 1),
                                                               top=b[0], left=b[1], width=b[2], height=b[3])
                                                      for b in t_box_xywh]
                                        valid_t_box = [c_i for c_i, c in enumerate(t_box_crop)
                                                       if c.shape[1] != 0 and c.shape[2] != 0]
                                        if len(valid_t_box) == 0:
                                            logger.info(
                                                f'No object according to object classifier at this cluster center'
                                                f' - box out of frame!')
                                            continue
                                        t_box_crop_resized = [tvf.resize(c, [generic_box_wh, generic_box_wh])
                                                              for c in t_box_crop if
                                                              c.shape[1] != 0 and c.shape[2] != 0]
                                        t_box_crop_resized = torch.stack(t_box_crop_resized)
                                        t_box_crop_resized = (t_box_crop_resized.float() / 255.0).to(
                                            OC_DEVICE)

                                        with torch.no_grad():
                                            t_box_patch_prediction = model(t_box_crop_resized)

                                        t_box_patch_label = torch.round(
                                            torch.sigmoid(t_box_patch_prediction))

                                        if t_box_patch_label.bool().item():
                                            # Note: Do not start track if bbox is out of frame
                                            if use_is_box_overlapping_live_boxes:
                                                if not (np.sign(t_box) < 0).any() and \
                                                        not is_box_overlapping_live_boxes(t_box,
                                                                                          [t.bbox for t in
                                                                                           running_tracks]):
                                                    # NOTE: the second check might result in killing potential tracks!
                                                    t_id = max(track_ids_used) + 1
                                                    running_tracks.append(Track(bbox=t_box, idx=t_id))
                                                    track_ids_used.append(t_id)
                                                    new_track_boxes.append(t_box)
                                                    new_track_ids.append(t_id)
                                            else:
                                                if not (np.sign(t_box) < 0).any():
                                                    t_id = max(track_ids_used) + 1
                                                    running_tracks.append(Track(bbox=t_box, idx=t_id))
                                                    track_ids_used.append(t_id)
                                                    new_track_boxes.append(t_box)
                                                    new_track_ids.append(t_id)
                                        else:
                                            logger.info(
                                                f'No object according to object classifier at this cluster center!')

                            except TimeoutException:
                                logger.warn('Clustering took too much time, skipping!')

                                # plot_features_with_circles(
                                #     all_cloud, features_covered, features_skipped, fg_mask, marker_size=8,
                                #     cluster_centers=final_cluster_centers,
                                #     num_clusters=final_cluster_centers.shape[0],
                                #     frame_number=frame_number, boxes=annotations[:, :-1], radius=radius + 50,
                                #     additional_text=
                                #     f'Original Cluster Center Count: {n_clusters}\nPruned Cluster Distribution: '
                                #     f'{[mean_shift.cluster_distribution[x] for x in final_cluster_centers_idx]}')

                    new_track_boxes = np.stack(new_track_boxes) if len(new_track_boxes) > 0 else np.empty(shape=(0,))
                    new_track_ids = np.stack(new_track_ids) if len(new_track_ids) > 0 else np.empty(shape=(0,))

                    # STEP 4i: save stuff and reiterate
                    accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                    object_features=object_features)})

                    if video_mode:
                        if filter_switch_boxes_based_on_angle_and_recent_history or compute_histories_for_plot:
                            frame_feats = accumulated_features[frame_number.item()]
                            tracks_histories = []
                            tracks_gt_histories = []
                            for obj_feature in frame_feats.object_features:
                                tracks_histories.extend(obj_feature.track_history)
                                tracks_gt_histories.extend(obj_feature.gt_history)

                            # no need to reverse
                            # tracks_histories.reverse()
                            # tracks_gt_histories.reverse()

                            tracks_histories = np.array(tracks_histories)
                            tracks_gt_histories = np.array(tracks_gt_histories)

                            if tracks_histories.size == 0:
                                tracks_histories = np.zeros(shape=(0, 2))
                            if tracks_gt_histories.size == 0:
                                tracks_gt_histories = np.zeros(shape=(0, 2))
                        else:
                            tracks_histories = np.zeros(shape=(0, 2))
                            tracks_gt_histories = np.zeros(shape=(0, 2))

                        # for gt_annotation_box in annotations[:, :-1]:
                        #     ground_truth_track_histories.append(get_bbox_center(gt_annotation_box).flatten())
                        # ground_truth_track_histories = np.array(ground_truth_track_histories)

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[a_boxes_idx.tolist(), r_boxes_idx],
                            generated_track_histories=tracks_histories,
                            gt_track_histories=tracks_gt_histories,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=video_mode, original_dims=original_dims, zero_shot=True)

                        canvas = FigureCanvas(fig)
                        canvas.draw()

                        buf = canvas.buffer_rgba()
                        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                        if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                            out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                            out_frame = (out_frame * 255).astype(np.uint8)
                        # out_frame = out_frame.reshape(1200, 1000, 3)
                        out.write(out_frame)

                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=36)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=201)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=212)
                        # process_plot_per_track_angle_and_history(frame, frame_number, plot_save_path,
                        #                                          track_based_accumulated_features,
                        #                                          track_id_to_plot=3)
                    else:
                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            additional_text=
                            f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                            f'Recall: {l2_distance_hungarian_recall}\n'
                            f'Center Inside based - Precision: {center_precision} | Recall: {center_recall}\n'
                            f'Precision: {precision} | Recall: {recall}\n'
                            f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                            f'Track Ids Killed: '
                            f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                            video_mode=False, original_dims=original_dims, zero_shot=True)

                    # : save stuff and reiterate - moved up
                    # accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                    #                                                                 object_features=object_features)})

                    second_last_frame = last_frame.copy()
                    last_frame = frame.copy()
                    last_frame_mask = fg_mask.copy()
                    last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

                    batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                        np.array(l2_distance_hungarian_fn_list).sum()
                    batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
                    batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
                    logger.info(f'Video: {VIDEO_LABEL.value} - {VIDEO_NUMBER} | Batch: {part_idx} | '
                                f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

                    if save_checkpoint:
                        resume_dict = {'frame_number': frame_number,
                                       'part_idx': part_idx,
                                       'second_last_frame': second_last_frame,
                                       'last_frame': last_frame,
                                       'last_frame_mask': last_frame_mask,
                                       'last_frame_live_tracks': last_frame_live_tracks,
                                       'running_tracks': running_tracks,
                                       'track_ids_used': track_ids_used,
                                       'new_track_boxes': new_track_boxes,
                                       'precision': precision_list,
                                       'recall': recall_list,
                                       'tp_list': tp_list,
                                       'fp_list': fp_list,
                                       'fn_list': fn_list,
                                       'meter_tp_list': meter_tp_list,
                                       'meter_fp_list': meter_fp_list,
                                       'meter_fn_list': meter_fn_list,
                                       'center_inside_tp_list': center_inside_tp_list,
                                       'center_inside_fp_list': center_inside_fp_list,
                                       'center_inside_fn_list': center_inside_fn_list,
                                       'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                       'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                       'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                       'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                       'accumulated_features': accumulated_features}
            if save_every_n_batch_itr is not None:
                if part_idx % save_every_n_batch_itr == 0 and part_idx != 0:
                    save_dict = {'frame_number': frame_number,
                                 'part_idx': part_idx,
                                 'current_track_idx': current_track_idx,
                                 'second_last_frame': second_last_frame,
                                 'last_frame': last_frame,
                                 'last_frame_mask': last_frame_mask,
                                 'last_frame_live_tracks': last_frame_live_tracks,
                                 'running_tracks': running_tracks,
                                 'track_ids_used': track_ids_used,
                                 'new_track_boxes': new_track_boxes,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'tp_list': tp_list,
                                 'fp_list': fp_list,
                                 'fn_list': fn_list,
                                 'meter_tp_list': meter_tp_list,
                                 'meter_fp_list': meter_fp_list,
                                 'meter_fn_list': meter_fn_list,
                                 'center_inside_tp_list': center_inside_tp_list,
                                 'center_inside_fp_list': center_inside_fp_list,
                                 'center_inside_fn_list': center_inside_fn_list,
                                 'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                 'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                 'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                 'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                 'track_based_accumulated_features': track_based_accumulated_features,
                                 'accumulated_features': accumulated_features}
                    Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
                    f_n = f'features_dict_part_{part_idx}.pt'
                    torch.save(save_dict, features_save_path + 'parts/' + f_n)

                    accumulated_features = {}
                    live_track_ids = [live_track.idx for live_track in last_frame_live_tracks]
                    track_based_accumulated_features = remove_from_dict_except_entries(live_track_ids,
                                                                                       track_based_accumulated_features)
                    logger.info(f'Saved part {part_idx} at frame {frame_number}')
            # gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)
            # if video_mode:
            #     out.release()
    except KeyboardInterrupt:
        if video_mode:
            logger.info('Saving video before exiting!')
            out.release()
        if premature_kill_save:
            premature_save_dict = {'frame_number': frame_number,
                                   'part_idx': part_idx,
                                   'second_last_frame': second_last_frame,
                                   'last_frame': last_frame,
                                   'last_frame_mask': last_frame_mask,
                                   'last_frame_live_tracks': last_frame_live_tracks,
                                   'running_tracks': running_tracks,
                                   'track_ids_used': track_ids_used,
                                   'new_track_boxes': new_track_boxes,
                                   'precision': precision_list,
                                   'recall': recall_list,
                                   'tp_list': tp_list,
                                   'fp_list': fp_list,
                                   'fn_list': fn_list,
                                   'meter_tp_list': meter_tp_list,
                                   'meter_fp_list': meter_fp_list,
                                   'meter_fn_list': meter_fn_list,
                                   'center_inside_tp_list': center_inside_tp_list,
                                   'center_inside_fp_list': center_inside_fp_list,
                                   'center_inside_fn_list': center_inside_fn_list,
                                   'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                                   'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                                   'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'track_based_accumulated_features': track_based_accumulated_features,
                                   'accumulated_features': accumulated_features}
            Path(features_save_path).mkdir(parents=True, exist_ok=True)
            f_n = f'premature_kill_features_dict.pt'
            torch.save(premature_save_dict, features_save_path + f_n)

        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')
    finally:
        tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Precision: {precision} | Recall: {recall}')

        # Distance Based
        tp_sum, fp_sum, fn_sum = \
            np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
            np.array(l2_distance_hungarian_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

        # Center Inside Based
        tp_sum, fp_sum, fn_sum = \
            np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
            np.array(center_inside_fn_list).sum()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

        premature_save_dict = {'frame_number': frame_number,
                               'part_idx': part_idx,
                               'current_track_idx': current_track_idx,
                               'second_last_frame': second_last_frame,
                               'last_frame': last_frame,
                               'last_frame_mask': last_frame_mask,
                               'last_frame_live_tracks': last_frame_live_tracks,
                               'running_tracks': running_tracks,
                               'track_ids_used': track_ids_used,
                               'new_track_boxes': new_track_boxes,
                               'precision': precision_list,
                               'recall': recall_list,
                               'tp_list': tp_list,
                               'fp_list': fp_list,
                               'fn_list': fn_list,
                               'meter_tp_list': meter_tp_list,
                               'meter_fp_list': meter_fp_list,
                               'meter_fn_list': meter_fn_list,
                               'center_inside_tp_list': center_inside_tp_list,
                               'center_inside_fp_list': center_inside_fp_list,
                               'center_inside_fn_list': center_inside_fn_list,
                               'l2_distance_hungarian_tp_list': l2_distance_hungarian_tp_list,
                               'l2_distance_hungarian_fp_list': l2_distance_hungarian_fp_list,
                               'l2_distance_hungarian_fn_list': l2_distance_hungarian_fn_list,
                               'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                               'track_based_accumulated_features': track_based_accumulated_features,
                               'accumulated_features': accumulated_features}

        Path(features_save_path + 'parts/').mkdir(parents=True, exist_ok=True)
        # f_n = f'accumulated_features_from_finally.pt'
        f_n = f'features_dict_part_{part_idx}.pt'
        torch.save(premature_save_dict, features_save_path + 'parts/' + f_n)

    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')

    # Distance Based
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')

    # Center Inside Based
    tp_sum, fp_sum, fn_sum = \
        np.array(center_inside_tp_list).sum(), np.array(center_inside_fp_list).sum(), \
        np.array(center_inside_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Center Inside Based - Precision: {precision} | Recall: {recall}')

    out.release()
    return accumulated_features


if __name__ == '__main__':
    eval_mode = False

    Path(video_save_path).mkdir(parents=True, exist_ok=True)
    Path(features_save_path).mkdir(parents=True, exist_ok=True)
    Path(plot_save_path).mkdir(parents=True, exist_ok=True)

    if not eval_mode and EXECUTE_STEP == STEP.UNSUPERVISED:
        resume = RESUME_MODE
        param = ObjectDetectionParameters.BEV_TIGHT.value

        if resume:
            features_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                 f'/parts/'
            every_part_file = np.array(os.listdir(features_base_path))
            part_idx = np.array([int(s[:-3].split('_')[-1]) for s in every_part_file]).argsort()
            every_part_file = every_part_file[part_idx]

            accumulated_features: Dict[Any, Any] = torch.load(features_base_path + every_part_file[-1])
            logger.info(f'Resuming from batch {accumulated_features["part_idx"]}')
            feats = preprocess_data_zero_shot_resumable(
                last_saved_features_dict=accumulated_features,
                var_threshold=None, plot=False, radius=param['radius'],
                video_mode=True, video_save_path=video_save_path + 'extraction_resumed.avi',
                desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
                min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
                use_circle_to_keep_track_alive=False, custom_video_shape=False,
                extra_radius=param['extra_radius'], generic_box_wh=param['generic_box_wh'],
                use_is_box_overlapping_live_boxes=True, save_per_part_path=None,
                save_every_n_batch_itr=BATCH_CHECKPOINT, drop_last_batch=True,
                detect_shadows=param['detect_shadows'],
                filter_switch_boxes_based_on_angle_and_recent_history=True,
                compute_histories_for_plot=True)
        else:
            feats = preprocess_data_zero_shot(
                var_threshold=None, plot=False, radius=param['radius'],
                video_mode=True, video_save_path=video_save_path + 'extraction.avi',
                desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
                min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
                use_circle_to_keep_track_alive=False, custom_video_shape=False,
                extra_radius=param['extra_radius'], generic_box_wh=param['generic_box_wh'],
                use_is_box_overlapping_live_boxes=True, save_per_part_path=None,
                save_every_n_batch_itr=BATCH_CHECKPOINT, drop_last_batch=True,
                detect_shadows=param['detect_shadows'],
                filter_switch_boxes_based_on_angle_and_recent_history=True,
                compute_histories_for_plot=True)
    elif not eval_mode and EXECUTE_STEP == STEP.VERIFY_ANNOTATIONS:
        video_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                          f'/video_annotation_generated/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)

        # annotation_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
        #                        f'/csv_annotation/'
        annotation_base_path = f'{BASE_PATH}filtered_generated_annotations/{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/'
        annotation_filename = 'generated_annotations.csv'
        annotation_path = annotation_base_path + annotation_filename

        visualize_annotations(
            pd.read_csv(annotation_path), batch_size=32, drop_last_batch=True, custom_video_shape=False,
            plot_scale_factor=1, video_mode=True, save_path_for_video=video_save_path + 'gen_video.avi', desired_fps=5)
    elif not eval_mode and EXECUTE_STEP == STEP.GENERATE_ANNOTATIONS:
        use_v2 = True
        track_length_threshold = 5
        csv_save_filename = f'generated_annotations_{track_length_threshold}.csv'  # 'generated_annotations.csv'

        if not GENERATE_BUNDLED_ANNOTATIONS and use_v2:
            features_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                 f'/minimal_zero_shot/parts/'
            # features_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
            #                      f'/parts/'

            csv_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/csv_annotation/'
            Path(csv_path).mkdir(parents=True, exist_ok=True)

            every_part_file = np.array(os.listdir(features_base_path))
            part_idx = np.array([int(s[:-3].split('_')[-1]) for s in every_part_file]).argsort()
            every_part_file = every_part_file[part_idx]

            combine_features_generate_annotations_v2(files_base_path=features_base_path, files_list=every_part_file,
                                                     min_track_length_threshold=track_length_threshold,
                                                     csv_save_path=csv_path,
                                                     do_filter=True, low_memory_mode=False,
                                                     csv_filename=csv_save_filename)
        elif use_v2 and GENERATE_BUNDLED_ANNOTATIONS:
            for v_id, v_clz in enumerate(BUNDLED_ANNOTATIONS_VIDEO_CLASSES_LIST):
                for v_num in BUNDLED_ANNOTATIONS_PER_CLASSES_VIDEO_LIST[v_id]:
                    features_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{v_clz.value}{v_num}' \
                                         f'/minimal_zero_shot/parts/'
                    csv_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{v_clz.value}{v_num}/csv_annotation/'
                    Path(csv_path).mkdir(parents=True, exist_ok=True)
                    features_chunk_list = np.array(os.listdir(features_base_path))
                    part_idx = np.array([int(s[:-3].split('_')[-1]) for s in features_chunk_list]).argsort()
                    features_chunk_list = features_chunk_list[part_idx]

                    logger.info(f'Generating annotations for Video: {v_clz.value}{v_num}')
                    combine_features_generate_annotations_v2(files_base_path=features_base_path,
                                                             files_list=features_chunk_list,
                                                             min_track_length_threshold=track_length_threshold,
                                                             csv_save_path=csv_path,
                                                             do_filter=True, low_memory_mode=False)
        else:
            features_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                 f'/minimal_zero_shot/parts/'
            # features_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
            #                      f'/parts/'

            csv_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/csv_annotation/'
            Path(csv_path).mkdir(parents=True, exist_ok=True)

            every_part_file = np.array(os.listdir(features_base_path))
            part_idx = np.array([int(s[:-3].split('_')[-1]) for s in every_part_file]).argsort()
            every_part_file = every_part_file[part_idx]

            combine_features_generate_annotations(files_base_path=features_base_path, files_list=every_part_file,
                                                  min_track_length_threshold=track_length_threshold,
                                                  csv_save_path=csv_path,
                                                  do_filter=True, low_memory_mode=False)
    elif not eval_mode and EXECUTE_STEP == STEP.MINIMAL:
        csv_mode = CSV_MODE
        resume_mode = RESUME_MODE
        use_timeout = TIMEOUT_MODE
        logger.info('MINIMAL MODE')

        video_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                          f'/minimal_zero_shot/'
        plot_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                         f'/minimal_zero_shot/'
        features_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                             f'/minimal_zero_shot/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)
        Path(features_save_path).mkdir(parents=True, exist_ok=True)

        param = ObjectDetectionParameters.BEV_TIGHT.value

        if csv_mode:
            logger.info('MINIMAL - CSV MODE')
            accumulated_features_path_filename = 'accumulated_features_from_finally.pt'
            # accumulated_features_path_filename = 'accumulated_features_from_finally_tight.pt'
            track_length_threshold = 5

            accumulated_features_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                        f'/minimal_zero_shot/{accumulated_features_path_filename}'
            # accumulated_features_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
            #                             f'/{accumulated_features_path_filename}'
            accumulated_features: Dict[int, Any] = torch.load(accumulated_features_path)
            per_track_features: Dict[int, TrackFeatures] = accumulated_features['track_based_accumulated_features']
            per_frame_features: Dict[int, FrameFeatures] = accumulated_features['accumulated_features']

            video_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                              f'/processed_features/{track_length_threshold}/'
            Path(video_save_path).mkdir(parents=True, exist_ok=True)
            extracted_features_in_csv(track_based_features=per_track_features,
                                      frame_based_features=per_frame_features,
                                      do_filter=True,
                                      min_track_length_threshold=track_length_threshold,
                                      csv_save_path=features_save_path)
        elif resume_mode:
            if use_timeout:
                resume_method = preprocess_data_zero_shot_minimal_resumable_with_timeout
            else:
                resume_method = preprocess_data_zero_shot_minimal_resumable
            features_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                 f'/minimal_zero_shot/parts/'
            # features_base_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
            #                      f'/parts/'
            every_part_file = np.array(os.listdir(features_base_path))
            part_idx = np.array([int(s[:-3].split('_')[-1]) for s in every_part_file]).argsort()
            every_part_file = every_part_file[part_idx]

            accumulated_features: Dict[Any, Any] = torch.load(features_base_path + every_part_file[-1])
            logger.info(f'Resuming from batch {accumulated_features["part_idx"]}')

            feats = resume_method(
                last_saved_features_dict=accumulated_features,
                var_threshold=None, plot=False, radius=param['radius'],
                video_mode=True, video_save_path=video_save_path + 'extraction.avi',
                desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
                min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
                use_circle_to_keep_track_alive=False, custom_video_shape=False,
                extra_radius=param['extra_radius'], generic_box_wh=param['generic_box_wh'],
                use_is_box_overlapping_live_boxes=True, save_per_part_path=None,
                save_every_n_batch_itr=BATCH_CHECKPOINT, drop_last_batch=True,
                detect_shadows=param['detect_shadows'],
                filter_switch_boxes_based_on_angle_and_recent_history=True,
                compute_histories_for_plot=True)
        elif WITH_OBJECT_CLASSIFIER:
            feats = preprocess_data_zero_shot_minimal_with_timeout_and_classifier(
                var_threshold=None, plot=False, radius=param['radius'],
                video_mode=True, video_save_path=video_save_path + 'extraction.avi',
                desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
                min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
                use_circle_to_keep_track_alive=False, custom_video_shape=False,
                extra_radius=param['extra_radius'], generic_box_wh=param['generic_box_wh'],
                use_is_box_overlapping_live_boxes=True, save_per_part_path=None,
                save_every_n_batch_itr=BATCH_CHECKPOINT, drop_last_batch=True,
                detect_shadows=param['detect_shadows'],
                filter_switch_boxes_based_on_angle_and_recent_history=True,
                compute_histories_for_plot=True)
        else:
            if use_timeout:
                method_to_call = preprocess_data_zero_shot_minimal_with_timeout
            else:
                method_to_call = preprocess_data_zero_shot_minimal
            feats = method_to_call(
                var_threshold=None, plot=False, radius=param['radius'],
                video_mode=True, video_save_path=video_save_path + 'extraction.avi',
                desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
                min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
                use_circle_to_keep_track_alive=False, custom_video_shape=False,
                extra_radius=param['extra_radius'], generic_box_wh=param['generic_box_wh'],
                use_is_box_overlapping_live_boxes=True, save_per_part_path=None,
                save_every_n_batch_itr=BATCH_CHECKPOINT, drop_last_batch=True,
                detect_shadows=param['detect_shadows'],
                filter_switch_boxes_based_on_angle_and_recent_history=True,
                compute_histories_for_plot=True)
    elif not eval_mode and EXECUTE_STEP == STEP.CUSTOM_VIDEO:
        param = ObjectDetectionParameters.SLANTED.value

        video_class = 'Oxford'  # 'Virat'
        video_name = 'input_video_s2_l1_08'
        custom_video_dict = {
            'dataset': SimpleVideoDatasetBase,
            # 'video_path': f'{ROOT_PATH}Datasets/Virat/VIRAT_S_000201_00_000018_000380.mp4',
            'video_path': f'{ROOT_PATH}Datasets/{video_class}/{video_name}.mp4',
            'start': 0,
            'end': 10,
            'pts_unit': 'sec'
        }

        video_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/custom/{video_class}/{video_name}/zero_shot/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)

        features_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/custom/{video_class}/{video_name}/zero_shot/'
        Path(features_save_path).mkdir(parents=True, exist_ok=True)

        feats = preprocess_data_zero_shot_custom_video(
            var_threshold=None, plot=False, radius=param['radius'],
            video_mode=True, video_save_path=video_save_path + 'extraction.avi',
            desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
            min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
            use_circle_to_keep_track_alive=False, custom_video_shape=False,
            extra_radius=param['extra_radius'], generic_box_wh=param['generic_box_wh'],
            use_is_box_overlapping_live_boxes=True, save_per_part_path=None,
            save_every_n_batch_itr=BATCH_CHECKPOINT, drop_last_batch=True,
            detect_shadows=param['detect_shadows'],
            filter_switch_boxes_based_on_angle_and_recent_history=True,
            compute_histories_for_plot=True, custom_video=custom_video_dict)
    elif not eval_mode and EXECUTE_STEP == STEP.SEMI_SUPERVISED:
        video_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/one_shot/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)
        feats = preprocess_data_one_shot(var_threshold=None, plot=False, radius=60, save_per_part_path=None,
                                         video_mode=True, video_save_path=video_save_path + 'extraction.avi',
                                         desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
                                         min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
                                         use_circle_to_keep_track_alive=False, custom_video_shape=False,
                                         extra_radius=0, generic_box_wh=50, use_is_box_overlapping_live_boxes=True,
                                         save_every_n_batch_itr=BATCH_CHECKPOINT, drop_last_batch=True,
                                         detect_shadows=False)
    elif not eval_mode and EXECUTE_STEP == STEP.FILTER_FEATURES:
        # accumulated_features_path_filename = 'accumulated_features_from_finally.pt'
        accumulated_features_path_filename = 'accumulated_features_from_finally_tight.pt'
        # accumulated_features_path_filename = 'accumulated_features_from_finally_filtered.pt'
        track_length_threshold = 5

        accumulated_features_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                    f'/{accumulated_features_path_filename}'
        accumulated_features: Dict[int, Any] = torch.load(accumulated_features_path)
        per_track_features: Dict[int, TrackFeatures] = accumulated_features['track_based_accumulated_features']
        per_frame_features: Dict[int, FrameFeatures] = accumulated_features['accumulated_features']

        video_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                          f'/processed_features' \
                          f'/{track_length_threshold}/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)
        evaluate_extracted_features(track_based_features=per_track_features, frame_based_features=per_frame_features,
                                    video_save_location=video_save_path + 'extraction_filter.avi', do_filter=True,
                                    min_track_length_threshold=track_length_threshold, desired_fps=1, video_mode=False,
                                    skip_plot_save=True)
        logger.info(f'Track length threshold: {track_length_threshold}')
    elif not eval_mode and EXECUTE_STEP == STEP.NN_EXTRACTION:
        # accumulated_features_path_filename = 'accumulated_features_from_finally.pt'
        accumulated_features_path_filename = 'features_dict_from_finally.pt'
        # accumulated_features_path_filename = 'accumulated_features_from_finally_filtered.pt'

        accumulated_features_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                    f'/zero_shot/frames12apart/features/{accumulated_features_path_filename}'

        accumulated_features: Dict[int, Any] = torch.load(accumulated_features_path)
        per_track_features: Dict[int, TrackFeatures] = accumulated_features['track_based_accumulated_features']
        per_frame_features: Dict[int, FrameFeatures] = accumulated_features['total_accumulated_features']

        print()
    elif not eval_mode and EXECUTE_STEP == STEP.DEBUG:
        # Scrapped ###################################################################################################
        # time_step_between_frames = 12
        # for frame_number, frame_features in tqdm(extracted_features.items()):
        #     for object_feature in frame_features.object_features:
        #         current_track_idx = object_feature.idx
        #         obj_flow = object_feature.flow
        #         obj_past_flow = object_feature.past_flow
        #         for ts in range(1, time_step_between_frames):
        #             next_frame_features: FrameFeatures = extracted_features[frame_number + ts]
        #             next_frame_number: int = next_frame_features.frame_number
        #             next_frame_obj_features: Sequence[ObjectFeatures] = next_frame_features.object_features
        #             for next_frame_object_feature in next_frame_obj_features:
        #                 if next_frame_object_feature == object_feature:
        #                     print()
        #                 else:
        #                     print()
        #                 print()
        ##############################################################################################################
        # out = process_complex_features_rnn(extracted_features, time_steps=5)
        # csv1 = pd.read_csv('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Plots/baseline_v2/v0/hyang2/'
        #                    'csv_annotation/generated_annotations.csv')
        # csv2 = pd.read_csv('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Plots/baseline_v2/v0/hyang2/'
        #                    'csv_annotationgenerated_annotations.csv')
        # eqls = csv1.equals(csv2)
        # pd.testing.assert_frame_equal(csv1, csv2)
        print()
    elif not eval_mode and EXECUTE_STEP == STEP.EXTRACTION:
        # accumulated_features_path_filename = 'accumulated_features_from_finally.pt'
        accumulated_features_path_filename = 'accumulated_features_from_finally_filtered.pt'
        track_length_threshold = 60

        accumulated_features_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                    f'/{accumulated_features_path_filename}'
        accumulated_features: Dict[int, Any] = torch.load(accumulated_features_path)
        per_track_features: Dict[int, TrackFeatures] = accumulated_features['track_based_accumulated_features']
        per_frame_features: Dict[int, FrameFeatures] = accumulated_features['accumulated_features']

        filtered_track_based_features, filtered_frame_based_features = filter_tracks_through_all_steps(
            per_track_features, per_frame_features, track_length_threshold
        )

        plot_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/track_based/'
        Path(plot_save_path).mkdir(parents=True, exist_ok=True)
        video_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/' \
                          f'zero_shot/frames12apart/'
        # + 'cancelled/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)
        features_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/' \
                             f'zero_shot/frames12apart/features/'
        Path(features_save_path).mkdir(parents=True, exist_ok=True)
        feats = preprocess_data_zero_shot_12_frames_apart(
            extracted_features=filtered_frame_based_features,
            var_threshold=None, plot=False, radius=60, save_per_part_path=None,
            video_mode=True, save_path_for_video=video_save_path + 'extraction.avi',
            desired_fps=5, overlap_percent=0.4, save_path_for_plot=plot_save_path,
            min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
            use_circle_to_keep_track_alive=False, custom_video_shape=False,
            extra_radius=0, generic_box_wh=50, use_is_box_overlapping_live_boxes=True,
            save_every_n_batch_itr=BATCH_CHECKPOINT, frame_by_frame_estimation=False,
            filter_switch_boxes_based_on_angle_and_rfecfefnft_history=True,
            compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
            angle_threshold_to_filter=120, save_path_for_features=features_save_path)
    else:
        feat_file_path = f'{ROOT_PATH}Plots/baseline_v2/v0/deathCircle4/' \
                         'use_is_box_overlapping_live_boxes/premature_kill_features_dict.pt'
        eval_metrics(feat_file_path)
    # TODO:
    #  -> Add gt info - associate gt (iou) + check 12 frames later that gt was actually the gt we had - !!Done!!
    #  - test-viz!
    #  -> Frame_by_frame estimation for 12 frames, check the dict fix for other function - !!Done!!
    #  -> plot to verify -> track-based+frame_based - !!Done!! - 12 frames apart and frame-by-frame both works,
    #  later slow
    #  -> box switch stuff - can reduce the track ids count a lot - Important to fix - !!Done!!
    #  -> viz via plot angle behaviour - momentum - !!!ON HOLD!!!
    #  -> make it configurable to switch on/off - !!Done!!
    #  -> make generic boxes (w, h) instead of square
    #  -> extract ans set up NN
    # NOTE:
    #  -> setting use_circle_to_keep_track_alive=False to avoid noisy new tracks to pick up true live tracks
    #  -> Crowded - Death Circle [ smaller one 4]
    #  -> Good one - Little - 0 smaller, 3 big n good
    #  -> radius=60, extra_radius=0, generic_box_wh=50, use_circle_to_keep_track_alive=False, min_points_in_cluster=16
    #  , use_is_box_overlapping_live_boxes=True --- looks good
    #  -> too less frames, bg fails
    #  -> Interesting: gates[2,3,8, 4--], hyang[0,3,4,7s, 8,9], nexus[3?,4?,], quad[1,], deatCircle[2, 4]
    #  -> Extraction time: gates3 - 1day
    #  -> What about stationary objects? Decreasing Recall
    #  -> side by side walking - consider them as one?
    #  -> combines objects into one
    #  -> shadow activations into grass! invalid area
    #  -> some plots for past goes negative idx and pcks up wrong image?
    #  -> track 3 & 8, identity switch
    #  -> shadows mimic the motion - so not so bad
