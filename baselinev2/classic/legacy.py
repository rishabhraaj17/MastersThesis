import copy
from pathlib import Path
from typing import Dict

import cv2 as cv
import numpy as np
import scipy
import skimage
import torch
import torchvision
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.feature_extractor import FeatureExtractor
from average_image.utils import is_inside_bbox
from baselinev2.config import DATASET_META, META_LABEL, VIDEO_LABEL, VIDEO_NUMBER, SAVE_PATH, BASE_PATH, \
    features_save_path
from baselinev2.structures import Track, ObjectFeatures, FrameFeatures, TrackFeatures, AgentFeatures
from baselinev2.utils import first_frame_processing_and_gt_association, optical_flow_processing, \
    get_mog2_foreground_mask, extract_features_per_bounding_box, evaluate_shifted_bounding_box, get_bbox_center, \
    extract_features_inside_circle, features_filter_append_preprocessing, filter_features, append_features, \
    filter_for_one_to_one_matches, features_included_in_live_tracks, mean_shift_clustering, prune_clusters, \
    is_box_overlapping_live_boxes, associate_frame_with_ground_truth, remove_entries_from_dict
from baselinev2.plot_utils import plot_mask_matching_bbox, plot_for_video, plot_for_video_current_frame, \
    plot_processing_steps
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames

initialize_logging()
logger = get_logger('baselinev2.legacy')


def preprocess_data(save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1, plot=False,
                    radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None, plot_scale_factor=1,
                    desired_fps=5, custom_video_shape=True, plot_save_path=None, save_checkpoint=False,
                    begin_track_mode=True, use_circle_to_keep_track_alive=True, iou_threshold=0.5, generic_box_wh=100,
                    extra_radius=50, use_is_box_overlapping_live_boxes=True, detect_shadows=True):
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size)
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
    selected_track_distances = []
    accumulated_features = {}

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (1200, 1000))  # (video_shape[0], video_shape[1]))
            # (video_shape[1], video_shape[0]))
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (1200, 1000))  # (video_shape[0], video_shape[1]))

    for part_idx, data in enumerate(tqdm(data_loader)):
        frames, frame_numbers = data
        frames = frames.squeeze()
        frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        frames_count = frames.shape[0]
        original_shape = new_shape = [frames.shape[1], frames.shape[2]]
        for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)), total=len(frame_numbers)):
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
                            object_features.append(ObjectFeatures(idx=current_track_idx,
                                                                  xy=xy_current_frame,
                                                                  past_xy=xy,
                                                                  final_xy=xy_current_frame,
                                                                  flow=xy_displacement,
                                                                  past_flow=past_xy_displacement,
                                                                  past_bbox=box,
                                                                  final_bbox=np.array(shifted_box),
                                                                  is_track_live=False,
                                                                  frame_number=frame_number))
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
                                              xy_cloud_current_frame=xy_current_frame, frame_number=frame_number.item(),
                                              track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                              selected_current=closest_n_xy_current_frame_pair)
                    # STEP 4g: save the information gathered
                    object_features.append(ObjectFeatures(idx=current_track_idx,
                                                          xy=xy_current_frame,
                                                          past_xy=xy,
                                                          final_xy=final_features_xy,
                                                          flow=xy_displacement,
                                                          past_flow=past_xy_displacement,
                                                          past_bbox=box,
                                                          final_bbox=np.array(shifted_box),
                                                          frame_number=frame_number))
                    if current_track_idx not in track_ids_used:
                        track_ids_used.append(current_track_idx)

                _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
                ratio = float(meta_info.flatten()[-1])

                # NOTE: running ADE/FDE
                r_boxes = [b.bbox for b in running_tracks]
                r_boxes_idx = [b.idx for b in running_tracks]
                select_track_idx = 4

                # bbox_distance_to_of_centers = []
                # for a_box in annotations[:, :-1]:
                #     a_box_center = get_bbox_center(a_box).flatten()
                #     a_box_distance_with_centers = []
                #     for r_box in r_boxes:
                #         a_box_distance_with_centers.append(
                #             np.linalg.norm((a_box_center, get_bbox_center(r_box).flatten()), 2) * ratio)
                #     bbox_distance_to_of_centers.append([a_box, np.min(a_box_distance_with_centers) * ratio,
                #                                         r_boxes[np.argmin(a_box_distance_with_centers).item()]])

                r_boxes = torch.tensor(r_boxes, dtype=torch.int)
                a_boxes = torch.from_numpy(annotations[:, :-1])
                iou_boxes = torchvision.ops.box_iou(a_boxes, r_boxes).numpy()
                iou_boxes_threshold = iou_boxes.copy()
                iou_boxes_threshold[iou_boxes_threshold < iou_threshold] = 0
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

                bbox_distance_to_of_centers_iou_based = []
                boxes_distance = []
                r_boxes, a_boxes = r_boxes.numpy(), a_boxes.numpy()
                for a_box_idx, r_box_idx in zip(*match_idx):
                    dist = np.linalg.norm((get_bbox_center(a_boxes[a_box_idx]).flatten() -
                                           get_bbox_center(r_boxes[r_box_idx]).flatten()), 2) * ratio
                    boxes_distance.append([(a_box_idx, r_box_idx), dist])
                    bbox_distance_to_of_centers_iou_based.append([a_boxes[a_box_idx], dist, r_boxes[r_box_idx],
                                                                  iou_boxes[a_box_idx, r_box_idx]])
                    if select_track_idx == [r_boxes_idx[i] for i, b in enumerate(r_boxes)
                                            if (b == r_boxes[r_box_idx]).all()][0]:
                        selected_track_distances.append(dist)

                plot_mask_matching_bbox(fg_mask, bbox_distance_to_of_centers_iou_based, frame_number,
                                        save_path=f'{plot_save_path}iou_distance{min_points_in_cluster}/')

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
                                            not is_box_overlapping_live_boxes(t_box, [t.bbox for t in running_tracks]):
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
                    fig = plot_for_video(
                        gt_rgb=frame, gt_mask=fg_mask, last_frame_rgb=last_frame,
                        last_frame_mask=last_frame_mask, current_frame_rgb=frame,
                        current_frame_mask=fg_mask, gt_annotations=annotations[:, :-1],
                        last_frame_annotation=[t.bbox for t in last_frame_live_tracks],
                        current_frame_annotation=[t.bbox for t in running_tracks],
                        new_track_annotation=new_track_boxes,
                        frame_number=frame_number,
                        additional_text=
                        # f'Track Ids Used: {track_ids_used}\n'
                        f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        f'Track Ids Killed: '
                        f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                        video_mode=video_mode)

                    canvas = FigureCanvas(fig)
                    canvas.draw()

                    buf = canvas.buffer_rgba()
                    out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                    out_frame = out_frame.reshape(1200, 1000, 3)
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
                        f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                        f'Track Ids Killed: '
                        f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                        video_mode=False,
                        save_path=f'{plot_save_path}plots{min_points_in_cluster}/')
                    # fig = plot_for_one_track(
                    #     gt_rgb=frame, gt_mask=fg_mask, last_frame_rgb=last_frame,
                    #     last_frame_mask=last_frame_mask, current_frame_rgb=frame,
                    #     current_frame_mask=fg_mask, gt_annotations=annotations[:, :-1],
                    #     last_frame_annotation=[t.bbox for t in last_frame_live_tracks],
                    #     current_frame_annotation=[t.bbox for t in running_tracks],
                    #     new_track_annotation=new_track_boxes,
                    #     frame_number=frame_number,
                    #     additional_text=
                    #     f'Track Idx Shown: {last_frame_live_tracks[4].gt_track_idx}\n'
                    #     f'Track Ids Active: {[t.idx for t in running_tracks]}\n'
                    #     f'Track Ids Killed: '
                    #     f'{np.setdiff1d([t.idx for t in last_frame_live_tracks], [t.idx for t in running_tracks])}',
                    #     video_mode=False,
                    #     save_path=None,
                    #     track_idx=4)
                    # save_path=f'{plot_save_path}plots{min_points_in_cluster}/')

                # STEP 4i: save stuff and reiterate
                accumulated_features.update({frame_number.item(): FrameFeatures(frame_number=frame_number.item(),
                                                                                object_features=object_features)})

                second_last_frame = last_frame.copy()
                last_frame = frame.copy()
                last_frame_mask = fg_mask.copy()
                last_frame_live_tracks = np.stack(running_tracks) if len(running_tracks) != 0 else []

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
                                   'matching_boxes_with_iou_list': matching_boxes_with_iou_list,
                                   'accumulated_features': accumulated_features}
        gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
        if save_per_part_path is not None:
            Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
            f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
            torch.save(accumulated_features, save_per_part_path + f_n)
        if video_mode:
            out.release()

    return accumulated_features


def preprocess_data_zero_shot_v0(save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1,
                                 radius=50, min_points_in_cluster=5, video_mode=False, video_save_path=None,
                                 plot_scale_factor=1, desired_fps=5, custom_video_shape=True, plot_save_path=None,
                                 save_checkpoint=False, plot=False, begin_track_mode=True, generic_box_wh=100,
                                 use_circle_to_keep_track_alive=True, iou_threshold=0.5, extra_radius=50,
                                 use_is_box_overlapping_live_boxes=True, premature_kill_save=False,
                                 distance_threshold=2, save_every_n_batch_itr=None, drop_last_batch=True,
                                 detect_shadows=True):
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
                                if not (np.sign(t_box) < 0).any() and \
                                        not is_box_overlapping_live_boxes(t_box, [t.bbox for t in running_tracks]):
                                    # NOTE: the second check might result in killing potential tracks!
                                    t_id = max(track_ids_used) + 1 if len(track_ids_used) else 0
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
                        # STEP 4a: Get features inside the bounding box
                        xy = extract_features_per_bounding_box(box, last_frame_mask)

                        if xy.size == 0:  # Check! This should always be false, since track started coz feats was there
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
                                # STEP 4e: b>Kill the track if corresponding features are not detected
                                #  in the next time-step
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
                        closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair = \
                            features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame)

                        # points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair)

                        filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                        final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                        # TODO: shift box again?

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

                        fig = plot_for_video_current_frame(
                            gt_rgb=frame, current_frame_rgb=frame,
                            gt_annotations=annotations[:, :-1],
                            current_frame_annotation=[t.bbox for t in running_tracks],
                            new_track_annotation=new_track_boxes,
                            frame_number=frame_number,
                            box_annotation=[a_boxes_idx.tolist(), r_boxes_idx],
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


def twelve_frames_feature_extraction_zero_shot_v0(frames, n, frames_to_build_model, extracted_features,
                                                  var_threshold=None,
                                                  time_gap_within_frames=3, frame_numbers=None, remaining_frames=None,
                                                  remaining_frames_idx=None, past_12_frames_optical_flow=None,
                                                  last_frame_from_last_used_batch=None, last12_bg_sub_mask=None,
                                                  resume_mode=False, detect_shadows=True, overlap_percent=0.4,
                                                  track_based_accumulated_features=None, frame_time_gap=12,
                                                  save_path_for_plot=None, df=None, ratio=None):
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

                    if object_feature.gt_track_idx != match_rows_tracks_idx_future[0]:
                        logger.info(f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                                    f' {match_rows_tracks_idx_future[0]}')
                    if object_feature.gt_track_idx != match_rows_tracks_idx_past[0]:
                        logger.info(f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                                    f' {match_rows_tracks_idx_past[0]}')

                    if activations_future_frame.size == 0:
                        logger.info(f'Ending Track! No features found in future!')
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
                                                                                         match_cols_past[0]],
                            past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]],
                            future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]],
                            gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                                                                             match_cols_future[0]],
                            past_gt_track_idx=match_rows_tracks_idx_past[0],
                            future_gt_track_idx=match_rows_tracks_idx_future[0],
                            future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0],
                            past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0],
                            frame_by_frame_estimation=False
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
                                                                                     match_cols_past[0]],
                        past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]],
                        future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]],
                        gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                                                                         match_cols_future[0]],
                        past_gt_track_idx=match_rows_tracks_idx_past[0],
                        future_gt_track_idx=match_rows_tracks_idx_future[0],
                        future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0],
                        past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0],
                        frame_by_frame_estimation=False
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

    return data_all_frames, frames[interest_fr:, ...], \
           torch.arange(interest_fr + frame_numbers[0], frame_numbers[-1] + 1), \
           last_frame_from_last_used_batch, past_12_frames_optical_flow, last12_bg_sub_mask, \
           track_based_accumulated_features


def twelve_frame_by_frame_feature_extraction_zero_shot_v0(
        frames, n, frames_to_build_model, extracted_features, df,
        var_threshold=None, time_gap_within_frames=3, frame_numbers=None,
        remaining_frames=None, remaining_frames_idx=None, past_12_frames_optical_flow=None,
        last_frame_from_last_used_batch=None, last12_bg_sub_mask=None,
        resume_mode=False, detect_shadows=True, overlap_percent=0.4, ratio=None,
        track_based_accumulated_features=None, frame_time_gap=12, save_path_for_plot=None):
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

                        if object_feature.gt_track_idx != match_rows_tracks_idx_future[0]:
                            logger.info(
                                f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                                f' {match_rows_tracks_idx_future[0]}')
                        if object_feature.gt_track_idx != match_rows_tracks_idx_past[0]:
                            logger.info(
                                f'GT track id mismatch! Per Frame:{object_feature.gt_track_idx} vs 12 frames apart'
                                f' {match_rows_tracks_idx_past[0]}')

                        if activations_future_frame_per_frame.size == 0 or activations_past_frame_per_frame.size == 0:
                            logger.info(f'Ending Track {object_feature.idx}! No features found in '
                                        f'{"past" if activations_past_frame_per_frame.size == 0 else "future"} '
                                        f'at {running_idx} frames apart from frame {actual_fr.item()}')
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
                                gt_past_current_distance=l2_distance_boxes_score_matrix_past[match_rows_past[0],
                                                                                             match_cols_past[0]],
                                past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]],
                                future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]],
                                gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                                                                                 match_cols_future[0]],
                                past_gt_track_idx=match_rows_tracks_idx_past[0],
                                future_gt_track_idx=match_rows_tracks_idx_future[0],
                                future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0],
                                past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0],
                                frame_by_frame_estimation=True
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
                        gt_past_current_distance=l2_distance_boxes_score_matrix_past[match_rows_past[0],
                                                                                     match_cols_past[0]],
                        past_gt_box=past_gt_track_box_mapping[match_rows_tracks_idx_past[0]],
                        future_gt_box=future_gt_track_box_mapping[match_rows_tracks_idx_future[0]],
                        gt_current_future_distance=l2_distance_boxes_score_matrix_future[match_rows_future[0],
                                                                                         match_cols_future[0]],
                        past_gt_track_idx=match_rows_tracks_idx_past[0],
                        future_gt_track_idx=match_rows_tracks_idx_future[0],
                        future_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_future[0],
                        past_box_inconsistent=object_feature.gt_track_idx == match_rows_tracks_idx_past[0],
                        frame_by_frame_estimation=True
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

    return data_all_frames, frames[interest_fr:, ...], \
           torch.arange(interest_fr + frame_numbers[0], frame_numbers[-1] + 1), \
           last_frame_from_last_used_batch, past_12_frames_optical_flow, last12_bg_sub_mask, \
           track_based_accumulated_features
