import copy
from typing import List, Union

import cv2 as cv
import numpy as np
import pandas as pd
import scipy
import skimage
import timeout_decorator
import torch
import torchvision
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.feature_clustering import MeanShiftClustering
from average_image.feature_extractor import FeatureExtractor
from baseline.extracted_of_optimization import find_points_inside_circle, is_point_inside_circle, \
    clouds_distance_matrix, smallest_n_indices
from baselinev2.config import DATASET_META, META_LABEL, VIDEO_LABEL, VIDEO_NUMBER, BASE_PATH, plot_save_path, \
    CLUSTERING_TIMEOUT
from baselinev2.exceptions import TimeoutException
from baselinev2.plot_utils import plot_features_with_mask, \
    plot_track_history_with_angle_info_with_track_plot, plot_for_video_current_frame
from baselinev2.structures import ObjectFeatures, AgentFeatures
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames

initialize_logging()
logger = get_logger('baselinev2.utils')


def find_points_inside_box(points, box):
    points_to_alter = points.copy()
    x1, y1, x2, y2 = box

    points_to_alter = points_to_alter[x1 < points_to_alter[..., 0]]
    points_to_alter = points_to_alter[points_to_alter[..., 0] < x2]

    points_to_alter = points_to_alter[y1 < points_to_alter[..., 1]]
    points_to_alter = points_to_alter[points_to_alter[..., 1] < y2]

    return points_to_alter


def first_violation_till_now(direction, angle):
    direction = np.where(np.isnan(direction), 0, direction)
    for d in direction[:-1]:
        if d > angle:
            return False
    return True


def mean_shift_clustering(data, bandwidth: float = 0.1, min_bin_freq: int = 3, max_iter: int = 300,
                          bin_seeding: bool = False, cluster_all: bool = True):
    mean_shift = MeanShiftClustering(data=data, bandwidth=bandwidth, min_bin_freq=min_bin_freq, max_iter=max_iter,
                                     bin_seeding=bin_seeding, cluster_all=cluster_all)
    mean_shift.cluster(renormalize=False)
    labels_unique, points_per_cluster = np.unique(mean_shift.labels, return_counts=True)
    mean_shift.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
    n_clusters_ = len(labels_unique)
    return mean_shift, n_clusters_


@timeout_decorator.timeout(seconds=CLUSTERING_TIMEOUT, timeout_exception=TimeoutException)
def mean_shift_clustering_with_timeout(data, bandwidth: float = 0.1, min_bin_freq: int = 3, max_iter: int = 300,
                                       bin_seeding: bool = False, cluster_all: bool = True):
    mean_shift = MeanShiftClustering(data=data, bandwidth=bandwidth, min_bin_freq=min_bin_freq, max_iter=max_iter,
                                     bin_seeding=bin_seeding, cluster_all=cluster_all)
    mean_shift.cluster(renormalize=False)
    labels_unique, points_per_cluster = np.unique(mean_shift.labels, return_counts=True)
    mean_shift.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
    n_clusters_ = len(labels_unique)
    return mean_shift, n_clusters_


def prune_cluster_centers_proximity_based_v0(cluster_centers, radius, mean_shift):
    rejected_cluster_centers = []
    pruned_cluster_centers = []
    pruned_cluster_centers_idx = []
    for cluster_center in cluster_centers:
        if not np.isin(cluster_center, pruned_cluster_centers).all() \
                or not np.isin(cluster_center, rejected_cluster_centers).all():
            centers_inside_idx = find_points_inside_circle(cluster_centers,
                                                           circle_center=cluster_center,
                                                           circle_radius=radius)
            # sort on the basis of cluster distribution
            centers_inside_idx = centers_inside_idx.tolist()
            centers_inside_idx.sort(key=lambda x: mean_shift.cluster_distribution[x], reverse=True)
            centers_inside_idx = np.array(centers_inside_idx)

            if not np.isin(cluster_centers[centers_inside_idx[0]], pruned_cluster_centers).all():
                pruned_cluster_centers.append(cluster_centers[centers_inside_idx[0]])
                pruned_cluster_centers_idx.append(centers_inside_idx[0])
            if len(centers_inside_idx) > 1:
                rejected_cluster_centers.extend(
                    [cluster_centers[c_idx] for c_idx in centers_inside_idx[1:]
                     if not np.isin(c_idx, pruned_cluster_centers).all()])
    pruned_cluster_centers = np.stack(pruned_cluster_centers)
    return pruned_cluster_centers, pruned_cluster_centers_idx


def prune_cluster_centers_proximity_based_v1(cluster_centers, radius, cluster_distribution):
    rejected_cluster_centers = []
    rejected_cluster_centers_idx = []
    pruned_cluster_centers = []
    pruned_cluster_centers_idx = []
    for cluster_center in cluster_centers:
        if not np.isin(cluster_center, pruned_cluster_centers).all() \
                or not np.isin(cluster_center, rejected_cluster_centers).all():

            if not is_cluster_center_in_the_radius_of_one_of_pruned_centers(
                    cluster_center, pruned_cluster_centers, radius, rejected_cluster_centers):

                centers_inside_idx = find_points_inside_circle(cluster_centers,
                                                               circle_center=cluster_center,
                                                               circle_radius=radius)
                # sort on the basis of cluster distribution
                centers_inside_idx = centers_inside_idx.tolist()
                centers_inside_idx.sort(key=lambda x: cluster_distribution[x], reverse=True)
                centers_inside_idx = np.array(centers_inside_idx)

                if not np.isin(cluster_centers[centers_inside_idx[0]], pruned_cluster_centers).all():
                    pruned_cluster_centers.append(cluster_centers[centers_inside_idx[0]])
                    pruned_cluster_centers_idx.append(centers_inside_idx[0])

                if len(centers_inside_idx) > 1:
                    for c_idx in centers_inside_idx[1:]:
                        if not np.isin(cluster_centers[c_idx], pruned_cluster_centers).all() or \
                                not np.isin(cluster_centers[c_idx], rejected_cluster_centers).all():
                            rejected_cluster_centers.append(cluster_centers[c_idx])
                            rejected_cluster_centers_idx.append(c_idx)

    pruned_cluster_centers = np.stack(pruned_cluster_centers)
    return pruned_cluster_centers, pruned_cluster_centers_idx


def prune_cluster_centers_proximity_based(cluster_centers, radius, cluster_distribution):
    rejected_cluster_centers = []
    rejected_cluster_centers_idx = []
    pruned_cluster_centers = []
    pruned_cluster_centers_idx = []
    for cluster_center in cluster_centers:
        if not np.isin(cluster_center, pruned_cluster_centers).all() \
                or not np.isin(cluster_center, rejected_cluster_centers).all():

            centers_inside_idx = find_points_inside_circle(cluster_centers,
                                                           circle_center=cluster_center,
                                                           circle_radius=radius)
            # sort on the basis of cluster distribution
            centers_inside_idx = centers_inside_idx.tolist()
            centers_inside_idx.sort(key=lambda x: cluster_distribution[x], reverse=True)
            centers_inside_idx = np.array(centers_inside_idx)

            for center_inside_idx in centers_inside_idx:
                if not is_cluster_center_in_the_radius_of_one_of_pruned_centers(
                        cluster_centers[center_inside_idx], pruned_cluster_centers, radius, rejected_cluster_centers):

                    if not np.isin(cluster_centers[center_inside_idx], pruned_cluster_centers).all() and \
                            not np.isin(cluster_centers[center_inside_idx], rejected_cluster_centers).all():
                        pruned_cluster_centers.append(cluster_centers[center_inside_idx])
                        pruned_cluster_centers_idx.append(center_inside_idx)
                else:
                    if not np.isin(cluster_centers[center_inside_idx], rejected_cluster_centers).all() and \
                            not np.isin(cluster_centers[center_inside_idx], pruned_cluster_centers).all():
                        rejected_cluster_centers.append(cluster_centers[center_inside_idx])
                        rejected_cluster_centers_idx.append(center_inside_idx)

    pruned_cluster_centers = np.stack(pruned_cluster_centers)
    return pruned_cluster_centers, pruned_cluster_centers_idx


def is_cluster_center_in_the_radius_of_one_of_pruned_centers(cluster_center, pruned_cluster_centers, radius,
                                                             rejected_cluster_centers):
    for pruned_cluster_center in pruned_cluster_centers:
        if is_point_inside_circle(circle_x=pruned_cluster_center[0], circle_y=pruned_cluster_center[1],
                                  rad=radius, x=cluster_center[0], y=cluster_center[1]):
            # rejected_cluster_centers.append(cluster_center)
            return True

    return False


def prune_based_on_cluster_density(mean_shift, pruned_cluster_centers, pruned_cluster_centers_idx,
                                   min_points_in_cluster=5):
    final_cluster_centers, final_cluster_centers_idx = [], []
    for cluster_center, cluster_center_idx in zip(pruned_cluster_centers, pruned_cluster_centers_idx):
        if mean_shift.cluster_distribution[cluster_center_idx] > min_points_in_cluster:
            final_cluster_centers.append(cluster_center)
            final_cluster_centers_idx.append(cluster_center_idx)
    final_cluster_centers = np.stack(final_cluster_centers) \
        if len(final_cluster_centers) > 0 else np.empty(shape=(0,))
    return final_cluster_centers, final_cluster_centers_idx


def prune_clusters(cluster_centers, mean_shift, radius, min_points_in_cluster=5):
    pruned_cluster_centers, pruned_cluster_centers_idx = prune_cluster_centers_proximity_based(
        cluster_centers, radius, mean_shift.cluster_distribution)
    final_cluster_centers, final_cluster_centers_idx = prune_based_on_cluster_density(
        mean_shift, pruned_cluster_centers, pruned_cluster_centers_idx, min_points_in_cluster=min_points_in_cluster)
    return final_cluster_centers, final_cluster_centers_idx


def extract_features_inside_circle(fg_mask, radius, circle_center):
    # all_cloud = extract_features_per_bounding_box([0, 0, fg_mask.shape[0], fg_mask.shape[1]], fg_mask)
    all_cloud = extract_mask_features(fg_mask)
    points_current_frame_inside_circle_of_validity_idx = find_points_inside_circle(
        cloud=all_cloud,
        circle_radius=radius,
        circle_center=circle_center
    )
    features_inside_circle = all_cloud[points_current_frame_inside_circle_of_validity_idx]
    return all_cloud, features_inside_circle


def features_included_in_live_tracks(annotations, fg_mask, radius, running_tracks, plot=False):
    feature_idx_covered = []
    # all_cloud = extract_features_per_bounding_box([0, 0, fg_mask.shape[0], fg_mask.shape[1]],
    #                                               fg_mask)
    all_cloud = extract_mask_features(fg_mask)
    for t in running_tracks:
        b_center = get_bbox_center(t.bbox).flatten()
        points_current_frame_inside_circle_of_validity_idx = find_points_inside_circle(
            cloud=all_cloud,
            circle_radius=radius,
            circle_center=b_center
        )
        features_inside_circle = all_cloud[points_current_frame_inside_circle_of_validity_idx]
        if plot:
            plot_features_with_mask(all_cloud, features_inside_circle, center=b_center,
                                    radius=radius, mask=fg_mask, box=t.bbox, m_size=1,
                                    current_boxes=annotations[:, :-1])
        feature_idx_covered.extend(points_current_frame_inside_circle_of_validity_idx)
    features_covered = all_cloud[feature_idx_covered]
    return all_cloud, feature_idx_covered, features_covered


def get_track_history(track_id, track_features):
    if len(track_features) == 0 or track_id not in track_features.keys():
        return []
    else:
        track_feature = track_features[track_id]
        assert track_id == track_feature.track_id
        track_object_features: List[ObjectFeatures] = track_feature.object_features
        history = [get_bbox_center(obj_feature.final_bbox).flatten() for obj_feature in track_object_features]
        return history


def get_agent_track_history(track_id, track_features):
    if len(track_features) == 0 or track_id not in track_features.keys():
        return []
    else:
        track_feature = track_features[track_id]
        assert track_id == track_feature.track_id
        track_object_features: List[AgentFeatures] = track_feature.object_features
        history = [get_bbox_center(obj_feature.bbox_t_plus_one).flatten() for obj_feature in track_object_features]
        return history


def get_gt_track_history(track_id, track_features):
    if len(track_features) == 0 or track_id not in track_features.keys():
        return []
    else:
        track_feature = track_features[track_id]
        assert track_id == track_feature.track_id
        track_object_features: Union[List[AgentFeatures], List[ObjectFeatures]] = track_feature.object_features
        history = [get_bbox_center(obj_feature.gt_box).flatten() for obj_feature in track_object_features
                   if obj_feature.gt_box is not None]
        return history


def get_track_velocity_history(track_id, track_features, past_flow=False):
    if len(track_features) == 0 or track_id not in track_features.keys():
        return []
    else:
        track_feature = track_features[track_id]
        assert track_id == track_feature.track_id
        track_object_features = track_feature.object_features
        if past_flow:
            history = [obj_feature.past_flow.mean(0) for obj_feature in track_object_features]
        else:
            history = [obj_feature.flow.mean(0) for obj_feature in track_object_features]
        return history


def get_agent_track_velocity_history(track_id, track_features, past_flow=False):
    if len(track_features) == 0 or track_id not in track_features.keys():
        return []
    else:
        track_feature = track_features[track_id]
        assert track_id == track_feature.track_id
        track_object_features: List[AgentFeatures] = track_feature.object_features
        if past_flow:
            history = [obj_feature.past_flow.mean(0) for obj_feature in track_object_features]
        else:
            history = [obj_feature.future_flow.mean(0) for obj_feature in track_object_features]
        return history


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def filter_features(features, features_to_remove):
    out = features.copy()
    for point in features_to_remove:
        if np.isin(point, out).all():
            unique, count = np.unique(np.argwhere(point == out), return_counts=True)
            try:
                idx = unique[np.where(count == 2)[-1][-1]].item()
            except IndexError:
                idx = unique[0].item()
            out = np.delete(out, idx, axis=0)
    return out


def append_features(features, features_to_append):
    return np.concatenate((features, features_to_append), axis=0)


def filter_for_one_to_one_matches(iou_matrix):
    for d_i, dimension in enumerate(iou_matrix):
        count = np.unique(np.where(dimension))
        if len(count) > 1:
            max_score = np.max(dimension)
            dimension = [c if c == max_score else 0 for c in dimension]
            iou_matrix[d_i] = dimension
    return iou_matrix


def get_bbox_center(b_box):
    x_min = b_box[0]
    y_min = b_box[1]
    x_max = b_box[2]
    y_max = b_box[3]
    x_mid = (x_min + (x_max - x_min) / 2.).astype('int')
    y_mid = (y_min + (y_max - y_min) / 2.).astype('int')

    return np.vstack((x_mid, y_mid)).T


def min_max_to_centroids(bbox):
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    width = x_max - x_min
    height = y_max - y_min
    centroid_x, centroid_y = get_bbox_center(bbox).flatten()
    return [centroid_x, centroid_y, width, height]


def centroids_to_min_max(bbox):
    centroid_x, centroid_y, width, height = bbox
    w_shift, h_shift = width / 2, height / 2
    x_min, y_min = centroid_x - w_shift, centroid_y - h_shift
    x_max, y_max = centroid_x + w_shift, centroid_y + h_shift
    return np.ceil([x_min, y_min, x_max, y_max]).astype(np.int)


def min_max_to_corner_width_height(bbox, bottom_left=True):
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    width = x_max - x_min
    height = y_max - y_min
    if bottom_left:
        return [x_min, y_min, width, height]
    return [x_max, y_max, width, height]


def corner_width_height_to_min_max(bbox, bottom_left=True):
    if bottom_left:
        x_min, y_min, width, height = bbox
        x_max = width + x_min
        y_max = height + y_min
    else:
        x_max, y_max, width, height = bbox
        x_min = x_max - width
        y_min = y_max - height
    return [x_min, y_min, x_max, y_max]


def remove_entries_from_dict(entries, the_dict):
    return_dict = copy.deepcopy(the_dict)
    for key in entries:
        if key in return_dict:
            del return_dict[key]
    return return_dict


def remove_from_dict_except_entries(entries, the_dict):
    return_dict = {}
    for key in entries:
        if key in the_dict.keys():
            return_dict.update({key: the_dict[key]})
    return return_dict


def calculate_flexible_bounding_box(cluster_center_idx, cluster_center_x, cluster_center_y, mean_shift):
    points_in_current_cluster_idx = np.argwhere(mean_shift.labels == cluster_center_idx)
    points_in_current_cluster = mean_shift.data[points_in_current_cluster_idx].squeeze()
    max_value_x = max(points_in_current_cluster[:, 0])
    min_value_x = min(points_in_current_cluster[:, 0])
    max_value_y = max(points_in_current_cluster[:, 1])
    min_value_y = min(points_in_current_cluster[:, 1])
    flexible_height = max_value_y - min_value_y
    flexible_width = max_value_x - min_value_x
    flexible_box = torchvision.ops.box_convert(
        torch.tensor([cluster_center_x, cluster_center_y, flexible_width, flexible_height]),
        'cxcywh', 'xyxy').int().numpy()
    return flexible_box, points_in_current_cluster


def eval_metrics(features_file):
    features = torch.load(features_file)
    tp_list, fp_list, fn_list = features['tp_list'], features['fp_list'], features['fn_list']
    tp_sum, fp_sum, fn_sum = np.array(tp_list).sum(), np.array(fp_list).sum(), np.array(fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f'Precision: {precision} | Recall: {recall}')


def is_box_overlapping_live_boxes(box, live_boxes, threshold=0):
    box = torch.from_numpy(box).unsqueeze(0)
    for l_box in live_boxes:
        l_box = torch.from_numpy(l_box).unsqueeze(0)
        iou = torchvision.ops.box_iou(l_box, box).squeeze().item()
        if iou > threshold and iou > 0:
            return True
    return False


def extract_features_per_bounding_box(box, mask):
    temp_mask = np.zeros_like(mask)
    temp_mask[box[1]:box[3], box[0]:box[2]] = mask[box[1]:box[3], box[0]:box[2]]
    xy = np.argwhere(temp_mask)
    rolled = np.rollaxis(xy, -1).tolist()
    data_x, data_y = rolled[1], rolled[0]
    xy = np.stack([data_x, data_y]).T
    return xy


def extract_flow_inside_box(box, flow):
    extracted_flow = flow[box[1]:box[3], box[0]:box[2]]
    return extracted_flow


def extract_mask_features(mask):
    xy = np.argwhere(mask)
    rolled = np.rollaxis(xy, -1).tolist()
    data_x, data_y = rolled[1], rolled[0]
    xy = np.stack([data_x, data_y]).T
    return xy


def evaluate_shifted_bounding_box(box, shifted_xy, xy):
    xy_center = np.round(xy.mean(axis=0)).astype(np.int)
    shifted_xy_center = np.round(shifted_xy.mean(axis=0)).astype(np.int)
    center_shift = shifted_xy_center - xy_center
    # box_c_x, box_c_y, w, h = min_max_to_centroids(box)
    box_c_x, box_c_y, w, h = torchvision.ops.box_convert(torch.from_numpy(box), 'xyxy', 'cxcywh').numpy()
    # shifted_box = centroids_to_min_max([box_c_x + center_shift[0], box_c_y + center_shift[1], w, h])
    shifted_box = torchvision.ops.box_convert(torch.tensor([box_c_x + center_shift[0], box_c_y + center_shift[1], w, h])
                                              , 'cxcywh', 'xyxy').int().numpy()
    return shifted_box, shifted_xy_center


def calculate_difference_between_centers(box, shifted_box):
    box_center = get_bbox_center(box)
    shifted_box_center = get_bbox_center(shifted_box)
    return shifted_box_center - box_center


def optical_flow_processing(frame, last_frame, second_last_frame, return_everything=False):
    second_last_frame_gs = cv.cvtColor(second_last_frame, cv.COLOR_BGR2GRAY)
    last_frame_gs = cv.cvtColor(last_frame, cv.COLOR_BGR2GRAY)
    frame_gs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    past_flow, past_rgb, past_mag, past_ang = FeatureExtractor.get_optical_flow(
        previous_frame=second_last_frame_gs,
        next_frame=last_frame_gs,
        all_results_out=True)
    flow, rgb, mag, ang = FeatureExtractor.get_optical_flow(
        previous_frame=last_frame_gs,
        next_frame=frame_gs,
        all_results_out=True)
    if return_everything:
        return [[flow, rgb, mag, ang], [past_flow, past_rgb, past_mag, past_ang]]
    return flow, past_flow


def points_pair_stat_analysis(closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair):
    xy_distance_closest_n_points = np.linalg.norm(
        np.expand_dims(closest_n_xy_current_frame_pair, 0) -
        np.expand_dims(closest_n_shifted_xy_pair, 0), 2, axis=0)
    xy_distance_closest_n_points_mean = xy_distance_closest_n_points.mean()
    xy_per_dimension_overlap = np.equal(closest_n_xy_current_frame_pair, closest_n_shifted_xy_pair) \
        .astype(np.float).mean(0)
    xy_overall_dimension_overlap = np.equal(
        closest_n_xy_current_frame_pair, closest_n_shifted_xy_pair).astype(np.float).mean()
    logger.debug(f'xy_distance_closest_n_points_mean: {xy_distance_closest_n_points_mean}\n'
                 f'xy_per_dimension_overlap: {xy_per_dimension_overlap}'
                 f'xy_overall_dimension_overlap: {xy_overall_dimension_overlap}')


def features_filter_append_preprocessing(overlap_percent, shifted_xy, xy_current_frame):
    distance_matrix = clouds_distance_matrix(xy_current_frame, shifted_xy)
    n_point_pair_count = int(min(xy_current_frame.shape[0], shifted_xy.shape[0]) * overlap_percent)
    n_point_pair_count = n_point_pair_count if n_point_pair_count > 0 \
        else min(xy_current_frame.shape[0], shifted_xy.shape[0])
    closest_n_point_pair_idx = smallest_n_indices(distance_matrix, n_point_pair_count)
    closest_n_xy_current_frame_pair = xy_current_frame[closest_n_point_pair_idx[..., 0]]
    closest_n_shifted_xy_pair = shifted_xy[closest_n_point_pair_idx[..., 1]]
    return closest_n_shifted_xy_pair, closest_n_xy_current_frame_pair


def first_frame_processing_and_gt_association(df, first_frame_mask, frame_idx, frame_number, frames, frames_count,
                                              kernel, n, new_shape, original_shape, step, var_threshold,
                                              detect_shadows=True):
    first_frame_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                time_gap_within_frames=3,
                                                total_frames=frames_count, step=step, n=n,
                                                kernel=kernel, var_threshold=var_threshold,
                                                detect_shadows=detect_shadows)
    first_frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
    first_annotations, first_bbox_centers = scale_annotations(first_frame_annotation,
                                                              original_scale=original_shape,
                                                              new_scale=new_shape, return_track_id=False,
                                                              tracks_with_annotations=True)
    return first_annotations, first_frame_mask


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
                             var_threshold, detect_shadows=True):
    selected_past = [(interest_frame_idx - i * time_gap_within_frames) % total_frames for i in range(1, step + 1)]
    selected_future = [(interest_frame_idx + i * time_gap_within_frames) % total_frames for i in range(1, step + 1)]
    selected_frames = selected_past + selected_future
    frames_building_model = [frames[s] for s in selected_frames]

    algo = cv.createBackgroundSubtractorMOG2(history=n, varThreshold=var_threshold, detectShadows=detect_shadows)
    _ = build_mog2_bg_model(n, frames_building_model, kernel, algo)

    mask = algo.apply(frames[interest_frame_idx], learningRate=0)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask


def filter_low_length_tracks(track_based_features, frame_based_features, threshold, low_memory_mode=False):
    logger.info('Level 1 filtering\n')
    if low_memory_mode:
        # f_per_track_features = track_based_features  # RuntimeError: dictionary changed size during iteration
        f_per_track_features = copy.deepcopy(track_based_features)
        f_per_frame_features = frame_based_features
    else:
        # copy to alter the data
        logger.info('copy to alter the data')
        f_per_track_features = copy.deepcopy(track_based_features)
        f_per_frame_features = copy.deepcopy(frame_based_features)

    for track_id, track_features in track_based_features.items():
        dict_track_id = track_features.track_id
        dict_track_object_features = track_features.object_features

        if len(dict_track_object_features) < threshold:
            for track_obj_feature in dict_track_object_features:
                frame_containing_current_track = track_obj_feature.frame_number
                current_track_frame_features = f_per_frame_features[frame_containing_current_track]
                assert frame_containing_current_track == current_track_frame_features.frame_number
                current_track_frame_features_list = copy.deepcopy(current_track_frame_features.object_features)

                frame_for_track_idx = 0
                frame_for_track_idx_debug_count = 0
                for c_frame_idx, current_track_frame_feature in enumerate(current_track_frame_features_list):
                    if current_track_frame_feature.idx == dict_track_id:
                        frame_for_track_idx = c_frame_idx
                        frame_for_track_idx_debug_count += 1

                if frame_for_track_idx_debug_count > 1:
                    logger.info(f'Count: {frame_for_track_idx_debug_count}')
                current_track_frame_features.object_features = \
                    current_track_frame_features_list[0:frame_for_track_idx]
                current_track_frame_features.object_features.extend(
                    current_track_frame_features_list[frame_for_track_idx + 1:])
                logger.info(f'Removed index {frame_for_track_idx} from frame number '
                            f'{current_track_frame_features.frame_number}')

            del f_per_track_features[dict_track_id]
            logger.info(f'Deleted key {dict_track_id} from dict')

    return f_per_track_features, f_per_frame_features


def filter_low_length_tracks_with_skips(track_based_features, frame_based_features, threshold, low_memory_mode=False,
                                        track_ids_to_skip=()):
    logger.info('Level 1 filtering\n')
    if low_memory_mode:
        # f_per_track_features = track_based_features  # RuntimeError: dictionary changed size during iteration
        f_per_track_features = copy.deepcopy(track_based_features)
        f_per_frame_features = frame_based_features
    else:
        # copy to alter the data
        logger.info('copy to alter the data')
        f_per_track_features = copy.deepcopy(track_based_features)
        f_per_frame_features = copy.deepcopy(frame_based_features)

    for track_id, track_features in track_based_features.items():
        dict_track_id = track_features.track_id
        dict_track_object_features = track_features.object_features

        if len(dict_track_object_features) < threshold and dict_track_id not in track_ids_to_skip:
            for track_obj_feature in dict_track_object_features:
                frame_containing_current_track = track_obj_feature.frame_number
                current_track_frame_features = f_per_frame_features[frame_containing_current_track]
                assert frame_containing_current_track == current_track_frame_features.frame_number
                current_track_frame_features_list = copy.deepcopy(current_track_frame_features.object_features)

                frame_for_track_idx = 0
                frame_for_track_idx_debug_count = 0
                for c_frame_idx, current_track_frame_feature in enumerate(current_track_frame_features_list):
                    if current_track_frame_feature.idx == dict_track_id:
                        frame_for_track_idx = c_frame_idx
                        frame_for_track_idx_debug_count += 1

                if frame_for_track_idx_debug_count > 1:
                    logger.info(f'Count: {frame_for_track_idx_debug_count}')
                current_track_frame_features.object_features = \
                    current_track_frame_features_list[0:frame_for_track_idx]
                current_track_frame_features.object_features.extend(
                    current_track_frame_features_list[frame_for_track_idx + 1:])
                logger.info(f'Removed index {frame_for_track_idx} from frame number '
                            f'{current_track_frame_features.frame_number}')

            del f_per_track_features[dict_track_id]
            logger.info(f'Deleted key {dict_track_id} from dict')

    return f_per_track_features, f_per_frame_features


def filter_low_length_tracks_lvl2(track_based_features, frame_based_features, low_memory_mode=False):
    logger.info('\nLevel 2 filtering\n')
    allowed_tracks = list(track_based_features.keys())
    if low_memory_mode:
        f_per_frame_features = frame_based_features
    else:
        logger.info('copy to alter the data')
        f_per_frame_features = copy.deepcopy(frame_based_features)

    for frame_number, frame_features in frame_based_features.items():
        assert frame_number == frame_features.frame_number
        frame_object_features = frame_features.object_features

        for list_idx, object_feature in enumerate(frame_object_features):
            if object_feature.idx in allowed_tracks:
                continue
            else:
                editable_list_dict_value = f_per_frame_features[frame_number]
                editable_list_dict_value_object_features = editable_list_dict_value.object_features

                logger.info(f'The track id {object_feature.idx} in frame {frame_number}'
                            f' was not in allowed track. Hence removed!')

                editable_list_dict_value_object_features.remove(object_feature)

    return f_per_frame_features


def filter_tracks_through_all_steps(track_based_features, frame_based_features, min_track_length_threshold,
                                    low_memory_mode=False):
    track_based_features, frame_based_features = filter_low_length_tracks(
        track_based_features=track_based_features,
        frame_based_features=frame_based_features,
        threshold=min_track_length_threshold,
        low_memory_mode=low_memory_mode)

    frame_based_features = filter_low_length_tracks_lvl2(track_based_features, frame_based_features,
                                                         low_memory_mode=low_memory_mode)

    return track_based_features, frame_based_features


def evaluate_extracted_features(track_based_features, frame_based_features, batch_size=32, do_filter=False,
                                drop_last_batch=True, plot_scale_factor=1, desired_fps=5, custom_video_shape=False,
                                video_mode=True, video_save_location=None, min_track_length_threshold=5,
                                skip_plot_save=False, low_memory_mode=False):
    # frame_track_distribution_pre_filter = {k: len(v.object_features) for k, v in frame_based_features.items()}
    if do_filter:
        track_based_features, frame_based_features = filter_low_length_tracks(
            track_based_features=track_based_features,
            frame_based_features=frame_based_features,
            threshold=min_track_length_threshold,
            low_memory_mode=low_memory_mode)

        frame_based_features = filter_low_length_tracks_lvl2(track_based_features, frame_based_features,
                                                             low_memory_mode=low_memory_mode)
    # frame_track_distribution_post_filter = {k: len(v.object_features) for k, v in frame_based_features.items()}

    frame_based_features_length = len(frame_based_features)

    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size, drop_last=drop_last_batch)
    df = sdd_simple.annotations_df

    l2_distance_hungarian_tp_list, l2_distance_hungarian_fp_list, l2_distance_hungarian_fn_list = [], [], []

    out = None
    frames_shape = sdd_simple.original_shape
    video_shape = (1200, 1000) if custom_video_shape else frames_shape
    original_dims = None
    if video_mode:
        if frames_shape[0] < frames_shape[1]:
            original_dims = (
                frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_location, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[1], video_shape[0]))
            video_shape[0], video_shape[1] = video_shape[1], video_shape[0]
        else:
            original_dims = (
                frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_save_location, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (video_shape[0], video_shape[1]))

    _, meta_info = DATASET_META.get_meta(META_LABEL, VIDEO_NUMBER)
    ratio = float(meta_info.flatten()[-1])

    for part_idx, data in enumerate(tqdm(data_loader)):
        frames, frame_numbers = data
        frames = frames.squeeze()
        frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        frames_count = frames.shape[0]
        original_shape = new_shape = [frames.shape[1], frames.shape[2]]

        for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)), total=len(frame_numbers)):
            frame_number = frame_number.item()
            frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number)
            annotations, bbox_centers = scale_annotations(frame_annotation,
                                                          original_scale=original_shape,
                                                          new_scale=new_shape, return_track_id=False,
                                                          tracks_with_annotations=True)
            gt_boxes = annotations[:, :-1]
            gt_boxes_idx = annotations[:, -1]

            if frame_number < frame_based_features_length:
                next_frame_features = frame_based_features[frame_number + 1]
            else:
                logger.info('Frames remaining, features exhausted!')
                continue

            assert (frame_number + 1) == next_frame_features.frame_number
            next_frame_object_features = next_frame_features.object_features
            generated_boxes = np.array([f.past_bbox for f in next_frame_object_features])
            generated_boxes_idx = np.array([f.idx for f in next_frame_object_features])

            a_boxes_np, r_boxes_np = gt_boxes, generated_boxes
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
                match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                matching_distribution = [[i, j, l2_distance_boxes_score_matrix[i, j]] for i, j in zip(
                    match_rows, match_cols)]
                actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]
            else:
                match_rows, match_cols = np.array([]), np.array([])

            if len(match_rows) != 0:
                if len(match_rows) != len(match_cols):
                    logger.info('Matching arrays length not same!')
                l2_distance_hungarian_tp = len(match_rows)
                l2_distance_hungarian_fp = len(r_boxes_np) - len(match_rows)
                l2_distance_hungarian_fn = len(a_boxes_np) - len(match_rows)

                l2_distance_hungarian_precision = \
                    l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fp)
                l2_distance_hungarian_recall = \
                    l2_distance_hungarian_tp / (l2_distance_hungarian_tp + l2_distance_hungarian_fn)
            else:
                l2_distance_hungarian_tp = 0
                l2_distance_hungarian_fp = 0
                l2_distance_hungarian_fn = len(a_boxes_np)

                l2_distance_hungarian_precision = 0
                l2_distance_hungarian_recall = 0

            l2_distance_hungarian_tp_list.append(l2_distance_hungarian_tp)
            l2_distance_hungarian_fp_list.append(l2_distance_hungarian_fp)
            l2_distance_hungarian_fn_list.append(l2_distance_hungarian_fn)

            if video_mode:
                fig = plot_for_video_current_frame(
                    gt_rgb=frame, current_frame_rgb=frame,
                    gt_annotations=annotations[:, :-1],
                    current_frame_annotation=generated_boxes,
                    new_track_annotation=[],
                    frame_number=frame_number,
                    additional_text=
                    f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                    f'Recall: {l2_distance_hungarian_recall}\n',
                    video_mode=video_mode, original_dims=original_dims, zero_shot=True,
                    box_annotation=[gt_boxes_idx, generated_boxes_idx])

                canvas = FigureCanvas(fig)
                canvas.draw()

                buf = canvas.buffer_rgba()
                out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
                if out_frame.shape[0] != video_shape[1] or out_frame.shape[1] != video_shape[0]:
                    out_frame = skimage.transform.resize(out_frame, (video_shape[1], video_shape[0]))
                    out_frame = (out_frame * 255).astype(np.uint8)

                out.write(out_frame)
            else:
                if not skip_plot_save:
                    fig = plot_for_video_current_frame(
                        gt_rgb=frame, current_frame_rgb=frame,
                        gt_annotations=annotations[:, :-1],
                        current_frame_annotation=generated_boxes,
                        new_track_annotation=[],
                        frame_number=frame_number,
                        additional_text=
                        f'Distance based - Precision: {l2_distance_hungarian_precision} | '
                        f'Recall: {l2_distance_hungarian_recall}\n',
                        video_mode=False, original_dims=original_dims, zero_shot=True,
                        save_path=f'{plot_save_path}zero_shot/plots_filtered/')

            batch_tp_sum, batch_fp_sum, batch_fn_sum = \
                np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
                np.array(l2_distance_hungarian_fn_list).sum()
            batch_precision = batch_tp_sum / (batch_tp_sum + batch_fp_sum)
            batch_recall = batch_tp_sum / (batch_tp_sum + batch_fn_sum)
            logger.info(f'Batch: {part_idx}, '
                        f'L2 Distance Based - Precision: {batch_precision} | Recall: {batch_recall}')

    if video_mode:
        out.release()
    tp_sum, fp_sum, fn_sum = \
        np.array(l2_distance_hungarian_tp_list).sum(), np.array(l2_distance_hungarian_fp_list).sum(), \
        np.array(l2_distance_hungarian_fn_list).sum()
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info('Final')
    logger.info(f'L2 Distance Based - Precision: {precision} | Recall: {recall}')


def extracted_features_in_csv(track_based_features, frame_based_features, do_filter=False,
                              min_track_length_threshold=5, csv_save_path=None, low_memory_mode=False,
                              return_list=False, track_ids_to_skip=None):
    # frame_track_distribution_pre_filter = {k: len(v.object_features) for k, v in frame_based_features.items()}
    if do_filter:
        if track_ids_to_skip is not None:
            track_based_features, frame_based_features = filter_low_length_tracks_with_skips(
                track_based_features=track_based_features,
                frame_based_features=frame_based_features,
                threshold=min_track_length_threshold,
                low_memory_mode=low_memory_mode,
                track_ids_to_skip=track_ids_to_skip)
        else:
            track_based_features, frame_based_features = filter_low_length_tracks(
                track_based_features=track_based_features,
                frame_based_features=frame_based_features,
                threshold=min_track_length_threshold,
                low_memory_mode=low_memory_mode)

        frame_based_features = filter_low_length_tracks_lvl2(track_based_features, frame_based_features,
                                                             low_memory_mode=low_memory_mode)
    # frame_track_distribution_post_filter = {k: len(v.object_features) for k, v in frame_based_features.items()}

    # frame_based_features_length = len(frame_based_features)

    csv_data = []
    for frame_number, frame_feature in tqdm(frame_based_features.items()):
        for object_feature in frame_feature.object_features:
            x_min, y_min, x_max, y_max = object_feature.past_bbox
            center_x, center_y = get_bbox_center(object_feature.past_bbox).flatten()

            if object_feature.past_gt_box is not None:
                gt_x_min, gt_y_min, gt_x_max, gt_y_max = object_feature.past_gt_box
                gt_center_x, gt_center_y = get_bbox_center(object_feature.past_gt_box).flatten()
            else:
                gt_x_min, gt_y_min, gt_x_max, gt_y_max = None, None, None, None
                gt_center_x, gt_center_y = None, None

            csv_data.append([object_feature.idx, x_min, y_min, x_max, y_max, object_feature.frame_number - 1, 'object',
                             center_x, center_y, gt_x_min, gt_y_min, gt_x_max, gt_y_max, gt_center_x, gt_center_y])

    if return_list:
        return csv_data

    df = pd.DataFrame(data=csv_data, columns=['track_id', 'x_min', 'y_min', 'x_max', 'y_max', 'frame_number', 'label',
                                              'center_x', 'center_y', 'gt_x_min', 'gt_y_min', 'gt_x_max', 'gt_y_max',
                                              'gt_center_x', 'gt_center_y'])
    if csv_save_path is not None:
        df.to_csv(csv_save_path + 'generated_annotations.csv', index=False)
        logger.info('CSV saved!')
    return df


def associate_frame_with_ground_truth(frames, frame_numbers):
    return 0


def get_generated_frame_annotations(df: pd.DataFrame, frame_number: int):
    idx: pd.DataFrame = df.loc[df["frame_number"] == frame_number]
    return idx.to_numpy()


def process_plot_per_track_angle_and_history(frame, frame_number, save_plot_path, track_based_accumulated_features,
                                             track_id_to_plot):
    track_to_plot_features = track_based_accumulated_features[track_id_to_plot].object_features[-1] \
        if track_id_to_plot in track_based_accumulated_features.keys() else []
    if isinstance(track_to_plot_features, list):
        track_to_plot_box, track_to_plot_histories, track_to_plot_direction = [], [], []
    else:
        track_to_plot_box = track_to_plot_features.final_bbox

        track_to_plot_histories = []
        for obj_feature in track_based_accumulated_features[track_id_to_plot].object_features:
            track_to_plot_histories.extend(obj_feature.track_history)

        # no need to reverse
        # track_to_plot_histories.reverse()
        track_to_plot_histories = np.array(track_to_plot_histories)

        if track_to_plot_histories.size == 0:
            track_to_plot_histories = np.zeros(shape=(0, 2))

        track_to_plot_direction = track_to_plot_features.velocity_direction[-1] \
            if len(track_to_plot_features.velocity_direction) != 0 else None
    plot_track_history_with_angle_info_with_track_plot(
        img=frame,
        box=track_to_plot_box,
        history=track_to_plot_histories,
        direction=track_to_plot_direction,
        frame_number=frame_number,
        track_id=track_id_to_plot,
        save_path=save_plot_path + f'per_track/{track_id_to_plot}/'
    )
