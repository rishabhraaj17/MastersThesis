import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from average_image.constants import SDDVideoClasses
from average_image.utils import is_inside_bbox
from log import get_logger, initialize_logging

initialize_logging()
logger = get_logger(__name__)


def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)


def least_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n least indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, array.shape)


def cost_function(of_points, center, top_k, alpha, bbox, weight_points_inside_bbox_more=True):
    points_inside_bbox_idx = []
    if weight_points_inside_bbox_more:
        points_inside_bbox = [is_inside_bbox(point, bbox) for point in of_points]
        points_inside_bbox_idx = (np.array(points_inside_bbox) > 0).nonzero()[0]
    center = np.expand_dims(center, axis=0)
    num_points, dim = of_points.shape
    center_stacked = np.repeat(center, repeats=num_points, axis=0)
    dist = np.linalg.norm((of_points - center_stacked), ord=2, axis=1, keepdims=True)
    z = (1 / dist).sum()
    weighted_of_points = np.power((1 / dist) * of_points, alpha)
    weighted_of_points = (1 / z) * weighted_of_points  # normalize 0-1?
    if weight_points_inside_bbox_more and len(points_inside_bbox_idx) > 0:
        weighted_of_points[points_inside_bbox_idx] = weighted_of_points[points_inside_bbox_idx] * 100
    top_k_indices = largest_indices(weighted_of_points, top_k)[0]
    return weighted_of_points, of_points[top_k_indices]


def of_optimization(data_features_x, data_features_y, bounding_box_x, bounding_box_y, bounding_box_centers_x,
                    bounding_box_centers_y, alpha):
    for features_x, bboxs_x, bbox_centers_x, features_y, bboxs_y, bbox_centers_y in \
            zip(data_features_x, bounding_box_x, bounding_box_centers_x, data_features_y, bounding_box_y,
                bounding_box_centers_y):
        for feature_x, bbox_x, bbox_center_x, feature_y, bbox_y, bbox_center_y in \
                zip(features_x, bboxs_x, bbox_centers_x, features_y, bboxs_y, bbox_centers_y):
            t0_points_xy, t0_points_uv, t0_points_past_uv = feature_x[:, :2], feature_x[:, 2:4], feature_x[:, 4:]
            t1_points_xy, t1_points_uv, t1_points_past_uv = feature_y[:, :2], feature_y[:, 2:4], feature_y[:, 4:]
            t0_x_min, t0_y_min, t0_x_max, t0_y_max = bbox_x
            t0_x, t0_y, t0_w, t0_h = t0_x_min, t0_y_min, (t0_x_max - t0_x_min), (t0_y_max - t0_y_min)
            t1_x_min, t1_y_min, t1_x_max, t1_y_max = bbox_y
            t1_x, t1_y, t1_w, t1_h = t1_x_min, t1_y_min, (t1_x_max - t1_x_min), (t1_y_max - t1_y_min)

            shifted_points = t0_points_xy + t0_points_uv

            weighted_shifted_points, weighted_shifted_points_top_k = cost_function(shifted_points, bbox_center_y,
                                                                                   top_k=10, alpha=alpha, bbox=bbox_y,
                                                                                   weight_points_inside_bbox_more=True)

            plot_basic_analysis(bbox_center_y, shifted_points, t1_h, t1_points_xy, t1_w, t1_x, t1_y,
                                weighted_shifted_points_top_k)


def plot_basic_analysis(bbox_center_y, shifted_points, t1_h, t1_points_xy, t1_w, t1_x, t1_y,
                        weighted_shifted_points_top_k=None):
    plt.plot(shifted_points[:, 0], shifted_points[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8, label='OF Shifted')
    plt.plot(t1_points_xy[:, 0], t1_points_xy[:, 1], '*', markerfacecolor='green', markeredgecolor='k',
             markersize=8, label='GT Points')
    plt.plot(bbox_center_y[0], bbox_center_y[1], '+', markerfacecolor='black', markeredgecolor='k',
             markersize=8, label='Bbox Center')
    if weighted_shifted_points_top_k is not None:
        plt.plot(weighted_shifted_points_top_k[:, 0], weighted_shifted_points_top_k[:, 1], '*', markerfacecolor='pink',
                 markeredgecolor='k', markersize=8, label='Weighted TopK')
    plt.gca().add_patch(Rectangle((t1_x, t1_y), t1_w, t1_h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    time_steps = 5
    base_path = "../Datasets/SDD/"
    save_base_path = "../Datasets/SDD_Features/"
    vid_label = SDDVideoClasses.LITTLE
    video_number = 3

    save_path = f'{save_base_path}{vid_label.value}/video{video_number}/'
    file_name = f'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_' \
                f'center_based_gt_velocity_t{time_steps}.pt'

    logger.info('Loading Dataset')
    dataset_feats = torch.load(save_path + file_name)
    logger.info('Dataset Loaded')

    dataset_x, dataset_y, dataset_frames, dataset_track_ids, dataset_center_x, dataset_center_y, dataset_bbox_x, \
    dataset_bbox_y, dataset_center_data = dataset_feats['x'], dataset_feats['y'], dataset_feats['frames'], \
                                          dataset_feats['track_ids'], dataset_feats['bbox_center_x'], \
                                          dataset_feats['bbox_center_y'], dataset_feats['bbox_x'], \
                                          dataset_feats['bbox_y'], dataset_feats['center_based']

    of_optimization(data_features_x=dataset_x, data_features_y=dataset_y, bounding_box_x=dataset_bbox_x,
                    bounding_box_y=dataset_bbox_y, bounding_box_centers_x=dataset_center_x,
                    bounding_box_centers_y=dataset_center_y, alpha=1)
