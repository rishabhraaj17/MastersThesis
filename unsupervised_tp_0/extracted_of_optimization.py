import torch

import numpy as np

from average_image.constants import SDDVideoClasses
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


def cost_function(of_points, center, top_k):
    num_points, dim = of_points.shape
    center_stacked = np.repeat(center, repeats=num_points, axis=0)
    dist = np.linalg.norm((of_points - center_stacked), ord=2, axis=1)
    z = (1 / dist).sum()
    weighted_of_points = (1 / dist) * of_points
    weighted_of_points = (1 / z) * weighted_of_points
    top_k_indices = largest_indices(weighted_of_points, top_k)[0]
    return weighted_of_points, weighted_of_points[top_k_indices]


def of_optimization(data_features, bounding_box, bounding_box_centers):
    print()


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

    dataset_x, dataset_y, dataset_frames, dataset_track_ids, dataset_center_x, dataset_center_y, \
    dataset_bbox_x, dataset_bbox_y, dataset_center_data = dataset_feats['x'], dataset_feats['y'], \
                                                          dataset_feats['frames'], \
                                                          dataset_feats['track_ids'], dataset_feats[
                                                              'bbox_center_x'], \
                                                          dataset_feats['bbox_center_y'], dataset_feats[
                                                              'bbox_x'], \
                                                          dataset_feats['bbox_y'], dataset_feats['center_based']
    of_optimization(data_features=dataset_x, bounding_box=dataset_bbox_x, bounding_box_centers=dataset_center_x)
