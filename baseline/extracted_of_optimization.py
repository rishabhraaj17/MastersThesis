from pathlib import Path

import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses
from average_image.utils import is_inside_bbox
from log import get_logger, initialize_logging

initialize_logging()
logger = get_logger(__name__)

SHIFT_X = 15
SHIFT_Y = 12
CLOSEST_N_POINTS = 10


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
    try:
        top_k_indices = largest_indices(weighted_of_points, top_k)[0]
    except ValueError:
        top_k_indices = largest_indices(weighted_of_points, len(weighted_of_points))[0]
    return weighted_of_points, of_points[top_k_indices]


def of_optimization(data_features_x, data_features_y, bounding_box_x, bounding_box_y, bounding_box_centers_x,
                    bounding_box_centers_y, data_frames, data_track_ids, alpha, save_plots=True, plot_save_path=None):
    for idx, (features_x, bboxs_x, bbox_centers_x, features_y, bboxs_y, bbox_centers_y, frames, track_ids) in \
            enumerate(tqdm(zip(data_features_x, bounding_box_x, bounding_box_centers_x, data_features_y, bounding_box_y,
                               bounding_box_centers_y, data_frames, data_track_ids), total=len(data_features_x))):
        for feature_x, bbox_x, bbox_center_x, feature_y, bbox_y, bbox_center_y, frame, track_id in \
                zip(features_x, bboxs_x, bbox_centers_x, features_y, bboxs_y, bbox_centers_y, frames, track_ids):
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
            if idx % 5000 == 0:
                plot_basic_analysis(bbox_center_y, shifted_points, t1_h, t1_points_xy, t1_w, t1_x, t1_y,
                                    save=save_plots, weighted_shifted_points_top_k=weighted_shifted_points_top_k,
                                    plot_save_path=plot_save_path,
                                    plt_file_name=f'frame-{frame}-track_id-{track_id}.png')


def plot_basic_analysis(bbox_center_y, shifted_points, t1_h, t1_points_xy, t1_w, t1_x, t1_y, save,
                        weighted_shifted_points_top_k=None, plot_save_path=None, plt_file_name=None):
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
    if save and plot_save_path is not None:
        Path(plot_save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_save_path + plt_file_name)
        plt.close()
    else:
        plt.show()


def plot_images_with_bbox(img1, img2, box1, box2, line_width=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='none', sharey='none',
                                   figsize=(12, 10))
    ax1.imshow(img1)
    # for box in bbox:
    rect1 = patches.Rectangle(xy=(box1[0], box1[1]), width=box1[2] - box1[0], height=box1[3] - box1[1], fill=False,
                              linewidth=line_width, edgecolor='green')
    ax2.imshow(img2)
    # for box in bbox:
    rect2 = patches.Rectangle(xy=(box2[0], box2[1]), width=box2[2] - box2[0], height=box2[3] - box2[1], fill=False,
                              linewidth=line_width, edgecolor='green')
    ax1.set_title('Previous')
    ax2.set_title('Next')
    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    plt.show()


def plot_images(img1, img2):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='none', sharey='none',
                                   figsize=(12, 10))
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax1.set_title('First')
    ax2.set_title('Last')
    plt.show()


def plot_points_with_bbox(points1, points2, box1, box2, line_width=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='none', sharey='none',
                                   figsize=(12, 6))
    ax1.plot(points1[:, 0], points1[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8, label='OF Shifted')
    # for box in bbox:
    rect1 = patches.Rectangle(xy=(box1[0], box1[1]), width=box1[2] - box1[0], height=box1[3] - box1[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    ax2.plot(points2[:, 0], points2[:, 1], '*', markerfacecolor='green', markeredgecolor='k',
             markersize=8, label='GT Points')
    # for box in bbox:
    rect2 = patches.Rectangle(xy=(box2[0], box2[1]), width=box2[2] - box2[0], height=box2[3] - box2[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    ax1.set_title('OF Shifted')
    ax2.set_title('GT Points')
    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    plt.show()


def plot_point_with_bbox(points, box, line_width=None):
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none')
    ax.plot(points[:, 0], points[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
            markersize=8, label='Points')
    rect1 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    ax.add_patch(rect1)
    plt.show()


def plot_point_with_center_and_bbox(points, box, center, line_width=None):
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none')
    ax.plot(points[:, 0], points[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
            markersize=8, label='Points')
    ax.plot(center[0], center[1], '*', markerfacecolor='red', markeredgecolor='k',
            markersize=8, label='Center')
    rect1 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    ax.add_patch(rect1)
    plt.show()


def plot_point_with_circle_around_center_and_bbox(points, box, center, circle_radius, line_width=None):
    circle = plt.Circle((center[0], center[1]), circle_radius, color='green', fill=False)
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(6, 6))
    ax.plot(points[:, 0], points[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
            markersize=8, label='Points')
    ax.plot(center[0], center[1], '*', markerfacecolor='red', markeredgecolor='k',
            markersize=8, label='Center')
    rect1 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    ax.add_patch(rect1)
    ax.add_artist(circle)
    plt.show()


def plot_point_in_and_out_with_circle_around_center_and_bbox(points, box, center, circle_radius, points_inside,
                                                             line_width=None):
    circle = plt.Circle((center[0], center[1]), circle_radius, color='green', fill=False)
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(6, 6))
    ax.plot(points[:, 0], points[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
            markersize=8, label='Points')
    ax.plot(center[0], center[1], '*', markerfacecolor='red', markeredgecolor='k',
            markersize=8, label='Center')
    ax.plot(points_inside[:, 0], points_inside[:, 1], 'o', markerfacecolor='yellow', markeredgecolor='k',
            markersize=8, label='Points')
    rect1 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    ax.add_patch(rect1)
    ax.add_artist(circle)
    plt.show()


def plot_clouds_in_and_out_with_circle_around_center_and_bbox(cloud1, cloud2, box, center, circle_radius,
                                                              points_inside_cloud1, points_inside_cloud2,
                                                              points_matched_cloud1, points_matched_cloud2,
                                                              points_in_pair_cloud1, points_in_pair_cloud2,
                                                              line_width=None):
    circle1 = plt.Circle((center[0], center[1]), circle_radius, color='green', fill=False)
    circle2 = plt.Circle((center[0], center[1]), circle_radius, color='green', fill=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 6))
    # cloud1
    ax1.plot(cloud1[:, 0], cloud1[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_inside_cloud1[:, 0], points_inside_cloud1[:, 1], 'o', markerfacecolor='yellow', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_matched_cloud1[:, 0], points_matched_cloud1[:, 1], 'o', markerfacecolor='aqua', markeredgecolor='k',
             markersize=8)
    ax1.plot(center[0], center[1], '*', markerfacecolor='red', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_in_pair_cloud1[..., 0], points_in_pair_cloud1[..., 1], 'o', markerfacecolor='orange',
             markeredgecolor='k', markersize=8)
    rect1 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    # cloud2
    ax2.plot(cloud2[:, 0], cloud2[:, 1], 'o', markerfacecolor='magenta', markeredgecolor='k',
             markersize=8)
    ax2.plot(points_inside_cloud2[:, 0], points_inside_cloud2[:, 1], 'o', markerfacecolor='yellow', markeredgecolor='k',
             markersize=8)
    ax2.plot(points_matched_cloud2[:, 0], points_matched_cloud2[:, 1], 'o', markerfacecolor='aqua', markeredgecolor='k',
             markersize=8)
    ax2.plot(center[0], center[1], '*', markerfacecolor='red', markeredgecolor='k',
             markersize=8)
    ax2.plot(points_in_pair_cloud2[..., 0], points_in_pair_cloud2[..., 1], 'o', markerfacecolor='orange',
             markeredgecolor='k', markersize=8)
    rect2 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    ax1.add_patch(rect1)
    ax1.add_artist(circle1)
    ax2.add_patch(rect2)
    ax2.add_artist(circle2)

    legends_dict = {'blue': 'Points at T',
                    'magenta': '(T-1) Shifted points at T',
                    'yellow': 'Points inside circle',
                    'red': 'Shifted point closest to Points at T Center + Bounding Box',
                    'green': 'Circle',
                    'aqua': 'Common points in two clusters',
                    'orange': 'Closest point pair'}

    legend_patches = [mpatches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.show()


def plot_clouds_in_and_out_with_circle_around_center_and_bbox_same_plot(cloud1, cloud2, box, center, circle_radius,
                                                                        points_inside_cloud1, points_inside_cloud2,
                                                                        points_matched_cloud1, points_matched_cloud2,
                                                                        points_in_pair_cloud1, points_in_pair_cloud2,
                                                                        line_width=None):
    circle = plt.Circle((center[0], center[1]), circle_radius, color='green', fill=False)

    fig, ax1 = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(6, 6))
    # cloud1
    ax1.plot(cloud1[:, 0], cloud1[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_inside_cloud1[:, 0], points_inside_cloud1[:, 1], 'o', markerfacecolor='yellow', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_matched_cloud1[:, 0], points_matched_cloud1[:, 1], 'o', markerfacecolor='aqua', markeredgecolor='k',
             markersize=8)
    ax1.plot(center[0], center[1], '*', markerfacecolor='red', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_in_pair_cloud1[..., 0], points_in_pair_cloud1[..., 1], 'o', markerfacecolor='orange',
             markeredgecolor='k', markersize=8)
    rect1 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    # cloud2
    ax1.plot(cloud2[:, 0], cloud2[:, 1], 'o', markerfacecolor='magenta', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_inside_cloud2[:, 0], points_inside_cloud2[:, 1], 'o', markerfacecolor='yellow', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_matched_cloud2[:, 0], points_matched_cloud2[:, 1], 'o', markerfacecolor='aqua', markeredgecolor='k',
             markersize=8)
    ax1.plot(points_in_pair_cloud2[..., 0], points_in_pair_cloud2[..., 1], 'o', markerfacecolor='orange',
             markeredgecolor='k', markersize=8)

    ax1.add_patch(rect1)
    ax1.add_artist(circle)

    legends_dict = {'blue': 'Points at T',
                    'magenta': '(T-1) Shifted points at T',
                    'yellow': 'Points inside circle',
                    'red': 'Shifted point closest to Points at T Center + Bounding Box',
                    'green': 'Circle',
                    'aqua': 'Common points in two clusters',
                    'orange': 'Closest point pair'}

    legend_patches = [mpatches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.show()


def is_point_inside_circle(circle_x, circle_y, rad, x, y):
    if (x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= rad * rad:
        return True
    return False


def find_points_inside_circle(cloud, circle_center, circle_radius):
    x, y = circle_center
    points_inside_circle_idx = ((cloud[..., 0] - x) * (cloud[..., 0] - x)) + \
                               ((cloud[..., 1] - y) * (cloud[..., 1] - y)) <= circle_radius * circle_radius
    return np.where(points_inside_circle_idx == True)[0]


def find_cloud_center(cloud, p=2):
    cloud_mean = cloud.mean(axis=0)
    distance_with_mean = np.linalg.norm(cloud - cloud_mean, p, axis=1)
    return cloud[np.argmin(distance_with_mean)]


def closest_point_in_cloud_to_a_point(cloud, point, p=2):
    distance_with_point = np.linalg.norm(cloud - point, p, axis=1)
    return cloud[np.argmin(distance_with_point)]


def clouds_distance_matrix(cloud1, cloud2, p=2):
    distance_matrix = []
    for point in cloud1:
        distance_matrix.append(np.linalg.norm(cloud2 - point, p, axis=1))
    return np.stack(distance_matrix)


def smallest_n_indices(a, n):
    idx = a.ravel().argsort()[:n]
    return np.stack(np.unravel_index(idx, a.shape)).T


def distance_weighted_optimization(features_path, plot_save_path, alpha=1, save_plots=True):
    logger.info('Loading Dataset')
    dataset_feats = torch.load(features_path)
    logger.info('Dataset Loaded')

    dataset_x, dataset_y, dataset_frames, dataset_track_ids, dataset_center_x, dataset_center_y, dataset_bbox_x, \
    dataset_bbox_y, dataset_center_data = dataset_feats['x'], dataset_feats['y'], dataset_feats['frames'], \
                                          dataset_feats['track_ids'], dataset_feats['bbox_center_x'], \
                                          dataset_feats['bbox_center_y'], dataset_feats['bbox_x'], \
                                          dataset_feats['bbox_y'], dataset_feats['center_based']

    of_optimization(data_features_x=dataset_x, data_features_y=dataset_y, bounding_box_x=dataset_bbox_x,
                    bounding_box_y=dataset_bbox_y, bounding_box_centers_x=dataset_center_x,
                    bounding_box_centers_y=dataset_center_y, alpha=alpha, save_plots=save_plots,
                    plot_save_path=plot_save_path, data_frames=dataset_frames, data_track_ids=dataset_track_ids)


def optimize_optical_flow_object_level_for_frames_old(df, foreground_masks, optical_flow_between_frames, original_shape,
                                                      new_shape):
    for (bg_sub_mask_key, bg_sub_mask_value), (of_between_frames_key, of_between_frames_value) in \
            zip(foreground_masks.items(), optical_flow_between_frames.items()):
        past_frame_activations = (bg_sub_mask_value > 0).nonzero()
        past_frame_flow_idx = of_between_frames_value[past_frame_activations[0], past_frame_activations[1]]
        past_frame_activations_stacked = np.stack(past_frame_activations)
        past_frame_flow_idx = past_frame_flow_idx.T
        optical_flow_shifted_past_frame_activations = np.zeros_like(past_frame_activations_stacked, dtype=np.float)
        optical_flow_shifted_past_frame_activations[0] = past_frame_activations[0] + past_frame_flow_idx[0]
        optical_flow_shifted_past_frame_activations[1] = past_frame_activations[1] + past_frame_flow_idx[1]
        optical_flow_shifted_past_frame_activations = np.round(optical_flow_shifted_past_frame_activations) \
            .astype(np.int)
        optical_flow_shifted_frame = np.zeros_like(bg_sub_mask_value)
        optical_flow_shifted_frame[optical_flow_shifted_past_frame_activations[0],
                                   optical_flow_shifted_past_frame_activations[1]] = 255

        bg_sub_mask_value_next = foreground_masks[bg_sub_mask_key + 1]

        frame_annotation = get_frame_annotations_and_skip_lost(df, bg_sub_mask_key)
        annotations, bbox_centers = scale_annotations(frame_annotation, original_scale=original_shape,
                                                      new_scale=new_shape, return_track_id=False,
                                                      tracks_with_annotations=True)
        # for id_ in range(annotations.shape[0]):
        #     past_mask = np.zeros_like(bg_sub_mask_value)
        #     past_mask[annotations[id_][1]:annotations[id_][3],
        #     annotations[id_][0]:annotations[id_][2]] = \
        #         bg_sub_mask_value[annotations[id_][1]:annotations[id_][3],
        #         annotations[id_][0]:annotations[id_][2]]
        #     object_idx = (past_mask > 0).nonzero()
        #     # plot_points_only(object_idx[0], object_idx[1])
        #     if object_idx[0].size != 0:
        #         past_flow_intensities = past_mask[object_idx[0], object_idx[1]]
        #         past_flow_idx = of_between_frames_value[object_idx[0], object_idx[1]]
        #         # past_flow_idx = optical_flow_till_current_frame[object_idx[0], object_idx[1]]
        print()
    return None


def optimize_optical_flow_object_level_for_frames(df, foreground_masks, optical_flow_between_frames, original_shape,
                                                  new_shape, circle_radius):
    processed_frames = 0
    processed_tracks = 0
    tracks_skipped = 0
    updated_optical_flow_between_frames = {}
    for (bg_sub_mask_key, bg_sub_mask_value), (of_between_frames_key, of_between_frames_value) in \
            zip(foreground_masks.items(), optical_flow_between_frames.items()):

        flow_dict_for_frame = {}

        past_frame_annotation = get_frame_annotations_and_skip_lost(df, bg_sub_mask_key)
        past_annotations, past_bbox_centers, past_track_ids = scale_annotations(past_frame_annotation,
                                                                                original_scale=original_shape,
                                                                                new_scale=new_shape,
                                                                                return_track_id=True,
                                                                                tracks_with_annotations=True)
        next_bg_sub_frame = foreground_masks[bg_sub_mask_key + 1]
        frame_annotation = get_frame_annotations_and_skip_lost(df, bg_sub_mask_key + 1)
        annotations, bbox_centers, track_ids = scale_annotations(frame_annotation, original_scale=original_shape,
                                                                 new_scale=new_shape, return_track_id=True,
                                                                 tracks_with_annotations=True)
        for id_ in range(past_annotations.shape[0]):
            past_mask = np.zeros_like(bg_sub_mask_value)
            past_mask[past_annotations[id_][1]:past_annotations[id_][3],
            past_annotations[id_][0]:past_annotations[id_][2]] = \
                bg_sub_mask_value[past_annotations[id_][1]:past_annotations[id_][3],
                past_annotations[id_][0]:past_annotations[id_][2]]

            past_track_id = past_annotations[id_][-1].item()
            past_bbox_center = past_bbox_centers[id_]
            past_bbox = past_annotations[id_][:4]

            past_object_idx = (past_mask > 0).nonzero()
            past_object_idx = list(past_object_idx)
            past_object_idx[0], past_object_idx[1] = past_object_idx[1], past_object_idx[0]

            if past_object_idx[0].size != 0:
                past_flow_idx = of_between_frames_value[past_object_idx[0], past_object_idx[1]]
                past_object_idx_stacked = np.stack(past_object_idx)
                past_flow_idx = past_flow_idx.T
                past_flow_shifted_points = np.zeros_like(past_object_idx_stacked, dtype=np.float)
                past_flow_shifted_points[0] = past_object_idx[0] + past_flow_idx[0] + SHIFT_X  # x -> u
                past_flow_shifted_points[1] = past_object_idx[1] + past_flow_idx[1] + SHIFT_Y  # y -> v
                past_flow_shifted_points = np.round(past_flow_shifted_points).astype(np.int)
                # plot_point_with_bbox(past_flow_shifted_points.T, box=past_bbox)

                try:
                    track_idx_next_frame = track_ids.tolist().index(past_track_id)
                except ValueError:
                    logger.info(f'SKIPPING: Track id {past_track_id} absent in next frame!')
                    tracks_skipped += 1
                    continue

                mask = np.zeros_like(next_bg_sub_frame)
                mask[annotations[track_idx_next_frame][1]:annotations[track_idx_next_frame][3],
                annotations[track_idx_next_frame][0]:annotations[track_idx_next_frame][2]] = \
                    bg_sub_mask_value[annotations[track_idx_next_frame][1]:annotations[track_idx_next_frame][3],
                    annotations[track_idx_next_frame][0]:annotations[track_idx_next_frame][2]]

                track_id = annotations[track_idx_next_frame][-1].item()
                bbox_center = bbox_centers[track_idx_next_frame]
                bbox = annotations[track_idx_next_frame][:4]
                x_min, y_min, x_max, y_max = bbox
                x, y, w, h = x_min, y_min, (x_max - x_min), (y_max - y_min)

                object_idx = (mask > 0).nonzero()
                object_idx = list(object_idx)
                object_idx[0], object_idx[1] = object_idx[1], object_idx[0]
                object_idx_stacked = np.stack(object_idx)

                object_idx_stacked_center = find_cloud_center(object_idx_stacked.T)
                points_object_idx_inside_circle = find_points_inside_circle(cloud=object_idx_stacked.T,
                                                                            circle_center=
                                                                            object_idx_stacked_center,
                                                                            circle_radius=circle_radius)
                # plot_point_in_and_out_with_circle_around_center_and_bbox(object_idx_stacked.T, box=bbox,
                #                                                          center=object_idx_stacked_center,
                #                                                          circle_radius=circle_radius,
                #                                                          points_inside=
                #                                                          object_idx_stacked.T[
                #                                                              points_object_idx_inside_circle])

                distance_matrix = clouds_distance_matrix(object_idx_stacked.T, past_flow_shifted_points.T)
                closest_point_pair_idx = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
                closest_n_point_pair_idx = smallest_n_indices(distance_matrix, CLOSEST_N_POINTS)
                closest_n_true_point_pair = object_idx_stacked.T[closest_n_point_pair_idx[..., 0]]
                closest_n_shifted_point_pair = past_flow_shifted_points.T[closest_n_point_pair_idx[..., 1]]
                true_point_in_pair = object_idx_stacked.T[closest_point_pair_idx[0]]
                shifted_point_in_pair = past_flow_shifted_points.T[closest_point_pair_idx[1]]

                # flow for closest N points

                closest_shifted_point_to_object_idx_stacked_center = closest_point_in_cloud_to_a_point(
                    cloud=past_flow_shifted_points.T, point=object_idx_stacked_center
                )
                # closest_shifted_point_to_object_idx_stacked_center = shifted_point_in_pair  # todo: use this as center
                shifted_points_inside_circle_idx = find_points_inside_circle(
                    cloud=past_flow_shifted_points.T,
                    circle_center=
                    closest_shifted_point_to_object_idx_stacked_center,
                    circle_radius=circle_radius)
                true_points_inside_circle_idx = find_points_inside_circle(
                    cloud=object_idx_stacked.T,
                    circle_center=
                    closest_shifted_point_to_object_idx_stacked_center,
                    circle_radius=circle_radius)

                shifted_points_inside_circle = past_flow_shifted_points.T[shifted_points_inside_circle_idx]
                true_points_inside_circle = object_idx_stacked.T[true_points_inside_circle_idx]
                # common_points_inside_circle_bool = (shifted_points_inside_circle == true_points_inside_circle).all(1)
                # common_points_inside_circle_idx = np.where(common_points_inside_circle_bool == True)[0]
                # common_points_inside_circle = true_points_inside_circle[common_points_inside_circle_idx]
                intersect = np.intersect1d(true_points_inside_circle[:, 0], shifted_points_inside_circle[:, 0])
                true_points_matches = true_points_inside_circle[np.any(true_points_inside_circle[:, 0]
                                                                       == intersect[:, None], axis=0)]
                shifted_points_matches = shifted_points_inside_circle[np.any(shifted_points_inside_circle[:, 0]
                                                                             == intersect[:, None], axis=0)]

                plot_clouds_in_and_out_with_circle_around_center_and_bbox(
                    cloud1=object_idx_stacked.T,
                    cloud2=past_flow_shifted_points.T, box=bbox,
                    center=closest_shifted_point_to_object_idx_stacked_center,
                    circle_radius=circle_radius,
                    points_inside_cloud1=true_points_inside_circle,
                    points_inside_cloud2=shifted_points_inside_circle,
                    points_matched_cloud1=true_points_matches,
                    points_matched_cloud2=shifted_points_matches,
                    points_in_pair_cloud1=closest_n_true_point_pair,
                    points_in_pair_cloud2=closest_n_shifted_point_pair)

                plot_clouds_in_and_out_with_circle_around_center_and_bbox_same_plot(
                    cloud1=object_idx_stacked.T,
                    cloud2=past_flow_shifted_points.T, box=bbox,
                    center=closest_shifted_point_to_object_idx_stacked_center,
                    circle_radius=circle_radius,
                    points_inside_cloud1=true_points_inside_circle,
                    points_inside_cloud2=shifted_points_inside_circle,
                    points_matched_cloud1=true_points_matches,
                    points_matched_cloud2=shifted_points_matches,
                    points_in_pair_cloud1=closest_n_true_point_pair,
                    points_in_pair_cloud2=closest_n_shifted_point_pair
                )

                # plot_point_with_bbox(object_idx_stacked.T, box=bbox)

                # plot_basic_analysis(bbox_center_y=bbox_center, shifted_points=past_flow_shifted_points.T,
                #                     t1_points_xy=object_idx_stacked.T, t1_h=h, t1_w=w, t1_x=x, t1_y=y, save=False)

                # plot_images_with_bbox(img1=bg_sub_mask_value, img2=next_bg_sub_frame, box1=past_bbox, box2=bbox)
                # plot_images_with_bbox(img1=past_mask, img2=mask, box1=past_bbox, box2=bbox)

                # plot_points_with_bbox(points1=past_flow_shifted_points.T, points2=object_idx_stacked.T,
                #                       box1=past_bbox, box2=bbox)

                processed_tracks += 1

                # todo: optimization and update the points and keep those flow vectors

                flow_dict_for_frame.update({past_track_id: {'features_xy': past_object_idx,
                                                            'flow_uv': past_flow_idx,
                                                            'shifted_xy': past_flow_shifted_points,
                                                            'shifted_points_inside_circle': shifted_points_matches}})
        updated_flow_map = np.zeros_like(of_between_frames_value)
        for k, v in flow_dict_for_frame.items():
            features_x, features_y = v['features_xy'][0], v['features_xy'][1]
            updated_flow_map[features_x, features_y] = v['flow_uv'].T  # fixme: or flow for chosen points
        updated_optical_flow_between_frames.update({of_between_frames_key: updated_flow_map})
        processed_frames += 1

    final_12_frames_flow = np.zeros((new_shape[0], new_shape[1], 2), dtype=np.float)
    for key, value in updated_optical_flow_between_frames.items():
        final_12_frames_flow += value

    # todo: move bg_sub_frame_0 with this flow and should be close to the 12th (last) frame and capture all objects
    #  , same for future frames -> distance is minimum
    activations = (foreground_masks[0] > 0).nonzero()
    frame_flow_idx = final_12_frames_flow[activations[0], activations[1]]
    activations_stacked = np.stack(activations)
    frame_flow_idx = frame_flow_idx.T
    optical_flow_shifted_past_frame_activations = np.zeros_like(activations_stacked, dtype=np.float)
    optical_flow_shifted_past_frame_activations[0] = activations[0] + frame_flow_idx[0]
    optical_flow_shifted_past_frame_activations[1] = activations[1] + frame_flow_idx[1]
    optical_flow_shifted_past_frame_activations = np.round(optical_flow_shifted_past_frame_activations) \
        .astype(np.int)
    optical_flow_shifted_frame = np.zeros_like(foreground_masks[0])
    optical_flow_shifted_frame[optical_flow_shifted_past_frame_activations[0],
                               optical_flow_shifted_past_frame_activations[1]] = 255

    bg_sub_mask_value_next = foreground_masks[12]
    plot_images(optical_flow_shifted_frame, bg_sub_mask_value_next)
    # can also verify object-level
    logger.info(f'Frames processed: {processed_frames} | Tracks Processed: {processed_tracks} | '
                f'Tracks Skipped: {tracks_skipped}')
    logger.info(f'Overall L2 Distance: {np.linalg.norm(optical_flow_shifted_frame - bg_sub_mask_value_next, 2)}')
    return final_12_frames_flow


if __name__ == '__main__':
    time_steps = 20
    base_path = "../Datasets/SDD/"
    save_base_path = "../Datasets/SDD_Features/"
    vid_label = SDDVideoClasses.LITTLE
    video_number = 3

    save_path = f'{save_base_path}{vid_label.value}/video{video_number}/'
    file_name = f'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_' \
                f'center_based_gt_velocity_t{time_steps}.pt'
    plt_save_path = f'../Plots/optimized_of/T={time_steps}/'

    # distance_weighted_optimization(features_path=save_path + file_name, plot_save_path=plt_save_path, alpha=1,
    #                                save_plots=True)
    data = torch.load('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/baseline/debug_of_opt_batch.pt')
    df_, fg_masks, of_flows, o_shape, n_shape = data['df'], data['foreground_masks'], \
                                                data['optical_flow_between_frames'], data['original_shape'], \
                                                data['new_shape']
    optimize_optical_flow_object_level_for_frames(df=df_, foreground_masks=fg_masks,
                                                  optical_flow_between_frames=of_flows,
                                                  original_shape=o_shape, new_shape=n_shape,
                                                  circle_radius=6)
