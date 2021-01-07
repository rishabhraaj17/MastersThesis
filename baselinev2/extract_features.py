from pathlib import Path
from typing import Sequence

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import patches
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations, cal_centers
from average_image.constants import SDDVideoClasses, OBJECT_CLASS_COLOR_MAPPING, ObjectClasses
from average_image.feature_extractor import MOG2, FeatureExtractor
from baseline.extracted_of_optimization import clouds_distance_matrix, smallest_n_indices
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames

initialize_logging()
logger = get_logger(__name__)

SAVE_BASE_PATH = "../Datasets/SDD_Features/"
# SAVE_BASE_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD_Features/"
BASE_PATH = "../Datasets/SDD/"
# BASE_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/"
VIDEO_LABEL = SDDVideoClasses.QUAD
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


class ObjectFeatures(object):
    def __init__(self, idx, xy, uv, bbox, bbox_center):
        super(ObjectFeatures, self).__init__()
        self.idx = idx
        self.xy = xy
        self.uv = uv
        self.bbox = bbox
        self.bbox_center = bbox_center


class FrameFeatures(object):
    def __init__(self, frame_number: int, object_features: Sequence[ObjectFeatures]):
        super(FrameFeatures, self).__init__()
        self.frame_number = frame_number
        self.object_features = object_features


def plot_image(im):
    plt.imshow(im, cmap='gray')
    plt.show()


def plot_with_one_bboxes(img, boxes):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none',
                            figsize=(12, 10))
    axs.imshow(img, cmap='gray')
    for box in boxes:
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                 edgecolor='r', fill=False,
                                 linewidth=None)
        axs.add_patch(rect)
    plt.show()


def plot_processing_steps(xy_cloud, shifted_xy_cloud, xy_box, shifted_xy_box,
                          final_cloud, xy_cloud_current_frame, frame_number, track_id,
                          selected_past, selected_current,
                          true_cloud_key_point=None, shifted_cloud_key_point=None,
                          overlap_threshold=None, shift_corrected_cloud_key_point=None,
                          key_point_criteria=None, shift_correction=None,
                          line_width=None, plot_save_path=None):
    fig, ax = plt.subplots(2, 2, sharex='none', sharey='none', figsize=(14, 12))
    ax1, ax2, ax3, ax4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    # cloud1
    ax1.plot(xy_cloud[:, 0], xy_cloud[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8)
    ax1.plot(selected_past[:, 0], selected_past[:, 1], 'o', markerfacecolor='silver',
             markeredgecolor='k', markersize=8)
    # ax1.plot(true_cloud_key_point[0], true_cloud_key_point[1], '*', markerfacecolor='silver', markeredgecolor='k',
    #          markersize=9)
    rect1 = patches.Rectangle(xy=(xy_box[0], xy_box[1]), width=xy_box[2] - xy_box[0],
                              height=xy_box[3] - xy_box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    # cloud2
    ax2.plot(xy_cloud_current_frame[:, 0], xy_cloud_current_frame[:, 1], 'o', markerfacecolor='magenta',
             markeredgecolor='k', markersize=8)
    ax2.plot(selected_current[:, 0], selected_current[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=8)
    # ax2.plot(shifted_cloud_key_point[0], shifted_cloud_key_point[1], '*', markerfacecolor='yellow',
    #          markeredgecolor='k', markersize=9)
    rect2 = patches.Rectangle(xy=(shifted_xy_box[0], shifted_xy_box[1]), width=shifted_xy_box[2] - shifted_xy_box[0],
                              height=shifted_xy_box[3] - shifted_xy_box[1], fill=False,
                              linewidth=line_width, edgecolor='teal')

    # cloud1 + cloud2
    # cloud1
    ax3.plot(xy_cloud_current_frame[:, 0], xy_cloud_current_frame[:, 1], '*', markerfacecolor='blue',
             markeredgecolor='k', markersize=8)
    rect3 = patches.Rectangle(xy=(xy_box[0], xy_box[1]), width=xy_box[2] - xy_box[0],
                              height=xy_box[3] - xy_box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    # cloud2
    ax3.plot(shifted_xy_cloud[:, 0], shifted_xy_cloud[:, 1], 'o', markerfacecolor='magenta',
             markeredgecolor='k', markersize=8)
    rect3_shifted = patches.Rectangle(xy=(shifted_xy_box[0], shifted_xy_box[1]),
                                      width=shifted_xy_box[2] - shifted_xy_box[0],
                                      height=shifted_xy_box[3] - shifted_xy_box[1], fill=False,
                                      linewidth=line_width, edgecolor='teal')
    ax3.plot(selected_past[:, 0], selected_past[:, 1], 'o', markerfacecolor='silver',
             markeredgecolor='k', markersize=8)
    ax3.plot(selected_current[:, 0], selected_current[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=8)
    # ax3.plot(true_cloud_key_point[0], true_cloud_key_point[1], '*', markerfacecolor='silver', markeredgecolor='k',
    #          markersize=9)
    # ax3.plot(shifted_cloud_key_point[0], shifted_cloud_key_point[1], '*', markerfacecolor='yellow',
    #          markeredgecolor='k', markersize=9)

    # cloud1 + cloud2 - final selected cloud
    # cloud1
    ax4.plot(final_cloud[:, 0], final_cloud[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8)
    # rect4 = patches.Rectangle(xy=(xy_box[0], xy_box[1]), width=xy_box[2] - xy_box[0],
    #                           height=xy_box[3] - xy_box[1], fill=False,
    #                           linewidth=line_width, edgecolor='r')
    # # cloud2
    # ax4.plot(shifted_xy_cloud[:, 0], shifted_xy_cloud[:, 1], 'o', markerfacecolor='magenta', markeredgecolor='k',
    #          markersize=8)
    rect4_shifted = patches.Rectangle(xy=(shifted_xy_box[0], shifted_xy_box[1]),
                                      width=shifted_xy_box[2] - shifted_xy_box[0],
                                      height=shifted_xy_box[3] - shifted_xy_box[1], fill=False,
                                      linewidth=line_width, edgecolor='teal')
    # ax4.plot(true_cloud_key_point[0], true_cloud_key_point[1], '*', markerfacecolor='silver', markeredgecolor='k',
    #          markersize=9)
    # ax4.plot(shift_corrected_cloud_key_point[0], shift_corrected_cloud_key_point[1], '*', markerfacecolor='plum',
    #          markeredgecolor='k', markersize=9)

    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    ax3.add_patch(rect3)
    # ax4.add_patch(rect4)
    ax3.add_patch(rect3_shifted)
    ax4.add_patch(rect4_shifted)

    # original_error = np.linalg.norm(true_cloud_key_point - shifted_cloud_key_point, 2)
    # optimized_error = np.linalg.norm(true_cloud_key_point - shift_corrected_cloud_key_point, 2)

    ax1.set_title('XY Past')
    ax2.set_title('XY Current')
    ax3.set_title(f'XY Past Shifted + XY Current')
    ax4.set_title(f'Final Selected XY')

    legends_dict = {'blue': 'Points at (T-1)',
                    'magenta': 'Points at T',
                    'r': '(T-1) Bounding Box',
                    'silver': 'Selected XY',
                    # 'plum': f'Shift Corrected {key_point_criteria}',
                    'yellow': 'Selected XY Current',
                    'teal': '(T-1) OF Shifted Bounding Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)
    fig.suptitle(f'Frame: {frame_number} | Track Id: {track_id}')  # \nShift Correction: {shift_correction}\n'
    # f'Overlap Threshold: {overlap_threshold}')

    if plot_save_path is None:
        plt.show()
    else:
        Path(plot_save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_save_path + f'fig_frame_{frame_number}_track_{track_id}.png')
        plt.close()


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


def preprocess_data(save_per_part_path=SAVE_PATH, batch_size=32, var_threshold=None, overlap_percent=0.1):
    # feature_extractor = MOG2.for_frames()
    sdd_simple = SDDSimpleDataset(root=BASE_PATH, video_label=VIDEO_LABEL, frames_per_clip=1, num_workers=8,
                                  num_videos=1, video_number_to_use=VIDEO_NUMBER,
                                  step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                  single_track_mode=False, track_id=5, multiple_videos=False)
    data_loader = DataLoader(sdd_simple, batch_size)
    df = sdd_simple.annotations_df
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    n = 30
    step = n // 2
    save_per_part_path += 'parts/'
    first_frame_bounding_boxes, first_frame_mask, last_frame, second_last_frame = None, None, None, None
    last_frame_bounding_boxes, last_frame_mask = None, None
    current_track_idx, track_ids_used = 0, []
    accumulated_features = {}
    for part_idx, data in enumerate(tqdm(data_loader)):
        frames, frame_numbers = data
        frames = frames.squeeze()
        frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        frames_count = frames.shape[0]
        original_shape = new_shape = [frames.shape[1], frames.shape[2]]
        for frame_idx, (frame, frame_number) in enumerate(zip(frames, frame_numbers)):
            if part_idx == 0 and frame_idx == 0:
                first_frame_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                            time_gap_within_frames=3,
                                                            total_frames=frames_count, step=step, n=n,
                                                            kernel=kernel, var_threshold=var_threshold)
                first_frame_annotation = get_frame_annotations_and_skip_lost(df, frame_number.item())
                first_annotations, first_bbox_centers = scale_annotations(first_frame_annotation,
                                                                          original_scale=original_shape,
                                                                          new_scale=new_shape, return_track_id=False,
                                                                          tracks_with_annotations=True)
                first_frame_bounding_boxes = first_annotations[:, :-1]
                last_frame = frame.copy()
                second_last_frame = last_frame.copy()
                last_frame_bounding_boxes = first_frame_bounding_boxes.copy()
                last_frame_mask = first_frame_mask.copy()
            else:
                bounding_boxes = []
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
                fg_mask = get_mog2_foreground_mask(frames=frames, interest_frame_idx=frame_idx,
                                                   time_gap_within_frames=3,
                                                   total_frames=frames_count, step=step, n=n,
                                                   kernel=kernel, var_threshold=var_threshold)
                for b_idx, box in enumerate(last_frame_bounding_boxes):
                    temp_mask = np.zeros_like(last_frame_mask)
                    temp_mask[box[1]:box[3], box[0]:box[2]] = last_frame_mask[box[1]:box[3], box[0]:box[2]]
                    xy = np.argwhere(temp_mask)

                    rolled = np.rollaxis(xy, -1).tolist()
                    data_x, data_y = rolled[1], rolled[0]
                    xy = np.stack([data_x, data_y]).T

                    if xy.size == 0:
                        continue

                    # calculate flow for the features
                    xy_displacement = flow[xy[:, 1], xy[:, 0]]
                    past_xy_displacement = past_flow[xy[:, 1], xy[:, 0]]
                    # xy_displacement = flow[xy[:, 0], xy[:, 1]]
                    # past_xy_displacement = past_flow[xy[:, 0], xy[:, 1]]

                    # shift bounding box by the average flow for localization
                    shifted_xy = xy + xy_displacement
                    xy_center = np.round(xy.mean(axis=0)).astype(np.int)
                    shifted_xy_center = np.round(shifted_xy.mean(axis=0)).astype(np.int)
                    center_shift = shifted_xy_center - xy_center
                    box_c_x, box_c_y, w, h = min_max_to_centroids(box)
                    shifted_box = centroids_to_min_max([box_c_x + center_shift[0], box_c_y + center_shift[1], w, h])
                    box_center = get_bbox_center(box)
                    shifted_box_center = get_bbox_center(shifted_box)
                    box_center_diff = shifted_box_center - box_center
                    bounding_boxes.append(shifted_box)

                    # features to keep - throw N% and keep N%

                    # get activations
                    temp_mask_current_frame = np.zeros_like(fg_mask)
                    temp_mask_current_frame[shifted_box[1]:shifted_box[3], shifted_box[0]:shifted_box[2]] = \
                        fg_mask[shifted_box[1]:shifted_box[3], shifted_box[0]:shifted_box[2]]
                    xy_current_frame = np.argwhere(temp_mask_current_frame)
                    rolled_cf = np.rollaxis(xy_current_frame, -1).tolist()
                    data_x_cf, data_y_cf = rolled_cf[1], rolled_cf[0]
                    xy_current_frame = np.stack([data_x_cf, data_y_cf]).T

                    # compare activations to keep and throw
                    distance_matrix = clouds_distance_matrix(xy_current_frame, shifted_xy)
                    closest_n_point_pair_idx = smallest_n_indices(
                        distance_matrix,
                        int(min(xy_current_frame.shape[0], shifted_xy.shape[0]) * overlap_percent))

                    closest_n_xy_current_frame_pair = xy_current_frame[closest_n_point_pair_idx[..., 0]]
                    closest_n_shifted_xy_pair = shifted_xy[closest_n_point_pair_idx[..., 1]]

                    xy_distance_closest_n_points = np.linalg.norm(
                        np.expand_dims(closest_n_xy_current_frame_pair, 0) -
                        np.expand_dims(closest_n_shifted_xy_pair, 0), 2, axis=0)
                    xy_distance_closest_n_points_mean = xy_distance_closest_n_points.mean()
                    xy_per_dimension_overlap = np.equal(closest_n_xy_current_frame_pair, closest_n_shifted_xy_pair) \
                        .astype(np.float).mean(0)
                    xy_overall_dimension_overlap = np.equal(
                        closest_n_xy_current_frame_pair, closest_n_shifted_xy_pair).astype(np.float).mean()
                    logger.info(f'xy_distance_closest_n_points_mean: {xy_distance_closest_n_points_mean}\n'
                                f'xy_per_dimension_overlap: {xy_per_dimension_overlap}'
                                f'xy_overall_dimension_overlap: {xy_overall_dimension_overlap}')
                    filtered_shifted_xy = filter_features(shifted_xy, closest_n_shifted_xy_pair)
                    final_features_xy = append_features(filtered_shifted_xy, closest_n_xy_current_frame_pair)
                    plot_processing_steps(xy_cloud=xy, shifted_xy_cloud=shifted_xy, xy_box=box,
                                          shifted_xy_box=shifted_box, final_cloud=final_features_xy,
                                          xy_cloud_current_frame=xy_current_frame, frame_number=frame_idx,
                                          track_id=current_track_idx, selected_past=closest_n_shifted_xy_pair,
                                          selected_current=closest_n_xy_current_frame_pair)
                    track_ids_used.append(current_track_idx)
                    current_track_idx += 1
                    print()

                second_last_frame = last_frame.copy()
                last_frame = frame.copy()
        gt_associated_frame = associate_frame_with_ground_truth(frames, frame_numbers)
        if save_per_part_path is not None:
            Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
            f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part_gt_velocity{part_idx}.pt'
            torch.save(accumulated_features, save_per_part_path + f_n)

    return accumulated_features


if __name__ == '__main__':
    feats = preprocess_data(var_threshold=150)
