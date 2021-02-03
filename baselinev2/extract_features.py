import copy
from enum import Enum
import math
from itertools import cycle
from pathlib import Path
from typing import Sequence, Dict, List, Any, Union

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.lines as mlines
import pytorch_lightning.metrics as plm
import scipy
import skimage
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations, cal_centers
from average_image.constants import SDDVideoClasses, OBJECT_CLASS_COLOR_MAPPING, ObjectClasses, SDDVideoDatasets
from average_image.feature_clustering import MeanShiftClustering
from average_image.feature_extractor import MOG2, FeatureExtractor
from average_image.utils import SDDMeta, is_inside_bbox
from baseline.extracted_of_optimization import clouds_distance_matrix, smallest_n_indices, find_points_inside_circle, \
    is_point_inside_circle
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames, SimpleVideoDatasetBase

initialize_logging()
logger = get_logger(__name__)

SAVE_BASE_PATH = "../Datasets/SDD_Features/"
# SAVE_BASE_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD_Features/"
BASE_PATH = "../Datasets/SDD/"
# BASE_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/"
VIDEO_LABEL = SDDVideoClasses.HYANG
VIDEO_NUMBER = 7
SAVE_PATH = f'{SAVE_BASE_PATH}{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/baseline_v2/'
FILE_NAME_STEP_1 = 'features_v0.pt'
LOAD_FILE_STEP_1 = SAVE_PATH + FILE_NAME_STEP_1
TIME_STEPS = 5

ENABLE_OF_OPTIMIZATION = True
ALPHA = 1
TOP_K = 1
WEIGHT_POINTS_INSIDE_BBOX_MORE = True

META_PATH = '../Datasets/SDD/H_SDD.txt'
DATASET_META = SDDMeta(META_PATH)
META_LABEL = SDDVideoDatasets.HYANG


class STEP(Enum):
    SEMI_SUPERVISED = 0
    UNSUPERVISED = 1
    EXTRACTION = 3
    ALGO_VERSION_1 = 4
    DEBUG = 5
    METRICS = 6
    FILTER_FEATURES = 7
    NN_EXTRACTION = 8


class ObjectDetectionParameters(Enum):
    BEV_TIGHT = {
        'radius': 60,
        'extra_radius': 0,
        'generic_box_wh': 50,
        'detect_shadows': True
    }
    BEV_RELAXED = {
        'radius': 90,
        'extra_radius': 50,
        'generic_box_wh': 100,
        'detect_shadows': True
    }
    SLANTED = {
        'radius': 90,
        'extra_radius': 50,
        'generic_box_wh': 150,
        'detect_shadows': True
    }


EXECUTE_STEP = STEP.UNSUPERVISED


class ObjectFeatures(object):
    def __init__(self, idx, xy, past_xy, final_xy, flow, past_flow, past_bbox, final_bbox, frame_number, history=None,
                 is_track_live=True, gt_history=None, track_direction=None, velocity_history=None,
                 velocity_direction=None, running_velocity=None, per_step_distance=None):
        super(ObjectFeatures, self).__init__()
        self.idx = idx
        self.xy = xy
        self.flow = flow
        self.past_bbox = past_bbox
        self.final_bbox = final_bbox
        self.past_flow = past_flow
        self.final_xy = final_xy
        self.past_xy = past_xy
        self.is_track_live = is_track_live
        self.frame_number = frame_number
        self.gt_box = None
        self.past_gt_box = None
        self.gt_track_idx = None
        self.gt_past_current_distance = None
        self.gt_history = gt_history
        self.track_history = history
        self.track_direction = track_direction
        self.velocity_history = velocity_history
        self.velocity_direction = velocity_direction
        self.per_step_distance = per_step_distance
        self.running_velocity = running_velocity

    def __eq__(self, other):
        return self.idx == other.idx


class AgentFeatures(object):
    def __init__(self, track_idx, activations_t, activations_t_minus_one, activations_t_plus_one, future_flow,
                 past_flow, bbox_t, bbox_t_plus_one, bbox_t_minus_one, frame_number, activations_future_frame,
                 activations_past_frame, final_features_future_activations, is_track_live=True, gt_box=None,
                 past_gt_box=None, gt_track_idx=None, gt_past_current_distance=None, frame_number_t=None,
                 frame_number_t_minus_one=None, frame_number_t_plus_one=None, past_frames_used_in_of_estimation=None,
                 frame_by_frame_estimation=False, future_frames_used_in_of_estimation=None, future_gt_box=None,
                 past_gt_track_idx=None, future_gt_track_idx=None, gt_current_future_distance=None,
                 past_box_inconsistent=False, future_box_inconsistent=False, gt_history=None, history=None,
                 track_direction=None, velocity_history=None, velocity_direction=None):
        super(AgentFeatures, self).__init__()
        self.frame_number_t = frame_number_t
        self.frame_number_t_minus_one = frame_number_t_minus_one
        self.frame_number_t_plus_one = frame_number_t_plus_one
        self.past_frames_used_in_of_estimation = past_frames_used_in_of_estimation
        self.future_frames_used_in_of_estimation = future_frames_used_in_of_estimation
        self.frame_by_frame_estimation = frame_by_frame_estimation
        self.track_idx = track_idx
        self.activations_t = activations_t
        self.future_flow = future_flow
        self.bbox_t = bbox_t
        self.bbox_t_plus_one = bbox_t_plus_one
        self.bbox_t_minus_one = bbox_t_minus_one
        self.past_flow = past_flow
        self.activations_t_plus_one = activations_t_plus_one
        self.activations_t_minus_one = activations_t_minus_one
        self.is_track_live = is_track_live
        self.frame_number = frame_number
        self.activations_past_frame = activations_past_frame
        self.activations_future_frame = activations_future_frame
        self.final_features_future_activations = final_features_future_activations
        self.gt_box = gt_box
        self.future_gt_box = future_gt_box
        self.past_gt_box = past_gt_box
        self.gt_track_idx = gt_track_idx
        self.future_gt_track_idx = future_gt_track_idx
        self.past_gt_track_idx = past_gt_track_idx
        self.gt_past_current_distance = gt_past_current_distance
        self.gt_current_future_distance = gt_current_future_distance
        self.future_box_inconsistent = future_box_inconsistent
        self.past_box_inconsistent = past_box_inconsistent
        self.gt_history = gt_history
        self.track_history = history
        self.track_direction = track_direction
        self.velocity_history = velocity_history
        self.velocity_direction = velocity_direction

    def __eq__(self, other):
        return self.track_idx == other.track_idx


class FrameFeatures(object):
    def __init__(self, frame_number: int, object_features: Union[List[ObjectFeatures], List[AgentFeatures]]
                 , flow=None, past_flow=None):
        super(FrameFeatures, self).__init__()
        self.frame_number = frame_number
        self.object_features = object_features
        self.flow = flow
        self.past_flow = past_flow


class TrackFeatures(object):
    def __init__(self, track_id: int):
        super(TrackFeatures, self).__init__()
        self.track_id = track_id
        self.object_features: Union[List[ObjectFeatures], List[AgentFeatures]] = []

    def __eq__(self, other):
        return self.track_id == other.track_id


class Track(object):
    def __init__(self, bbox, idx, history=None, gt_track_idx=None):
        super(Track, self).__init__()
        self.idx = idx
        self.bbox = bbox
        self.gt_track_idx = gt_track_idx
        self.history = history


def plot_image(im):
    plt.imshow(im, cmap='gray')
    plt.show()


def plot_image_simple(im, bbox=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.imshow(im, cmap='gray')
    if bbox is not None:
        add_box_to_axes(axs, bbox)
    plt.show()


def plot_image_set_of_boxes(im, bbox1=None, bbox2=None, overlay=True, annotate=None):
    if overlay:
        fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
        axs.imshow(im, cmap='gray')
        if bbox1 is not None or bbox2 is not None:
            if annotate is None:
                add_box_to_axes(axs, bbox1, 'r')
                add_box_to_axes(axs, bbox2, 'aqua')
            else:
                add_box_to_axes_with_annotation(axs, bbox1, annotate[0], 'r')
                add_box_to_axes_with_annotation(axs, bbox2, annotate[1], 'aqua')

            legends_dict = {'r': 'GT Bounding Box',
                            'aqua': 'Generated Bounding Box'}

            legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
            fig.legend(handles=legend_patches, loc=2)
    else:
        fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
        axs[0].imshow(im, cmap='gray')
        axs[1].imshow(im, cmap='gray')
        if bbox1 is not None or bbox2 is not None:
            if annotate is None:
                add_box_to_axes(axs[0], bbox1, 'r')
                add_box_to_axes(axs[1], bbox2, 'aqua')
            else:
                add_box_to_axes_with_annotation(axs[0], bbox1, annotate[0], 'r')
                add_box_to_axes_with_annotation(axs[1], bbox2, annotate[1], 'aqua')

            legends_dict = {'r': 'GT Bounding Box',
                            'aqua': 'Generated Bounding Box'}

            legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
            fig.legend(handles=legend_patches, loc=2)
    plt.show()


def plot_random_legends():
    blue_star = mlines.Line2D([], [], color='white', marker='*', linestyle='None',
                              markersize=10, label='Cluster\'s Centroid', markeredgecolor='black')
    red_square = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                               markersize=10, label='Region of Validity', markerfacecolor='white')
    purple_triangle = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                                    markersize=10, label='Cluster Pool Region', markerfacecolor='white')
    purple_triangle0 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                                     markersize=10, label='Inside Region of Validity')
    purple_triangle1 = mlines.Line2D([], [], color='aqua', marker='o', linestyle='None',
                                     markersize=10, label='New track candidate activations')

    plt.legend(handles=[blue_star, red_square, purple_triangle, purple_triangle0, purple_triangle1])

    plt.show()
    # fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    # legends_dict = {'*': 'Cluster\'s Centroid',
    #                 'g': 'Cluster Pool Region',
    #                 'r': 'Ground Truth Bounding Box'}
    #
    # legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    # fig.legend(handles=legend_patches, loc=2)
    # plt.show()


def plot_features_simple(features, bbox=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=8)
    if bbox is not None:
        add_box_to_axes(axs, bbox)
    plt.show()


def plot_features_with_mask_simple(features, mask, bbox=None):
    fig, axs = plt.subplots(2, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(mask, cmap='gray')
    axs[1].plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=8)
    if bbox is not None:
        add_box_to_axes(axs[0], bbox)
        add_box_to_axes(axs[1], bbox)
    plt.show()


def plot_tracks_with_features(frame_t, frame_t_minus_one, frame_t_plus_one, features_t, features_t_minus_one, file_idx,
                              features_t_plus_one, box_t, box_t_minus_one, box_t_plus_one, frame_number, marker_size=8,
                              annotations=None, additional_text='', video_mode=False, save_path=None, track_id=None):
    fig, axs = plt.subplots(1, 3, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(frame_t_minus_one, cmap='gray')
    axs[1].imshow(frame_t, cmap='gray')
    axs[2].imshow(frame_t_plus_one, cmap='gray')

    axs[0].plot(features_t_minus_one[:, 0], features_t_minus_one[:, 1], 'o',
                markerfacecolor='blue', markeredgecolor='k', markersize=marker_size)
    axs[1].plot(features_t[:, 0], features_t[:, 1], 'o',
                markerfacecolor='blue', markeredgecolor='k', markersize=marker_size)
    axs[2].plot(features_t_plus_one[:, 0], features_t_plus_one[:, 1], 'o',
                markerfacecolor='blue', markeredgecolor='k', markersize=marker_size)

    if annotations is not None:
        add_box_to_axes_with_annotation(axs[0], box_t_minus_one, annotations[0])
        add_box_to_axes_with_annotation(axs[1], box_t, annotations[1])
        add_box_to_axes_with_annotation(axs[2], box_t_plus_one, annotations[2])
    else:
        add_box_to_axes(axs[0], box_t_minus_one)
        add_box_to_axes(axs[1], box_t)
        add_box_to_axes(axs[2], box_t_plus_one)

    axs[0].set_title('T-1')
    axs[1].set_title('T')
    axs[2].set_title('T+1')

    fig.suptitle(f'Track - Past|Present|Future\nFrame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'blue': 'Features'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            save_path = save_path + f'{track_id if track_id is not None else "all"}/'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"{file_idx}_track_plot_frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()


def plot_mask_matching_bbox(mask, bboxes, frame_num, save_path=None):
    fig, axs = plt.subplots(3, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(mask, cmap='gray')
    axs[1].imshow(mask, cmap='gray')
    axs[2].imshow(mask, cmap='gray')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    iou = {}
    dist = {}
    for box, color in zip(bboxes, colors):
        a_box = box[0]
        r_box = box[2]
        iou.update({color: box[3]})
        dist.update({color: box[1]})
        add_one_box_to_axis(axs[0], color, a_box)
        add_one_box_to_axis(axs[0], color, r_box)
        add_one_box_to_axis(axs[1], color, a_box)
        add_one_box_to_axis(axs[2], color, r_box)
    axs[0].set_title('Both')
    axs[1].set_title('GT')
    axs[2].set_title('OF')

    fig.suptitle(f'Frame number: {frame_num}\nIOU: {iou}\nL2: {dist}\n')
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path + f"frame_{frame_num}.png")
        plt.close()
    else:
        plt.show()


def add_one_box_to_axis(axs, color, box):
    rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                             edgecolor=color, fill=False, linewidth=None)
    axs.add_patch(rect)


def find_points_inside_box(points, box):
    points_to_alter = points.copy()
    x1, y1, x2, y2 = box

    points_to_alter = points_to_alter[x1 < points_to_alter[..., 0]]
    points_to_alter = points_to_alter[points_to_alter[..., 0] < x2]

    points_to_alter = points_to_alter[y1 < points_to_alter[..., 1]]
    points_to_alter = points_to_alter[points_to_alter[..., 1] < y2]

    return points_to_alter


def plot_features_overlayed_mask_simple(features, mask, bbox=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.imshow(mask, cmap='gray')
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=8)
    if bbox is not None:
        add_box_to_axes(axs, bbox)
    plt.show()


def plot_features_with_mask_and_rgb_simple(features, mask, rgb, bbox=None):
    fig, axs = plt.subplots(3, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(mask, cmap='gray')
    axs[1].plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=8)
    axs[2].imshow(rgb)
    if bbox is not None:
        add_box_to_axes(axs[0], bbox)
        add_box_to_axes(axs[1], bbox)
        add_box_to_axes(axs[2], bbox)
    plt.show()


def plot_features_with_circle(features, features_inside_circle, center, radius, mask=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    circle = plt.Circle((center[0], center[1]), radius, color='green', fill=False)
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8)
    axs.plot(features_inside_circle[:, 0], features_inside_circle[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=8)
    axs.add_artist(circle)
    if mask is not None:
        axs.imshow(mask, 'gray')
    plt.show()


def plot_features(features, features_inside_circle, features_skipped=None, mask=None, cluster_centers=None,
                  marker_size=1, num_clusters=None, frame_number=None, additional_text=None, boxes=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=marker_size)
    axs.plot(features_inside_circle[:, 0], features_inside_circle[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=marker_size)
    if mask is not None:
        axs.imshow(mask, 'gray')
    if features_skipped is not None:
        axs.plot(features_skipped[:, 0], features_skipped[:, 1], 'o', markerfacecolor='aqua',
                 markeredgecolor='k', markersize=marker_size)
    if cluster_centers is not None:
        axs.plot(cluster_centers[:, 0], cluster_centers[:, 1], '*', markerfacecolor='lavender', markeredgecolor='k',
                 markersize=marker_size + 8)
        fig.suptitle(f'Frame: {frame_number} | Clusters Count: {num_clusters}\n {additional_text}')
    if boxes is not None:
        add_box_to_axes(axs, boxes)

    legends_dict = {'yellow': 'Inside Circle',
                    'blue': 'Features',
                    'aqua': 'Skipped Features'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.show()


def plot_features_with_circles(features, features_inside_circle, features_skipped=None, mask=None, cluster_centers=None,
                               marker_size=1, num_clusters=None, frame_number=None, additional_text=None, boxes=None,
                               radius=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=marker_size)
    axs.plot(features_inside_circle[:, 0], features_inside_circle[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=marker_size)
    if mask is not None:
        axs.imshow(mask, 'gray')
    if features_skipped is not None:
        axs.plot(features_skipped[:, 0], features_skipped[:, 1], 'o', markerfacecolor='aqua',
                 markeredgecolor='k', markersize=marker_size)
    if cluster_centers is not None:
        axs.plot(cluster_centers[:, 0], cluster_centers[:, 1], '*', markerfacecolor='lavender', markeredgecolor='k',
                 markersize=marker_size + 8)
        fig.suptitle(f'Frame: {frame_number} | Clusters Count: {num_clusters}\n {additional_text}')
        for c_center in cluster_centers:
            axs.add_artist(plt.Circle((c_center[0], c_center[1]), radius, color='green', fill=False))
    if boxes is not None:
        add_box_to_axes(axs, boxes)
        # for box in boxes:
        #     box_center = get_bbox_center(box).flatten()
        #     axs.add_artist(plt.Circle((box_center[0], box_center[1]), 70.71, color='red', fill=False))

    legends_dict = {'yellow': 'Inside Circle',
                    'blue': 'Features',
                    'aqua': 'Skipped Features'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.show()


def plot_features_with_mask(features, features_inside_circle, center, radius, mask, box=None, m_size=4,
                            current_boxes=None):
    fig, axs = plt.subplots(2, 1, sharex='none', sharey='none', figsize=(12, 10))
    circle = plt.Circle((center[0], center[1]), radius, color='green', fill=False)
    axs[0].plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
                markersize=m_size)
    axs[0].plot(features_inside_circle[:, 0], features_inside_circle[:, 1], 'o', markerfacecolor='yellow',
                markeredgecolor='k', markersize=m_size)
    axs[0].add_artist(circle)
    axs[0].imshow(mask, 'binary')
    axs[1].imshow(mask, 'gray')
    if box is not None:
        rect0 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                  edgecolor='r', fill=False, linewidth=None)
        rect1 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                  edgecolor='r', fill=False, linewidth=None)
        axs[0].add_patch(rect0)
        axs[1].add_patch(rect1)
    if current_boxes is not None:
        add_box_to_axes(axs[0], current_boxes, 'orange')
        add_box_to_axes(axs[1], current_boxes, 'orange')
    plt.show()


def plot_track_history_with_angle_info(img, box, history, direction, frame_number, track_id, additional_text='',
                                       save_path=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.imshow(img, cmap='gray')
    add_box_to_axes(axs, box)
    add_features_to_axis(axs, history, marker_size=1, marker_color='g')

    fig.suptitle(f'Frame: {frame_number} | Track Id: {track_id}'
                 f'\nAngle bw velocity vectors: {direction}\n {additional_text}')

    legends_dict = {'red': 'Bounding Box',
                    'g': 'Track'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path + f"track_id_{track_id}_frame_{frame_number}.png")
        plt.close()
    else:
        plt.show()


def plot_track_history_with_angle_info_with_track_plot(img, box, history, direction, frame_number, track_id,
                                                       additional_text='', save_path=None):
    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(img, cmap='gray')
    if len(box) != 0:
        add_one_box_to_axis(axs[0], 'r', box)
        add_one_box_to_axis(axs[1], 'r', box)

    if len(history) != 0:
        add_features_to_axis(axs[0], history, marker_size=1, marker_color='g')
        add_features_to_axis(axs[1], history, marker_size=1, marker_color='g')

    fig.suptitle(f'Frame: {frame_number} | Track Id: {track_id}'
                 f'\nAngle bw velocity vectors: {direction}\n {additional_text}')

    legends_dict = {'red': 'Bounding Box',
                    'g': 'Track'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path + f"track_id_{track_id}_frame_{frame_number}.png")
        plt.close()
    else:
        plt.show()


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


def plot_one_with_bounding_boxes(img, boxes):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none',
                            figsize=(12, 10))
    axs.imshow(img, cmap='gray')
    for box in boxes:
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                 edgecolor='r', fill=False,
                                 linewidth=None)
        axs.add_patch(rect)
    plt.show()


def plot_two_with_bounding_boxes(img0, boxes0, img1, boxes1, frame_number):
    fig, ax = plt.subplots(1, 2, sharex='none', sharey='none',
                           figsize=(12, 10))
    ax[0].imshow(img0, cmap='gray')
    ax[1].imshow(img1, cmap='gray')
    add_box_to_axes(ax[0], boxes0)
    add_box_to_axes(ax[1], boxes1)
    ax[0].set_title('GT')
    ax[1].set_title('OF')
    fig.suptitle(f'Frame: {frame_number}')
    plt.show()


def plot_two_with_bounding_boxes_and_rgb(img0, boxes0, img1, boxes1, rgb0, rgb1, frame_number, additional_text=None):
    fig, ax = plt.subplots(2, 2, sharex='none', sharey='none', figsize=(12, 10))
    ax[0, 0].imshow(img0, cmap='gray')
    ax[0, 1].imshow(img1, cmap='gray')
    ax[1, 0].imshow(rgb0)
    ax[1, 1].imshow(rgb1)
    add_box_to_axes(ax[0, 0], boxes0)
    add_box_to_axes(ax[0, 1], boxes1)
    add_box_to_axes(ax[1, 0], boxes0)
    add_box_to_axes(ax[1, 1], boxes1)
    ax[0, 0].set_title('GT/FG Mask')
    ax[0, 1].set_title('OF/FG Mask')
    ax[1, 0].set_title('GT/RGB')
    ax[1, 1].set_title('OF/RGB')
    fig.suptitle(f'Frame: {frame_number} | {additional_text}')
    plt.show()


def plot_for_video(gt_rgb, gt_mask, last_frame_rgb, last_frame_mask, current_frame_rgb, current_frame_mask,
                   gt_annotations, last_frame_annotation, current_frame_annotation, new_track_annotation,
                   frame_number, additional_text=None, video_mode=False, original_dims=None, save_path=None):
    fig, ax = plt.subplots(3, 2, sharex='none', sharey='none', figsize=original_dims or (12, 10))
    ax_gt_rgb, ax_gt_mask, ax_last_frame_rgb, ax_last_frame_mask, ax_current_frame_rgb, ax_current_frame_mask = \
        ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[2, 0], ax[2, 1]
    ax_gt_rgb.imshow(gt_rgb)
    ax_gt_mask.imshow(gt_mask, cmap='gray')
    ax_last_frame_rgb.imshow(last_frame_rgb)
    ax_last_frame_mask.imshow(last_frame_mask, cmap='gray')
    ax_current_frame_rgb.imshow(current_frame_rgb)
    ax_current_frame_mask.imshow(current_frame_mask, cmap='gray')

    add_box_to_axes(ax_gt_rgb, gt_annotations)
    add_box_to_axes(ax_gt_mask, gt_annotations)
    add_box_to_axes(ax_last_frame_rgb, last_frame_annotation)
    add_box_to_axes(ax_last_frame_mask, last_frame_annotation)
    add_box_to_axes(ax_current_frame_rgb, current_frame_annotation)
    add_box_to_axes(ax_current_frame_mask, current_frame_annotation)
    add_box_to_axes(ax_current_frame_rgb, new_track_annotation, 'green')
    add_box_to_axes(ax_current_frame_mask, new_track_annotation, 'green')

    ax_gt_rgb.set_title('GT/RGB')
    ax_gt_mask.set_title('GT/FG Mask')
    ax_last_frame_rgb.set_title('(T-1)/RGB')
    ax_last_frame_mask.set_title('(T-1)/FG Mask')
    ax_current_frame_rgb.set_title('(T)/RGB')
    ax_current_frame_mask.set_title('(T)/FG Mask')

    fig.suptitle(f'Frame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'green': 'New track Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()

    return fig


def add_features_to_axis(ax, features, marker_size=8, marker_shape='o', marker_color='blue'):
    ax.plot(features[:, 0], features[:, 1], marker_shape, markerfacecolor=marker_color, markeredgecolor='k',
            markersize=marker_size)


def plot_for_video_current_frame(gt_rgb, current_frame_rgb, gt_annotations, current_frame_annotation,
                                 new_track_annotation, frame_number, additional_text=None, video_mode=False,
                                 original_dims=None, save_path=None, zero_shot=False, box_annotation=None,
                                 generated_track_histories=None, gt_track_histories=None, track_marker_size=1,
                                 return_figure_only=False):
    fig, ax = plt.subplots(1, 2, sharex='none', sharey='none', figsize=original_dims or (12, 10))
    ax_gt_rgb, ax_current_frame_rgb = ax[0], ax[1]
    ax_gt_rgb.imshow(gt_rgb)
    ax_current_frame_rgb.imshow(current_frame_rgb)

    if box_annotation is None:
        add_box_to_axes(ax_gt_rgb, gt_annotations)
        add_box_to_axes(ax_current_frame_rgb, current_frame_annotation)
        add_box_to_axes(ax_current_frame_rgb, new_track_annotation, 'green')
    else:
        add_box_to_axes_with_annotation(ax_gt_rgb, gt_annotations, box_annotation[0])
        add_box_to_axes_with_annotation(ax_current_frame_rgb, current_frame_annotation, box_annotation[1])
        add_box_to_axes_with_annotation(ax_current_frame_rgb, new_track_annotation, [], 'green')

    if gt_track_histories is not None:
        add_features_to_axis(ax_gt_rgb, gt_track_histories, marker_size=track_marker_size, marker_color='g')

    if generated_track_histories is not None:
        add_features_to_axis(ax_current_frame_rgb, generated_track_histories, marker_size=track_marker_size,
                             marker_color='g')

    ax_gt_rgb.set_title('GT')
    ax_current_frame_rgb.set_title('Our Method')

    fig.suptitle(f'{"Zero Shot" if zero_shot else "One Shot"} Version\nFrame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'green': 'New track Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if return_figure_only:
        plt.close()
        return fig

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()

    return fig


def plot_for_one_track(gt_rgb, gt_mask, last_frame_rgb, last_frame_mask, current_frame_rgb, current_frame_mask,
                       gt_annotations, last_frame_annotation, current_frame_annotation, new_track_annotation,
                       frame_number, track_idx, additional_text=None, video_mode=False, original_dims=None,
                       save_path=None):
    fig, ax = plt.subplots(3, 2, sharex='none', sharey='none', figsize=original_dims or (12, 10))
    ax_gt_rgb, ax_gt_mask, ax_last_frame_rgb, ax_last_frame_mask, ax_current_frame_rgb, ax_current_frame_mask = \
        ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[2, 0], ax[2, 1]
    ax_gt_rgb.imshow(gt_rgb)
    ax_gt_mask.imshow(gt_mask, cmap='gray')
    ax_last_frame_rgb.imshow(last_frame_rgb)
    ax_last_frame_mask.imshow(last_frame_mask, cmap='gray')
    ax_current_frame_rgb.imshow(current_frame_rgb)
    ax_current_frame_mask.imshow(current_frame_mask, cmap='gray')

    add_one_box_to_axis(ax_gt_rgb, box=gt_annotations[track_idx], color='r')
    add_one_box_to_axis(ax_gt_mask, box=gt_annotations[track_idx], color='r')
    add_one_box_to_axis(ax_last_frame_rgb, box=last_frame_annotation[track_idx], color='r')
    add_one_box_to_axis(ax_last_frame_mask, box=last_frame_annotation[track_idx], color='r')
    add_one_box_to_axis(ax_current_frame_rgb, box=current_frame_annotation[track_idx], color='r')
    add_one_box_to_axis(ax_current_frame_mask, box=current_frame_annotation[track_idx], color='r')
    if new_track_annotation.size != 0:
        add_one_box_to_axis(ax_current_frame_rgb, box=new_track_annotation[track_idx], color='r')
        add_one_box_to_axis(ax_current_frame_mask, box=new_track_annotation[track_idx], color='r')

    ax_gt_rgb.set_title('GT/RGB')
    ax_gt_mask.set_title('GT/FG Mask')
    ax_last_frame_rgb.set_title('(T-1)/RGB')
    ax_last_frame_mask.set_title('(T-1)/FG Mask')
    ax_current_frame_rgb.set_title('(T)/RGB')
    ax_current_frame_mask.set_title('(T)/FG Mask')

    fig.suptitle(f'Frame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'green': 'New track Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()

    return fig


def add_box_to_axes(ax, boxes, edge_color='r'):
    for box in boxes:
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                 edgecolor=edge_color, fill=False,
                                 linewidth=None)
        ax.add_patch(rect)


def add_box_to_axes_with_annotation(ax, boxes, annotation, edge_color='r'):
    for a, box in zip(annotation, boxes):
        if box is None:
            continue
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                 edgecolor=edge_color, fill=False,
                                 linewidth=None)
        ax.add_patch(rect)

        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0

        ax.annotate(a, (cx, cy), color='w', weight='bold', fontsize=6, ha='center', va='center')


def plot_processing_steps(xy_cloud, shifted_xy_cloud, xy_box, shifted_xy_box,
                          final_cloud, xy_cloud_current_frame, frame_number, track_id,
                          selected_past, selected_current,
                          true_cloud_key_point=None, shifted_cloud_key_point=None,
                          overlap_threshold=None, shift_corrected_cloud_key_point=None,
                          key_point_criteria=None, shift_correction=None,
                          line_width=None, save_path=None):
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

    if save_path is None:
        plt.show()
    else:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + f'fig_frame_{frame_number}_track_{track_id}.png')
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


def filter_low_length_tracks(track_based_features, frame_based_features, threshold):
    logger.info('Level 1 filtering\n')
    # copy to alter the data
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


def filter_low_length_tracks_lvl2(track_based_features, frame_based_features):
    logger.info('\nLevel 2 filtering\n')
    allowed_tracks = list(track_based_features.keys())
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


def filter_tracks_through_all_steps(track_based_features, frame_based_features, min_track_length_threshold):
    track_based_features, frame_based_features = filter_low_length_tracks(
        track_based_features=track_based_features,
        frame_based_features=frame_based_features,
        threshold=min_track_length_threshold)

    frame_based_features = filter_low_length_tracks_lvl2(track_based_features, frame_based_features)

    return track_based_features, frame_based_features


def evaluate_extracted_features(track_based_features, frame_based_features, batch_size=32, do_filter=False,
                                drop_last_batch=True, plot_scale_factor=1, desired_fps=5, custom_video_shape=False,
                                video_mode=True, video_save_location=None, min_track_length_threshold=5,
                                skip_plot_save=False):
    # frame_track_distribution_pre_filter = {k: len(v.object_features) for k, v in frame_based_features.items()}
    if do_filter:
        track_based_features, frame_based_features = filter_low_length_tracks(
            track_based_features=track_based_features,
            frame_based_features=frame_based_features,
            threshold=min_track_length_threshold)

        frame_based_features = filter_low_length_tracks_lvl2(track_based_features, frame_based_features)
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


def associate_frame_with_ground_truth(frames, frame_numbers):
    return 0


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


if __name__ == '__main__':
    # feats = preprocess_data(var_threshold=150, plot=False)
    eval_mode = False
    version = 0
    video_save_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/zero_shot/'
    plot_save_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/'
    features_save_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/'
    Path(video_save_path).mkdir(parents=True, exist_ok=True)
    Path(features_save_path).mkdir(parents=True, exist_ok=True)
    if not eval_mode and EXECUTE_STEP == STEP.UNSUPERVISED:
        # use_tight_parameters = True
        # tight_parameters = {
        #     'radius': 60,
        #     'extra_radius': 0,
        #     'generic_box_wh': 50,
        #     'detect_shadows': True
        # }
        # relaxed_parameters = {
        #     'radius': 90,
        #     'extra_radius': 50,
        #     'generic_box_wh': 100,
        #     'detect_shadows': True
        # }
        # if use_tight_parameters:
        #     param = tight_parameters
        # else:
        #     param = relaxed_parameters

        param = ObjectDetectionParameters.SLANTED.value

        custom_video_dict = {
            'dataset': SimpleVideoDatasetBase,
            'video_path': f'../Datasets/Virat/VIRAT_S_000201_00_000018_000380.mp4',
            'start': 0,
            'end': 15,
            'pts_unit': 'sec'
        }
        custom_video_mode = True
        if custom_video_mode:
            unsupervised_method = preprocess_data_zero_shot_custom_video
        else:
            unsupervised_method = preprocess_data_zero_shot
        feats = unsupervised_method(var_threshold=None, plot=False, radius=param['radius'],
                                    video_mode=True, video_save_path=video_save_path + 'extraction.avi',
                                    desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
                                    min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
                                    use_circle_to_keep_track_alive=False, custom_video_shape=False,
                                    extra_radius=param['extra_radius'], generic_box_wh=param['generic_box_wh'],
                                    use_is_box_overlapping_live_boxes=True, save_per_part_path=None,
                                    save_every_n_batch_itr=50, drop_last_batch=True,
                                    detect_shadows=param['detect_shadows'],
                                    filter_switch_boxes_based_on_angle_and_recent_history=True,
                                    compute_histories_for_plot=True, custom_video=custom_video_dict)
        # torch.save(feats, features_save_path + 'features.pt')
    elif not eval_mode and EXECUTE_STEP == STEP.SEMI_SUPERVISED:
        video_save_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/one_shot/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)
        feats = preprocess_data_one_shot(var_threshold=None, plot=False, radius=60, save_per_part_path=None,
                                         video_mode=True, video_save_path=video_save_path + 'extraction.avi',
                                         desired_fps=5, overlap_percent=0.4, plot_save_path=plot_save_path,
                                         min_points_in_cluster=16, begin_track_mode=True, iou_threshold=0.5,
                                         use_circle_to_keep_track_alive=False, custom_video_shape=False,
                                         extra_radius=0, generic_box_wh=50, use_is_box_overlapping_live_boxes=True,
                                         save_every_n_batch_itr=50, drop_last_batch=True, detect_shadows=False)
    elif not eval_mode and EXECUTE_STEP == STEP.FILTER_FEATURES:
        # accumulated_features_path_filename = 'accumulated_features_from_finally.pt'
        accumulated_features_path_filename = 'accumulated_features_from_finally_tight.pt'
        # accumulated_features_path_filename = 'accumulated_features_from_finally_filtered.pt'
        track_length_threshold = 5

        accumulated_features_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                    f'/{accumulated_features_path_filename}'
        accumulated_features: Dict[int, Any] = torch.load(accumulated_features_path)
        per_track_features: Dict[int, TrackFeatures] = accumulated_features['track_based_accumulated_features']
        per_frame_features: Dict[int, FrameFeatures] = accumulated_features['accumulated_features']

        video_save_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/processed_features' \
                          f'/{track_length_threshold}/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)
        evaluate_extracted_features(track_based_features=per_track_features, frame_based_features=per_frame_features,
                                    video_save_location=video_save_path + 'extraction_filter.avi', do_filter=True,
                                    min_track_length_threshold=track_length_threshold, desired_fps=1, video_mode=False,
                                    skip_plot_save=True)
        logger.info(f'Track length threshold: {track_length_threshold}')
    elif not eval_mode and EXECUTE_STEP == STEP.NN_EXTRACTION:
        accumulated_features_path_filename = 'accumulated_features_from_finally.pt'
        # accumulated_features_path_filename = 'accumulated_features_from_finally_filtered.pt'

        accumulated_features_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                    f'/{accumulated_features_path_filename}'

        accumulated_features: Dict[int, Any] = torch.load(accumulated_features_path)
        per_track_features: Dict[int, TrackFeatures] = accumulated_features['track_based_accumulated_features']
        per_frame_features: Dict[int, FrameFeatures] = accumulated_features['accumulated_features']

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
        print()
    elif not eval_mode and EXECUTE_STEP == STEP.EXTRACTION:
        # accumulated_features_path_filename = 'accumulated_features_from_finally.pt'
        accumulated_features_path_filename = 'accumulated_features_from_finally_filtered.pt'
        track_length_threshold = 60

        accumulated_features_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}' \
                                    f'/{accumulated_features_path_filename}'
        accumulated_features: Dict[int, Any] = torch.load(accumulated_features_path)
        per_track_features: Dict[int, TrackFeatures] = accumulated_features['track_based_accumulated_features']
        per_frame_features: Dict[int, FrameFeatures] = accumulated_features['accumulated_features']

        filtered_track_based_features, filtered_frame_based_features = filter_tracks_through_all_steps(
            per_track_features, per_frame_features, track_length_threshold
        )

        plot_save_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/track_based/'
        Path(plot_save_path).mkdir(parents=True, exist_ok=True)
        video_save_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/zero_shot/frames12apart/'
        # + 'cancelled/'
        Path(video_save_path).mkdir(parents=True, exist_ok=True)
        features_save_path = f'../Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/' \
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
            save_every_n_batch_itr=50, frame_by_frame_estimation=False,
            filter_switch_boxes_based_on_angle_and_recent_history=True,
            compute_histories_for_plot=True, min_track_length_to_filter_switch_box=20,
            angle_threshold_to_filter=120, save_path_for_features=features_save_path)
    else:
        feat_file_path = '../Plots/baseline_v2/v0/deathCircle4/' \
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
