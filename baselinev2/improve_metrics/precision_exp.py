import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2 as cv
import hydra
import numpy as np
import scipy
import skimage
import torch
import torchvision
import torchvision.transforms.functional as tvf
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import seaborn as sns

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import BASE_PATH, DATASET_META, SERVER_PATH
from baselinev2.improve_metrics.crop_utils import show_image_with_crop_boxes
from baselinev2.improve_metrics.dataset import SDDDatasetV0
from baselinev2.improve_metrics.model import make_conv_blocks, Activations, PersonClassifier, people_collate_fn, \
    make_classifier_block
from baselinev2.improve_metrics.modules import resnet18, resnet9
from baselinev2.plot_utils import add_box_to_axes, add_box_to_axes_with_annotation, \
    add_features_to_axis
from baselinev2.utils import get_generated_frame_annotations, get_generated_track_annotations_for_frame
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames

# sns.set_theme(style="ticks")

initialize_logging()
logger = get_logger('improve_metrics.precision_exp')

DATASET_TO_MODEL = {
    SDDVideoClasses.BOOKSTORE: 376647,
    SDDVideoClasses.COUPA: 377095,
    SDDVideoClasses.GATES: 373993,
    SDDVideoClasses.HYANG: 373994,
    SDDVideoClasses.LITTLE: 376650,
    SDDVideoClasses.NEXUS: 377688,
    SDDVideoClasses.QUAD: 377576,
    SDDVideoClasses.DEATH_CIRCLE: 11
}


class MetricPerTrack(object):
    def __init__(self, track_id: int):
        super(MetricPerTrack, self).__init__()
        self.track_id = track_id
        self.tp = []
        self.fp = []
        self.frames = []
        self.track_len = 0
        self.precision = 0.


class PerTrajectoryPR(object):
    def __init__(self, video_class: SDDVideoClasses, video_number: int, video_meta: SDDVideoDatasets,
                 gt_annotation_root_path: str = '../Datasets/SDD/annotations/',
                 generated_annotation_root_path: str = '../Plots/baseline_v2/v0/',
                 overlap_threshold: float = 2, num_workers: int = 12, batch_size: int = 32,
                 drop_last_batch: bool = True, custom_video_shape: bool = False,
                 video_mode: bool = True, save_path_for_video: str = None, desired_fps: int = 5,
                 plot_scale_factor: int = 1, save_path_for_features: str = None,
                 object_classifier: Optional[Module] = None, additional_crop_h: int = 0,
                 additional_crop_w: int = 0, bounding_box_size: int = 50, cfg: DictConfig = None):
        super(PerTrajectoryPR, self).__init__()

        self.object_classifier = object_classifier

        self.additional_crop_w = additional_crop_w
        self.additional_crop_h = additional_crop_h
        self.bounding_box_size = bounding_box_size
        self.cfg = cfg

        self.video_class = video_class
        self.video_number = video_number
        self.video_meta = video_meta
        self.gt_annotation_path = gt_annotation_root_path
        self.generated_annotation_path = generated_annotation_root_path
        self.overlap_threshold = overlap_threshold

        self.dataset = SDDDatasetV0(root='../../' + BASE_PATH, video_label=video_class, frames_per_clip=1,
                                    num_workers=num_workers, num_videos=1, video_number_to_use=video_number,
                                    step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                    single_track_mode=False, track_id=5, multiple_videos=False)

        self.data_loader = DataLoader(self.dataset, batch_size, drop_last=drop_last_batch)
        self.gt_annotations = self.dataset.annotations_df
        self.generated_annotations = pd.read_csv(f'{self.generated_annotation_path}{video_class.value}{video_number}/'
                                                 f'csv_annotation/generated_annotations.csv')

        frames_shape = self.dataset.original_shape
        self.video_frame_shape = (1200, 1000) if custom_video_shape else frames_shape
        self.original_dims = None
        self.video_mode = video_mode

        _, meta_info = DATASET_META.get_meta(video_meta, video_number)
        self.ratio = float(meta_info.flatten()[-1])

        self.track_metrics = {}
        self.boosted_track_metrics = {}
        self.save_path_for_features = save_path_for_features

        if video_mode:
            if frames_shape[0] < frames_shape[1]:
                self.original_dims = (
                    frames_shape[1] / 100 * plot_scale_factor, frames_shape[0] / 100 * plot_scale_factor)
                self.out = cv.VideoWriter(save_path_for_video, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                          (self.video_frame_shape[1], self.video_frame_shape[0]))
                self.video_frame_shape[0], self.video_frame_shape[1] = \
                    self.video_frame_shape[1], self.video_frame_shape[0]
            else:
                self.original_dims = (
                    frames_shape[0] / 100 * plot_scale_factor, frames_shape[1] / 100 * plot_scale_factor)
                self.out = cv.VideoWriter(save_path_for_video, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                          (self.video_frame_shape[0], self.video_frame_shape[1]))

    def destroy(self):
        self.out.release()

    @staticmethod
    def get_bbox_center(b_box):
        x_min = b_box[0]
        y_min = b_box[1]
        x_max = b_box[2]
        y_max = b_box[3]
        x_mid = np.array((x_min + (x_max - x_min) / 2.), dtype=np.int)
        y_mid = np.array((y_min + (y_max - y_min) / 2.), dtype=np.int)

        return np.vstack((x_mid, y_mid)).T

    def extract_metrics(self):
        tp_list, fp_list, fn_list = [], [], []
        try:
            for p_idx, data in enumerate(tqdm(self.data_loader)):
                frames, frame_numbers = data
                frames = frames.squeeze()
                frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                frames_count = frames.shape[0]
                original_shape = new_shape = [frames.shape[1], frames.shape[2]]

                for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                             total=len(frame_numbers)):
                    gt_frame_annotation = get_frame_annotations_and_skip_lost(self.gt_annotations, frame_number.item())
                    gt_annotations, gt_bbox_centers = scale_annotations(gt_frame_annotation,
                                                                        original_scale=original_shape,
                                                                        new_scale=new_shape, return_track_id=False,
                                                                        tracks_with_annotations=True)
                    gt_boxes = gt_annotations[:, :-1]
                    gt_track_idx = gt_annotations[:, -1]

                    generated_frame_annotation = get_generated_frame_annotations(self.generated_annotations,
                                                                                 frame_number.item())
                    generated_boxes = generated_frame_annotation[:, 1:5]
                    generated_track_idx = generated_frame_annotation[:, 0]

                    for generated_t_idx in generated_track_idx:
                        if generated_t_idx not in self.track_metrics.keys():
                            self.track_metrics.update({generated_t_idx: MetricPerTrack(track_id=generated_t_idx)})

                    l2_distance_boxes_score_matrix = np.zeros(shape=(len(gt_boxes), len(generated_boxes)))
                    if generated_boxes.size != 0:
                        for a_i, a_box in enumerate(gt_boxes):
                            for r_i, r_box in enumerate(generated_boxes):
                                dist = np.linalg.norm((self.get_bbox_center(a_box).flatten() -
                                                       self.get_bbox_center(r_box).flatten()), 2) * self.ratio
                                l2_distance_boxes_score_matrix[a_i, r_i] = dist

                        l2_distance_boxes_score_matrix = self.overlap_threshold - l2_distance_boxes_score_matrix
                        l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
                        # Hungarian
                        # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
                        actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]
                        match_rows_tracks_idx = [gt_track_idx[m].item() for m in match_rows]
                        match_cols_tracks_idx = [generated_track_idx[m] for m in match_cols]

                        for generated_t_idx in generated_track_idx:
                            if generated_t_idx in match_cols_tracks_idx:
                                self.track_metrics[generated_t_idx].tp.append(1)
                            else:
                                self.track_metrics[generated_t_idx].fp.append(1)
                            self.track_metrics[generated_t_idx].frames.append(frame_number.item())

                        # gt_track_box_mapping = {a[-1]: a[:-1] for a in gt_annotations}
                        # for m_c_idx, matched_c in enumerate(match_cols_tracks_idx):
                        #     gt_t_idx = match_rows_tracks_idx[m_c_idx]
                        #     # gt_box_idx = np.argwhere(gt_track_idx == gt_t_idx)
                        #     # track_based_accumulated_features[matched_c].object_features[-1].gt_track_idx = gt_t_idx
                        #     # track_based_accumulated_features[matched_c].object_features[-1].gt_box = \
                        #     #     gt_track_box_mapping[gt_t_idx]
                        #     try:
                        #         # track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = \
                        #         #     last_frame_gt_tracks[gt_t_idx]
                        #         gt_distance = np.linalg.norm(
                        #             (self.get_bbox_center(gt_track_box_mapping[gt_t_idx]) -
                        #              self.get_bbox_center(last_frame_gt_tracks[gt_t_idx])), 2, axis=0)
                        #         # track_based_accumulated_features[matched_c].object_features[-1]. \
                        #         #     gt_past_current_distance = gt_distance
                        #     except KeyError:
                        #         track_based_accumulated_features[matched_c].object_features[-1].past_gt_box = None
                        #         track_based_accumulated_features[matched_c].object_features[-1]. \
                        #             gt_past_current_distance = [0, 0]

                        # last_frame_gt_tracks = copy.deepcopy(gt_track_box_mapping)

                        matched_distance_array = [(i, j, l2_distance_boxes_score_matrix[i, j])
                                                  for i, j in zip(match_rows, match_cols)]
                    else:
                        match_rows, match_cols = np.array([]), np.array([])
                        match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])

                    if len(match_rows) != 0:
                        if len(match_rows) != len(match_cols):
                            logger.warning('Matching arrays length not same!')
                        tp = len(match_rows)
                        fp = len(generated_boxes) - len(match_rows)
                        fn = len(gt_boxes) - len(match_rows)

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                    else:
                        tp = 0
                        fp = 0
                        fn = len(gt_boxes)

                        precision = 0
                        recall = 0

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    skipped_idx = np.setdiff1d(np.arange(len(generated_track_idx)), match_cols).astype(np.int)
                    logger.info(f'{self.video_class.name} - {self.video_number} || Precision: {precision} |'
                                f' Recall: {recall}')
                    self.plot(frame, frame_number, generated_boxes, generated_track_idx, gt_boxes, gt_track_idx,
                              precision, recall,
                              matched_gt_track_idx=gt_track_idx[match_rows] if match_rows.size != 0 else [],
                              matched_generated_track_idx=
                              generated_track_idx[match_cols] if match_cols.size != 0 else [],
                              matched_gt_boxes=gt_boxes[match_rows] if match_rows.size != 0 else [],
                              matched_generated_boxes=generated_boxes[match_cols] if match_cols.size != 0 else [],
                              skipped_generated_boxes=generated_boxes[skipped_idx] if skipped_idx.size != 0 else [],
                              skipped_generated_track_idx=
                              generated_track_idx[skipped_idx] if skipped_idx.size != 0 else [])
        except KeyboardInterrupt:
            if self.video_mode:
                logger.info('Saving video before exiting!')
                self.destroy()
        finally:
            if self.video_mode:
                self.destroy()
            torch.save(self.track_metrics, self.save_path_for_features)
        logger.info('Finished extracting metrics!')

    def extract_metrics_with_boosted_precision(self, plot: bool = False):
        track_ids_killed = []
        tp_list, fp_list, fn_list = [], [], []
        tp_boosted_list, fp_boosted_list, fn_boosted_list = [], [], []
        try:
            for p_idx, data in enumerate(tqdm(self.data_loader)):
                frames, frame_numbers, _ = data
                frames = frames.squeeze()
                frames = (frames * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                frames_count = frames.shape[0]
                original_shape = new_shape = [frames.shape[1], frames.shape[2]]

                for frame_idx, (frame, frame_number) in tqdm(enumerate(zip(frames, frame_numbers)),
                                                             total=len(frame_numbers)):
                    gt_frame_annotation = get_frame_annotations_and_skip_lost(self.gt_annotations[0],
                                                                              frame_number.item())
                    gt_annotations, gt_bbox_centers = scale_annotations(gt_frame_annotation,
                                                                        original_scale=original_shape,
                                                                        new_scale=new_shape, return_track_id=False,
                                                                        tracks_with_annotations=True)
                    gt_boxes = gt_annotations[:, :-1]
                    gt_track_idx = gt_annotations[:, -1]

                    generated_frame_annotation = get_generated_frame_annotations(self.generated_annotations,
                                                                                 frame_number.item())
                    generated_boxes = generated_frame_annotation[:, 1:5]
                    generated_track_idx = generated_frame_annotation[:, 0]

                    # generated_frame_annotation = get_generated_frame_annotations(self.generated_annotations,
                    #                                                              frame_number.item())
                    # generated_track_annotations = get_generated_track_annotations_for_frame(self.generated_annotations,
                    #                                                                         frame_number.item())
                    # generated_track_lengths = np.array([t.shape[0] for t in generated_track_annotations])
                    #
                    # feasible_generated_track_length = generated_track_lengths > 60
                    # feasible_generated_frame_annotations = generated_frame_annotation[feasible_generated_track_length]
                    # generated_boxes = torch.from_numpy(
                    #     feasible_generated_frame_annotations[:, 1:5].astype(np.int)).numpy()
                    # generated_track_idx = feasible_generated_frame_annotations[:, 0]

                    # classify patches
                    generated_boxes_xywh = torchvision.ops.box_convert(torch.from_numpy(generated_boxes.astype(np.int)),
                                                                       'xyxy', 'xywh')
                    generated_boxes_xywh = [torch.tensor((b[1], b[0], b[2] + self.additional_crop_h,
                                                          b[3] + self.additional_crop_w)) for b in generated_boxes_xywh]
                    try:
                        generated_boxes_xywh = torch.stack(generated_boxes_xywh)

                        generated_crops = [tvf.crop(torch.from_numpy(frame).permute(2, 0, 1),
                                                    top=b[0], left=b[1], width=b[2], height=b[3])
                                           for b in generated_boxes_xywh]
                        generated_crops_resized = [tvf.resize(c, [self.bounding_box_size, self.bounding_box_size])
                                                   for c in generated_crops if c.shape[1] != 0 and c.shape[2] != 0]
                        generated_valid_boxes = [c_i for c_i, c in enumerate(generated_crops)
                                                 if c.shape[1] != 0 and c.shape[2] != 0]
                        generated_boxes_xywh = generated_boxes_xywh[generated_valid_boxes]
                        generated_track_idx = generated_track_idx[generated_valid_boxes]
                        generated_boxes = generated_boxes[generated_valid_boxes]
                        generated_crops_resized = torch.stack(generated_crops_resized)
                        generated_crops_resized = (generated_crops_resized.float() / 255.0).to(self.cfg.eval.device)

                        # plot
                        if plot:
                            show_image_with_crop_boxes(frame,
                                                       [], generated_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
                                                       title='xywh')
                            gt_crops_grid = torchvision.utils.make_grid(generated_crops_resized)
                            plt.imshow(gt_crops_grid.cpu().permute(1, 2, 0))
                            plt.show()

                        with torch.no_grad():
                            patch_predictions = self.object_classifier(generated_crops_resized)

                        pred_labels = torch.round(torch.sigmoid(patch_predictions))

                        valid_boxes_idx = (pred_labels > 0.5).squeeze().cpu()

                        if valid_boxes_idx.ndim == 0:
                            if valid_boxes_idx.item():
                                valid_boxes_idx = 0
                                valid_boxes = generated_boxes_xywh[valid_boxes_idx]
                                invalid_boxes = []

                                valid_track_idx = [generated_track_idx[valid_boxes_idx]]
                                invalid_track_idx = []
                                valid_generated_boxes = np.expand_dims(generated_boxes[valid_boxes_idx], 0)
                            else:
                                valid_boxes_idx = 0
                                valid_boxes = []
                                invalid_boxes = generated_boxes_xywh[valid_boxes_idx]

                                valid_track_idx = []
                                invalid_track_idx = [generated_track_idx[valid_boxes_idx]]
                                valid_generated_boxes = np.array([])

                            # valid_generated_boxes = np.expand_dims(generated_boxes[valid_boxes_idx], 0)
                        else:
                            valid_boxes = generated_boxes_xywh[valid_boxes_idx]
                            invalid_boxes = generated_boxes_xywh[~valid_boxes_idx]

                            # plot removed boxes
                            if plot:
                                show_image_with_crop_boxes(frame,
                                                           invalid_boxes, valid_boxes, xywh_mode_v2=False,
                                                           xyxy_mode=False,
                                                           title='xywh')

                            valid_track_idx = generated_track_idx[valid_boxes_idx]
                            invalid_track_idx = generated_track_idx[~valid_boxes_idx]
                            valid_generated_boxes = generated_boxes[valid_boxes_idx]

                        track_ids_killed = np.union1d(track_ids_killed, invalid_track_idx)
                    except RuntimeError:
                        valid_generated_boxes, valid_track_idx = np.array([]), np.array([])

                    for generated_t_idx in generated_track_idx:
                        if generated_t_idx not in self.track_metrics.keys():
                            self.track_metrics.update({generated_t_idx: MetricPerTrack(track_id=generated_t_idx)})

                    for valid_t_idx in generated_track_idx:
                        if valid_t_idx not in self.boosted_track_metrics.keys() \
                                and not np.isin(valid_t_idx, track_ids_killed):
                            self.boosted_track_metrics.update({valid_t_idx: MetricPerTrack(track_id=valid_t_idx)})

                    fn, fp, match_cols, match_rows, precision, recall, tp = self.calculate_precision_recall(
                        frame_number, generated_boxes, generated_track_idx, gt_boxes, gt_track_idx,
                        self.track_metrics, killed_track_ids=None)

                    fn_boosted, fp_boosted, match_cols_boosted, match_rows_boosted, \
                    precision_boosted, recall_boosted, tp_boosted = self.calculate_precision_recall(
                        frame_number, valid_generated_boxes, valid_track_idx, gt_boxes, gt_track_idx,
                        self.boosted_track_metrics, killed_track_ids=track_ids_killed)

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

                    tp_boosted_list.append(tp_boosted)
                    fp_boosted_list.append(fp_boosted)
                    fn_boosted_list.append(fn_boosted)

                    skipped_idx = np.setdiff1d(np.arange(len(generated_track_idx)), match_cols).astype(np.int)
                    logger.info(f'{self.video_class.name} - {self.video_number} || Precision: {precision} |'
                                f' Recall: {recall}')
                    logger.info('Boosted')
                    logger.info(f'{self.video_class.name} - {self.video_number} || Precision: {precision_boosted} |'
                                f' Recall: {recall_boosted}')
                    self.plot(frame, frame_number, generated_boxes, generated_track_idx, gt_boxes, gt_track_idx,
                              precision, recall,
                              matched_gt_track_idx=gt_track_idx[match_rows] if match_rows.size != 0 else [],
                              matched_generated_track_idx=
                              generated_track_idx[match_cols] if match_cols.size != 0 else [],
                              matched_gt_boxes=gt_boxes[match_rows] if match_rows.size != 0 else [],
                              matched_generated_boxes=generated_boxes[match_cols] if match_cols.size != 0 else [],
                              skipped_generated_boxes=generated_boxes[skipped_idx] if skipped_idx.size != 0 else [],
                              skipped_generated_track_idx=
                              generated_track_idx[skipped_idx] if skipped_idx.size != 0 else [])
        except KeyboardInterrupt:
            if self.video_mode:
                logger.info('Saving video before exiting!')
                self.destroy()
        finally:
            if self.video_mode:
                self.destroy()
            if not os.path.exists(os.path.split(self.save_path_for_features)[0]):
                os.makedirs(os.path.split(self.save_path_for_features)[0])
            torch.save({'original': self.track_metrics, 'boosted': self.boosted_track_metrics,
                        'frame_based_metrics': {'original': {'tp': tp_list, 'fp': fp_list, 'fn': fn_list},
                                                'boosted': {'tp': tp_boosted_list,
                                                            'fp': fp_boosted_list,
                                                            'fn': fn_boosted_list}}},
                       self.save_path_for_features)
        logger.info('Finished extracting metrics!')

    def calculate_precision_recall(self, frame_number, generated_boxes, generated_track_idx, gt_boxes, gt_track_idx,
                                   track_metrics, killed_track_ids=None):
        l2_distance_boxes_score_matrix = np.zeros(shape=(len(gt_boxes), len(generated_boxes)))
        if generated_boxes.size != 0:
            for a_i, a_box in enumerate(gt_boxes):
                for r_i, r_box in enumerate(generated_boxes):
                    dist = np.linalg.norm((self.get_bbox_center(a_box).flatten() -
                                           self.get_bbox_center(r_box).flatten()), 2) * self.ratio
                    l2_distance_boxes_score_matrix[a_i, r_i] = dist

            l2_distance_boxes_score_matrix = self.overlap_threshold - l2_distance_boxes_score_matrix
            l2_distance_boxes_score_matrix[l2_distance_boxes_score_matrix < 0] = 10
            # Hungarian
            # match_rows, match_cols = scipy.optimize.linear_sum_assignment(-l2_distance_boxes_score_matrix)
            match_rows, match_cols = scipy.optimize.linear_sum_assignment(l2_distance_boxes_score_matrix)
            actually_matched_mask = l2_distance_boxes_score_matrix[match_rows, match_cols] < 10
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]
            match_rows_tracks_idx = [gt_track_idx[m].item() for m in match_rows]
            match_cols_tracks_idx = [generated_track_idx[m] for m in match_cols]

            if killed_track_ids is None:
                for generated_t_idx in generated_track_idx:
                    if generated_t_idx in match_cols_tracks_idx:
                        track_metrics[generated_t_idx].tp.append(1)
                    else:
                        track_metrics[generated_t_idx].fp.append(1)
                    track_metrics[generated_t_idx].frames.append(frame_number.item())
            else:
                for generated_t_idx in generated_track_idx:
                    if not np.isin(generated_t_idx, killed_track_ids):
                        if generated_t_idx in match_cols_tracks_idx:
                            track_metrics[generated_t_idx].tp.append(1)
                        else:
                            track_metrics[generated_t_idx].fp.append(1)
                        track_metrics[generated_t_idx].frames.append(frame_number.item())
        else:
            match_rows, match_cols = np.array([]), np.array([])
            match_rows_tracks_idx, match_cols_tracks_idx = np.array([]), np.array([])
        if len(match_rows) != 0:
            if len(match_rows) != len(match_cols):
                logger.warning('Matching arrays length not same!')
            tp = len(match_rows)
            fp = len(generated_boxes) - len(match_rows)
            fn = len(gt_boxes) - len(match_rows)

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            tp = 0
            fp = 0
            fn = len(gt_boxes)

            precision = 0
            recall = 0
        return fn, fp, match_cols, match_rows, precision, recall, tp

    @staticmethod
    def plot_for_video_current_frame(gt_rgb, current_frame_rgb, gt_annotations, current_frame_annotation,
                                     new_track_annotation, frame_number, additional_text=None, video_mode=False,
                                     original_dims=None, save_path=None, zero_shot=False, box_annotation=None,
                                     generated_track_histories=None, gt_track_histories=None, track_marker_size=1,
                                     return_figure_only=False, plot_gt_bbox_on_generated=False, plot_matched_only=False,
                                     matched_array=None):
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
            if plot_matched_only:
                add_box_to_axes_with_annotation(ax_current_frame_rgb, matched_array[1], matched_array[3])
                add_box_to_axes_with_annotation(ax_current_frame_rgb, matched_array[0], matched_array[2], 'g')
                add_box_to_axes_with_annotation(ax_current_frame_rgb, matched_array[4], matched_array[5], 'magenta')
            else:
                add_box_to_axes_with_annotation(ax_current_frame_rgb, current_frame_annotation, box_annotation[1])
                if plot_gt_bbox_on_generated:
                    add_box_to_axes_with_annotation(ax_current_frame_rgb, gt_annotations, box_annotation[0], 'orange')
            add_box_to_axes_with_annotation(ax_current_frame_rgb, new_track_annotation, [], 'green')

        if gt_track_histories is not None:
            add_features_to_axis(ax_gt_rgb, gt_track_histories, marker_size=track_marker_size, marker_color='g')

        if generated_track_histories is not None:
            add_features_to_axis(ax_current_frame_rgb, generated_track_histories, marker_size=track_marker_size,
                                 marker_color='g')

        ax_gt_rgb.set_title('GT')
        ax_current_frame_rgb.set_title('Our Method')

        fig.suptitle(f'{"Unsupervised" if zero_shot else "One Shot"} Version\nFrame: {frame_number}\n{additional_text}')

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

    def plot(self, frame, frame_number, generated_boxes, generated_track_idx, gt_boxes, gt_track_idx, precision,
             recall, matched_generated_track_idx, matched_gt_track_idx, matched_gt_boxes, matched_generated_boxes,
             skipped_generated_boxes, skipped_generated_track_idx):
        if self.video_mode:
            fig = self.plot_for_video_current_frame(
                gt_rgb=frame, current_frame_rgb=frame,
                gt_annotations=gt_boxes,
                current_frame_annotation=generated_boxes,
                new_track_annotation=[],
                frame_number=frame_number,
                box_annotation=[gt_track_idx, generated_track_idx],
                generated_track_histories=None,
                gt_track_histories=None,
                additional_text=f'Precision: {precision} | Recall: {recall}',
                plot_gt_bbox_on_generated=True,
                plot_matched_only=True,
                matched_array=[matched_gt_boxes, matched_generated_boxes,
                               matched_gt_track_idx, matched_generated_track_idx,
                               skipped_generated_boxes, skipped_generated_track_idx],
                video_mode=self.video_mode, original_dims=self.original_dims, zero_shot=True)

            canvas = FigureCanvas(fig)
            canvas.draw()

            buf = canvas.buffer_rgba()
            out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
            if out_frame.shape[0] != self.video_frame_shape[1] or \
                    out_frame.shape[1] != self.video_frame_shape[0]:
                out_frame = skimage.transform.resize(out_frame,
                                                     (self.video_frame_shape[1], self.video_frame_shape[0]))
                out_frame = (out_frame * 255).astype(np.uint8)
            self.out.write(out_frame)
        else:
            pass
            # fig = self.plot_for_video_current_frame(
            #     gt_rgb=frame, current_frame_rgb=frame,
            #     gt_annotations=gt_boxes,
            #     current_frame_annotation=generated_boxes,
            #     new_track_annotation=[],
            #     frame_number=frame_number,
            #     additional_text='',
            #     video_mode=False, original_dims=self.original_dims, zero_shot=True)

    @staticmethod
    def analyze_feature(path):
        features: Dict[int, MetricPerTrack] = torch.load(path)
        track_len_to_precision = {}

        for key, value in tqdm(features.items()):
            value.track_length = len(value.frames)

            tp = np.array(value.tp).sum()
            fp = np.array(value.fp).sum()
            value.precision = tp / (tp + fp)

            if value.track_length in track_len_to_precision.keys():
                track_len_to_precision[value.track_length].append(value.precision)
            else:
                track_len_to_precision.update({value.track_length: [value.precision]})

        for key, value in tqdm(track_len_to_precision.items()):
            track_len_to_precision[key] = np.array(value).mean()

        lengths = list(track_len_to_precision.keys())
        precisions = list(track_len_to_precision.values())
        plt.bar(lengths, precisions)
        plt.xlabel('Track Length')
        plt.ylabel('Precision')
        plt.title('Precision vs Track Length')
        plt.suptitle(f'{video_clz.name} - {video_num}')
        plt.show()

        sns_data = pd.DataFrame.from_dict({'lengths': lengths, 'precision': precisions})
        # sns.displot(sns_data, x="lengths", y='precision', cbar=True)
        sns.jointplot(data=sns_data, x="lengths", y='precision')
        plt.show()

        print()

    @staticmethod
    def analyze_multiple_features(paths, mode='mean', boosted: bool = False):
        features: List[Dict[str, Dict[int, MetricPerTrack]]] = [torch.load(path) for path in paths]
        track_len_to_precision = {}

        for feat in features:
            if boosted:
                feat = feat['boosted']
            else:
                feat = feat['original']
            for key, value in tqdm(feat.items()):
                value.track_length = len(value.frames)

                tp = np.array(value.tp).sum()
                fp = np.array(value.fp).sum()
                value.precision = tp / (tp + fp)

                if value.track_length in track_len_to_precision.keys():
                    track_len_to_precision[value.track_length].append(value.precision)
                else:
                    track_len_to_precision.update({value.track_length: [value.precision]})

        for key, value in tqdm(track_len_to_precision.items()):
            if mode == 'mean':
                track_len_to_precision[key] = np.array(value).mean()
            else:
                track_len_to_precision[key] = np.median(np.array(value))

        lengths = list(track_len_to_precision.keys())
        precisions = list(track_len_to_precision.values())
        plt.bar(lengths, precisions)
        plt.xlabel('Track Length')
        plt.ylabel('Precision')
        plt.title('Precision vs Track Length')
        plt.suptitle(f'{video_clz.name}')
        plt.show()

        sns_data = pd.DataFrame.from_dict({'lengths': lengths, 'precision': precisions})
        # sns.displot(sns_data, x="lengths", y='precision', cbar=True)
        sns.jointplot(data=sns_data, x="lengths", y='precision')
        plt.show()

        print()

    @staticmethod
    def combine_multiple_features(paths):
        # paths.remove('../Plots/baseline_v2/v0/experiments/combined.pt')
        features: List[Dict[str, MetricPerTrack]] = [torch.load(path) for path in paths]

        original_features, boosted_features = [], []
        for f in features:
            original_features.append(f['original'])
            boosted_features.append(f['boosted'])

        track_len_to_precision = {}

        for feat in original_features:
            for key, value in tqdm(feat.items()):
                value.track_length = len(value.frames)

                tp = np.array(value.tp).sum()
                fp = np.array(value.fp).sum()
                value.precision = tp / (tp + fp)

                if value.track_length in track_len_to_precision.keys():
                    track_len_to_precision[value.track_length].append(value.precision)
                else:
                    track_len_to_precision.update({value.track_length: [value.precision]})

        return {'features': features, 'out': track_len_to_precision}

    @staticmethod
    def combine_multiple_features_v2(paths):
        features: List[Dict[str, MetricPerTrack]] = [torch.load(path) for path in paths]
        video_list = [os.path.split(p)[-1][6:-3] for p in paths]

        original_features, boosted_features, frame_based_metrics = [], [], []
        for f in features:
            original_features.append(f['original'])
            boosted_features.append(f['boosted'])
            frame_based_metrics.append(f['frame_based_metrics'])

        track_len_to_precision_original = {}
        track_len_to_precision_boosted = {}
        metric_per_video = {}

        for frame_metrics, video_name_number in zip(frame_based_metrics, video_list):
            original_metrics = frame_metrics['original']
            boosted_metrics = frame_metrics['boosted']

            original_precision = np.array(original_metrics['tp']).sum() / \
                                 (np.array(original_metrics['tp']).sum() + np.array(original_metrics['fp']).sum())
            original_recall = np.array(original_metrics['tp']).sum() / \
                              (np.array(original_metrics['tp']).sum() + np.array(original_metrics['fn']).sum())

            boosted_precision = np.array(boosted_metrics['tp']).sum() / \
                                (np.array(boosted_metrics['tp']).sum() + np.array(boosted_metrics['fp']).sum())
            boosted_recall = np.array(boosted_metrics['tp']).sum() / \
                             (np.array(boosted_metrics['tp']).sum() + np.array(boosted_metrics['fn']).sum())

            metric_per_video.update(
                {video_name_number: {
                    'original': {'precision': original_precision, 'recall': original_recall},
                    'boosted': {'precision': boosted_precision, 'recall': boosted_recall}}})

        for original_feat in original_features:
            for key, value in tqdm(original_feat.items()):
                value.track_length = len(value.frames)

                tp = np.array(value.tp).sum()
                fp = np.array(value.fp).sum()
                value.precision = tp / (tp + fp)

                if value.track_length in track_len_to_precision_original.keys():
                    track_len_to_precision_original[value.track_length].append(value.precision)
                else:
                    track_len_to_precision_original.update({value.track_length: [value.precision]})

        for boosted_feat in boosted_features:
            for key, value in tqdm(boosted_feat.items()):
                value.track_length = len(value.frames)

                tp = np.array(value.tp).sum()
                fp = np.array(value.fp).sum()
                value.precision = tp / (tp + fp)

                if value.track_length in track_len_to_precision_boosted.keys():
                    track_len_to_precision_boosted[value.track_length].append(value.precision)
                else:
                    track_len_to_precision_boosted.update({value.track_length: [value.precision]})

        return {'features': features,
                'out': {'original': track_len_to_precision_original,
                        'boosted': track_len_to_precision_boosted,
                        'metrics': metric_per_video}}

    @staticmethod
    def just_plot(feat_path, mode='mean'):
        feat = torch.load(feat_path)
        track_len_to_precision = feat['out']
        for key, value in tqdm(track_len_to_precision.items()):
            if mode == 'mean':
                track_len_to_precision[key] = np.array(value).mean()
            else:
                track_len_to_precision[key] = np.median(np.array(value))

        lengths = list(track_len_to_precision.keys())
        precisions = list(track_len_to_precision.values())
        plt.bar(lengths, precisions)
        plt.xlabel('Track Length')
        plt.ylabel('Precision')
        plt.title('Precision vs Track Length')
        plt.suptitle(f'Whole Dataset')
        plt.show()

        sns_data = pd.DataFrame.from_dict({'lengths': lengths, 'precision': precisions})
        # sns.displot(sns_data, x="lengths", y='precision', cbar=True)
        sns.jointplot(data=sns_data, x="lengths", y='precision')
        plt.show()

    @staticmethod
    def just_plot_v2(feat_path, mode='mean', boosted=False):
        feat = torch.load(feat_path)
        track_len_to_precision = feat['out']['original'] if not boosted else feat['out']['boosted']
        metrics = feat['out']['metrics']

        video_list, v_num_list, p_list, p_b_list, r_list, r_b_list, p_delta, r_delta = [], [], [], [], [], [], [], []
        df_data = {}
        for vid, m in metrics.items():
            try:
                v_clz, v_num = vid.split('_')
            except ValueError:
                v_clz, v_clz_add, v_num = vid.split('_')
                v_clz += '_' + v_clz_add

            video_list.append(v_clz)
            v_num_list.append(v_num)
            p_list.append(m['original']['precision'])
            p_b_list.append(m['boosted']['precision'])
            p_delta.append(m['boosted']['precision'] - m['original']['precision'])
            r_list.append(m['original']['recall'])
            r_b_list.append(m['boosted']['recall'])
            r_delta.append(m['boosted']['recall'] - m['original']['recall'])

        df_data.update({
            'VIDEO': video_list,
            'VIDEO_NUMBER': v_num_list,
            'Precision': p_list,
            'Precision - Boosted': p_b_list,
            'Delta Precision': p_delta,
            'Recall': r_list,
            'Recall - Boosted': r_b_list,
            'Delta Recall': r_delta,
        })
        df = pd.DataFrame.from_dict(df_data, orient='index').transpose()
        df = df.sort_values(['VIDEO', 'VIDEO_NUMBER'])

        df.to_markdown('../Plots/baseline_v2/v0/experimentsv2/metrics.md', index=False)
        import json
        with open('../Plots/baseline_v2/v0/experimentsv2/metrics.json', 'w+') as f:
            json.dump(metrics, f)
        for key, value in tqdm(track_len_to_precision.items()):
            if mode == 'mean':
                track_len_to_precision[key] = np.array(value).mean()
            else:
                track_len_to_precision[key] = np.median(np.array(value))

        lengths = list(track_len_to_precision.keys())
        precisions = list(track_len_to_precision.values())
        plt.bar(lengths, precisions)
        plt.xlabel('Track Length')
        plt.ylabel('Precision')
        plt.title('Precision vs Track Length')
        plt.suptitle(f'Whole Dataset')
        plt.savefig(f"../Plots/baseline_v2/v0/experimentsv2/whole_dataset_{'boosted' if boosted else 'original'}_0.png")
        plt.show()

        sns_data = pd.DataFrame.from_dict({'lengths': lengths, 'precision': precisions})
        # sns.displot(sns_data, x="lengths", y='precision', cbar=True)
        sns.jointplot(data=sns_data, x="lengths", y='precision')
        plt.savefig(f"../Plots/baseline_v2/v0/experimentsv2/whole_dataset_{'boosted' if boosted else 'original'}_1.png")
        plt.show()


@hydra.main(config_path="config", config_name="config")
def boost_precision(cfg):
    logger.info(f'Setting up model...')

    if cfg.eval.use_resnet:
        conv_layers = resnet18(pretrained=cfg.eval.use_pretrained) \
            if not cfg.eval.smaller_resnet else resnet9(pretrained=cfg.eval.use_pretrained,
                                                        first_in_channel=cfg.eval.first_in_channel,
                                                        first_stride=cfg.eval.first_stride,
                                                        first_padding=cfg.eval.first_padding)
    else:
        conv_layers = make_conv_blocks(cfg.input_dim, cfg.out_channels, cfg.kernel_dims, cfg.stride, cfg.padding,
                                       cfg.batch_norm, non_lin=Activations.RELU, dropout=cfg.dropout)
    classifier_layers = make_classifier_block(cfg.in_feat, cfg.out_feat, Activations.RELU)

    model = PersonClassifier(conv_block=conv_layers, classifier_block=classifier_layers,
                             train_dataset=None, val_dataset=None, batch_size=cfg.eval.batch_size,
                             num_workers=cfg.eval.num_workers, shuffle=cfg.eval.shuffle,
                             pin_memory=cfg.eval.pin_memory, lr=cfg.lr, collate_fn=people_collate_fn,
                             hparams=cfg)
    checkpoint_path = f'{cfg.eval.checkpoint.path}{cfg.eval.checkpoint.object_classifier_version}/checkpoints/'
    checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]
    load_dict = torch.load(checkpoint_file)

    model.load_state_dict(load_dict['state_dict'])
    model.to(cfg.eval.device)
    model.eval()

    video_save_path = f'../../../Plots/baseline_v2/v0/experiments/video_{video_clz.name}_{video_num}.avi'
    feats_save_path = f'../../../Plots/baseline_v2/v0/experiments/feats_{video_clz.name}_{video_num}.pt'

    per_trajectory_pr = PerTrajectoryPR(video_class=video_clz, video_number=video_num, video_meta=video_clz_meta,
                                        num_workers=12, save_path_for_video=video_save_path,
                                        save_path_for_features=feats_save_path, video_mode=False,
                                        object_classifier=model, cfg=cfg,
                                        generated_annotation_root_path='../../../Plots/baseline_v2/v0/',
                                        additional_crop_h=cfg.eval.dataset.additional_h,
                                        additional_crop_w=cfg.eval.dataset.additional_w)
    # per_trajectory_pr.extract_metrics()
    per_trajectory_pr.extract_metrics_with_boosted_precision()

    return model


@hydra.main(config_path="config", config_name="config")
def boosted_precision_for_all_clips(cfg):
    logger.info(f'Setting up model...')
    if cfg.eval.use_resnet:
        conv_layers = resnet18(pretrained=cfg.eval.use_pretrained) \
            if not cfg.eval.smaller_resnet else resnet9(pretrained=cfg.eval.use_pretrained,
                                                        first_in_channel=cfg.eval.first_in_channel,
                                                        first_stride=cfg.eval.first_stride,
                                                        first_padding=cfg.eval.first_padding)
    else:
        conv_layers = make_conv_blocks(cfg.input_dim, cfg.out_channels, cfg.kernel_dims, cfg.stride, cfg.padding,
                                       cfg.batch_norm, non_lin=Activations.RELU, dropout=cfg.dropout)
    classifier_layers = make_classifier_block(cfg.in_feat, cfg.out_feat, Activations.RELU)

    video_clazzes = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.GATES,
                     SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS, SDDVideoClasses.QUAD]
    video_metas = [SDDVideoDatasets.BOOKSTORE, SDDVideoDatasets.COUPA, SDDVideoDatasets.GATES,
                   SDDVideoDatasets.HYANG, SDDVideoDatasets.LITTLE, SDDVideoDatasets.NEXUS, SDDVideoDatasets.QUAD]
    video_numbers = [[i for i in range(7)], [i for i in range(4)], [i for i in range(9)], [i for i in range(15)],
                     [i for i in range(4)], [i for i in range(12)], [i for i in range(4)]]

    for idx, (v_clz, v_meta) in tqdm(enumerate(zip(video_clazzes, video_metas))):
        model = PersonClassifier(conv_block=conv_layers, classifier_block=classifier_layers,
                                 train_dataset=None, val_dataset=None, batch_size=cfg.eval.batch_size,
                                 num_workers=cfg.eval.num_workers, shuffle=cfg.eval.shuffle,
                                 pin_memory=cfg.eval.pin_memory, lr=cfg.lr, collate_fn=people_collate_fn,
                                 hparams=cfg)

        logger.info(f"Loading weights for the dataset: {v_clz.name}")
        checkpoint_path = f'{cfg.eval.checkpoint.path}{DATASET_TO_MODEL[v_clz]}/checkpoints/'
        checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]
        logger.info(f"Checkpoint file: {checkpoint_file}")
        load_dict = torch.load(checkpoint_file)

        model.load_state_dict(load_dict['state_dict'])
        model.to(cfg.eval.device)
        model.eval()
        for v_num in video_numbers[idx]:
            logger.info(f"************** Processing clip: {v_clz.name} - {v_num} *******************************")

            video_save_path = f'../../../Plots/baseline_v2/v0/experiments/video_{v_clz.name}_{v_num}.avi'
            feats_save_path = f'../../../Plots/baseline_v2/v0/experimentsv2/feats_{v_clz.name}_{v_num}.pt'

            # video_save_path = f'Plots/baseline_v2/v0/experiments/video_{v_clz.name}_{v_num}.avi'
            # feats_save_path = f'Plots/baseline_v2/v0/experiments/feats_{v_clz.name}_{v_num}.pt'

            per_trajectory_pr = PerTrajectoryPR(video_class=v_clz, video_number=v_num,
                                                video_meta=v_meta,
                                                num_workers=12,
                                                # save_path_for_video=SERVER_PATH + video_save_path,
                                                save_path_for_video=video_save_path,
                                                # save_path_for_features=SERVER_PATH + feats_save_path,
                                                save_path_for_features=feats_save_path,
                                                video_mode=False,
                                                object_classifier=model, cfg=cfg,
                                                # generated_annotation_root_path=
                                                # SERVER_PATH + '../../../Plots/baseline_v2/v0/',
                                                generated_annotation_root_path=
                                                '../../../Plots/baseline_v2/v0/',
                                                additional_crop_h=cfg.eval.dataset.additional_h,
                                                additional_crop_w=cfg.eval.dataset.additional_w)
            per_trajectory_pr.extract_metrics_with_boosted_precision()


if __name__ == '__main__':
    analyze = False
    all_dataset = False
    all_dataset_boosted = False
    combine_features = False
    plot_only = True

    video_clz = SDDVideoClasses.DEATH_CIRCLE
    video_clz_meta = SDDVideoDatasets.DEATH_CIRCLE
    video_num = 4

    video_save_path = f'../Plots/baseline_v2/v0/experiments/video_{video_clz.name}_{video_num}.avi'
    feats_save_path = f'../Plots/baseline_v2/v0/experiments/feats_{video_clz.name}_{video_num}.pt'

    if analyze:
        # PerTrajectoryPR.analyze_feature(feats_save_path)
        start, end = 0, 4
        PerTrajectoryPR.analyze_multiple_features([
            f'../Plots/baseline_v2/v0/experiments/feats_{video_clz.name}_{i}.pt' for i in range(start, end)
        ], mode='median', boosted=False)
        PerTrajectoryPR.analyze_multiple_features([
            f'../Plots/baseline_v2/v0/experiments/feats_{video_clz.name}_{i}.pt' for i in range(start, end)
        ], mode='median', boosted=True)
    elif plot_only:
        feat_path = f'../Plots/baseline_v2/v0/experimentsv2/combined.pt'
        PerTrajectoryPR.just_plot_v2(feat_path, 'median', boosted=False)
        PerTrajectoryPR.just_plot_v2(feat_path, 'median', boosted=True)
    elif combine_features:
        feat_paths = os.listdir(SERVER_PATH + 'Plots/baseline_v2/v0/experimentsv2/')
        feat_paths = [SERVER_PATH + 'Plots/baseline_v2/v0/experimentsv2/' + p for p in feat_paths]

        # feat_paths = os.listdir('../' + 'Plots/baseline_v2/v0/experimentsv2/')
        # feat_paths = ['../' + 'Plots/baseline_v2/v0/experimentsv2/' + p for p in feat_paths]

        # feat_paths = os.listdir('../' + 'Plots/baseline_v2/v0/experiments/')
        # feat_paths = ['../' + 'Plots/baseline_v2/v0/experiments/' + p for p in feat_paths]

        out = PerTrajectoryPR.combine_multiple_features_v2(feat_paths)
        # out = PerTrajectoryPR.combine_multiple_features(feat_paths)
        torch.save(out, SERVER_PATH + 'Plots/baseline_v2/v0/experimentsv2/combined.pt')
    elif all_dataset:
        video_clazzes = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.GATES,
                         SDDVideoClasses.HYANG, SDDVideoClasses.NEXUS, SDDVideoClasses.QUAD]
        video_metas = [SDDVideoDatasets.BOOKSTORE, SDDVideoDatasets.COUPA, SDDVideoDatasets.GATES,
                       SDDVideoDatasets.HYANG, SDDVideoDatasets.NEXUS, SDDVideoDatasets.QUAD]
        video_numbers = [[i for i in range(7)], [i for i in range(4)], [i for i in range(9)], [i for i in range(15)],
                         [i for i in range(12)], [i for i in range(4)]]

        for idx, (v_clz, v_meta) in tqdm(enumerate(zip(video_clazzes, video_metas))):
            for v_num in video_numbers[idx]:
                # video_save_path = f'../Plots/baseline_v2/v0/experiments/video_{v_clz.name}_{v_num}.avi'
                # feats_save_path = f'../Plots/baseline_v2/v0/experiments/feats_{v_clz.name}_{v_num}.pt'
                video_save_path = f'Plots/baseline_v2/v0/experiments/video_{v_clz.name}_{v_num}.avi'
                feats_save_path = f'Plots/baseline_v2/v0/experiments/feats_{v_clz.name}_{v_num}.pt'
                per_trajectory_pr = PerTrajectoryPR(video_class=v_clz, video_number=v_num,
                                                    video_meta=v_meta,
                                                    num_workers=4, save_path_for_video=SERVER_PATH + video_save_path,
                                                    save_path_for_features=SERVER_PATH + feats_save_path,
                                                    video_mode=False,
                                                    generated_annotation_root_path=
                                                    SERVER_PATH + 'Plots/baseline_v2/v0/')
                per_trajectory_pr.extract_metrics()
    elif all_dataset_boosted:
        boosted_precision_for_all_clips()
    else:
        boost_precision()
