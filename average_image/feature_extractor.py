from itertools import cycle
from typing import Union, Optional

import cv2 as cv
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from skimage.transform import resize
from skimage import color
from sklearn.cluster import cluster_optics_dbscan
from tqdm import tqdm

from bbox_utils import add_bbox_to_axes, resize_v_frames, get_frame_annotations, preprocess_annotations, \
    scale_annotations
from constants import OBJECT_CLASS_COLOR_MAPPING, SDDVideoClasses
from deep_networks_avg import get_vgg_layer_activations, get_resnet_layer_activations, \
    get_densenet_filtered_layer_activations
from feature_clustering import MeanShiftClustering, HierarchicalClustering, AffinityPropagationClustering, \
    DBSCANClustering, OPTICSClustering, Clustering, BirchClustering
from layers import min_pool2d
from utils import precision_recall, evaluate_clustering_per_frame, normalize, SDDMeta, evaluate_clustering_non_cc


class FeatureExtractor(object):
    def __init__(self, video_path):
        super(FeatureExtractor, self).__init__()
        self.start_frame = None
        self.end_frame = None
        self.video_path = video_path
        self.video_frames = None
        self.start_sec = 0
        self.end_sec = 0
        self.method = None
        self.original_shape = None
        self.algo = None

    def get_frames(self, start: Union[int, float], end: Union[int, float], dtype: str = 'float'):
        self.start_sec, self.end_sec = start, end
        self.video_frames, a_frames, meta_info = torchvision.io.read_video(filename=self.video_path, start_pts=start,
                                                                           end_pts=end,
                                                                           pts_unit="sec")
        self.original_shape = (self.video_frames.size(1), self.video_frames.size(2))
        if dtype == 'float':
            video_frames = self.video_frames.float() / 255.0
        else:
            video_frames = self.video_frames
        return video_frames, a_frames, meta_info

    def get_activations(self):
        return NotImplemented

    def get_min_pooled_activations(self, kernel_size: int, iterations: int = 1):
        return NotImplemented

    @staticmethod
    def get_optical_flow(previous_frame, next_frame, all_results_out=False):
        flow = cv.calcOpticalFlowFarneback(previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros((next_frame.shape[0], next_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # 0-1 normalize
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        if all_results_out:
            return flow, rgb, mag, ang

        return flow, rgb

    def plot(self):
        return NotImplemented

    def plot_all_steps(self, nrows, ncols, plot_dims, original_frame, processed_img, threshold_image,
                       optical_flow_image, frame_annotation, num_clusters, cluster_centers, frame_evaluation,
                       frame_number, video_label, video_number, video_out_save_path, video_mode=False,
                       all_objects=None):  # todo: cluster_project changed
        original_shape = (original_frame.shape[0], original_frame.shape[1])
        downscaled_shape = (processed_img.shape[0], processed_img.shape[1])
        fig, axs = plt.subplots(nrows, ncols, sharex='none', sharey='none',
                                figsize=plot_dims)
        axs[0, 0].imshow(original_frame)
        axs[0, 1].imshow(processed_img, cmap='binary')
        axs[0, 2].imshow(threshold_image, cmap='binary')
        axs[1, 0].imshow(processed_img, cmap='binary')
        axs[1, 1].imshow(resize(original_frame, output_shape=downscaled_shape))
        axs[1, 2].imshow(optical_flow_image)
        # axs[1, 2].imshow(thresholded_img, cmap='binary')

        axs[0, 0].set_title('Image')
        axs[0, 1].set_title('Processed Image')
        axs[0, 2].set_title('Threshold Image')
        axs[1, 0].set_title('Clustered - Processed')
        axs[1, 1].set_title('Clustered - Image')
        axs[1, 2].set_title('Optical Flow')
        # axs[1, 2].set_title('Clustered - Thresholded')

        add_bbox_to_axes(axs[0, 0], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=False, use_dnn=False)
        add_bbox_to_axes(axs[0, 1], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=True, use_dnn=False)
        add_bbox_to_axes(axs[1, 1], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=True, use_dnn=False)

        # self.project_cluster_centers(axs, cluster_centers, num_clusters)
        self.project_cluster_centers_simple(axs, cluster_centers, num_clusters)

        patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
        fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label} | Video Number: {video_number} | Method: {self.method}"
                     f"\nPrecision: {frame_evaluation[frame_number]['precision']}"
                     f" | Recall: {frame_evaluation[frame_number]['recall']}\n"
                     f"Objects Count: {all_objects} | Clusters Found: {num_clusters}",
                     fontsize=14, fontweight='bold')
        if not video_mode:
            fig.savefig(video_out_save_path + f"frame_{frame_number}.png")

        return fig

    def plot_all_steps_rescaled(self, nrows, ncols, plot_dims, original_frame, processed_img, threshold_image,
                                optical_flow_image, frame_annotation, num_clusters, cluster_centers, frame_evaluation,
                                frame_number, video_label, video_number, video_out_save_path, video_mode=False,
                                all_objects=None):
        original_shape = self.original_shape
        downscaled_shape = (processed_img.shape[0], processed_img.shape[1])
        fig, axs = plt.subplots(nrows, ncols, sharex='none', sharey='none',
                                figsize=plot_dims)
        axs[0, 0].imshow(original_frame)
        axs[0, 1].imshow(processed_img, cmap='binary')
        axs[0, 2].imshow(threshold_image, cmap='binary')
        axs[1, 0].imshow(processed_img, cmap='binary')
        axs[1, 1].imshow(resize(original_frame, output_shape=downscaled_shape))
        axs[1, 2].imshow(optical_flow_image)
        # axs[1, 2].imshow(thresholded_img, cmap='binary')

        axs[0, 0].set_title('Image')
        axs[0, 1].set_title('Processed Image')
        axs[0, 2].set_title('Threshold Image')
        axs[1, 0].set_title('Clustered - Processed')
        axs[1, 1].set_title('Clustered - Image')
        axs[1, 2].set_title('Optical Flow')
        # axs[1, 2].set_title('Clustered - Thresholded')

        add_bbox_to_axes(axs[0, 0], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=True, use_dnn=False)
        add_bbox_to_axes(axs[0, 1], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=True, use_dnn=False)
        add_bbox_to_axes(axs[1, 1], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=True, use_dnn=False)

        self.project_cluster_centers(axs, cluster_centers, num_clusters)

        patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
        fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label} | Video Number: {video_number} | Method: {self.method}"
                     f"\nPrecision: {frame_evaluation[frame_number]['precision']}"
                     f" | Recall: {frame_evaluation[frame_number]['recall']}\n"
                     f"Objects Count: {all_objects}",
                     fontsize=14, fontweight='bold')
        if not video_mode:
            fig.savefig(video_out_save_path + f"frame_{frame_number}.png")

        return fig

    @staticmethod
    def project_cluster_centers(axs, cluster_centers, num_clusters):
        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        colors = cm.rainbow(np.linspace(0, 1, 50))
        marker = cycle('o*v^><12348sphH+xXD')
        for k, col, m in zip(range(num_clusters), colors, marker):
            cluster_center = cluster_centers[k]
            axs[1, 0].plot(cluster_center[0], cluster_center[1], m, markerfacecolor=col,  # 'o'
                           markeredgecolor='k', markersize=8)
            axs[1, 1].plot(cluster_center[0], cluster_center[1], m, markerfacecolor=col,
                           markeredgecolor='k', markersize=8)
            # axs[1, 2].plot(cluster_center[0], cluster_center[1], m, markerfacecolor=col,
            #                markeredgecolor='k', markersize=8)

    @staticmethod
    def project_cluster_centers_simple(axs, cluster_centers, num_clusters):
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(num_clusters), colors):
            cluster_center = cluster_centers[k]
            axs[1, 0].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  # 'o'
                           markeredgecolor='k', markersize=8)
            axs[1, 1].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                           markeredgecolor='k', markersize=8)
        # plt.show()

    def plot_3(self, nrows, ncols, plot_dims, original_frame, processed_img, threshold_image,
               frame_annotation, num_clusters, cluster_centers, frame_evaluation,
               frame_number, video_label, video_number, video_out_save_path):
        original_shape = (original_frame.shape[0], original_frame.shape[1])
        downscaled_shape = (processed_img.shape[0], processed_img.shape[1])
        fig, axs = plt.subplots(nrows, ncols, sharex='none', sharey='none',
                                figsize=plot_dims)
        axs[0, 0].imshow(original_frame)
        axs[0, 2].imshow(threshold_image, cmap='binary')
        axs[1, 1].imshow(resize(original_frame, output_shape=downscaled_shape))

        axs[0, 0].set_title('Image')
        axs[0, 2].set_title('Threshold Image')
        axs[1, 1].set_title('Clustered - Image')

        add_bbox_to_axes(axs[0, 0], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=False, use_dnn=False)
        add_bbox_to_axes(axs[0, 1], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=True, use_dnn=False)
        add_bbox_to_axes(axs[1, 1], annotations=frame_annotation, only_pedestrians=False,
                         original_spatial_dim=original_shape,
                         pooled_spatial_dim=downscaled_shape,
                         min_pool=True, use_dnn=False)

        self.project_cluster_centers(axs, cluster_centers, num_clusters)

        patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
        fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {video_number}\nMethod: {self.method}"
                     f"\nPrecision: {frame_evaluation[frame_number]['precision']}"
                     f"\nRecall: {frame_evaluation[frame_number]['recall']}",
                     fontsize=14, fontweight='bold')
        fig.savefig(video_out_save_path + f"frame_{frame_number}.png")

    def plot_frame_processed_with_of(self, nrows, ncols, plot_dims, original_frame, processed_img, optical_flow_image,
                                     frame_annotation, frame_number, video_label, video_number, video_out_save_path,
                                     show_bbox=False):
        original_shape = (original_frame.shape[0], original_frame.shape[1])
        downscaled_shape = (processed_img.shape[0], processed_img.shape[1])
        fig, axs = plt.subplots(nrows, ncols, sharex='none', sharey='none',
                                figsize=plot_dims)
        axs[0].imshow(original_frame)
        axs[1].imshow(processed_img, cmap='binary')
        axs[2].imshow(optical_flow_image)

        axs[0].set_title('Image')
        axs[1].set_title('MOG2')
        axs[2].set_title('Optical Flow')

        if show_bbox:
            add_bbox_to_axes(axs[0], annotations=frame_annotation, only_pedestrians=False,
                             original_spatial_dim=original_shape,
                             pooled_spatial_dim=downscaled_shape,
                             min_pool=False, use_dnn=False)
            add_bbox_to_axes(axs[1], annotations=frame_annotation, only_pedestrians=False,
                             original_spatial_dim=original_shape,
                             pooled_spatial_dim=downscaled_shape,
                             min_pool=True, use_dnn=False)
            # add_bbox_to_axes(axs[2], annotations=frame_annotation, only_pedestrians=False,
            #                  original_spatial_dim=original_shape,
            #                  pooled_spatial_dim=downscaled_shape,
            #                  min_pool=True, use_dnn=False)

            patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
            fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {video_number}\nMethod: {self.method}",
                     fontsize=14, fontweight='bold')
        fig.savefig(video_out_save_path + f"fpof_frame_{frame_number}.png")

    def plot_mog2_steps(self, nrows, ncols, plot_dims, original_frame, processed_img,
                        frame_annotation, frame_number, video_label, video_number, video_out_save_path,
                        show_bbox=False):
        original_shape = (original_frame.shape[0], original_frame.shape[1])
        downscaled_shape = (processed_img.shape[0], processed_img.shape[1])
        fig, axs = plt.subplots(nrows, ncols, sharex='none', sharey='none',
                                figsize=plot_dims)
        axs[0].imshow(original_frame)
        axs[1].imshow(processed_img, cmap='binary')
        axs[2].imshow(self.algo.getBackgroundImage())

        axs[0].set_title('Image')
        axs[1].set_title('MOG2')
        axs[2].set_title('Background Image')

        if show_bbox:
            add_bbox_to_axes(axs[0], annotations=frame_annotation, only_pedestrians=False,
                             original_spatial_dim=original_shape,
                             pooled_spatial_dim=downscaled_shape,
                             min_pool=False, use_dnn=False)
            add_bbox_to_axes(axs[1], annotations=frame_annotation, only_pedestrians=False,
                             original_spatial_dim=original_shape,
                             pooled_spatial_dim=downscaled_shape,
                             min_pool=True, use_dnn=False)
            # add_bbox_to_axes(axs[2], annotations=frame_annotation, only_pedestrians=False,
            #                  original_spatial_dim=original_shape,
            #                  pooled_spatial_dim=downscaled_shape,
            #                  min_pool=True, use_dnn=False)

            patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
            fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {video_number}\nMethod: {self.method}",
                     fontsize=14, fontweight='bold')
        fig.savefig(video_out_save_path + f"fpof_frame_{frame_number}.png")

    def get_per_frame_results(self, processed_data, video_label, video_number, frames_out_save_path, annotations_df,
                              plot_scale_factor: int = 1, plot: bool = True, with_optical_flow: bool = True):
        if processed_data.shape[1] < processed_data.shape[2]:
            original_dims = (processed_data.shape[2] / 100 * plot_scale_factor, processed_data.shape[1] / 100 *
                             plot_scale_factor)
        else:
            original_dims = (processed_data.shape[1] / 100 * plot_scale_factor, processed_data.shape[2] / 100 *
                             plot_scale_factor)

        cap = cv.VideoCapture(self.video_path)
        cap_count = 0
        previous = None

        gt_bbox_cluster_center_dict = {}
        pr_for_frames = {}

        for fr in tqdm(range(0, processed_data.shape[0])):
            ret, frame = cap.read()
            if previous is None:
                previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                continue

            if cap_count < self.start_frame:
                continue
            next_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            flow, rgb = self.get_optical_flow(previous_frame=previous, next_frame=next_frame)

            if with_optical_flow:
                data, data_, max_0, max_1, min_0, min_1, threshold_img = self._prepare_data_xyuv(flow, fr,
                                                                                                 processed_data)
                mean_shift, n_clusters_ = self._perform_clustering(data, max_0, max_1, min_0, min_1, bandwidth=0.1)
            else:
                data, data_, threshold_img = self._prepare_data_xy(fr, processed_data)
                mean_shift, n_clusters_ = self._perform_clustering(data, 0, 0, 0, 0, renormalize=False, bandwidth=0.1)

            annotation_, annotation_full = self._process_frame_annotation(annotations_df, data_, fr)
            gt_bbox_cluster_center_dict.update({fr: {'gt_bbox': annotation_,
                                                     'cluster_centers': mean_shift.cluster_centers}})
            frame_results = evaluate_clustering_per_frame(fr, {'gt_bbox': annotation_,
                                                               'cluster_centers': mean_shift.cluster_centers})
            pre_rec = precision_recall(frame_results)
            pr_for_frames.update(pre_rec)

            if plot:
                fig = self.plot_all_steps(nrows=2, ncols=3, plot_dims=original_dims, original_frame=frame,
                                          processed_img=data_,
                                          threshold_image=threshold_img, optical_flow_image=rgb,
                                          frame_annotation=annotation_full,
                                          num_clusters=n_clusters_, cluster_centers=mean_shift.cluster_centers,
                                          frame_evaluation=pre_rec, video_out_save_path=frames_out_save_path,
                                          frame_number=fr, video_label=video_label, video_number=video_number)

            cap_count += 1
            previous = next_frame

        cap.release()
        return gt_bbox_cluster_center_dict, pr_for_frames

    def evaluate_sequence(self, num_frames, video_label, video_number, frames_out_save_path, annotations_df,
                          plot_scale_factor: int = 1, plot: bool = True, with_optical_flow: bool = True,
                          history: int = 120,
                          detect_shadows: bool = True, var_threshold: int = 100):
        original_dims = None
        cap = cv.VideoCapture(self.video_path)
        cap_count = 0
        previous = None

        gt_bbox_cluster_center_dict = {}
        pr_for_frames = {}

        kernel = self._core_setup_algo(detect_shadows, history, var_threshold)

        for fr in tqdm(range(0, num_frames)):
            ret, frame = cap.read()
            if previous is None or original_dims is None:
                if frame.shape[0] < frame.shape[1]:
                    original_dims = (frame.shape[1] / 100 * plot_scale_factor, frame.shape[0] / 100 *
                                     plot_scale_factor)
                    self.original_shape = (frame.shape[1], frame.shape[0])
                else:
                    original_dims = (frame.shape[0] / 100 * plot_scale_factor, frame.shape[1] / 100 *
                                     plot_scale_factor)
                    self.original_shape = (frame.shape[0], frame.shape[1])
                previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                processed_data = self._core_processing(frame, kernel)
                continue

            if cap_count < self.start_frame:
                continue

            processed_data = self._core_processing(frame, kernel)

            next_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            flow, rgb = self.get_optical_flow(previous_frame=previous, next_frame=next_frame)

            if with_optical_flow:
                data, data_, max_0, max_1, min_0, min_1, threshold_img = self._prepare_data_xyuv(flow, fr,
                                                                                                 processed_data,
                                                                                                 evaluation_mode=True)
                mean_shift, n_clusters_ = self._perform_clustering(data, max_0, max_1, min_0, min_1, bandwidth=0.1)
            else:
                data, data_, threshold_img = self._prepare_data_xy(fr, processed_data, evaluation_mode=True)
                mean_shift, n_clusters_ = self._perform_clustering(data, 0, 0, 0, 0, renormalize=False, bandwidth=0.1)

            annotation_, annotation_full = self._process_frame_annotation(annotations_df, data_, fr)
            gt_bbox_cluster_center_dict.update({fr: {'gt_bbox': annotation_,
                                                     'cluster_centers': mean_shift.cluster_centers}})
            frame_results = evaluate_clustering_per_frame(fr, {'gt_bbox': annotation_,
                                                               'cluster_centers': mean_shift.cluster_centers})
            pre_rec = precision_recall(frame_results)
            pr_for_frames.update(pre_rec)

            if plot:
                fig = self.plot_all_steps(nrows=2, ncols=3, plot_dims=original_dims, original_frame=frame,
                                          processed_img=data_,
                                          threshold_image=threshold_img, optical_flow_image=rgb,
                                          frame_annotation=annotation_full,
                                          num_clusters=n_clusters_, cluster_centers=mean_shift.cluster_centers,
                                          frame_evaluation=pre_rec, video_out_save_path=frames_out_save_path,
                                          frame_number=fr, video_label=video_label, video_number=video_number)

            cap_count += 1
            previous = next_frame

        cap.release()
        return gt_bbox_cluster_center_dict, pr_for_frames

    def generate_processed_image_with_optical_flow(self, num_frames, video_label, video_number, frames_out_save_path,
                                                   annotations_df,
                                                   plot_scale_factor: int = 1, plot: bool = True,
                                                   history: int = 120,
                                                   detect_shadows: bool = True, var_threshold: int = 100,
                                                   show_bbox: bool = False):
        original_dims = None
        cap = cv.VideoCapture(self.video_path)
        cap_count = 0
        previous = None

        kernel = self._core_setup_algo(detect_shadows, history, var_threshold)

        for fr in tqdm(range(0, num_frames)):
            ret, frame = cap.read()
            if previous is None or original_dims is None:
                if frame.shape[0] < frame.shape[1]:
                    original_dims = (frame.shape[1] / 100 * plot_scale_factor, frame.shape[0] / 100 *
                                     plot_scale_factor)
                    self.original_shape = (frame.shape[1], frame.shape[0])
                else:
                    original_dims = (frame.shape[0] / 100 * plot_scale_factor, frame.shape[1] / 100 *
                                     plot_scale_factor)
                    self.original_shape = (frame.shape[0], frame.shape[1])
                previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                processed_data = self._core_processing(frame, kernel)
                continue

            if cap_count < self.start_frame:
                continue

            processed_data = self._core_processing(frame, kernel)

            next_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            flow, rgb = self.get_optical_flow(previous_frame=previous, next_frame=next_frame)

            annotation_, annotation_full = self._process_frame_annotation(annotations_df, processed_data, fr)

            if plot:
                self.plot_frame_processed_with_of(nrows=1, ncols=3, plot_dims=original_dims, original_frame=frame,
                                                  processed_img=processed_data,
                                                  optical_flow_image=rgb, frame_annotation=annotation_full,
                                                  frame_number=fr, video_label=video_label, video_number=video_number,
                                                  video_out_save_path=frames_out_save_path, show_bbox=show_bbox)

            cap_count += 1
            previous = next_frame

        cap.release()

    def generate_mog2_steps(self, num_frames, video_label, video_number, frames_out_save_path,
                            annotations_df,
                            plot_scale_factor: int = 1, plot: bool = True,
                            history: int = 120,
                            detect_shadows: bool = True, var_threshold: int = 100,
                            show_bbox: bool = False):
        original_dims = None
        cap = cv.VideoCapture(self.video_path)
        cap_count = 0

        kernel = self._core_setup_algo(detect_shadows, history, var_threshold)

        for fr in tqdm(range(0, num_frames)):
            ret, frame = cap.read()
            if original_dims is None:
                if frame.shape[0] < frame.shape[1]:
                    original_dims = (frame.shape[1] / 100 * plot_scale_factor, frame.shape[0] / 100 *
                                     plot_scale_factor)
                    self.original_shape = (frame.shape[1], frame.shape[0])
                else:
                    original_dims = (frame.shape[0] / 100 * plot_scale_factor, frame.shape[1] / 100 *
                                     plot_scale_factor)
                    self.original_shape = (frame.shape[0], frame.shape[1])
                processed_data = self._core_processing(frame, kernel)
                continue

            if cap_count < self.start_frame:
                continue

            processed_data = self._core_processing(frame, kernel)

            annotation_, annotation_full = self._process_frame_annotation(annotations_df, processed_data, fr)

            if plot:
                self.plot_mog2_steps(nrows=1, ncols=3, plot_dims=original_dims, original_frame=frame,
                                     processed_img=processed_data,
                                     frame_annotation=annotation_full,
                                     frame_number=fr, video_label=video_label, video_number=video_number,
                                     video_out_save_path=frames_out_save_path, show_bbox=show_bbox)

            cap_count += 1

        cap.release()

    @staticmethod
    def _prepare_data_xyuv(flow, fr, processed_data, use_intensities=False, evaluation_mode: bool = False):
        if evaluation_mode:
            data_ = processed_data
        else:
            data_ = processed_data[fr]
        data_ = np.abs(data_)
        threshold_img = np.zeros_like(data_)
        object_idx = (data_ > 0).nonzero()
        intensities = data_[object_idx[0], object_idx[1]]
        flow_idx = flow[object_idx[0], object_idx[1]]
        flow_idx_normalized_0, f_max_0, f_min_0 = normalize(flow_idx[..., 0])
        flow_idx_normalized_1, f_max_1, f_min_1 = normalize(flow_idx[..., 1])
        threshold_img[object_idx[0], object_idx[1]] = data_[object_idx[0], object_idx[1]]
        object_idx_normalized_0, max_0, min_0 = normalize(object_idx[0])
        object_idx_normalized_1, max_1, min_1 = normalize(object_idx[1])
        if use_intensities:
            data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
                             flow_idx_normalized_0, intensities)).transpose()
        else:
            data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
                             flow_idx_normalized_0)).transpose()
        return data, data_, max_0, max_1, min_0, min_1, threshold_img

    @staticmethod
    def _prepare_data_xy_weighted_uv(flow, fr, processed_data, mag, use_intensities=False,
                                     evaluation_mode: bool = False,
                                     weight: float = 1.0):
        if evaluation_mode:
            data_ = processed_data
        else:
            data_ = processed_data[fr]
        data_ = np.abs(data_)
        threshold_img = np.zeros_like(data_)
        object_idx = (data_ > 0).nonzero()
        intensities = data_[object_idx[0], object_idx[1]]
        flow_idx = flow[object_idx[0], object_idx[1]]
        flow_idx_normalized_0, f_max_0, f_min_0 = normalize(flow_idx[..., 0])
        flow_idx_normalized_1, f_max_1, f_min_1 = normalize(flow_idx[..., 1])

        mag_idx = mag[object_idx[0], object_idx[1]]
        mag_idx_normalized, mag_idx_max, mag_idx_min = normalize(mag_idx)
        mag_idx_normalized *= weight
        flow_idx_normalized_0, flow_idx_normalized_1 = flow_idx_normalized_0 * mag_idx_normalized \
            , flow_idx_normalized_1 * mag_idx_normalized

        threshold_img[object_idx[0], object_idx[1]] = data_[object_idx[0], object_idx[1]]
        object_idx_normalized_0, max_0, min_0 = normalize(object_idx[0])
        object_idx_normalized_1, max_1, min_1 = normalize(object_idx[1])
        if use_intensities:
            data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
                             flow_idx_normalized_0, intensities)).transpose()
        else:
            data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
                             flow_idx_normalized_0)).transpose()
        return data, data_, max_0, max_1, min_0, min_1, threshold_img

    @staticmethod
    def _prepare_data_xyuv_color(image, flow, fr, processed_data, use_intensities=False, evaluation_mode: bool = False,
                                 lab_space: bool = False):
        if evaluation_mode:
            data_ = processed_data
            im = image
        else:
            data_ = processed_data[fr]
            im = image[fr]
        if im.shape[0] != data_.shape[0]:
            im = (resize(im, output_shape=(data_.shape[0], data_.shape[1])) * 255).astype(np.uint8)
        if lab_space:
            im = color.rgb2lab(im)
        data_ = np.abs(data_)
        threshold_img = np.zeros_like(data_)
        object_idx = (data_ > 0).nonzero()
        intensities = data_[object_idx[0], object_idx[1]]
        colors = im[object_idx[0], object_idx[1]]
        colors = colors / 255
        flow_idx = flow[object_idx[0], object_idx[1]]
        flow_idx_normalized_0, f_max_0, f_min_0 = normalize(flow_idx[..., 0])
        flow_idx_normalized_1, f_max_1, f_min_1 = normalize(flow_idx[..., 1])
        threshold_img[object_idx[0], object_idx[1]] = data_[object_idx[0], object_idx[1]]
        object_idx_normalized_0, max_0, min_0 = normalize(object_idx[0])
        object_idx_normalized_1, max_1, min_1 = normalize(object_idx[1])
        if use_intensities:
            data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
                             flow_idx_normalized_0, intensities)).transpose()
        else:
            data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
                             flow_idx_normalized_0, colors[..., 0], colors[..., 1], colors[..., 2])).transpose()
        return data, data_, max_0, max_1, min_0, min_1, threshold_img

    @staticmethod
    def _prepare_data_xy(fr, processed_data, use_intensities=False, evaluation_mode: bool = False):
        if evaluation_mode:
            data_ = processed_data
        else:
            data_ = processed_data[fr]
        data_ = np.abs(data_)
        threshold_img = np.zeros_like(data_)
        object_idx = (data_ > 0).nonzero()
        intensities = data_[object_idx[0], object_idx[1]]
        threshold_img[object_idx[0], object_idx[1]] = data_[object_idx[0], object_idx[1]]
        if use_intensities:
            data = np.stack((object_idx[1], object_idx[0])).transpose()
        else:
            data = np.stack((object_idx[1], object_idx[0], intensities)).transpose()
        return data, data_, threshold_img

    @staticmethod
    def _perform_clustering(data, max_0, max_1, min_0, min_1, renormalize=True, bandwidth: float = 0.1,
                            min_bin_freq: int = 3, max_iter: int = 300):
        mean_shift = MeanShiftClustering(data=data, bandwidth=bandwidth, min_bin_freq=min_bin_freq, max_iter=max_iter)
        mean_shift_dict = {'max_0': max_0,
                           'min_0': min_0,
                           'max_1': max_1,
                           'min_1': min_1}
        mean_shift.cluster(renormalize=renormalize, options=mean_shift_dict)
        labels_unique, points_per_cluster = np.unique(mean_shift.labels, return_counts=True)
        mean_shift.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        return mean_shift, n_clusters_

    @staticmethod
    def _perform_clustering_affinity_propagation(data, max_0, max_1, min_0, min_1, renormalize=True,
                                                 preference: Optional[int] = -50,
                                                 damping: float = 0.5):
        affinity_clustering = AffinityPropagationClustering(data=data, preference=preference, damping=damping)
        scaling_dict = {'max_0': max_0,
                        'min_0': min_0,
                        'max_1': max_1,
                        'min_1': min_1}
        affinity_clustering.cluster(renormalize=renormalize, options=scaling_dict)
        labels_unique, points_per_cluster = np.unique(affinity_clustering.labels, return_counts=True)
        affinity_clustering.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        return affinity_clustering, n_clusters_

    @staticmethod
    def _perform_clustering_dbscan(data, max_0, max_1, min_0, min_1, renormalize=True,
                                   min_samples: int = 5,
                                   eps: float = 0.1):
        dbscan_clustering = DBSCANClustering(data=data, eps=0.1, min_samples=5)
        scaling_dict = {'max_0': max_0,
                        'min_0': min_0,
                        'max_1': max_1,
                        'min_1': min_1}
        dbscan_clustering.cluster(renormalize=renormalize, options=scaling_dict)
        labels_unique, points_per_cluster = np.unique(dbscan_clustering.labels, return_counts=True)
        dbscan_clustering.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        return dbscan_clustering, n_clusters_

    @staticmethod
    def _perform_clustering_optics(data, max_0, max_1, min_0, min_1, renormalize=True,
                                   min_samples: int = 5,
                                   min_cluster_size: Optional[float] = 0.1,
                                   max_eps: float = np.inf):
        optics_clustering = OPTICSClustering(data=data, min_cluster_size=min_cluster_size, min_samples=min_samples,
                                             max_eps=max_eps)
        scaling_dict = {'max_0': max_0,
                        'min_0': min_0,
                        'max_1': max_1,
                        'min_1': min_1}
        optics_clustering.cluster(renormalize=False, options=scaling_dict)
        labels_unique, points_per_cluster = np.unique(optics_clustering.labels, return_counts=True)
        optics_clustering.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        return optics_clustering, n_clusters_

    @staticmethod
    def _perform_clustering_birch(data, max_0, max_1, min_0, min_1, renormalize=True,
                                  branching_factor: int = 50,
                                  threshold: float = 0.5,
                                  compute_labels: bool = True,
                                  n_clusters: Optional[int] = None):
        birch_clustering = BirchClustering(data=data, threshold=threshold, branching_factor=branching_factor,
                                           compute_labels=compute_labels, n_clusters=n_clusters)
        scaling_dict = {'max_0': max_0,
                        'min_0': min_0,
                        'max_1': max_1,
                        'min_1': min_1}
        birch_clustering.cluster(renormalize=False, options=scaling_dict)
        labels_unique, points_per_cluster = np.unique(birch_clustering.labels, return_counts=True)
        birch_clustering.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        return birch_clustering, n_clusters_

    @staticmethod
    def _perform_clustering_hierarchical(data, max_0, max_1, min_0, min_1, renormalize=True,
                                         n_clusters: Optional[int] = 2, affinity: str = 'euclidean',
                                         memory: Optional[str] = None,
                                         connectivity=None, compute_full_tree='auto', linkage='ward',
                                         distance_threshold: float = None):
        hierarchical_clustering = HierarchicalClustering(data=data, n_clusters=n_clusters, affinity=affinity,
                                                         memory=memory,
                                                         connectivity=connectivity, compute_full_tree=compute_full_tree,
                                                         linkage=linkage, distance_threshold=distance_threshold)
        scaling_dict = {'max_0': max_0,
                        'min_0': min_0,
                        'max_1': max_1,
                        'min_1': min_1}
        hierarchical_clustering.cluster(renormalize=False, options=scaling_dict)
        labels_unique, points_per_cluster = np.unique(hierarchical_clustering.labels, return_counts=True)
        hierarchical_clustering.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        return hierarchical_clustering, n_clusters_

    def _process_frame_annotation(self, annotations_df, data_, fr):
        annotation = get_frame_annotations(annotations_df, frame_number=fr)  # check-out +/- 1
        annotation = preprocess_annotations(annotation)
        annotation_ = scale_annotations(annotation, self.original_shape, (data_.shape[0], data_.shape[1]))
        return annotation_, annotation

    def make_video(self, processed_data, video_label, video_number, video_out_save_path, annotations_df,
                   plot_scale_factor: int = 1, desired_fps=5, with_optical_flow: bool = True):
        if processed_data.shape[1] < processed_data.shape[2]:
            original_dims = (
                processed_data.shape[2] / 100 * plot_scale_factor, processed_data.shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (processed_data.shape[2], processed_data.shape[1]))
        else:
            original_dims = (
                processed_data.shape[1] / 100 * plot_scale_factor, processed_data.shape[2] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (processed_data.shape[1], processed_data.shape[2]))

        cap = cv.VideoCapture(self.video_path)
        cap_count = 0
        previous = None

        gt_bbox_cluster_center_dict = {}
        pr_for_frames = {}

        for fr in tqdm(range(0, processed_data.shape[0])):
            ret, frame = cap.read()
            if previous is None:
                previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                continue

            if cap_count < self.start_frame:
                continue
            next_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            flow, rgb = self.get_optical_flow(previous_frame=previous, next_frame=next_frame)

            if with_optical_flow:
                data, data_, max_0, max_1, min_0, min_1, threshold_img = self._prepare_data_xyuv(flow, fr,
                                                                                                 processed_data)
                mean_shift, n_clusters_ = self._perform_clustering(data, max_0, max_1, min_0, min_1, bandwidth=0.1)
            else:
                data, data_, threshold_img = self._prepare_data_xy(fr, processed_data)
                mean_shift, n_clusters_ = self._perform_clustering(data, 0, 0, 0, 0, renormalize=False, bandwidth=0.1)

            annotation_, annotation_full = self._process_frame_annotation(annotations_df, data_, fr)
            gt_bbox_cluster_center_dict.update({fr: {'gt_bbox': annotation_,
                                                     'cluster_centers': mean_shift.cluster_centers}})
            frame_results = evaluate_clustering_per_frame(fr, {'gt_bbox': annotation_,
                                                               'cluster_centers': mean_shift.cluster_centers})
            pre_rec = precision_recall(frame_results)
            pr_for_frames.update(pre_rec)

            fig = self.plot_all_steps(nrows=2, ncols=3, plot_dims=original_dims, original_frame=frame,
                                      processed_img=data_,
                                      threshold_image=threshold_img, optical_flow_image=rgb,
                                      frame_annotation=annotation_full,
                                      num_clusters=n_clusters_, cluster_centers=mean_shift.cluster_centers,
                                      frame_evaluation=pre_rec, video_out_save_path=video_out_save_path,
                                      frame_number=fr, video_label=video_label, video_number=video_number,
                                      video_mode=True)

            canvas = FigureCanvas(fig)
            canvas.draw()

            buf = canvas.buffer_rgba()
            out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
            out.write(out_frame)

            cap_count += 1
            previous = next_frame

        cap.release()
        out.release()
        return gt_bbox_cluster_center_dict, pr_for_frames

    @staticmethod
    def to_numpy_int(data):
        return (data * 255).int().squeeze().numpy()

    def _core_setup_algo(self, detect_shadows, history, var_threshold):
        return NotImplemented

    def _core_processing(self, frame, kernel):
        return NotImplemented


class AverageImageSubtraction(FeatureExtractor):
    def __init__(self, video_path):
        super(AverageImageSubtraction, self).__init__(video_path=video_path)
        self.average_image = None
        self.subtracted_images = None
        self.method = "Subtracted Image"

    def get_average_image(self):
        self.average_image = self.video_frames.mean(dim=0)
        return self.average_image

    def get_subtracted_image(self):
        self.get_average_image()
        average_frame_stacked = self.average_image.repeat(self.video_frames.size(0), 1, 1, 1)
        self.subtracted_images = (self.video_frames - average_frame_stacked).mean(dim=-1).unsqueeze(dim=-1)
        return self.subtracted_images

    def get_activations(self):
        return self.get_subtracted_image()

    def get_min_pooled_activations(self, kernel_size: int, iterations: int = 1, as_numpy_int: bool = True):
        processed_img = min_pool2d(self.subtracted_images.permute(0, 3, 1, 2), kernel_size=kernel_size)
        return self.to_numpy_int(processed_img) if as_numpy_int else processed_img


class BackgroundSubtraction(FeatureExtractor):
    def __init__(self, video_path, start_frame, end_frame):
        super(BackgroundSubtraction, self).__init__(video_path=video_path)
        self.method = "Background Subtraction"
        self.foreground_masks = None
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.algo = None

    def get_activations(self):
        return NotImplemented

    def get_activations_from_preloaded_frames(self, start_sec, end_sec, history=120, detect_shadows=True,
                                              var_threshold=100):
        return NotImplemented

    def get_min_pooled_activations(self, kernel_size: int, iterations: int = 1, as_numpy_int: bool = True):
        self.foreground_masks = self.foreground_masks / 255
        self.foreground_masks = min_pool2d(torch.from_numpy(self.foreground_masks).unsqueeze(1),
                                           kernel_size=kernel_size)
        return self.to_numpy_int(self.foreground_masks) if as_numpy_int else self.foreground_masks

    def _process_frames(self, cap, cap_count, kernel):
        out = None
        for _ in tqdm(range(0, self.end_frame)):
            ret, frame = cap.read()
            if out is None:
                out = np.zeros(shape=(0, frame.shape[0], frame.shape[1]))
                self.original_shape = (frame.shape[0], frame.shape[1])
            if cap_count < self.start_frame:
                continue

            mask = self.algo.apply(frame)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

            out = np.concatenate((out, np.expand_dims(mask, axis=0)), axis=0)
        return out

    def _process_preloaded_frames(self, frames, kernel):  # todo: set original_shape somewhere else
        out = None
        for frame in tqdm(range(0, frames.shape[0])):
            if out is None:
                out = np.zeros(shape=(0, frame.shape[1], frame.shape[2]))
                self.original_shape = (frame.shape[1], frame.shape[2])

            mask = self.algo.apply(frames[frame])
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

            out = np.concatenate((out, np.expand_dims(mask, axis=0)), axis=0)
        return out

    def _process_preloaded_n_frames(self, n, frames, kernel, algo):  # todo: set original_shape somewhere else
        out = None
        for frame in range(0, n):
            if out is None:
                out = np.zeros(shape=(0, frames[0].shape[0], frames[0].shape[1]))
                # self.original_shape = (frames[0].shape[0], frames[0].shape[1])

            mask = algo.apply(frames[frame])
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

            out = np.concatenate((out, np.expand_dims(mask, axis=0)), axis=0)
        return out

    def _core_processing(self, frame, kernel):
        mask = self.algo.apply(frame)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        return mask

    def _core_setup_algo(self, detect_shadows, history, var_threshold):
        return NotImplemented


class MOG2(BackgroundSubtraction):
    def __init__(self, video_path, start_frame, end_frame):
        super(MOG2, self).__init__(video_path=video_path, start_frame=start_frame, end_frame=end_frame)
        self.method = "MOG2"

    def get_activations(self, history=120, detect_shadows=True, var_threshold=100):
        cap = cv.VideoCapture(self.video_path)
        cap_count = 0
        kernel = self._core_setup_algo(detect_shadows, history, var_threshold)
        out = self._process_frames(cap, cap_count, kernel)
        return out

    def get_activations_from_preloaded_frames(self, start_sec, end_sec, history=120, detect_shadows=True,
                                              var_threshold=100):
        frames, _, _ = self.get_frames(start=start_sec, end=end_sec, dtype='int')
        frames = frames.numpy()
        kernel = self._core_setup_algo(detect_shadows, history, var_threshold)
        return self._process_preloaded_frames(frames, kernel)

    def _core_setup_algo(self, detect_shadows, history, var_threshold):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        if var_threshold is None:
            self.algo = cv.createBackgroundSubtractorMOG2(detectShadows=detect_shadows, history=history)
        else:
            self.algo = cv.createBackgroundSubtractorMOG2(detectShadows=detect_shadows, history=history,
                                                          varThreshold=var_threshold)
        return kernel

    def keyframe_based_experiment(self, start_sec, end_sec, save_path):
        frames, _, _ = self.get_frames(start=start_sec, end=end_sec, dtype='int')
        frames = frames.numpy()

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        frames_to_save = np.random.choice(len(frames), 2)
        n = 50
        step = len(frames) / (n + 1)
        for fr in tqdm(range(frames.shape[0])):
            selected_frames = [int((step * i) + fr) % len(frames) for i in range(1, n + 1)]
            frames_building_model = [frames[int((step * i) + fr) % len(frames)] for i in range(1, n + 1)]
            algo = cv.createBackgroundSubtractorMOG2(history=n)
            _ = self._process_preloaded_n_frames(n, frames_building_model, kernel, algo)

            mask = algo.apply(frames[fr], learningRate=0)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            fig, axs = plt.subplots(1, 2, sharex='none', sharey='none')
            axs[0].imshow(frames[fr])
            axs[1].imshow(mask, cmap='binary')

            axs[0].set_title(f'Image: {fr}')
            axs[1].set_title(f'Mask: {fr}')
            if fr in frames_to_save:
                fig.savefig(save_path + f"frame_{fr}.png")

    def keyframe_based_clustering(self, start_sec, end_sec, save_path, annotations_df, eval_frames, video_label,
                                  video_number, n, use_color, plot, weighted_of):
        frames, _, _ = self.get_frames(start=start_sec, end=end_sec, dtype='int')
        frames = frames.numpy()

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # frames_to_save = np.random.choice(len(frames), 2)
        step = len(frames) / (n + 1)
        pr_res = {}
        for fr in tqdm(range(frames.shape[0])):
            if fr in eval_frames:
                selected_frames = [int((step * i) + fr) % len(frames) for i in range(1, n + 1)]
                frames_building_model = [frames[int((step * i) + fr) % len(frames)] for i in range(1, n + 1)]

                algo = cv.createBackgroundSubtractorMOG2(history=n, varThreshold=100)
                # algo = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=n)

                _ = self._process_preloaded_n_frames(n, frames_building_model, kernel, algo)

                mask = algo.apply(frames[fr], learningRate=0)

                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

                previous = cv.cvtColor(frames[fr - 1], cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor(frames[fr], cv.COLOR_BGR2GRAY)

                # flow, rgb = self.get_optical_flow(previous_frame=previous, next_frame=next_frame)
                flow, rgb, mag, ang = self.get_optical_flow(previous_frame=previous, next_frame=next_frame,
                                                            all_results_out=True)

                if use_color:
                    data, data_, max_0, max_1, min_0, min_1, threshold_img = \
                        self._prepare_data_xyuv_color(frames[fr],
                                                      flow, fr,
                                                      mask,
                                                      evaluation_mode
                                                      =True,
                                                      lab_space=True)
                elif weighted_of:
                    data, data_, max_0, max_1, min_0, min_1, threshold_img = \
                        self._prepare_data_xy_weighted_uv(flow, fr,
                                                          mask,
                                                          mag,
                                                          evaluation_mode=True,
                                                          weight=10.0)

                else:
                    data, data_, max_0, max_1, min_0, min_1, threshold_img = \
                        self._prepare_data_xyuv(flow, fr,
                                                mask,
                                                evaluation_mode=True)

                cluster_algo, n_clusters_ = self._perform_clustering(data, max_0, max_1, min_0, min_1, bandwidth=0.1,
                                                                     min_bin_freq=3, max_iter=300)

                # cluster_algo, n_clusters_ = self._perform_clustering_affinity_propagation(data, max_0, max_1, min_0,
                #                                                                           min_1, preference=0,
                #                                                                           damping=0.7)

                annotation_, annotation_full = self._process_frame_annotation(annotations_df, data_, fr)

                frame_results = evaluate_clustering_per_frame(fr, {'gt_bbox': annotation_,
                                                                   'cluster_centers': cluster_algo.cluster_centers},
                                                              one_to_one=True)
                pre_rec = precision_recall(frame_results)
                pr_res.update(pre_rec)

                if frames[fr].shape[0] < frames[fr].shape[1]:
                    original_dims = (frames[fr].shape[1] / 100 * 1, frames[fr].shape[0] / 100 *
                                     1)
                else:
                    original_dims = (frames[fr].shape[0] / 100 * 1, frames[fr].shape[1] / 100 *
                                     1)

                self.original_shape = (frames[fr].shape[0], frames[fr].shape[1])

                if plot:
                    self.plot_all_steps(nrows=2, ncols=3, plot_dims=original_dims, original_frame=frames[fr],
                                        processed_img=data_,
                                        threshold_image=threshold_img, optical_flow_image=rgb,
                                        frame_annotation=annotation_full,
                                        num_clusters=n_clusters_, cluster_centers=cluster_algo.cluster_centers,
                                        frame_evaluation=pre_rec, video_out_save_path=save_path,
                                        frame_number=fr, video_label=video_label.value, video_number=video_number,
                                        video_mode=False, all_objects=len(annotation_))
        return pr_res

    def evaluate_clustering_algos(self, start_sec, end_sec, n, eval_frames, annotations_df):
        frames, _, _ = self.get_frames(start=start_sec, end=end_sec, dtype='int')
        frames = frames.numpy()

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # frames_to_save = np.random.choice(len(frames), 2)
        step = len(frames) / (n + 1)
        for fr in tqdm(range(frames.shape[0])):
            if fr in eval_frames:
                selected_frames = [int((step * i) + fr) % len(frames) for i in range(1, n + 1)]
                frames_building_model = [frames[int((step * i) + fr) % len(frames)] for i in range(1, n + 1)]

                algo = cv.createBackgroundSubtractorMOG2(history=n, varThreshold=100)
                # algo = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=n)

                _ = self._process_preloaded_n_frames(n, frames_building_model, kernel, algo)

                mask = algo.apply(frames[fr], learningRate=0)

                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

                previous = cv.cvtColor(frames[fr - 1], cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor(frames[fr], cv.COLOR_BGR2GRAY)

                # flow, rgb = self.get_optical_flow(previous_frame=previous, next_frame=next_frame)
                flow, rgb, mag, ang = self.get_optical_flow(previous_frame=previous, next_frame=next_frame,
                                                            all_results_out=True)

                data, data_, max_0, max_1, min_0, min_1, threshold_img = \
                    self._prepare_data_xyuv(flow, fr,
                                            mask,
                                            evaluation_mode=True)
                marker = cycle('o*v^><12348sphH+xXD')
                cluster_options_dict = {'max_0': max_0,
                                        'min_0': min_0,
                                        'max_1': max_1,
                                        'min_1': min_1}

                annotation_, annotation_full = self._process_frame_annotation(annotations_df, data_, fr)

                # AgglomerativeClustering
                # self._agglomerative_clustering(annotation_, cluster_options_dict, data, fr, max_0, max_1, min_0,
                #                                min_1)

                # BIRCH
                # self._birch_clustering(annotation_, cluster_options_dict, data, fr, frames, marker, max_0, max_1,
                #                        min_0, min_1)

                # OPTICS
                # self._optics_clustering(annotation_, cluster_options_dict, data, fr, frames, marker, max_0, max_1,
                #                         min_0, min_1, plot_reachability=False)

                # DBSCAN
                self._dbscan_cluster(annotation_, cluster_options_dict, data, fr, frames, marker, max_0, max_1, min_0,
                                     min_1)
                plt.close('all')

    def _dbscan_cluster(self, annotation_, cluster_options_dict, data, fr, frames, marker, max_0, max_1, min_0, min_1):
        cluster_algo, n_clusters_ = self._perform_clustering_dbscan(data, max_0, max_1, min_0,
                                                                    min_1, min_samples=13,
                                                                    eps=2)
        core_samples_mask = np.zeros_like(cluster_algo.labels, dtype=bool)
        core_samples_mask[cluster_algo.core_sample_indices] = True
        # Black removed and is used for noise instead.
        unique_labels = set(cluster_algo.labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        c_centers = []
        plt.imshow(frames[fr])
        for k, col, m in zip(unique_labels, colors, marker):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (cluster_algo.labels == k)

            xy = data[class_member_mask & core_samples_mask]
            xy = Clustering.renormalize_any_cluster(xy, cluster_options_dict)
            xy_c = xy.mean(axis=0)
            c_centers.append(xy_c)
            # plt.plot(xy[:, 0], xy[:, 1], m, markerfacecolor=tuple(col),
            #          markeredgecolor='k', markersize=8)
            plt.plot(xy_c[0], xy_c[1], m, markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=8)

            xy = data[class_member_mask & ~core_samples_mask]
            xy = Clustering.renormalize_any_cluster(xy, cluster_options_dict)
            plt.plot(xy[:, 0], xy[:, 1], '*', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        frame_results = evaluate_clustering_non_cc(fr, {'gt_bbox': annotation_,
                                                        'cluster_centers': c_centers},
                                                   one_to_one=True)
        pre_rec = precision_recall(frame_results)
        plt.title(f'DBSCAN Clusters: {len(unique_labels)} | P: {pre_rec[fr]["precision"]} | '
                  f'R: {pre_rec[fr]["recall"]}')
        plt.show()

    def _optics_clustering(self, annotation_, cluster_options_dict, data, fr, frames, marker, max_0, max_1, min_0,
                           min_1, plot_reachability=False):
        cluster_algo, n_clusters_ = self._perform_clustering_optics(data, max_0, max_1, min_0,
                                                                    min_1, min_samples=5,
                                                                    min_cluster_size=None,
                                                                    max_eps=15)
        clust = cluster_algo.algo
        labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=0.5)
        labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=2)
        space = np.arange(len(data))
        reachability = clust.reachability_[clust.ordering_]
        labels = clust.labels_[clust.ordering_]
        plt.figure(figsize=(15, 12))

        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(np.unique(labels)))]

        if plot_reachability:
            G = gridspec.GridSpec(2, 3)
            ax1 = plt.subplot(G[0, :])
            ax2 = plt.subplot(G[1, 0])
            ax3 = plt.subplot(G[1, 1])
            ax4 = plt.subplot(G[1, 2])

            # Reachability plot
            for klass, color in zip(range(0, 5), colors):
                Xk = space[labels == klass]
                Rk = reachability[labels == klass]
                ax1.plot(Xk, Rk, color, alpha=0.3)
            ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
            ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
            ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
            ax1.set_ylabel('Reachability (epsilon distance)')
            ax1.set_title('Reachability Plot')
        else:
            G = gridspec.GridSpec(1, 3)
            ax2 = plt.subplot(G[0, 0])
            ax3 = plt.subplot(G[0, 1])
            ax4 = plt.subplot(G[0, 2])

        ax2.imshow(frames[fr])
        ax3.imshow(frames[fr])
        ax4.imshow(frames[fr])
        c_center_optics = []
        c_center_db05 = []
        c_center_db2 = []

        # OPTICS
        for klass, color, m in zip(range(0, 5), colors, marker):
            Xk = data[clust.labels_ == klass]
            Xk = Clustering.renormalize_any_cluster(Xk, cluster_options_dict)
            xk_c = Xk.mean(axis=0)
            c_center_optics.append(xk_c.tolist())
            # ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
            ax2.plot(xk_c[0], xk_c[1], m, markerfacecolor=color, alpha=0.3, markeredgecolor='k', markersize=8)
        frame_results = evaluate_clustering_non_cc(fr, {'gt_bbox': annotation_,
                                                        'cluster_centers': c_center_optics},
                                                   one_to_one=True)
        pre_rec = precision_recall(frame_results)
        ax2.plot(data[clust.labels_ == -1, 0], data[clust.labels_ == -1, 1], 'k+', alpha=0.1)
        ax2.set_title(f'Automatic Clustering\nOPTICS\nP:{pre_rec[fr]["precision"]} |'
                      f' R:{pre_rec[fr]["recall"]}')

        # DBSCAN at 0.5
        for klass, color, m in zip(range(0, 6), colors, marker):
            Xk = data[labels_050 == klass]
            Xk = Clustering.renormalize_any_cluster(Xk, cluster_options_dict)
            xk_c = Xk.mean(axis=0)
            c_center_db05.append(xk_c.tolist())
            # ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
            ax2.plot(xk_c[0], xk_c[1], m, markerfacecolor=color, alpha=0.3, markeredgecolor='k', markersize=18)
        frame_results = evaluate_clustering_non_cc(fr, {'gt_bbox': annotation_,
                                                        'cluster_centers': c_center_db05},
                                                   one_to_one=True)
        pre_rec = precision_recall(frame_results)
        ax3.plot(data[labels_050 == -1, 0], data[labels_050 == -1, 1], 'k+', alpha=0.1)
        ax3.set_title(f'Clustering at 0.5 epsilon cut\nDBSCAN\nP:{pre_rec[fr]["precision"]} |'
                      f' R:{pre_rec[fr]["recall"]}')

        # DBSCAN at 2.
        for klass, color, m in zip(range(0, 4), colors, marker):
            Xk = data[labels_200 == klass]
            Xk = Clustering.renormalize_any_cluster(Xk, cluster_options_dict)
            xk_c = Xk.mean(axis=0)
            c_center_db2.append(xk_c.tolist())
            # ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
            ax2.plot(xk_c[0], xk_c[1], m, markerfacecolor=color, alpha=0.3, markeredgecolor='k', markersize=8)
        frame_results = evaluate_clustering_non_cc(fr, {'gt_bbox': annotation_,
                                                        'cluster_centers': c_center_db2},
                                                   one_to_one=True)
        pre_rec = precision_recall(frame_results)
        ax4.plot(data[labels_200 == -1, 0], data[labels_200 == -1, 1], 'k+', alpha=0.1)
        ax4.set_title(f'Clustering at 2.0 epsilon cut\nDBSCAN\nP:{pre_rec[fr]["precision"]} |'
                      f' R:{pre_rec[fr]["recall"]}')
        plt.tight_layout()
        plt.show()

    def _birch_clustering(self, annotation_, cluster_options_dict, data, fr, frames, marker, max_0, max_1, min_0,
                          min_1):
        cluster_algo, n_clusters_ = self._perform_clustering_birch(data, max_0, max_1, min_0,
                                                                   min_1, threshold=0.03, branching_factor=6)
        birch_model = cluster_algo.algo
        labels = birch_model.labels_
        centroids = birch_model.subcluster_centers_
        n_clusters = np.unique(labels).size
        print("n_clusters : %d" % n_clusters)
        centroids = Clustering.renormalize_any_cluster(centroids, cluster_options_dict)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(np.unique(labels)))]
        fig = plt.figure(figsize=(10, 8))
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(frames[fr])
        for this_centroid, k, col, m in zip(centroids, range(n_clusters), colors, marker):
            m_ = labels == k
            ax.scatter(data[m_, 0], data[m_, 1],
                       c='w', edgecolor=col, marker='.', alpha=0.5)
            if birch_model.n_clusters is None:
                ax.scatter(this_centroid[0], this_centroid[1], marker=m,
                           c='k', s=25)
        frame_results = evaluate_clustering_non_cc(fr, {'gt_bbox': annotation_,
                                                        'cluster_centers': centroids},
                                                   one_to_one=True)
        pre_rec = precision_recall(frame_results)
        ax.set_title(f'BIRCH Clusters:{n_clusters}\nP:{pre_rec[fr]["precision"]} |'
                     f'R:{pre_rec[fr]["recall"]}')
        plt.show()

    def _agglomerative_clustering(self, annotation_, cluster_options_dict, data, fr, frames, marker, max_0, max_1,
                                  min_0, min_1):
        cluster_algo, n_clusters_ = self._perform_clustering_hierarchical(data, max_0, max_1, min_0, min_1,
                                                                          n_clusters=None,
                                                                          distance_threshold=0.03)

        def plot_clustering(X_red, labels, title=None):
            center = []

            f = plt.figure(figsize=(10, 8))
            plt.imshow(frames[fr])
            for i, m in zip(range(cluster_algo.n_clusters), marker):
                Xk = X_red[cluster_algo.labels == i]
                xk_c = Xk.mean(axis=0)
                center.append(xk_c)
                plt.plot(xk_c[0], xk_c[1], m,
                         color=plt.cm.nipy_spectral(labels[i] / 10.), markersize=8)
                # plt.plot(Xk[i, 0], Xk[i, 1], m,
                #          color=plt.cm.nipy_spectral(labels[i] / 10.), markersize=8)

            if title is not None:
                plt.title(title, size=8)
            plt.axis('off')
            return center, f

        dd_data = Clustering.renormalize_any_cluster(data, cluster_options_dict)
        c_center, fig = plot_clustering(dd_data, cluster_algo.labels)
        frame_results = evaluate_clustering_non_cc(fr, {'gt_bbox': annotation_,
                                                        'cluster_centers': c_center},
                                                   one_to_one=True)
        pre_rec = precision_recall(frame_results)
        fig.suptitle(f'Agglomerative Clusters:{cluster_algo.n_clusters}'
                     f'\nP:{pre_rec[fr]["precision"]} |'
                     f'R:{pre_rec[fr]["recall"]}', fontsize=14, fontweight='bold')
        plt.show()

    def keyframe_based_rescaled_clustering(self, start_sec, end_sec, save_path, annotations_df, eval_frames,
                                           video_label,
                                           video_number, n, sdd_meta_path, dataset_type):
        frames, _, _ = self.get_frames(start=start_sec, end=end_sec, dtype='int')
        sdd_meta = SDDMeta(sdd_meta_path)
        new_scale = sdd_meta.get_new_scale_from_img(frames[0], dataset_type, video_number)
        new_scale_channels = new_scale[0], new_scale[1], 3
        # frames = torch.nn.functional.interpolate(frames, size=new_scale)  # memory-issue
        frames = frames.numpy()

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # frames_to_save = np.random.choice(len(frames), 2)
        step = len(frames) / (n + 1)
        for fr in tqdm(range(frames.shape[0])):
            if fr in eval_frames:
                selected_frames = [int((step * i) + fr) % len(frames) for i in range(1, n + 1)]
                frames_building_model = [(resize(frames[int((step * i) + fr) % len(frames)],
                                                 output_shape=new_scale) * 255).astype(np.uint8)
                                         for i in range(1, n + 1)]

                # algo = cv.createBackgroundSubtractorMOG2(history=n, varThreshold=50)
                algo = cv.createBackgroundSubtractorMOG2(history=n)

                _ = self._process_preloaded_n_frames(n, frames_building_model, kernel, algo)

                if frames.shape[1] < frames.shape[2]:
                    original_dims = (frames.shape[2] / 100 * 1, frames.shape[1] / 100 *
                                     1)
                else:
                    original_dims = (frames.shape[1] / 100 * 1, frames.shape[2] / 100 *
                                     1)

                self.original_shape = (frames.shape[1], frames.shape[2])

                mask = algo.apply((resize(frames[fr], output_shape=new_scale) * 255).astype(np.uint8), learningRate=0)

                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

                previous = cv.cvtColor((resize(frames[fr - 1], output_shape=new_scale) * 255).astype(np.uint8),
                                       cv.COLOR_BGR2GRAY)
                next_frame = cv.cvtColor((resize(frames[fr], output_shape=new_scale) * 255).astype(np.uint8),
                                         cv.COLOR_BGR2GRAY)

                flow, rgb = self.get_optical_flow(previous_frame=previous, next_frame=next_frame)

                data, data_, max_0, max_1, min_0, min_1, threshold_img = self._prepare_data_xyuv(flow, fr,
                                                                                                 mask,
                                                                                                 evaluation_mode=True)
                mean_shift, n_clusters_ = self._perform_clustering(data, max_0, max_1, min_0, min_1, bandwidth=0.1)

                annotation_, annotation_full = self._process_frame_annotation(annotations_df, data_, fr)

                frame_results = evaluate_clustering_per_frame(fr, {'gt_bbox': annotation_,
                                                                   'cluster_centers': mean_shift.cluster_centers})
                pre_rec = precision_recall(frame_results)

                in_frame = (resize(frames[fr], output_shape=new_scale) * 255).astype(np.uint8)

                self.plot_all_steps_rescaled(nrows=2, ncols=3, plot_dims=original_dims, original_frame=in_frame,
                                             processed_img=data_,
                                             threshold_image=threshold_img, optical_flow_image=rgb,
                                             frame_annotation=annotation_full,
                                             num_clusters=n_clusters_, cluster_centers=mean_shift.cluster_centers,
                                             frame_evaluation=pre_rec, video_out_save_path=save_path,
                                             frame_number=fr, video_label=video_label.value, video_number=video_number,
                                             video_mode=False, all_objects=len(annotation_))


class KNNBased(BackgroundSubtraction):
    def __init__(self, video_path, start_frame, end_frame):
        super(KNNBased, self).__init__(video_path=video_path, start_frame=start_frame, end_frame=end_frame)
        self.method = "KNN"

    def get_activations(self, history=120, detect_shadows=True, var_threshold=100):
        cap = cv.VideoCapture(self.video_path)
        cap_count = 0
        kernel = self._core_setup_algo(detect_shadows, history, var_threshold)

        out = self._process_frames(cap, cap_count, kernel)
        return out

    def get_activations_from_preloaded_frames(self, start_sec, end_sec, history=120, detect_shadows=True,
                                              var_threshold=100):
        frames, _, _ = self.get_frames(start=start_sec, end=end_sec, dtype='int')
        frames = frames.numpy()
        kernel = self._core_setup_algo(detect_shadows, history, var_threshold)

        return self._process_preloaded_frames(frames, kernel)

    def _core_setup_algo(self, detect_shadows, history, var_threshold):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        if var_threshold is None:
            self.algo = cv.createBackgroundSubtractorKNN(detectShadows=detect_shadows, history=history)
        else:
            self.algo = cv.createBackgroundSubtractorKNN(detectShadows=detect_shadows, history=history,
                                                         dist2Threshold=var_threshold)
        return kernel


class DNNFeatureExtractors(FeatureExtractor):
    def __init__(self, video_path):
        super(DNNFeatureExtractors, self).__init__(video_path=video_path)
        self.method = "DNN"

    def vgg_activations(self, layer_number: int = 3, scale_factor: Optional[float] = None):
        v_frames = self.video_frames.permute(0, 3, 1, 2)
        if scale_factor is not None:
            v_frames = resize_v_frames(v_frames=v_frames, scale_factor=scale_factor)
        vgg_out = get_vgg_layer_activations(x=v_frames, layer_number=layer_number)
        return vgg_out.permute(0, 2, 3, 1)

    def resnet_activations(self, scale_factor: Optional[float] = None):
        v_frames = self.video_frames.permute(0, 3, 1, 2)
        if scale_factor is not None:
            v_frames = resize_v_frames(v_frames=v_frames, scale_factor=scale_factor)
        out = get_resnet_layer_activations(x=v_frames)
        return out.permute(0, 2, 3, 1)

    def densenet_activations(self, layer_number: int = 3, scale_factor: Optional[float] = None):
        v_frames = self.video_frames.permute(0, 3, 1, 2)
        if scale_factor is not None:
            v_frames = resize_v_frames(v_frames=v_frames, scale_factor=scale_factor)
        # out = get_densenet_layer_activations(x=v_frames, layer_number=layer_number)
        out = get_densenet_filtered_layer_activations(x=v_frames, layer_number=layer_number)
        return out.permute(0, 2, 3, 1)
