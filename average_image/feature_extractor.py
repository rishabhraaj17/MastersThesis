from itertools import cycle
from typing import Union, Optional

import cv2 as cv
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

from bbox_utils import add_bbox_to_axes, resize_v_frames, get_frame_annotations, preprocess_annotations, \
    scale_annotations
from constants import OBJECT_CLASS_COLOR_MAPPING
from deep_networks_avg import get_vgg_layer_activations, get_resnet_layer_activations, \
    get_densenet_filtered_layer_activations
from feature_clustering import MeanShiftClustering
from layers import min_pool2d
from utils import precision_recall, evaluate_clustering_per_frame, normalize


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

    def get_optical_flow(self, previous_frame, next_frame):
        flow = cv.calcOpticalFlowFarneback(previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros((next_frame.shape[0], next_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # 0-1 normalize
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        return flow, rgb

    def plot(self):
        return NotImplemented

    def plot_all_steps(self, nrows, ncols, plot_dims, original_frame, processed_img, threshold_image,
                       optical_flow_image, frame_annotation, num_clusters, cluster_centers, frame_evaluation,
                       frame_number, video_label, video_number, video_out_save_path, video_mode=False):
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

        self.project_cluster_centers(axs, cluster_centers, num_clusters)

        patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
        fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {video_number}\nMethod: {self.method}"
                     f"\nPrecision: {frame_evaluation[frame_number]['precision']}"
                     f"\nRecall: {frame_evaluation[frame_number]['recall']}",
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

    def plot_3(self, nrows, ncols, plot_dims, original_frame, processed_img, threshold_image,
               frame_annotation, num_clusters, cluster_centers, frame_evaluation,
               frame_number, video_label, video_number, video_out_save_path):
        original_shape = (original_frame.shape[0], original_frame[1])
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

    @staticmethod
    def _prepare_data_xyuv(flow, fr, processed_data, use_intensities=False):
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
    def _prepare_data_xy(fr, processed_data, use_intensities=False):
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
    def _perform_clustering(data, max_0, max_1, min_0, min_1, renormalize=True, bandwidth: float = 0.1):
        mean_shift = MeanShiftClustering(data=data, bandwidth=bandwidth)
        mean_shift_dict = {'max_0': max_0,
                           'min_0': min_0,
                           'max_1': max_1,
                           'min_1': min_1}
        mean_shift.cluster(renormalize=renormalize, options=mean_shift_dict)
        labels_unique, points_per_cluster = np.unique(mean_shift.labels, return_counts=True)
        mean_shift.cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        return mean_shift, n_clusters_

    def _process_frame_annotation(self, annotations_df, data_, fr):
        annotation = get_frame_annotations(annotations_df, frame_number=fr)  # check-out +/- 1
        annotation = preprocess_annotations(annotation)
        annotation_ = scale_annotations(annotation, self.original_shape, (data_.shape[0], data_.shape[1]))
        return annotation_, annotation

    def make_video(self, processed_data, video_label, video_number, video_out_save_path, annotations_df,
                   plot_scale_factor: int = 1, desired_fps=5, with_optical_flow: bool = True):
        if processed_data.shape[1] < processed_data.shape[2]:
            original_dims = (processed_data.shape[2] / 100 * plot_scale_factor, processed_data.shape[1] / 100 * plot_scale_factor)
            out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                 (processed_data.shape[2], processed_data.shape[1]))
        else:
            original_dims = (processed_data.shape[1] / 100 * plot_scale_factor, processed_data.shape[2] / 100 * plot_scale_factor)
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


class MOG2(BackgroundSubtraction):
    def __init__(self, video_path, start_frame, end_frame):
        super(MOG2, self).__init__(video_path=video_path, start_frame=start_frame, end_frame=end_frame)
        self.method = "MOG2"

    def get_activations(self, history=120, detect_shadows=True, var_threshold=100):
        cap = cv.VideoCapture(self.video_path)
        cap_count = 0
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        if var_threshold is None:
            self.algo = cv.createBackgroundSubtractorMOG2(detectShadows=detect_shadows, history=history)
        else:
            self.algo = cv.createBackgroundSubtractorMOG2(detectShadows=detect_shadows, history=history,
                                                          varThreshold=var_threshold)
        out = self._process_frames(cap, cap_count, kernel)
        return out


class KNNBased(BackgroundSubtraction):
    def __init__(self, video_path, start_frame, end_frame):
        super(KNNBased, self).__init__(video_path=video_path, start_frame=start_frame, end_frame=end_frame)
        self.method = "KNN"

    def get_activations(self, history=120, detect_shadows=True, var_threshold=100):
        cap = cv.VideoCapture(self.video_path)
        cap_count = 0
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        if var_threshold is None:
            self.algo = cv.createBackgroundSubtractorKNN(detectShadows=detect_shadows, history=history)
        else:
            self.algo = cv.createBackgroundSubtractorKNN(detectShadows=detect_shadows, history=history,
                                                         dist2Threshold=var_threshold)
        out = self._process_frames(cap, cap_count, kernel)
        return out


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
