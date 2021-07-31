import os
import shutil
from pathlib import Path
from typing import List

import motmetrics as mm
import numpy as np
import pandas as pd
import scipy
import skimage
import torch
import torchvision.io
from matplotlib import pyplot as plt, patches
from omegaconf import OmegaConf
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.plot_utils import add_box_to_axes, add_box_to_axes_with_annotation
from src.position_maps.evaluate import get_image_array_from_figure, process_numpy_video_frame_to_tensor


class Track(object):
    def __init__(self, idx):
        super(Track, self).__init__()
        self.idx = idx
        self.gt_idx_list = []
        self.extracted_idx_list = []
        self.gt_frames = []
        self.extracted_frames = []
        self.gt_coordinates = []
        self.extracted_coordinates = []

    def __repr__(self):
        return f"Track: {self.idx}"


class ExtractedTrack(object):
    def __init__(self, idx):
        super(ExtractedTrack, self).__init__()
        self.idx = idx
        self.frames = []
        self.coordinates = []
        self.active = True

    def __repr__(self):
        return f"Track: {self.idx}"


class VideoSequenceTracksData(object):
    def __init__(self, video_class, video_number, tracks: List[Track]):
        super(VideoSequenceTracksData, self).__init__()
        self.video_class = video_class
        self.video_number = video_number
        self.tracks = tracks

    def __getitem__(self, item):
        return [t for t in self.tracks if t.idx == item][0]

    def __contains__(self, item):
        for t in self.tracks:
            if t.idx == item:
                return True
        return False

    def __repr__(self):
        return f"{self.video_class.name} | {self.video_number}\n{self.tracks}"

    def get_alive_gt_features(self):
        gt_features = [np.stack(t.gt_coordinates) for t in self.tracks]
        return gt_features

    def get_alive_extracted_features(self):
        gt_features = [np.stack(t.extracted_coordinates) for t in self.tracks]
        return gt_features


class ExtractedTracksData(object):
    def __init__(self, video_class, video_number, tracks: List[ExtractedTrack]):
        super(ExtractedTracksData, self).__init__()
        self.video_class = video_class
        self.video_number = video_number
        self.tracks = tracks

    def __getitem__(self, item):
        return [t for t in self.tracks if t.idx == item][0]

    def __contains__(self, item):
        for t in self.tracks:
            if t.idx == item:
                return True
        return False

    def __repr__(self):
        return f"{self.video_class.name} | {self.video_number}\n{self.tracks}"

    def get_alive_features(self):
        gt_features = [np.stack(t.coordinates) for t in self.tracks if t.active]
        return gt_features


class TracksAnalyzer(object):
    def __init__(self, config):
        self.config = config.metrics_analysis

        self.video_classes = []
        self.video_meta_classes = []
        for v_class in self.config.video_classes:
            self.video_classes.append(getattr(SDDVideoClasses, v_class))
            self.video_meta_classes.append(getattr(SDDVideoDatasets, v_class))

        self.video_numbers = self.config.video_numbers
        self.root = self.config.root

    @staticmethod
    def get_gt_df(gt_annotation_path):
        gt_df = pd.read_csv(gt_annotation_path)
        gt_df = gt_df.drop(gt_df.columns[[0]], axis=1)
        gt_df = gt_df.sort_values(by=['frame']).reset_index()
        gt_df = gt_df.drop(columns=['index'])
        return gt_df

    def get_extracted_centers(self, extracted_df, frame):
        extracted_centers = self.get_frame_annotations(df=extracted_df, frame_number=frame)
        return extracted_centers[:, 2:], extracted_centers[:, 1]

    @staticmethod
    def get_gt_annotation(frame_number, gt_annotation_df, original_shape):
        frame_annotation = get_frame_annotations_and_skip_lost(gt_annotation_df, frame_number)
        gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                            original_scale=original_shape,
                                                            new_scale=original_shape,
                                                            return_track_id=False,
                                                            tracks_with_annotations=True)
        supervised_boxes = gt_annotations[:, :-1]
        # dont need it
        # inside_boxes_idx = [b for b, box in enumerate(supervised_boxes)
        #                     if (box[0] > 0 and box[2] < original_shape[1])
        #                     and (box[1] > 0 and box[3] < original_shape[0])]
        # supervised_boxes = supervised_boxes[inside_boxes_idx]
        # gt_bbox_centers = gt_bbox_centers[inside_boxes_idx]
        return gt_bbox_centers, supervised_boxes, gt_annotations[:, -1]

    @staticmethod
    def get_frame_annotations(df: pd.DataFrame, frame_number: int):
        idx: pd.DataFrame = df.loc[df["frame"] == frame_number]
        return idx.to_numpy()

    def get_ratio(self, meta_class, video_number):
        sdd_meta = SDDMeta(self.root + 'H_SDD.txt')
        ratio = float(sdd_meta.get_meta(meta_class, video_number)[0]['Ratio'].to_numpy()[0])
        return ratio

    def get_associations_and_metrics(self, gt_centers, extracted_centers, max_distance, ratio, threshold):
        match_rows, match_cols = self.get_associations(gt_centers, extracted_centers, max_distance, ratio, threshold)
        fn, fp, precision, recall, tp = self.get_precision_and_recall(extracted_centers, gt_centers, match_rows)
        return (match_rows, match_cols), (fn, fp, precision, recall, tp)

    @staticmethod
    def get_precision_and_recall(extracted_centers, gt_centers, match_rows):
        tp = len(match_rows)
        fp = len(extracted_centers) - len(match_rows)
        fn = len(gt_centers) - len(match_rows)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return fn, fp, precision, recall, tp

    @staticmethod
    def get_associations(gt_centers, extracted_centers, max_distance, ratio, threshold):
        distance_matrix = np.sqrt(mm.distances.norm2squared_matrix(
            gt_centers, extracted_centers, max_d2=max_distance)) * ratio
        distance_matrix = threshold - distance_matrix
        distance_matrix[distance_matrix < 0] = 1000
        # Hungarian
        match_rows, match_cols = scipy.optimize.linear_sum_assignment(distance_matrix)
        actually_matched_mask = distance_matrix[match_rows, match_cols] < 1000
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]
        return match_rows, match_cols

    @staticmethod
    def get_frame_by_track_annotations(df: pd.DataFrame, frame_number: int, track_id: int, for_gt: bool,
                                       return_current_frame_only: bool = False):
        track_id_key = 'track_id' if for_gt else 'track'
        filtered_by_track_id = df.loc[df[track_id_key] == track_id]
        if return_current_frame_only:
            idx: pd.DataFrame = filtered_by_track_id.loc[filtered_by_track_id["frame"] == frame_number]
            return idx.to_numpy()
        return filtered_by_track_id.to_numpy()

    def construct_new_track(self, extracted_coordinates, extracted_track_id, frame, gt_coordinates, gt_track_id):
        track = Track(idx=gt_track_id)
        self.update_existing_track(track, extracted_coordinates, extracted_track_id, frame,
                                   gt_coordinates, gt_track_id)
        return track

    @staticmethod
    def update_existing_track(existing_track, extracted_coordinates, extracted_track_id, frame, gt_coordinates,
                              gt_track_id):
        existing_track.gt_idx_list.append(gt_track_id)
        existing_track.extracted_idx_list.append(int(extracted_track_id))
        existing_track.gt_frames.append(frame)
        existing_track.extracted_frames.append(frame)
        existing_track.gt_coordinates.append(gt_coordinates.tolist())
        existing_track.extracted_coordinates.append(extracted_coordinates.tolist())
        
    @staticmethod
    def get_frame_from_figure(fig, original_shape):
        video_frame = get_image_array_from_figure(fig)

        # if video_frame.shape[0] != original_shape[1] \
        #         or video_frame.shape[1] != original_shape[0]:
        #     video_frame = skimage.transform.resize(
        #         video_frame, (original_shape[1], original_shape[0]))
        #     video_frame = (video_frame * 255).astype(np.uint8)
        #
        #     video_frame = process_numpy_video_frame_to_tensor(video_frame)
        video_frame = process_numpy_video_frame_to_tensor(video_frame)
        return video_frame

    @staticmethod
    def add_features_to_axis(ax, features, marker_size=8, marker_shape='o', marker_color='blue'):
        for f in features:
            ax.plot(f[:, 0], f[:, 1], marker_shape, markerfacecolor=marker_color, markeredgecolor='k',
                    markersize=marker_size)
        
    def plot(self, frame, boxes, gt_features, extracted_features, frame_number, box_annotation,
             marker_size=1, radius=None, fig_title='', footnote_text='', video_mode=False,
             boxes_with_annotation=True):
        fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(8, 10))

        axs.imshow(frame)
        
        legends_dict = {}
        if gt_features is not None:
            self.add_features_to_axis(axs, gt_features, marker_size=marker_size, marker_color='b')
            legends_dict.update({'b': 'GT Locations'})

        if extracted_features is not None:
            self.add_features_to_axis(axs, extracted_features, marker_size=marker_size, marker_color='g')
            legends_dict.update({'g': 'Extracted Locations'})

        if boxes is not None:
            if boxes_with_annotation:
                add_box_to_axes_with_annotation(axs, boxes, box_annotation)
            else:
                add_box_to_axes(axs, boxes)
            legends_dict.update({'r': 'GT Boxes'})

        if radius is not None:
            for c_center in np.concatenate(extracted_features):
                axs.add_artist(plt.Circle((c_center[0], c_center[1]), radius, color='yellow', fill=False))
            legends_dict.update({'yellow': 'Neighbourhood Radius'})

        axs.set_title(f'Frame: {frame_number}')

        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout(pad=1.58)

        legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
        fig.legend(handles=legend_patches, loc=2)

        plt.suptitle(fig_title)
        plt.figtext(0.99, 0.01, footnote_text, horizontalalignment='right')

        if video_mode:
            plt.close()
        else:
            plt.show()

        return fig

    def perform_analysis_on_multiple_sequences(self, show_extracted_tracks_only=False):
        metrics = {}
        for v_idx, (v_clz, v_meta_clz) in enumerate(zip(self.video_classes, self.video_meta_classes)):
            for v_num in self.video_numbers[v_idx]:
                gt_annotation_path = f"{self.root}annotations/{v_clz.value}/video{v_num}/annotation_augmented.csv"
                extracted_annotation_path = f"{self.root}pm_extracted_annotations/{v_clz.value}/" \
                                            f"video{v_num}/trajectories.csv"
                ref_img = torchvision.io.read_image(f"{self.root}annotations/{v_clz.value}/video{v_num}/reference.jpg")
                video_path = f"{self.root}/videos/{v_clz.value}/video{v_num}/video.mov"
                if show_extracted_tracks_only:
                    self.construct_extracted_tracks_only(
                        extracted_annotation_path, v_clz, v_num, video_path, (ref_img.shape[1:]))
                else:
                    data, p, r = self.perform_analysis_on_single_sequence(
                        gt_annotation_path, extracted_annotation_path, v_clz, v_meta_clz, v_num, (ref_img.shape[1:]),
                        video_path)
                    ade, fde = self.calculate_ade_fde_for_associations(
                        video_sequence_track_data=data, meta_class=v_meta_clz, video_number=v_num)
                    if v_clz.name in metrics.keys():
                        metrics[v_clz.name][v_num] = {
                            'ade': ade, 'fde': fde,
                            'precision': p, 'recall': r,
                            'neighbourhood_radius': self.config.threshold
                        }
                    else:
                        metrics[v_clz.name] = {
                            v_num: {
                                'ade': ade, 'fde': fde,
                                'precision': p, 'recall': r,
                                'neighbourhood_radius': self.config.threshold
                            }
                        }
        self.save_as_csv(metrics=metrics)
        return metrics

    def perform_analysis_on_single_sequence(
            self, gt_annotation_path, extracted_annotation_path, video_class,
            video_meta_class, video_number, image_shape, video_path):
        video_sequence_track_data = VideoSequenceTracksData(
            video_class=video_class, video_number=video_number, tracks=[])
        ratio = self.get_ratio(meta_class=video_meta_class, video_number=video_number)
        video_frames = []
        tp_list, fp_list, fn_list = [], [], []

        gt_df = self.get_gt_df(gt_annotation_path)
        extracted_df = pd.read_csv(extracted_annotation_path)

        for frame in tqdm(gt_df.frame.unique()):
            gt_bbox_centers, supervised_boxes, gt_track_ids = self.get_gt_annotation(
                frame_number=frame, gt_annotation_df=gt_df, original_shape=tuple(image_shape))
            extracted_centers, extracted_track_ids = self.get_extracted_centers(extracted_df, frame)
            (match_rows, match_cols), (fn, fp, precision, recall, tp) = self.get_associations_and_metrics(
                gt_centers=gt_bbox_centers, extracted_centers=extracted_centers,
                max_distance=float(self.config.match_distance),
                ratio=ratio,
                threshold=self.config.threshold
            )
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            if frame == 0:
                for r, c in zip(match_rows, match_cols):
                    gt_coordinates = gt_bbox_centers[r]
                    extracted_coordinates = extracted_centers[c]

                    gt_track_id = gt_track_ids[r]
                    extracted_track_id = extracted_track_ids[c]

                    track = self.construct_new_track(extracted_coordinates, extracted_track_id, frame, gt_coordinates,
                                                     gt_track_id)

                    video_sequence_track_data.tracks.append(track)
            else:
                for r, c in zip(match_rows, match_cols):
                    gt_coordinates = gt_bbox_centers[r]
                    extracted_coordinates = extracted_centers[c]

                    gt_track_id = gt_track_ids[r]
                    extracted_track_id = extracted_track_ids[c]
                    gt_tracks_for_agents_in_frame = self.get_frame_by_track_annotations(
                        gt_df, frame_number=frame, track_id=gt_track_id, for_gt=True)
                    extracted_tracks_for_agents_in_frame = self.get_frame_by_track_annotations(
                        extracted_df, frame_number=frame, track_id=extracted_track_id, for_gt=False)

                    if gt_track_id in video_sequence_track_data:
                        existing_track = video_sequence_track_data[gt_track_id]
                        self.update_existing_track(existing_track, extracted_coordinates, extracted_track_id, frame,
                                                   gt_coordinates, gt_track_id)
                    else:
                        track = self.construct_new_track(extracted_coordinates, extracted_track_id, frame,
                                                         gt_coordinates,
                                                         gt_track_id)

                        video_sequence_track_data.tracks.append(track)
                        
            if self.config.show_plot or self.config.make_video:
                fig = self.plot(
                    frame=extract_frame_from_video(video_path, frame_number=frame),
                    boxes=supervised_boxes,
                    gt_features=video_sequence_track_data.get_alive_gt_features()
                    if self.config.plot_gt_features else [],
                    extracted_features=video_sequence_track_data.get_alive_extracted_features(),
                    frame_number=frame,
                    marker_size=self.config.marker_size,
                    radius=self.config.threshold,
                    fig_title=f"Precision: {precision} | Recall: {recall}",
                    footnote_text=f"{video_class.name} - {video_number}\n"
                                  f"Neighbourhood Radius: {self.config.threshold}m",
                    video_mode=self.config.make_video,
                    box_annotation=gt_track_ids,
                    boxes_with_annotation=True
                )
            if self.config.make_video:
                video_frames.append(self.get_frame_from_figure(fig, original_shape=image_shape))
        overall_precision = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fp_list).sum())
        overall_recall = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fn_list).sum())
        if self.config.make_video:
            print(f"Writing Video")
            Path(os.path.join(os.getcwd(), 'logs/analysis_videos/')).mkdir(parents=True, exist_ok=True)
            torchvision.io.write_video(
                f'logs/analysis_videos/{video_class.name}_'
                f'{video_number}_'
                f'neighbourhood_radius_{self.config.threshold}.avi',
                torch.cat(video_frames).permute(0, 2, 3, 1),
                self.config.video_fps)
        print(f"Analysis done for {video_class.name} - {video_number}")
        return video_sequence_track_data, overall_precision, overall_recall

    @staticmethod
    def split_track_ids_into_chunks_of_indices(idx_list):
        search_num = idx_list[0]
        splits_idx = []
        temp = []
        for i, e in enumerate(idx_list):
            if e == search_num:
                temp.append(i)
            else:
                splits_idx.append(temp)
                search_num = e
                temp = [i]
        splits_idx.append(temp)
        return splits_idx

    def calculate_ade_fde_for_associations(
            self, video_sequence_track_data: VideoSequenceTracksData, meta_class, video_number):
        ratio = self.get_ratio(meta_class=meta_class, video_number=video_number)
        global_ade, global_fde = [], []
        for track in video_sequence_track_data.tracks:
            local_ade, local_fde = [], []
            extracted_split_idx_list = self.split_track_ids_into_chunks_of_indices(np.array(track.extracted_idx_list))
            for split_idx in extracted_split_idx_list:
                split_gt_locations = np.array(track.gt_coordinates)[split_idx]
                split_extracted_locations = np.array(track.extracted_coordinates)[split_idx]
                ade = np.linalg.norm(
                    (split_gt_locations - split_extracted_locations), ord=2, axis=-1).mean().item() * ratio
                fde = np.linalg.norm(
                    (split_gt_locations[-1] - split_extracted_locations[-1]), ord=2, axis=-1).item() * ratio
                local_ade.append(ade)
                local_fde.append(fde)
            global_ade.append(np.array(local_ade).mean().item())
            global_fde.append(np.array(local_fde).mean().item())
        return np.array(global_ade).mean().item(), np.array(global_fde).mean().item()

    def save_as_csv(self, metrics):
        video_class, video_number, ade, fde, precision, recall, radius = [], [], [], [], [], [], []
        for k, v in metrics.items():
            for vk, vv in v.items():
                video_class.append(k)
                video_number.append(vk)
                ade.append(vv['ade'])
                fde.append(vv['fde'])
                precision.append(vv['precision'])
                recall.append(vv['recall'])
                radius.append(vv['neighbourhood_radius'])
        df: pd.DataFrame = pd.DataFrame({
            'class': video_class,
            'number': video_number,
            'ade': ade,
            'fde': fde,
            'precision': precision,
            'recall': recall,
            'neighbourhood_radius': radius
        })
        df.to_csv(f"{self.root}/pm_extracted_annotations/metrics.csv", index=False)

    def construct_extracted_tracks_only(
            self, extracted_annotation_path, video_class, video_number, video_path, image_shape):
        video_sequence_track_data = ExtractedTracksData(
            video_class=video_class, video_number=video_number, tracks=[])
        video_frames = []

        extracted_df = pd.read_csv(extracted_annotation_path)

        for frame in tqdm(extracted_df.frame.unique()):
            extracted_centers, extracted_track_ids = self.get_extracted_centers(extracted_df, frame)
            if frame == 0:
                for e_t_idx, e_center in zip(extracted_track_ids, extracted_centers):
                    track = ExtractedTrack(idx=int(e_t_idx))
                    track.frames.append(frame)
                    track.coordinates.append(e_center.tolist())
                    video_sequence_track_data.tracks.append(track)
            else:
                for e_t_idx, e_center in zip(extracted_track_ids, extracted_centers):
                    if e_t_idx in video_sequence_track_data:
                        track = video_sequence_track_data[e_t_idx]
                        track.frames.append(frame)
                        track.coordinates.append(e_center.tolist())
                    else:
                        track = ExtractedTrack(idx=int(e_t_idx))
                        track.frames.append(frame)
                        track.coordinates.append(e_center.tolist())
                        video_sequence_track_data.tracks.append(track)

            # filter out inactive tracks
            for t in video_sequence_track_data.tracks:
                if t.frames[-1] != frame:
                    t.active = False

            if self.config.show_plot or self.config.make_video:
                fig = self.plot(
                    frame=extract_frame_from_video(video_path, frame_number=frame),
                    boxes=[],
                    gt_features=[],
                    extracted_features=video_sequence_track_data.get_alive_features(),
                    frame_number=frame,
                    marker_size=self.config.marker_size,
                    radius=self.config.threshold,
                    fig_title="",
                    footnote_text=f"{video_class.name} - {video_number}",
                    video_mode=self.config.make_video,
                    box_annotation=extracted_track_ids,
                    boxes_with_annotation=True
                )
            if self.config.make_video:
                video_frames.append(self.get_frame_from_figure(fig, original_shape=tuple(image_shape)))
        if self.config.make_video:
            print(f"Writing Video")
            Path(os.path.join(os.getcwd(), 'logs/analysis_videos/')).mkdir(parents=True, exist_ok=True)
            torchvision.io.write_video(
                f'logs/analysis_videos/extracted_only_{video_class.name}_'
                f'{video_number}_'
                f'neighbourhood_radius_{self.config.threshold}.avi',
                torch.cat(video_frames).permute(0, 2, 3, 1),
                self.config.video_fps)
        print(f"Analysis done for {video_class.name} - {video_number}")
        return video_sequence_track_data


class TracksAnalyzerClassical(TracksAnalyzer):
    def __init__(self, config, use_patch_filtered=True):
        super(TracksAnalyzerClassical, self).__init__(config=config)
        self.use_patch_filtered = use_patch_filtered
        if use_patch_filtered:
            self.extracted_folder = 'filtered_generated_annotations'
        else:
            self.extracted_folder = 'generated_annotations'

    @staticmethod
    def get_frame_annotations(df: pd.DataFrame, frame_number: int):
        idx: pd.DataFrame = df.loc[df["frame_number"] == frame_number]
        return idx.to_numpy()

    def get_extracted_centers(self, extracted_df, frame):
        extracted_centers = self.get_frame_annotations(df=extracted_df, frame_number=frame)
        return extracted_centers[:, 7:9], extracted_centers[:, 0]

    def save_as_csv(self, metrics):
        video_class, video_number, ade, fde, precision, recall, radius = [], [], [], [], [], [], []
        for k, v in metrics.items():
            for vk, vv in v.items():
                video_class.append(k)
                video_number.append(vk)
                ade.append(vv['ade'])
                fde.append(vv['fde'])
                precision.append(vv['precision'])
                recall.append(vv['recall'])
                radius.append(vv['neighbourhood_radius'])
        df: pd.DataFrame = pd.DataFrame({
            'class': video_class,
            'number': video_number,
            'ade': ade,
            'fde': fde,
            'precision': precision,
            'recall': recall,
            'neighbourhood_radius': radius
        })
        df.to_csv(f"{self.root}/{self.extracted_folder}/metrics_{self.config.threshold}m.csv", index=False)

    def perform_analysis_on_multiple_sequences(self, show_extracted_tracks_only=False):
        metrics = {}
        for v_idx, (v_clz, v_meta_clz) in enumerate(zip(self.video_classes, self.video_meta_classes)):
            for v_num in self.video_numbers[v_idx]:
                gt_annotation_path = f"{self.root}annotations/{v_clz.value}/video{v_num}/annotation_augmented.csv"
                extracted_annotation_path = f"{self.root}{self.extracted_folder}/{v_clz.value}/" \
                                            f"video{v_num}/generated_annotations.csv"
                ref_img = torchvision.io.read_image(f"{self.root}annotations/{v_clz.value}/video{v_num}/reference.jpg")
                video_path = f"{self.root}/videos/{v_clz.value}/video{v_num}/video.mov"
                if show_extracted_tracks_only:
                    self.construct_extracted_tracks_only(
                        extracted_annotation_path, v_clz, v_num, video_path, (ref_img.shape[1:]))
                else:
                    data, p, r = self.perform_analysis_on_single_sequence(
                        gt_annotation_path, extracted_annotation_path, v_clz, v_meta_clz, v_num, (ref_img.shape[1:]),
                        video_path)
                    ade, fde = self.calculate_ade_fde_for_associations(
                        video_sequence_track_data=data, meta_class=v_meta_clz, video_number=v_num)
                    if v_clz.name in metrics.keys():
                        metrics[v_clz.name][v_num] = {
                            'ade': ade, 'fde': fde,
                            'precision': p, 'recall': r,
                            'neighbourhood_radius': self.config.threshold
                        }
                    else:
                        metrics[v_clz.name] = {
                            v_num: {
                                'ade': ade, 'fde': fde,
                                'precision': p, 'recall': r,
                                'neighbourhood_radius': self.config.threshold
                            }
                        }
        self.save_as_csv(metrics=metrics)
        return metrics

    def perform_analysis_on_single_sequence(
            self, gt_annotation_path, extracted_annotation_path, video_class,
            video_meta_class, video_number, image_shape, video_path):

        video_sequence_track_data = VideoSequenceTracksData(
            video_class=video_class, video_number=video_number, tracks=[])
        ratio = self.get_ratio(meta_class=video_meta_class, video_number=video_number)
        video_frames = []
        tp_list, fp_list, fn_list = [], [], []

        gt_df = self.get_gt_df(gt_annotation_path)
        extracted_df = pd.read_csv(extracted_annotation_path)

        for frame in tqdm(gt_df.frame.unique()):
            gt_bbox_centers, supervised_boxes, gt_track_ids = self.get_gt_annotation(
                frame_number=frame, gt_annotation_df=gt_df, original_shape=tuple(image_shape))
            extracted_centers, extracted_track_ids = self.get_extracted_centers(extracted_df, frame)
            (match_rows, match_cols), (fn, fp, precision, recall, tp) = self.get_associations_and_metrics(
                gt_centers=gt_bbox_centers, extracted_centers=extracted_centers,
                max_distance=float(self.config.match_distance),
                ratio=ratio,
                threshold=self.config.threshold
            )
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            if frame == 0:
                for r, c in zip(match_rows, match_cols):
                    gt_coordinates = gt_bbox_centers[r]
                    extracted_coordinates = extracted_centers[c]

                    gt_track_id = gt_track_ids[r]
                    extracted_track_id = extracted_track_ids[c]

                    track = self.construct_new_track(extracted_coordinates, extracted_track_id, frame, gt_coordinates,
                                                     gt_track_id)

                    video_sequence_track_data.tracks.append(track)
            else:
                for r, c in zip(match_rows, match_cols):
                    gt_coordinates = gt_bbox_centers[r]
                    extracted_coordinates = extracted_centers[c]

                    gt_track_id = gt_track_ids[r]
                    extracted_track_id = extracted_track_ids[c]
                    # gt_tracks_for_agents_in_frame = self.get_frame_by_track_annotations(
                    #     gt_df, frame_number=frame, track_id=gt_track_id, for_gt=True)
                    # extracted_tracks_for_agents_in_frame = self.get_frame_by_track_annotations(
                    #     extracted_df, frame_number=frame, track_id=extracted_track_id, for_gt=False)

                    if gt_track_id in video_sequence_track_data:
                        existing_track = video_sequence_track_data[gt_track_id]
                        self.update_existing_track(existing_track, extracted_coordinates, extracted_track_id, frame,
                                                   gt_coordinates, gt_track_id)
                    else:
                        track = self.construct_new_track(extracted_coordinates, extracted_track_id, frame,
                                                         gt_coordinates,
                                                         gt_track_id)

                        video_sequence_track_data.tracks.append(track)

            if self.config.show_plot or self.config.make_video:
                fig = self.plot(
                    frame=extract_frame_from_video(video_path, frame_number=frame),
                    boxes=supervised_boxes,
                    gt_features=video_sequence_track_data.get_alive_gt_features()
                    if self.config.plot_gt_features else [],
                    extracted_features=video_sequence_track_data.get_alive_extracted_features(),
                    frame_number=frame,
                    marker_size=self.config.marker_size,
                    radius=self.config.threshold,
                    fig_title=f"Precision: {precision} | Recall: {recall}",
                    footnote_text=f"{video_class.name} - {video_number}\n"
                                  f"Neighbourhood Radius: {self.config.threshold}m",
                    video_mode=self.config.make_video,
                    box_annotation=gt_track_ids,
                    boxes_with_annotation=True
                )
            if self.config.make_video:
                video_frames.append(self.get_frame_from_figure(fig, original_shape=image_shape))

        overall_precision = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fp_list).sum())
        overall_recall = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fn_list).sum())

        if self.config.make_video:
            print(f"Writing Video")
            Path(os.path.join(
                os.getcwd(), f'logs/analysis_videos/{self.extracted_folder}/')).mkdir(parents=True, exist_ok=True)
            torchvision.io.write_video(
                f'logs/analysis_videos/{self.extracted_folder}/{video_class.name}_'
                f'{video_number}_'
                f'neighbourhood_radius_{self.config.threshold}.avi',
                torch.cat(video_frames).permute(0, 2, 3, 1),
                self.config.video_fps)
        print(f"Analysis done for {video_class.name} - {video_number}")
        return video_sequence_track_data, overall_precision, overall_recall

    def construct_extracted_tracks_only(
            self, extracted_annotation_path, video_class, video_number, video_path, image_shape):
        video_sequence_track_data = ExtractedTracksData(
            video_class=video_class, video_number=video_number, tracks=[])
        video_frames = []

        extracted_df = pd.read_csv(extracted_annotation_path)

        for frame in tqdm(extracted_df.frame_number.unique()):
            extracted_centers, extracted_track_ids = self.get_extracted_centers(extracted_df, frame)
            if frame == 0:
                for e_t_idx, e_center in zip(extracted_track_ids, extracted_centers):
                    track = ExtractedTrack(idx=int(e_t_idx))
                    track.frames.append(frame)
                    track.coordinates.append(e_center.tolist())
                    video_sequence_track_data.tracks.append(track)
            else:
                for e_t_idx, e_center in zip(extracted_track_ids, extracted_centers):
                    if e_t_idx in video_sequence_track_data:
                        track = video_sequence_track_data[e_t_idx]
                        track.frames.append(frame)
                        track.coordinates.append(e_center.tolist())
                    else:
                        track = ExtractedTrack(idx=int(e_t_idx))
                        track.frames.append(frame)
                        track.coordinates.append(e_center.tolist())
                        video_sequence_track_data.tracks.append(track)

            # filter out inactive tracks
            for t in video_sequence_track_data.tracks:
                if t.frames[-1] != frame:
                    t.active = False

            if self.config.show_plot or self.config.make_video:
                fig = self.plot(
                    frame=extract_frame_from_video(video_path, frame_number=frame),
                    boxes=[],
                    gt_features=[],
                    extracted_features=video_sequence_track_data.get_alive_features(),
                    frame_number=frame,
                    marker_size=self.config.marker_size,
                    radius=self.config.threshold,
                    fig_title="",
                    footnote_text=f"{video_class.name} - {video_number}",
                    video_mode=self.config.make_video,
                    box_annotation=extracted_track_ids,
                    boxes_with_annotation=True
                )
            if self.config.make_video:
                video_frames.append(self.get_frame_from_figure(fig, original_shape=tuple(image_shape)))
        if self.config.make_video:
            print(f"Writing Video")
            Path(os.path.join(
                os.getcwd(), f'logs/analysis_videos/{self.extracted_folder}/')).mkdir(parents=True, exist_ok=True)
            torchvision.io.write_video(
                f'logs/analysis_videos/{self.extracted_folder}/extracted_only_{video_class.name}_'
                f'{video_number}_'
                f'neighbourhood_radius_{self.config.threshold}.avi',
                torch.cat(video_frames).permute(0, 2, 3, 1),
                self.config.video_fps)
        print(f"Analysis done for {video_class.name} - {video_number}")
        return video_sequence_track_data
    
    
def internal_csv_mover():
    video_classes_to_use = [
        SDDVideoClasses.GATES,
        SDDVideoClasses.HYANG,
        SDDVideoClasses.LITTLE,
        SDDVideoClasses.NEXUS,
        SDDVideoClasses.QUAD,
        SDDVideoClasses.BOOKSTORE,
        SDDVideoClasses.COUPA]
    video_numbers_to_use = [
        [i for i in range(9)],
        [i for i in range(15)],
        [i for i in range(4)],
        [i for i in range(12) if i not in [3, 4, 5]],
        [i for i in range(4)],
        [i for i in range(7)],
        [i for i in range(4)], ]

    from_path_root = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/src/position_maps/logs/Trajectories/'
    to_path_root = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/pm_extracted_annotations/'
    for v_idx, v_clz in enumerate(video_classes_to_use):
        for v_num in video_numbers_to_use[v_idx]:
            from_path = f"{from_path_root}{v_clz.name}/{v_num}/trajectories.csv"
            to_path = f"{to_path_root}{v_clz.value}/video{v_num}/trajectories.csv"
            shutil.copyfile(from_path, to_path)


if __name__ == '__main__':
    # internal_csv_mover()
    analyzer = TracksAnalyzer(OmegaConf.load('config/training/training.yaml'))
    # analyzer = TracksAnalyzerClassical(OmegaConf.load('config/training/training.yaml'), use_patch_filtered=True)
    out = analyzer.perform_analysis_on_multiple_sequences(show_extracted_tracks_only=False)
    print()
