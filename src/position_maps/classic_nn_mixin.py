import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as tvf
from matplotlib import pyplot as plt, patches
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.nn.functional import pad
from tqdm import tqdm

from average_image.constants import SDDVideoDatasets
from baselinev2.improve_metrics.crop_utils import show_image_with_crop_boxes
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.plot_utils import add_box_to_axes_with_annotation, add_box_to_axes
from log import get_logger
from src.position_maps.analysis import TracksAnalyzer
from src_lib.models_hub.crop_classifiers import CropClassifier

seed_everything(42)
logger = get_logger(__name__)


class Detections(object):
    def __init__(self, frame_number, gt_detections, classic_detections, pos_map_detections,
                 common_detections, candidate_detections, classified_detections=None):
        super(Detections, self).__init__()
        self.frame_number = frame_number
        self.gt_detections = gt_detections
        self.classic_detections = classic_detections
        self.pos_map_detections = pos_map_detections
        self.common_detections = common_detections
        self.candidate_detections = candidate_detections
        self.classified_detections = classified_detections

    def __repr__(self):
        return f"Frame: {self.frame_number}"


class CommonDetection(object):
    def __init__(self, classic_center, classic_box, classic_track_id, pos_map_center, pos_map_track_id):
        super(CommonDetection, self).__init__()
        self.classic_center = classic_center
        self.classic_box = classic_box
        self.classic_track_id = classic_track_id
        self.pos_map_center = pos_map_center
        self.pos_map_track_id = pos_map_track_id


class GenericDetection(object):
    def __init__(self, center, box, track_id):
        super(GenericDetection, self).__init__()
        self.center = center
        self.box = box
        self.track_id = track_id


class VideoDetections(object):
    def __init__(self, video_class, video_number, detections: List[Detections]):
        super(VideoDetections, self).__init__()
        self.video_class = video_class
        self.video_number = video_number
        self.detections = detections

    def __getitem__(self, item):
        return [d for d in self.detections if d.frame_number == item][0]

    def __contains__(self, item):
        for d in self.detections:
            if d.frame_number == item:
                return True
        return False

    def __repr__(self):
        return f"{self.video_class.name} | {self.video_number}\n{self.detections}"


class PosMapToConventional(TracksAnalyzer):
    def __init__(self, config, use_patch_filtered, classifier=None):
        super(PosMapToConventional, self).__init__(config=config)
        self.classifier = classifier
        classifier.to(config.device)
        self.use_patch_filtered = use_patch_filtered
        if use_patch_filtered:
            self.extracted_folder = 'filtered_generated_annotations'
        else:
            self.extracted_folder = 'generated_annotations'

    @staticmethod
    def get_classical_frame_annotations(df: pd.DataFrame, frame_number: int):
        idx: pd.DataFrame = df.loc[df["frame_number"] == frame_number]
        return idx.to_numpy()

    def get_classic_extracted_centers(self, extracted_df, frame):
        extracted_centers = self.get_classical_frame_annotations(df=extracted_df, frame_number=frame)
        return extracted_centers[:, 7:9], extracted_centers[:, 0], extracted_centers[:, 1:5]

    def plot_detections(self, frame, boxes, gt_features, extracted_features, common_features,
                        frame_number, box_annotation, marker_size=1, radius=None,
                        fig_title='', footnote_text='', video_mode=False,
                        boxes_with_annotation=True):
        fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(8, 10))

        axs.imshow(frame)

        legends_dict = {}
        if gt_features is not None:
            self.add_features_to_axis(axs, gt_features, marker_size=marker_size, marker_color='b')
            legends_dict.update({'b': 'Classic Locations'})

        if extracted_features is not None:
            self.add_features_to_axis(axs, extracted_features, marker_size=marker_size, marker_color='g')
            legends_dict.update({'g': 'Candidate Locations'})

        if common_features is not None:
            self.add_features_to_axis(axs, common_features, marker_size=marker_size, marker_color='pink')
            legends_dict.update({'pink': 'Common Locations'})

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

    def save_as_csv(self, metrics):
        video_class, video_number, classic_precision, classic_recall, \
        classic_nn_precision, classic_nn_recall, radius = [], [], [], [], [], [], []
        for k, v in metrics.items():
            for vk, vv in v.items():
                video_class.append(k)
                video_number.append(vk)
                classic_precision.append(vv['classic_precision'])
                classic_recall.append(vv['classic_recall'])
                classic_nn_precision.append(vv['classic_nn_precision'])
                classic_nn_recall.append(vv['classic_nn_recall'])
                radius.append(vv['neighbourhood_radius'])
        df: pd.DataFrame = pd.DataFrame({
            'class': video_class,
            'number': video_number,
            'classic_precision': classic_precision,
            'classic_recall': classic_recall,
            'classic_nn_precision': classic_nn_precision,
            'classic_nn_recall': classic_nn_recall,
            'neighbourhood_radius': radius
        })
        df.to_csv(f"{self.root}/classic_nn_extracted_annotations/metrics_{self.config.threshold}m.csv", index=False)

    def perform_analysis_on_multiple_sequences(self, show_extracted_tracks_only=False):
        metrics = {}
        for v_idx, (v_clz, v_meta_clz) in enumerate(zip(self.video_classes, self.video_meta_classes)):
            for v_num in self.video_numbers[v_idx]:
                gt_annotation_path = f"{self.root}annotations/{v_clz.value}/video{v_num}/annotation_augmented.csv"
                classic_extracted_annotation_path = f"{self.root}{self.extracted_folder}/{v_clz.value}/" \
                                                    f"video{v_num}/generated_annotations.csv"
                pos_map_extracted_annotation_path = f"{self.root}pm_extracted_annotations/{v_clz.value}/" \
                                                    f"video{v_num}/trajectories.csv"

                ref_img = torchvision.io.read_image(f"{self.root}annotations/{v_clz.value}/video{v_num}/reference.jpg")
                video_path = f"{self.root}/videos/{v_clz.value}/video{v_num}/video.mov"

                if show_extracted_tracks_only:
                    if self.config.show_extracted_tracks_for == 'gt':
                        self.construct_extracted_tracks_only(
                            gt_annotation_path, v_clz, v_num, video_path, (ref_img.shape[1:]))
                    elif self.config.show_extracted_tracks_for == 'classic':
                        self.construct_extracted_tracks_only(
                            classic_extracted_annotation_path, v_clz, v_num, video_path, (ref_img.shape[1:]))
                    elif self.config.show_extracted_tracks_for == 'pos_map':
                        self.construct_extracted_tracks_only(
                            pos_map_extracted_annotation_path, v_clz, v_num, video_path, (ref_img.shape[1:]))
                    else:
                        logger.error('Not Supported')
                else:
                    data, p, r = self.perform_detection_collection_on_single_sequence(
                        gt_annotation_path, classic_extracted_annotation_path, pos_map_extracted_annotation_path,
                        v_clz, v_meta_clz, v_num,
                        (ref_img.shape[1:]),
                        video_path)
                    df, \
                    (overall_gt_classic_precision, overall_gt_classic_recall), \
                    (overall_gt_classic_nn_precision, overall_gt_classic_nn_recall) = \
                        self.generate_annotation_from_data(data)
                    if v_clz.name in metrics.keys():
                        metrics[v_clz.name][v_num] = {
                            'classic_precision': overall_gt_classic_precision,
                            'classic_recall': overall_gt_classic_recall,
                            'classic_nn_precision': overall_gt_classic_nn_precision,
                            'classic_nn_recall': overall_gt_classic_nn_recall,
                            'neighbourhood_radius': self.config.threshold
                        }
                    else:
                        metrics[v_clz.name] = {
                            v_num: {
                                'classic_precision': overall_gt_classic_precision,
                                'classic_recall': overall_gt_classic_recall,
                                'classic_nn_precision': overall_gt_classic_nn_precision,
                                'classic_nn_recall': overall_gt_classic_nn_recall,
                                'neighbourhood_radius': self.config.threshold
                            }
                        }
        self.save_as_csv(metrics=metrics)
        return metrics

    def perform_detection_collection_on_single_sequence(
            self, gt_annotation_path, classic_extracted_annotation_path,
            pos_map_extracted_annotation_path, video_class,
            video_meta_class, video_number, image_shape, video_path):

        detections_list = VideoDetections(video_class, video_number, [])
        ratio = self.get_ratio(meta_class=video_meta_class, video_number=video_number)
        video_frames = []
        tp_list, fp_list, fn_list = [], [], []

        gt_df = self.get_gt_df(gt_annotation_path)
        classic_extracted_df = pd.read_csv(classic_extracted_annotation_path)
        pos_map_extracted_df = pd.read_csv(pos_map_extracted_annotation_path)

        for frame in tqdm(gt_df.frame.unique()):
            gt_bbox_centers, supervised_boxes, gt_track_ids = self.get_gt_annotation(
                frame_number=frame, gt_annotation_df=gt_df, original_shape=tuple(image_shape))
            pos_map_extracted_centers, pos_map_extracted_track_ids = self.get_extracted_centers(
                pos_map_extracted_df, frame)
            classic_extracted_centers, classic_extracted_track_ids, classic_extracted_boxes = \
                self.get_classic_extracted_centers(
                    classic_extracted_df, frame)
            (match_rows, match_cols), (fn, fp, precision, recall, tp) = self.get_associations_and_metrics(
                gt_centers=classic_extracted_centers, extracted_centers=pos_map_extracted_centers,
                max_distance=float(self.config.match_distance),
                ratio=ratio,
                threshold=self.config.threshold
            )
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

            gt_detection_list = [
                GenericDetection(center=c, box=b, track_id=t) for c, b, t in
                zip(gt_bbox_centers, supervised_boxes, gt_track_ids)
            ]
            classic_detection_list = [
                GenericDetection(center=c, box=b, track_id=t) for c, b, t in
                zip(classic_extracted_centers, classic_extracted_boxes, classic_extracted_track_ids)
            ]
            pos_map_detection_list = [
                GenericDetection(center=c, box=None, track_id=t) for c, t in
                zip(pos_map_extracted_centers, pos_map_extracted_track_ids)
            ]
            common_detection_list = []
            for r, c in zip(match_rows, match_cols):
                common_detection_list.append(CommonDetection(
                    classic_center=classic_extracted_centers[r],
                    classic_box=classic_extracted_boxes[r],
                    classic_track_id=classic_extracted_track_ids[r],
                    pos_map_center=pos_map_extracted_centers[c],
                    pos_map_track_id=pos_map_extracted_track_ids[c]
                ))

            candidate_pos_map_idx = np.setdiff1d(
                np.arange(start=0, stop=pos_map_extracted_centers.shape[0]), match_cols)

            candidate_detection_list = [
                GenericDetection(center=pos_map_extracted_centers[i],
                                 box=None,
                                 track_id=pos_map_extracted_track_ids[i])
                for i in candidate_pos_map_idx
            ]

            rgb_frame = extract_frame_from_video(video_path, frame_number=frame)

            if self.config.use_classifier:
                candidate_boxes, candidate_track_ids, candidate_centers = [], [], []
                for candidate in candidate_detection_list:
                    candidate_boxes.append([*np.round(candidate.center).astype(np.int), *self.config.crop_size])
                    candidate_track_ids.append(candidate.track_id)
                    candidate_centers.append(candidate.center)

                if len(candidate_pos_map_idx) != 0:
                    selected_boxes, selected_centers, selected_track_idx = self.get_classified_candidates(
                        candidate_boxes, candidate_centers, candidate_track_ids, rgb_frame)

                    classified_detection_list = [
                        GenericDetection(center=c, box=b, track_id=t)
                        for c, b, t in zip(selected_centers, selected_boxes, selected_track_idx)
                    ]
                else:
                    classified_detection_list = []

            detections_list.detections.append(Detections(
                frame_number=frame.item(),
                gt_detections=gt_detection_list,
                classic_detections=classic_detection_list,
                pos_map_detections=pos_map_detection_list,
                common_detections=common_detection_list,
                candidate_detections=candidate_detection_list,
                classified_detections=classified_detection_list if self.config.use_classifier else None
            ))

            if self.config.show_plot or self.config.make_video:
                fig = self.plot_detections(
                    frame=rgb_frame,
                    boxes=classic_extracted_boxes,
                    gt_features=classic_extracted_centers[:, None, :],
                    extracted_features=pos_map_extracted_centers[:, None, :],
                    common_features=np.stack([c.pos_map_center for c in common_detection_list])[:, None, :],
                    frame_number=frame,
                    marker_size=self.config.marker_size + 5,
                    radius=self.config.threshold,
                    fig_title=f"Precision: {precision} | Recall: {recall}",
                    footnote_text=f"{video_class.name} - {video_number}\n"
                                  f"Neighbourhood Radius: {self.config.threshold}m",
                    video_mode=self.config.make_video,
                    box_annotation=[],  # classic_extracted_track_ids,
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
        return detections_list, overall_precision, overall_recall

    @staticmethod
    def get_processed_patches_to_train_rgb_only(crops, crop_h, crop_w):
        crops_filtered, filtered_idx = [], []
        for f_idx, c in enumerate(crops):
            if c.numel() != 0:
                if c.shape[-1] != crop_w or c.shape[-2] != crop_h:
                    diff_h = crop_h - c.shape[-2]
                    diff_w = crop_w - c.shape[-1]

                    c = pad(c.unsqueeze(0), [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
                            mode='replicate').squeeze(0)
                crops_filtered.append(c)
                filtered_idx.append(f_idx)
        crops_filtered = torch.stack(crops_filtered) if len(crops_filtered) != 0 else []
        return crops_filtered, filtered_idx

    def get_classified_candidates(self, candidate_boxes, candidate_centers, candidate_track_ids, rgb_frame):
        candidate_track_ids = torch.tensor(candidate_track_ids)
        candidate_boxes = torch.tensor(candidate_boxes)
        candidate_centers = torch.tensor(candidate_centers)

        boxes_xywh = torchvision.ops.box_convert(candidate_boxes, 'cxcywh', 'xywh')
        boxes_xywh = [torch.tensor((b[1], b[0], b[2], b[3])) for b in boxes_xywh]
        boxes_xywh = torch.stack(boxes_xywh)

        crops = [tvf.crop(torch.from_numpy(rgb_frame).permute(2, 0, 1),
                          top=b[0], left=b[1], width=b[2], height=b[3])
                 for b in boxes_xywh.to(dtype=torch.int)]

        # feasible boxes
        valid_boxes = [c_i for c_i, c in enumerate(crops) if c.shape[1] != 0 and c.shape[2] != 0]
        boxes_xywh = boxes_xywh[valid_boxes]
        track_idx = candidate_track_ids[valid_boxes]
        candidate_boxes = candidate_boxes[valid_boxes]
        candidate_centers = candidate_centers[valid_boxes]

        crops = [(c.float() / 255.0).to(self.config.device) for i, c in enumerate(crops) if i in valid_boxes]
        crops, filtered_idx = self.get_processed_patches_to_train_rgb_only(
            crops, crop_h=self.config.crop_size[0], crop_w=self.config.crop_size[1])
        filtered_idx = torch.tensor(filtered_idx)
        boxes_xywh = boxes_xywh[filtered_idx]
        track_idx = candidate_track_ids[filtered_idx]
        candidate_boxes = candidate_boxes[filtered_idx]
        candidate_centers = candidate_centers[filtered_idx]

        # crops = torch.stack(crops)
        # crops = (crops.float() / 255.0).to(self.config.device)

        if self.config.debug.enabled:
            show_image_with_crop_boxes(rgb_frame,
                                       [], boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
                                       title='xywh')
            gt_crops_grid = torchvision.utils.make_grid(crops)
            plt.imshow(gt_crops_grid.cpu().permute(1, 2, 0))
            plt.show()

        with torch.no_grad():
            patch_predictions = self.classifier(crops)

        pred_labels = torch.round(torch.sigmoid(patch_predictions))
        selected_boxes_idx = torch.where(pred_labels.squeeze(-1))

        selected_track_idx = track_idx[selected_boxes_idx]
        selected_boxes = boxes_xywh[selected_boxes_idx]
        selected_centers = candidate_centers[selected_boxes_idx]

        return selected_boxes, selected_centers, selected_track_idx

    def generate_annotation_from_data(self, video_detection_data: VideoDetections):
        save_path = os.path.join(
            self.config.root,
            f"classic_nn_extracted_annotations/{video_detection_data.video_class.value}"
            f"/video{video_detection_data.video_number}/")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        ratio = self.get_ratio(
            meta_class=getattr(SDDVideoDatasets, video_detection_data.video_class.name),
            video_number=video_detection_data.video_number)

        frame, track_id, x, y = [], [], [], []

        gt_classic_tp_list, gt_classic_fp_list, gt_classic_fn_list = [], [], []
        gt_classic_nn_tp_list, gt_classic_nn_fp_list, gt_classic_nn_fn_list = [], [], []

        for detection_data in tqdm(video_detection_data.detections):
            for common_detection in detection_data.common_detections:
                frame.append(detection_data.frame_number)
                # giving classic data as we want to extend them
                track_id.append(common_detection.classic_track_id)
                x.append(common_detection.classic_center[0])
                y.append(common_detection.classic_center[1])
            for classified_detection in detection_data.classified_detections:
                frame.append(detection_data.frame_number)
                # giving -1 as track id as they are not associated to any track
                track_id.append(-1)
                x.append(classified_detection.center[0].item())
                y.append(classified_detection.center[1].item())

            (match_rows, match_cols), (fn, fp, precision_c, recall_c, tp) = self.get_associations_and_metrics(
                gt_centers=np.stack([b.center for b in detection_data.gt_detections]),
                extracted_centers=np.stack([b.center for b in detection_data.classic_detections])
                if len(detection_data.classic_detections) != 0 else [],
                max_distance=float(self.config.match_distance),
                ratio=ratio,
                threshold=self.config.threshold
            )
            gt_classic_tp_list.append(tp)
            gt_classic_fp_list.append(fp)
            gt_classic_fn_list.append(fn)

            (match_rows, match_cols), (fn, fp, precision_c_nn, recall_c_nn, tp) = self.get_associations_and_metrics(
                gt_centers=np.stack([b.center for b in detection_data.gt_detections]),
                extracted_centers=
                np.concatenate(
                    (np.stack([b.center for b in detection_data.classic_detections]),
                     np.stack([b.center for b in detection_data.classified_detections])),
                    axis=0) if ((len(detection_data.classic_detections) != 0)
                                and (len(detection_data.classified_detections) != 0)) else [],
                max_distance=float(self.config.match_distance),
                ratio=ratio,
                threshold=self.config.threshold
            )
            gt_classic_nn_tp_list.append(tp)
            gt_classic_nn_fp_list.append(fp)
            gt_classic_nn_fn_list.append(fn)

        annotation_df = pd.DataFrame.from_dict(
            {
                'frame': frame,
                'track_id': track_id,
                'x': x,
                'y': y,
            }
        )
        overall_gt_classic_precision = np.array(gt_classic_tp_list).sum() / \
                                       (np.array(gt_classic_tp_list).sum() + np.array(gt_classic_fp_list).sum())
        overall_gt_classic_recall = np.array(gt_classic_tp_list).sum() / \
                                    (np.array(gt_classic_tp_list).sum() + np.array(gt_classic_fn_list).sum())

        overall_gt_classic_nn_precision = np.array(gt_classic_nn_tp_list).sum() / \
                                          (np.array(gt_classic_nn_tp_list).sum() + np.array(
                                              gt_classic_nn_fp_list).sum())
        overall_gt_classic_nn_recall = np.array(gt_classic_nn_tp_list).sum() / \
                                       (np.array(gt_classic_nn_tp_list).sum() + np.array(gt_classic_nn_fn_list).sum())

        annotation_df.to_csv(os.path.join(save_path, 'annotation.csv'), index=False)
        return annotation_df, (overall_gt_classic_precision, overall_gt_classic_recall), \
               (overall_gt_classic_nn_precision, overall_gt_classic_nn_recall)


if __name__ == '__main__':
    cfg = OmegaConf.load('config/training/training.yaml')
    classifier_network = CropClassifier(config=cfg, train_dataset=None, val_dataset=None, desired_output_shape=None,
                                        loss_function=None)
    analyzer = PosMapToConventional(cfg, use_patch_filtered=True, classifier=classifier_network)
    # out = analyzer.perform_analysis_on_multiple_sequences(show_extracted_tracks_only=False)
    # out = analyzer.generate_annotation_from_data(torch.load(
    #     '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/src/position_maps/logs/dummy_classic_nn.pt'))
    print()
