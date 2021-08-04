import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt, patches
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

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

            detections_list.detections.append(Detections(
                frame_number=frame.item(),
                gt_detections=gt_detection_list,
                classic_detections=classic_detection_list,
                pos_map_detections=pos_map_detection_list,
                common_detections=common_detection_list,
                candidate_detections=candidate_detection_list
            ))

            if self.config.show_plot or self.config.make_video:
                fig = self.plot_detections(
                    frame=extract_frame_from_video(video_path, frame_number=frame),
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


if __name__ == '__main__':
    cfg = OmegaConf.load('config/training/training.yaml')
    classifier_network = CropClassifier(config=cfg, train_dataset=None, val_dataset=None, desired_output_shape=None,
                                        loss_function=None)
    analyzer = PosMapToConventional(cfg, use_patch_filtered=True, classifier=classifier_network)
    out = analyzer.perform_analysis_on_multiple_sequences(show_extracted_tracks_only=False)
    print()
