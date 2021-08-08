import os
from pathlib import Path
from typing import List

import cv2
import motmetrics as mm
import networkx as nx
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage
import torch
import torchvision
import torchvision.transforms.functional as tvf
from matplotlib import pyplot as plt, patches
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from scipy.spatial import cKDTree
from torch.nn.functional import pad
from tqdm import tqdm

from average_image.constants import SDDVideoDatasets, SDDVideoClasses
from baselinev2.improve_metrics.crop_utils import show_image_with_crop_boxes
from baselinev2.improve_metrics.model import make_conv_blocks, Activations, PersonClassifier, people_collate_fn, \
    make_classifier_block
from baselinev2.improve_metrics.modules import resnet18, resnet9
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.plot_utils import add_box_to_axes_with_annotation, add_box_to_axes
from baselinev2.stochastic.model import BaselineGAN
from log import get_logger
from src.position_maps.analysis import TracksAnalyzer
from src.position_maps.trajectory_utils import plot_trajectory_with_one_frame
from src_lib.models_hub.crop_classifiers import CropClassifier

seed_everything(42)
logger = get_logger(__name__)

# MODEL_ROOT = '/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/'
MODEL_ROOT = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/'

MODEL_PATH = f'{MODEL_ROOT}baselinev2/improve_metrics/logs/lightning_logs/version_'

DATASET_TO_MODEL = {
    SDDVideoClasses.BOOKSTORE: f"{MODEL_PATH}376647",
    SDDVideoClasses.COUPA: f"{MODEL_PATH}377095",
    SDDVideoClasses.GATES: f"{MODEL_PATH}373993",
    SDDVideoClasses.HYANG: f"{MODEL_PATH}373994",
    SDDVideoClasses.LITTLE: f"{MODEL_PATH}376650",
    SDDVideoClasses.NEXUS: f"{MODEL_PATH}377688",
    SDDVideoClasses.QUAD: f"{MODEL_PATH}377576",
    SDDVideoClasses.DEATH_CIRCLE: f"{MODEL_PATH}11",
}


class AgentTrack(object):
    def __init__(self, idx):
        super(AgentTrack, self).__init__()
        self.idx = idx
        self.frames = []
        self.locations = []
        self.inactive = 0  # 0 is active, -1 dead for sure
        self.extended_at_frames = []

    def __repr__(self):
        return f"Track: {self.idx} | Frames: {self.frames}"


class CandidateAgentTrack(object):
    def __init__(self, frame, location):
        super(CandidateAgentTrack, self).__init__()
        self.frame = frame
        self.location = location

    def __repr__(self):
        return f"Location: {self.location}"


class VideoSequenceAgentTracks(object):
    def __init__(self, video_class, video_number, tracks: List[AgentTrack]):
        super(VideoSequenceAgentTracks, self).__init__()
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


def setup_person_classifier(cfg, video_class):
    logger.info(f'Setting up PersonalClassifier model...')

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
    checkpoint_path = f'{DATASET_TO_MODEL[video_class]}/checkpoints/'
    checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]
    load_dict = torch.load(checkpoint_file)

    model.load_state_dict(load_dict['state_dict'])
    model.to(cfg.eval.device)
    model.eval()

    return model


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
            'classic_nn_precision': classic_nn_precision,
            'classic_recall': classic_recall,
            'classic_nn_recall': classic_nn_recall,
            'neighbourhood_radius': radius
        })
        df.to_csv(f"{self.root}/classic_nn_extracted_annotations/metrics_{self.config.threshold}m.csv", index=False)

    def perform_analysis_on_multiple_sequences(self, show_extracted_tracks_only=False, pm_extracted_version='v0',
                                               pm_extracted_filename='generated_annotations.csv'):
        # pm_extracted_filename='trajectories.csv'
        metrics = {}
        for v_idx, (v_clz, v_meta_clz) in enumerate(zip(self.video_classes, self.video_meta_classes)):
            for v_num in self.video_numbers[v_idx]:
                if self.config.use_classifier and self.config.use_old_model:
                    self.classifier = setup_person_classifier(
                        OmegaConf.merge(
                            OmegaConf.load(f'{MODEL_ROOT}baselinev2/improve_metrics/config/eval/eval.yaml'),
                            OmegaConf.load(f'{MODEL_ROOT}baselinev2/improve_metrics/config/model/model.yaml'),
                            OmegaConf.load(f'{MODEL_ROOT}baselinev2/improve_metrics/config/training/training.yaml'),
                        ),
                        video_class=v_clz
                    )
                    self.classifier.to(self.config.device)
                gt_annotation_path = f"{self.root}annotations/{v_clz.value}/video{v_num}/annotation_augmented.csv"
                classic_extracted_annotation_path = f"{self.root}{self.extracted_folder}/{v_clz.value}/" \
                                                    f"video{v_num}/generated_annotations.csv"
                pos_map_extracted_annotation_path = f"{self.root}pm_extracted_annotations/{v_clz.value}/" \
                                                    f"video{v_num}/{pm_extracted_version}/{pm_extracted_filename}"

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

    @staticmethod
    def plot_predictions(crops, pred_labels, n=4, batch_idx=-1, save_path=None, additional_text=''):
        k = 0
        fig, ax = plt.subplots(n, n, figsize=(16, 14))
        for i in range(n):
            for j in range(n):
                ax[i, j].axis('off')
                ax[i, j].set_title(f'{pred_labels[k].int().item()}')
                ax[i, j].imshow(crops[k].permute(1, 2, 0).cpu())

                k += 1
        plt.suptitle(f'Predictions\n GT | Prediction\n 1 -> Person/Object | 0 -> Background\n{additional_text}')

        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path + f'batch_idx_{batch_idx}.png')
            plt.close()
        else:
            plt.show()

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
        if filtered_idx.numel() != 0:
            boxes_xywh = boxes_xywh[filtered_idx]
            track_idx = candidate_track_ids[filtered_idx]
            candidate_boxes = candidate_boxes[filtered_idx]
            candidate_centers = candidate_centers[filtered_idx]

        # crops = torch.stack(crops)
        # crops = (crops.float() / 255.0).to(self.config.device)

        # for position map model
        if len(crops) == 0:
            return boxes_xywh, candidate_centers, track_idx

        if self.config.use_classifier and self.config.use_old_model:
            crops = torch.stack([tvf.resize(c, [50, 50]) for c in crops if c.shape[1] != 0 and c.shape[2] != 0])

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

        if self.config.debug.enabled:
            self.plot_predictions(crops, pred_labels.squeeze(), n=5)

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
                gt_centers=np.stack([b.center for b in detection_data.gt_detections])
                if len(detection_data.gt_detections) != 0 else np.zeros((0, 2)),
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
                gt_centers=np.stack([b.center for b in detection_data.gt_detections])
                if len(detection_data.gt_detections) != 0 else np.zeros((0, 2)),
                extracted_centers=
                np.concatenate(
                    (np.stack([b.center for b in detection_data.classic_detections])
                     if len(detection_data.gt_detections) != 0 else np.zeros((0, 2)),
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

    def get_metrics_for_multiple_sequences(self):
        metrics = {}
        for v_idx, (v_clz, v_meta_clz) in enumerate(zip(self.video_classes, self.video_meta_classes)):
            for v_num in self.video_numbers[v_idx]:
                ref_img = torchvision.io.read_image(f"{self.root}annotations/{v_clz.value}/video{v_num}/reference.jpg")
                video_path = f"{self.root}/videos/{v_clz.value}/video{v_num}/video.mov"
                (overall_gt_classic_precision, overall_gt_classic_recall), \
                (overall_gt_classic_nn_precision, overall_gt_classic_nn_recall) = \
                    self.get_metrics_from_data(v_clz=v_clz, v_num=v_num, image_shape=(ref_img.shape[1:]))
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

    def get_metrics_from_data(self, v_clz, v_num, image_shape):
        gt_annotation_path = f"{self.root}annotations/{v_clz.value}/video{v_num}/annotation_augmented.csv"
        nn_classic_extracted_annotation_path = f"{self.root}classic_nn_extracted_annotations/{v_clz.value}/" \
                                               f"video{v_num}/annotation.csv"
        classic_extracted_annotation_path = f"{self.root}{self.extracted_folder}/{v_clz.value}/" \
                                            f"video{v_num}/generated_annotations.csv"

        ratio = self.get_ratio(
            meta_class=getattr(SDDVideoDatasets, v_clz.name),
            video_number=v_num)

        gt_df = self.get_gt_df(gt_annotation_path)
        classic_extracted_df = pd.read_csv(classic_extracted_annotation_path)
        nn_classic_extracted_df = pd.read_csv(nn_classic_extracted_annotation_path)

        gt_classic_tp_list, gt_classic_fp_list, gt_classic_fn_list = [], [], []
        gt_classic_nn_tp_list, gt_classic_nn_fp_list, gt_classic_nn_fn_list = [], [], []

        for frame in tqdm(gt_df.frame.unique()):
            gt_bbox_centers, supervised_boxes, gt_track_ids = self.get_gt_annotation(
                frame_number=frame, gt_annotation_df=gt_df, original_shape=tuple(image_shape))
            classic_extracted_centers, classic_extracted_track_ids, classic_extracted_boxes = \
                self.get_classic_extracted_centers(
                    classic_extracted_df, frame)
            nn_classic_extracted_centers, nn_classic_extracted_track_ids = \
                self.get_extracted_centers(
                    nn_classic_extracted_df, frame)

            (match_rows, match_cols), (fn, fp, precision_c, recall_c, tp) = self.get_associations_and_metrics(
                gt_centers=np.stack(gt_bbox_centers)
                if len(gt_bbox_centers) != 0 else np.zeros((0, 2)),
                extracted_centers=np.stack(classic_extracted_centers)
                if len(classic_extracted_centers) != 0 else [],
                max_distance=float(self.config.match_distance),
                ratio=ratio,
                threshold=self.config.threshold
            )
            gt_classic_tp_list.append(tp)
            gt_classic_fp_list.append(fp)
            gt_classic_fn_list.append(fn)

            (match_rows, match_cols), (fn, fp, precision_c_nn, recall_c_nn, tp) = self.get_associations_and_metrics(
                gt_centers=np.stack(gt_bbox_centers)
                if len(gt_bbox_centers) != 0 else np.zeros((0, 2)),
                extracted_centers=np.stack(nn_classic_extracted_centers)
                if len(nn_classic_extracted_centers) != 0 else [],
                max_distance=float(self.config.match_distance),
                ratio=ratio,
                threshold=self.config.threshold
            )
            gt_classic_nn_tp_list.append(tp)
            gt_classic_nn_fp_list.append(fp)
            gt_classic_nn_fn_list.append(fn)

        overall_gt_classic_precision = np.array(gt_classic_tp_list).sum() / \
                                       (np.array(gt_classic_tp_list).sum() + np.array(gt_classic_fp_list).sum())
        overall_gt_classic_recall = np.array(gt_classic_tp_list).sum() / \
                                    (np.array(gt_classic_tp_list).sum() + np.array(gt_classic_fn_list).sum())

        overall_gt_classic_nn_precision = np.array(gt_classic_nn_tp_list).sum() / \
                                          (np.array(gt_classic_nn_tp_list).sum() + np.array(
                                              gt_classic_nn_fp_list).sum())
        overall_gt_classic_nn_recall = np.array(gt_classic_nn_tp_list).sum() / \
                                       (np.array(gt_classic_nn_tp_list).sum() + np.array(gt_classic_nn_fn_list).sum())

        return (overall_gt_classic_precision, overall_gt_classic_recall), \
               (overall_gt_classic_nn_precision, overall_gt_classic_nn_recall)

    @staticmethod
    def adjust_dataframe_framerate(df, frame_txt, frame_rate=30., time_step=0.4):
        df = df[df[frame_txt] % int(round(frame_rate * time_step)) == 0]
        df[frame_txt] /= int(round(frame_rate * time_step))
        return df

    @staticmethod
    def setup_trajectory_model():
        version = 18
        epoch = 109
        step = 929623

        base_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/baselinev2/stochastic/'
        model_path = f'{base_path}' + f'logs/lightning_logs/version_{version}/checkpoints/' \
                                      f'epoch={epoch}-step={step}.ckpt'
        hparam_path = f'{base_path}' + f'logs/lightning_logs/version_{version}/hparams.yaml'

        m = BaselineGAN.load_from_checkpoint(checkpoint_path=model_path, hparams_file=hparam_path,
                                             map_location='cuda:0')
        m.eval()
        return m

    def perform_collection_on_multiple_sequences(self, classic_nn_extracted_annotations_version='v0'):
        extended_tracks = {}
        for v_idx, (v_clz, v_meta_clz) in enumerate(zip(self.video_classes, self.video_meta_classes)):
            for v_num in self.video_numbers[v_idx]:
                # dummy
                model = self.setup_trajectory_model()
                model.to(self.config.device)
                # gt_annotation_path = f"{self.root}annotations/{v_clz.value}/video{v_num}/annotation_augmented.csv"
                classic_extracted_annotation_path = f"{self.root}{self.extracted_folder}/{v_clz.value}/" \
                                                    f"video{v_num}/generated_annotations.csv"
                nn_classic_extracted_annotation_path = \
                    f"{self.root}classic_nn_extracted_annotations_" \
                    f"{classic_nn_extracted_annotations_version}/{v_clz.value}/" \
                    f"video{v_num}/annotation.csv"

                ref_img = torchvision.io.read_image(f"{self.root}annotations/{v_clz.value}/video{v_num}/reference.jpg")
                video_path = f"{self.root}/videos/{v_clz.value}/video{v_num}/video.mov"

                extended_tracks_per_seq = self.perform_collection_on_single_sequence(
                    classic_extracted_annotation_path, nn_classic_extracted_annotation_path,
                    v_clz, v_meta_clz, v_num,
                    (ref_img.shape[1:]),
                    video_path, trajectory_model=model)

                if v_clz.name in extended_tracks.keys():
                    extended_tracks[v_clz.name][v_num] = extended_tracks_per_seq
                else:
                    extended_tracks[v_clz.name] = {
                        v_num: extended_tracks_per_seq
                    }

        return extended_tracks

    def perform_collection_on_single_sequence(
            self, classic_extracted_annotation_path,
            nn_classic_extracted_annotation_path, video_class, video_meta_class, video_number, image_shape,
            video_path, trajectory_model):
        extended_tracks = VideoSequenceAgentTracks(video_class, video_number, [])
        inactive_tracks = VideoSequenceAgentTracks(video_class, video_number, [])
        ratio = self.get_ratio(meta_class=video_meta_class, video_number=video_number)

        # turn into 0.4 seconds apart
        classic_extracted_df = pd.read_csv(classic_extracted_annotation_path)
        classic_extracted_df = self.adjust_dataframe_framerate(classic_extracted_df, 'frame_number', 30, 0.4)

        nn_classic_extracted_df = pd.read_csv(nn_classic_extracted_annotation_path)
        nn_classic_extracted_df = self.adjust_dataframe_framerate(nn_classic_extracted_df, 'frame', 30, 0.4)

        nn_classic_extracted_extended_df = pd.read_csv(nn_classic_extracted_annotation_path)
        nn_classic_extracted_extended_df = self.adjust_dataframe_framerate(
            nn_classic_extracted_extended_df, 'frame', 30, 0.4)

        running_track_idx = []

        for frame in tqdm(classic_extracted_df.frame_number.unique()):
            classic_extracted_centers, classic_extracted_track_ids, classic_extracted_boxes = \
                self.get_classic_extracted_centers(
                    classic_extracted_df, frame)
            nn_classic_extracted_centers, nn_classic_extracted_track_ids = \
                self.get_extracted_centers(
                    nn_classic_extracted_df, frame)

            track_id_to_location = dict(zip(classic_extracted_track_ids, classic_extracted_centers))

            frame = int(frame)
            if frame == 0:
                running_track_idx = classic_extracted_track_ids.tolist()
                # add to running tracks - only final tracks
                for track_idx, loc in track_id_to_location.items():
                    agent_track = AgentTrack(idx=track_idx)
                    agent_track.frames.append(frame)
                    agent_track.locations.append(loc.tolist())
                    extended_tracks.tracks.append(agent_track)
            else:
                # associate - try for next K frames - todo
                candidate_agents = [
                    CandidateAgentTrack(frame=frame, location=l)
                    for t, l in zip(nn_classic_extracted_track_ids, nn_classic_extracted_centers) if t == -1
                ]

                continuing_tracks = np.intersect1d(np.array(running_track_idx), classic_extracted_track_ids)
                for c_track in continuing_tracks:
                    continuing_track: AgentTrack = extended_tracks[c_track]
                    continuing_track.frames.append(frame)
                    continuing_track.locations.append(track_id_to_location[c_track].tolist())

                # look for a died track - track id not alive in next frame
                killed_tracks = np.setdiff1d(np.array(running_track_idx), classic_extracted_track_ids)
                new_tracks = np.setdiff1d(classic_extracted_track_ids, np.array(running_track_idx))

                for n_track in new_tracks:
                    agent_track = AgentTrack(idx=n_track)
                    agent_track.frames.append(frame)
                    agent_track.locations.append(track_id_to_location[n_track].tolist())
                    extended_tracks.tracks.append(agent_track)
                    running_track_idx.append(n_track)

                for k_track in killed_tracks:
                    candidate_track = extended_tracks[k_track]
                    if len(candidate_track.frames) > 1:
                        # predict
                        in_xy = np.stack(candidate_track.locations)
                        in_dxdy = np.diff(in_xy, axis=0)
                        batch = {
                            'in_xy': torch.tensor(in_xy).unsqueeze(1).float().cuda(),
                            'in_dxdy': torch.tensor(in_dxdy).unsqueeze(1).float().cuda()
                        }
                        with torch.no_grad():
                            trajectory_out = trajectory_model.test(batch)

                        out_locations = trajectory_out['out_xy'].squeeze(1)
                        # take 1st location - create a distance matrix with candidate agents
                        candidate_location = out_locations[0].cpu().numpy()
                        candidate_distance_matrix = np.sqrt(mm.distances.norm2squared_matrix(
                            objs=np.stack([c.location for c in candidate_agents]),
                            hyps=candidate_location,
                        )) * ratio

                        candidate_distance_matrix = self.config.threshold - candidate_distance_matrix
                        candidate_distance_matrix[candidate_distance_matrix < 0] = 1000
                        # Hungarian
                        match_rows, match_cols = scipy.optimize.linear_sum_assignment(candidate_distance_matrix)
                        actually_matched_mask = candidate_distance_matrix[match_rows, match_cols] < 1000
                        match_rows = match_rows[actually_matched_mask]
                        match_cols = match_cols[actually_matched_mask]

                        # if match add and extend
                        if len(match_rows) > 0:
                            chosen_candidate = candidate_agents[match_rows[0].item()]
                            killed_track = extended_tracks[k_track]
                            killed_track.frames.append(frame)
                            killed_track.locations.append(chosen_candidate.location.tolist())
                            killed_track.extended_at_frames.append(frame)
                            killed_track.inactive = 0

                            row_idx = nn_classic_extracted_extended_df[
                                (nn_classic_extracted_extended_df['frame'] == float(frame)) &
                                (nn_classic_extracted_extended_df['x'] == chosen_candidate.location[0]) &
                                (nn_classic_extracted_extended_df['y'] == chosen_candidate.location[1])].index.item()
                            nn_classic_extracted_extended_df.at[row_idx, 'track_id'] = killed_track.idx

                            if killed_track.idx in inactive_tracks:
                                inactive_tracks.tracks.remove(killed_track)
                        else:
                            if self.config.check_in_future:
                                for i in range(1, self.config.dead_threshold + 1):
                                    nn_classic_extracted_centers_future, nn_classic_extracted_track_ids_future = \
                                        self.get_extracted_centers(
                                            nn_classic_extracted_df, frame + i)
                                    candidate_agents_future = [
                                        CandidateAgentTrack(frame=frame, location=l)
                                        for t, l in
                                        zip(nn_classic_extracted_track_ids_future,
                                            nn_classic_extracted_centers_future) if t == -1
                                    ]

                                    candidate_location_f = out_locations[i].cpu().numpy()
                                    candidate_distance_matrix_f = np.sqrt(mm.distances.norm2squared_matrix(
                                        objs=np.stack([c.location for c in candidate_agents_future]),
                                        hyps=candidate_location_f,
                                    )) * ratio

                                    candidate_distance_matrix_f = self.config.threshold - candidate_distance_matrix_f
                                    candidate_distance_matrix_f[candidate_distance_matrix_f < 0] = 1000
                                    # Hungarian
                                    match_rows, match_cols = scipy.optimize.linear_sum_assignment(
                                        candidate_distance_matrix_f)
                                    actually_matched_mask = candidate_distance_matrix_f[match_rows, match_cols] < 1000
                                    match_rows = match_rows[actually_matched_mask]
                                    match_cols = match_cols[actually_matched_mask]

                                    if len(match_rows) > 0:
                                        chosen_candidate = candidate_agents_future[match_rows[0].item()]
                                        killed_track = extended_tracks[k_track]
                                        killed_track.frames.append(frame + i)
                                        killed_track.locations.append(chosen_candidate.location.tolist())
                                        killed_track.extended_at_frames.append(frame + i)
                                        killed_track.inactive = 0

                                        row_idx = nn_classic_extracted_extended_df[
                                            (nn_classic_extracted_extended_df['frame'] == float(frame + i)) &
                                            (nn_classic_extracted_extended_df['x'] == chosen_candidate.location[0]) &
                                            (nn_classic_extracted_extended_df['y'] == chosen_candidate.location[
                                                1])].index.item()
                                        nn_classic_extracted_extended_df.at[row_idx, 'track_id'] = killed_track.idx

                                        if killed_track.idx in inactive_tracks:
                                            inactive_tracks.tracks.remove(killed_track)
                                    else:
                                        continue
                            # can go in future but what if we grab some other track's identity going in future
                            # else add to inactives
                            inactive_track = extended_tracks[k_track]
                            # inactive_track.inactive += 1
                            inactive_tracks.tracks.append(inactive_track)
                            running_track_idx.remove(k_track)
                    else:
                        # don't remove it from extended tracks
                        # just replace it if it becomes active
                        # add to inactives
                        inactive_track = extended_tracks[k_track]
                        # inactive_track.inactive += 1
                        inactive_tracks.tracks.append(inactive_track)
                        running_track_idx.remove(k_track)

                # increment all inactives together
                for dead_track in inactive_tracks.tracks:
                    if dead_track.inactive == -1:
                        continue
                    dead_track.inactive += 1
                    if dead_track.inactive > self.config.dead_threshold:
                        dead_track.inactive = -1

        # debug
        predicted_tracks = [e for e in extended_tracks.tracks if len(e.extended_at_frames) > 0]
        for p_track in predicted_tracks:
            f_no = p_track.frames[0]
            plot_trajectory_with_one_frame(
                frame=extract_frame_from_video(video_path, f_no),
                last_frame=None,
                trajectory=np.stack(p_track.locations),
                obs_trajectory=None,
                frame_number=f_no,
                track_id=0,
                active_tracks=None,
                current_frame_locations=None,
                last_frame_locations=None,
                plot_first_and_last=False,
                use_lines=True,
                marker_size=2
            )
        return extended_tracks


def get_components(boolean_array, r=1, p=1):
    # find neighbours
    coordinates = list(zip(*np.where(boolean_array)))
    tree = cKDTree(coordinates)
    neighbours_by_pixel = tree.query_ball_tree(tree, r=r, p=p)
    # p=1 -> Manhatten distance; r=1 -> what would be 4-connectivity in 2D

    # create graph and find components
    G = nx.Graph()
    for ii, neighbours in enumerate(neighbours_by_pixel):
        if len(neighbours) > 1:
            G.add_edges_from([(ii, jj) for jj in neighbours[1:]])  # skip first neighbour as that is a self-loop
    components = nx.connected_components(G)
    components_out = list(nx.connected_components(G))

    # create output image
    output = np.zeros_like(boolean_array, dtype=np.int)
    for ii, component in enumerate(components):
        for idx in component:
            output[coordinates[idx]] = ii + 1

    return output, components_out


def find_connected_components():
    # skipping for now
    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(8, 6))
    original_ax, component_axis = axs

    blur_radius = 2.0
    threshold = 0.05

    masks = torch.load('logs/heat_masks_little3.pt')
    masks = masks[0]

    single_mask = masks[0].sigmoid().numpy()

    # 0 - ok - best so far
    single_mask = scipy.ndimage.gaussian_filter(single_mask, sigma=blur_radius)
    lab, n = scipy.ndimage.label(single_mask > threshold,
                                 structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    object_slices = scipy.ndimage.find_objects(lab)
    center_of_mass = scipy.ndimage.center_of_mass(single_mask, lab)

    # 1 - ok n similar
    # ret, thresh = cv2.threshold((single_mask * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # You need to choose 4 or 8 for connectivity type
    # connectivity = 4
    # # Perform the operation
    # output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # # Get the results
    # # The first cell is the number of labels
    # n = output[0]
    # # The second cell is the label matrix
    # lab = output[1]
    # # The third cell is the stat matrix
    # stats = output[2]
    # # The fourth cell is the centroid matrix
    # centroids = output[3]

    # 2 - ok but similar
    # ret, thresh = cv2.threshold((single_mask * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lab, components = get_components(thresh, p=2, r=8)
    # n = len(components)

    # 3 - not good
    # ret, thresh = cv2.threshold((single_mask * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # # blur and grayscale before thresholding
    # blur = skimage.filters.gaussian(thresh, sigma=blur_radius)
    #
    # # perform inverse binary thresholding
    # mask = blur < threshold
    #
    # # Perform CCA on the mask
    # lab, n = skimage.measure.label(mask, connectivity=4, return_num=True)

    original_ax.imshow(single_mask)
    component_axis.imshow(lab)

    original_ax.set_title('Mask')
    component_axis.set_title('Connected Components')

    # plt.scatter(centroids[:, 0], centroids[:, 1])

    plt.tight_layout(pad=1.58)
    plt.title(f"Components Found: {n}")
    plt.show()

    for l_idx in np.unique(lab):
        fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(8, 6))
        original_ax, component_axis = axs

        im = np.zeros_like(lab)
        im[lab == l_idx] = 255

        original_ax.imshow(single_mask)
        component_axis.imshow(im)
        plt.tight_layout(pad=1.58)
        plt.title(f"Component: {l_idx}")
        plt.show()
    print()


if __name__ == '__main__':
    # find_connected_components()
    conf = OmegaConf.load('config/training/training.yaml')
    # classifier_network = CropClassifier(config=conf, train_dataset=None, val_dataset=None, desired_output_shape=None,
    #                                     loss_function=None)
    # classifier_network = setup_person_classifier(
    #     OmegaConf.merge(
    #         OmegaConf.load(f'{MODEL_ROOT}baselinev2/improve_metrics/config/eval/eval.yaml'),
    #         OmegaConf.load(f'{MODEL_ROOT}baselinev2/improve_metrics/config/model/model.yaml'),
    #         OmegaConf.load(f'{MODEL_ROOT}baselinev2/improve_metrics/config/training/training.yaml'),
    #     ),
    #     SDDVideoClasses.BOOKSTORE
    # )
    analyzer = PosMapToConventional(conf, use_patch_filtered=True, classifier=None)
    # out = analyzer.perform_analysis_on_multiple_sequences(show_extracted_tracks_only=False)
    out = analyzer.perform_collection_on_multiple_sequences()
    # out = analyzer.get_metrics_for_multiple_sequences()
    # out = analyzer.generate_annotation_from_data(torch.load(
    #     '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/src/position_maps/logs/dummy_classic_nn.pt'))
    print()
