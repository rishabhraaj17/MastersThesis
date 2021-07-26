from typing import List

import motmetrics as mm
import numpy as np
import pandas as pd
import scipy
import torchvision.io
from omegaconf import OmegaConf

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta


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


class VideoSequenceTracksData(object):
    def __init__(self, video_class, video_number, tracks: List[Track]):
        super(VideoSequenceTracksData, self).__init__()
        self.video_class = video_class
        self.video_number = video_number
        self.tracks = tracks


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
        extracted_centers = extracted_centers[:, 2:]
        return extracted_centers

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
        return gt_bbox_centers, supervised_boxes

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

    def perform_analysis_on_multiple_sequences(self):
        for v_idx, (v_clz, v_meta_clz) in enumerate(zip(self.video_classes, self.video_meta_classes)):
            for v_num in self.video_numbers[v_idx]:
                gt_annotation_path = f"{self.root}annotations/{v_clz.value}/video{v_num}/annotation_augmented.csv"
                extracted_annotation_path = f"{self.root}pm_extracted_annotations/{v_clz.value}/" \
                                            f"video{v_num}/trajectories.csv"
                ref_img = torchvision.io.read_image(f"{self.root}annotations/{v_clz.value}/video{v_num}/reference.jpg")
                self.perform_analysis_on_single_sequence(
                    gt_annotation_path, extracted_annotation_path, v_clz, v_meta_clz, v_num, (ref_img.shape[1:]))

    def perform_analysis_on_single_sequence(
            self, gt_annotation_path, extracted_annotation_path, video_class,
            video_meta_class, video_number, image_shape):
        tracks_data: List[Track] = []
        ratio = self.get_ratio(meta_class=video_meta_class, video_number=video_number)

        gt_df = self.get_gt_df(gt_annotation_path)
        extracted_df = pd.read_csv(extracted_annotation_path)

        for frame in gt_df.frame.unique():
            gt_bbox_centers, supervised_boxes = self.get_gt_annotation(
                frame_number=frame, gt_annotation_df=gt_df, original_shape=tuple(image_shape))
            extracted_centers = self.get_extracted_centers(extracted_df, frame)
            (match_rows, match_cols), (fn, fp, precision, recall, tp) = self.get_associations_and_metrics(
                gt_centers=gt_bbox_centers, extracted_centers=extracted_centers,
                max_distance=float(self.config.match_distance),
                ratio=ratio,
                threshold=self.config.threshold
            )
            print()

        video_sequence_track_data = VideoSequenceTracksData(
            video_class=video_class, video_number=video_number, tracks=tracks_data)


if __name__ == '__main__':
    analyzer = TracksAnalyzer(OmegaConf.load('config/training/training.yaml'))
    analyzer.perform_analysis_on_multiple_sequences()
