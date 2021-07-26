from typing import List

import pandas as pd
from PIL.Image import Image
from omegaconf import OmegaConf

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses, SDDVideoDatasets


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
            self.video_classes.append(getattr(SDDVideoDatasets, v_class))

        self.video_numbers = self.config.video_numbers
        self.root = self.config.root

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

    def perform_analysis_on_multiple_sequences(self):
        for v_idx, (v_clz, v_meta_clz) in enumerate(zip(self.video_classes, self.video_meta_classes)):
            for v_num in self.video_numbers[v_idx]:
                gt_annotation_path = f"{self.root}annotations/{v_clz.value}/video{v_num}/annotation_augmented.csv"
                extracted_annotation_path = f"{self.root}pm_extracted_annotations/{v_clz.value}/" \
                                            f"video{v_num}/trajectories.csv"
                ref_img = Image.open(f"{self.root}annotations/{v_clz.value}/video{v_num}/reference.jpg")
                self.perform_analysis_on_single_sequence(
                    gt_annotation_path, extracted_annotation_path, v_clz, v_num, (ref_img.height, ref_img.width))

    def perform_analysis_on_single_sequence(
            self, gt_annotation_path, extracted_annotation_path, video_class, video_number, image_shape):
        tracks_data: List[Track] = []
        gt_df = pd.read_csv(gt_annotation_path)
        extracted_df = pd.read_csv(extracted_annotation_path)
        for frame in gt_df.frame.values:
            gt_bbox_centers, supervised_boxes = self.get_gt_annotation(
                frame_number=frame, gt_annotation_df=gt_df, original_shape=image_shape)
            extracted_centers = self.get_frame_annotations(df=extracted_df, frame_number=frame)
            print()

        video_sequence_track_data = VideoSequenceTracksData(
            video_class=video_class, video_number=video_number, tracks=tracks_data)


if __name__ == '__main__':
    analyzer = TracksAnalyzer(OmegaConf.load('config/training/training.yaml'))
    analyzer.perform_analysis_on_multiple_sequences()
