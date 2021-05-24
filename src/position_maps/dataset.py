import os
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from omegaconf import ListConfig
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips

from average_image.constants import SDDVideoClasses
from average_image.utils import SDDMeta
from position_maps.utils import generate_position_map, plot_samples
from unsupervised_tp_0.dataset import sort_list


class SDDFrameAndAnnotationDataset(Dataset):
    def __init__(
            self, root: str, video_label: SDDVideoClasses, frames_per_clip: int = 1, num_videos=-1,
            step_between_clips: int = 1, frame_rate: Optional[float] = 30., transform: Optional[Callable] = None,
            _precomputed_metadata: bool = None, num_workers: int = 1, _video_width: int = 0, _video_height: int = 0,
            _video_min_dimension: int = 0, _audio_samples: int = 0, scale: float = 1.0, video_number_to_use: int = 0,
            multiple_videos: bool = False, use_generated: bool = False, sigma: int = 10, plot: bool = False,
            desired_size: Tuple[int, int] = None, heatmap_shape: Tuple[int, int] = None,
            return_combined_heatmaps: bool = True):
        super(SDDFrameAndAnnotationDataset, self).__init__()

        _mid_path = video_label.value
        _annotation_decider = "filtered_generated_annotations/" if use_generated else "annotations/"
        video_path = root + "videos/" + _mid_path
        annotation_path = root + _annotation_decider + _mid_path
        video_extensions = ('mov',)
        annotation_extension = ('csv',)

        annotation_classes = list(sorted(list_dir(annotation_path),
                                         key=lambda x: int(x[-1]) if len(x) == 6 else int(x[-2:])))
        annotation_class_to_idx = {annotation_classes[i]: i for i in range(len(annotation_classes))}

        self.annotation_samples = make_dataset(directory=annotation_path, class_to_idx=annotation_class_to_idx,
                                               extensions=annotation_extension)
        self.annotation_list = [x[0] for x in self.annotation_samples]
        self.annotation_list_idx = [x[1] for x in self.annotation_samples]

        classes = list(sorted(list_dir(video_path),
                              key=lambda x: int(x[-1]) if len(x) == 6 else int(x[-2:])))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(directory=video_path, class_to_idx=class_to_idx,
                                    extensions=video_extensions)
        video_list = [x[0] for x in self.samples]
        video_list_idx = [x[1] for x in self.samples]

        # sort it
        self.video_list = sort_list(video_list, video_list_idx)
        self.video_list_idx = sorted(video_list_idx)

        self.annotation_list = sort_list(self.annotation_list, self.annotation_list_idx)
        self.annotation_list_idx = sorted(self.annotation_list_idx)

        self.use_generated = use_generated
        self.num_videos = num_videos
        self.video_number_to_use = video_number_to_use
        self.video_label = video_label
        self.multiple_videos = multiple_videos
        if multiple_videos and num_videos == -1:
            annotation_path = self.annotation_list
            self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            video_list_subset = self.video_list
            video_list_idx_subset = self.video_list_idx

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
        elif multiple_videos and (isinstance(num_videos, list) or isinstance(num_videos, ListConfig)):
            # restricted to number of videos
            num_videos = list(num_videos)
            annotation_path = [self.annotation_list[n] for n in num_videos]
            self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            video_list_subset = [self.video_list[n] for n in num_videos]
            video_list_idx_subset = [self.video_list_idx[n] for n in num_videos]

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
        elif multiple_videos:
            # restricted to number of videos
            annotation_path = self.annotation_list[:num_videos]
            self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            video_list_subset = self.video_list[:num_videos]
            video_list_idx_subset = self.video_list_idx[:num_videos]

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
        else:
            video_list_subset = [self.video_list[video_number_to_use]]
            video_list_idx_subset = [self.video_list_idx[video_number_to_use]]

            annotation_path = [self.annotation_list[video_list_idx_subset[0]]]
            self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]

        video_clips = VideoClips(video_list_subset,
                                 frames_per_clip,
                                 step_between_clips,
                                 frame_rate,
                                 _precomputed_metadata,
                                 num_workers=num_workers,
                                 _video_width=_video_width,
                                 _video_height=_video_height,
                                 _video_min_dimension=_video_min_dimension,
                                 _audio_samples=_audio_samples,
                                 )
        self.video_clips_metadata = video_clips.metadata
        self.video_clips = video_clips
        self.transform = transform

        meta_file = 'H_SDD.txt'
        self.sdd_meta = SDDMeta(root + meta_file)

        self.scale = scale

        self.new_scale = None
        self.use_generated = use_generated
        self.sigma = sigma
        self.plot = plot
        self.desired_size = desired_size
        self.heatmap_shape = heatmap_shape
        self.return_combined_heatmaps = return_combined_heatmaps

    @property
    def metadata(self):
        return self.video_clips.metadata

    @staticmethod
    def get_generated_frame_annotations(df: pd.DataFrame, frame_number: int):
        idx: pd.DataFrame = df.loc[df["frame_number"] == frame_number]
        return idx.to_numpy()

    @staticmethod
    def _read_annotation_file(path, use_generated=False):
        dff = pd.read_csv(path)
        if not use_generated:
            dff = dff.drop(dff.columns[[0]], axis=1)
        return dff

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, item):
        video, audio, info, video_idx = self.video_clips.get_clip(item)
        video = video.permute(0, 3, 1, 2)
        original_shape = new_shape = (video.shape[-2], video.shape[-1])

        bbox_centers, boxes, track_idx, class_labels = self.get_annotation_for_frame(item, video_idx, original_shape)

        while bbox_centers.size == 0 and boxes.size == 0:
            random_frame_num = np.random.choice(len(self), 1, replace=False).item()

            video, audio, info, video_idx = self.video_clips.get_clip(random_frame_num)
            video = video.permute(0, 3, 1, 2)

            bbox_centers, boxes, track_idx, class_labels = self.get_annotation_for_frame(item, video_idx,
                                                                                         original_shape)
            item = random_frame_num

        video = video.float() / 255.0
        if self.transform is not None:
            out = self.transform(image=video.squeeze(0).permute(1, 2, 0).numpy(), keypoints=bbox_centers,
                                 bboxes=boxes, class_labels=class_labels)
            img = out['image']
            boxes = np.stack(out['bboxes'])
            bbox_centers = np.stack(out['keypoints'])

            video = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            new_shape = (video.shape[-2], video.shape[-1])

        heat_mask = torch.from_numpy(
            generate_position_map(list(new_shape), bbox_centers, sigma=self.sigma, heatmap_shape=self.heatmap_shape,
                                  return_combined=self.return_combined_heatmaps, hw_mode=True))
        if self.plot:
            plot_samples(video.squeeze().permute(1, 2, 0), heat_mask, boxes, bbox_centers, plot_boxes=True,
                         additional_text=f'Frame Number: {item} | Video Idx: {video_idx}')

        meta = {'boxes': boxes, 'bbox_centers': bbox_centers,
                'track_idx': track_idx, 'item': item,
                'original_shape': original_shape, 'new_shape': new_shape,
                'video_idx': video_idx}

        heat_mask = heat_mask.float()

        position_map = torch.zeros_like(heat_mask)
        key_points = torch.round(torch.from_numpy(bbox_centers)).long()
        position_map[key_points[:, 1], key_points[:, 0]] = 1

        distribution_map = torch.zeros(size=(heat_mask.shape[0], heat_mask.shape[1], 3))
        variance_list = torch.tensor([self.sigma] * key_points.shape[0])
        distribution_map[key_points[:, 1], key_points[:, 0]] = torch.stack(
            (key_points[:, 1], key_points[:, 0], variance_list)).t().float()
        distribution_map = distribution_map.permute(2, 0, 1)

        return video, heat_mask, position_map, distribution_map, meta

    def get_annotation_for_frame(self, item, video_idx, original_shape):
        h, w = original_shape
        df = self.annotations_df[video_idx]
        frame_annotation = self.get_generated_frame_annotations(df, item)

        boxes = frame_annotation[:, 1:5].astype(np.int)
        track_idx = frame_annotation[:, 0].astype(np.int)
        bbox_centers = frame_annotation[:, 7:9].astype(np.int)

        inside_boxes_idx = [b for b, box in enumerate(boxes)
                            if (box[0] > 0 and box[2] < w) and (box[1] > 0 and box[3] < h)]

        boxes = boxes[inside_boxes_idx]
        track_idx = track_idx[inside_boxes_idx]
        bbox_centers = bbox_centers[inside_boxes_idx]

        labels = ['object'] * boxes.shape[0]

        return bbox_centers, boxes, track_idx, labels
