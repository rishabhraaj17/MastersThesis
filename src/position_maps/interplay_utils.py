import math
import os
from typing import Optional, Callable, Tuple, Sequence, List

import albumentations as A
import pandas as pd
import torch
import torchvision
from matplotlib import colors
from omegaconf import ListConfig, DictConfig
from torch.nn.functional import pad
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from log import get_logger
from patch_utils import quick_viz
from unsupervised_tp_0.dataset import sort_list

logger = get_logger(__name__)

# PAD_MODE = 'constant'
PAD_MODE = 'replicate'


class SDDFrameOnlyDataset(Dataset):
    def __init__(
            self, root: str, video_label: SDDVideoClasses, meta_label: SDDVideoDatasets,
            frames_per_clip: int = 1, num_videos=-1, step_between_clips: int = 1, frame_rate: Optional[float] = 30.,
            transform: Optional[Callable] = None, _precomputed_metadata: bool = None, num_workers: int = 1,
            _video_width: int = 0, _video_height: int = 0, _video_min_dimension: int = 0, _audio_samples: int = 0,
            scale: float = 1.0, video_number_to_use: int = 0, multiple_videos: bool = False,
            use_generated: bool = False, sigma: int = 10, plot: bool = False,
            desired_size: Tuple[int, int] = None, heatmap_shape: Tuple[int, int] = None,
            return_combined_heatmaps: bool = True, seg_map_objectness_threshold: float = 0.5,
            heatmap_region_limit_threshold: float = 0.4, downscale_only_target_maps: bool = True,
            rgb_transform: Optional[Callable] = None, rgb_new_shape: Tuple[int, int] = None,
            rgb_pad_value: Sequence[int] = None, target_pad_value: Sequence[int] = None,
            rgb_plot_transform: Optional[Callable] = None, common_transform: Optional[Callable] = None,
            using_replay_compose: bool = False, manual_annotation_processing: bool = False, config: DictConfig = None):
        super(SDDFrameOnlyDataset, self).__init__()

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
            # self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
            self.annotations_df = [self._read_annotation_file_and_filter(p, os, self.use_generated)
                                   for p, os in zip(annotation_path, self.original_shape)]

            video_list_subset = self.video_list
            video_list_idx_subset = self.video_list_idx

            # ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            # ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            # self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
        elif multiple_videos and (isinstance(num_videos, list) or isinstance(num_videos, ListConfig)):
            # restricted to number of videos
            num_videos = list(num_videos)
            annotation_path = [self.annotation_list[n] for n in num_videos]
            # self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
            self.annotations_df = [self._read_annotation_file_and_filter(p, os, self.use_generated)
                                   for p, os in zip(annotation_path, self.original_shape)]

            video_list_subset = [self.video_list[n] for n in num_videos]
            video_list_idx_subset = [self.video_list_idx[n] for n in num_videos]

            # ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            # ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            # self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
        elif multiple_videos:
            # restricted to number of videos
            annotation_path = self.annotation_list[:num_videos]
            # self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
            self.annotations_df = [self._read_annotation_file_and_filter(p, os, self.use_generated)
                                   for p, os in zip(annotation_path, self.original_shape)]

            video_list_subset = self.video_list[:num_videos]
            video_list_idx_subset = self.video_list_idx[:num_videos]

            # ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            # ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            # self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
        else:
            video_list_subset = [self.video_list[video_number_to_use]]
            video_list_idx_subset = [self.video_list_idx[video_number_to_use]]

            annotation_path = [self.annotation_list[video_list_idx_subset[0]]]
            # self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
            self.annotations_df = [self._read_annotation_file_and_filter(p, os, self.use_generated)
                                   for p, os in zip(annotation_path, self.original_shape)]

            # ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            # ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            # self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]

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
        self.rgb_transform = rgb_transform
        self.rgb_new_shape = rgb_new_shape
        self.rgb_pad_value = rgb_pad_value
        self.target_pad_value = target_pad_value
        self.common_transform = common_transform
        self.rgb_plot_transform = rgb_plot_transform

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
        self.seg_map_objectness_threshold = seg_map_objectness_threshold
        self.meta_label = meta_label
        self.heatmap_region_limit_threshold = heatmap_region_limit_threshold
        self.downscale_only_target_maps = downscale_only_target_maps
        self.using_replay_compose = using_replay_compose
        self.manual_annotation_processing = manual_annotation_processing
        self.config = config
        self.frame_rate = frame_rate
        self.frames_per_clip = frames_per_clip

    @property
    def metadata(self):
        return self.video_clips.metadata

    def get_ratio_from_sdd_meta(self):
        pixel_to_meter_ratio = float(self.sdd_meta.get_meta(
            self.meta_label, self.video_number_to_use)[0]['Ratio'].to_numpy()[0])
        return pixel_to_meter_ratio

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

    @staticmethod
    def _read_annotation_file_and_filter(path, original_shape, use_generated=False):
        dff = pd.read_csv(path)
        if not use_generated:
            dff = dff.drop(dff.columns[[0]], axis=1)
        # drop rows where bounding box is out of image
        h, w = original_shape
        h -= 1
        w -= 1
        dff = dff.drop(dff[(dff.x_min < 0) | (dff.x_max > w) | (dff.y_min < 0) | (dff.y_max > h)].index)
        return dff

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, item):
        if self.frames_per_clip > 1:
            meta, video = self._get_item_core_temporal(item)
        else:
            meta, video = self._get_item_core(item)

        return video, meta

    def _get_item_core(self, item):
        video, audio, info, video_idx = self.video_clips.get_clip(item)
        video = video.permute(0, 3, 1, 2)
        original_shape = new_shape = downscale_shape = (video.shape[-2], video.shape[-1])

        fps = info['video_fps']
        item = int(math.floor(item * (float(self.video_clips.video_fps[video_idx]) / fps)))

        video = video.float() / 255.0
        if self.transform is not None:
            out = self.transform(image=video.squeeze(0).permute(1, 2, 0).numpy())
            img = out['image']
            downscale_shape = (img.shape[0], img.shape[1])

            video = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        pre_padded_video = video.clone()
        video = pad(video, self.rgb_pad_value, mode=PAD_MODE)
        new_shape = (video.shape[-2], video.shape[-1])
        meta = {
            'item': item, 'pre_pad_rgb': pre_padded_video,
            'original_shape': original_shape, 'post_pad_shape': new_shape,
            'downscale_shape': downscale_shape,
            'video_idx': video_idx}

        return meta, video

    def _get_item_core_temporal(self, item):
        # not yet ready - fixme
        video, audio, info, video_idx = self.video_clips.get_clip(item)
        video = video.permute(0, 3, 1, 2)
        original_shape = new_shape = downscale_shape = (video.shape[-2], video.shape[-1])

        video = video.float() / 255.0
        if self.transform is not None:
            video = self.transform(image=video)

        downscale_shape = (video.shape[-2], video.shape[-1])

        pre_padded_video = video.clone()
        video = pad(video, self.rgb_pad_value, mode=PAD_MODE)
        new_shape = (video.shape[-2], video.shape[-1])
        meta = {
            'item': item, 'pre_pad_rgb': pre_padded_video,
            'original_shape': original_shape, 'post_pad_shape': new_shape,
            'downscale_shape': downscale_shape,
            'video_idx': video_idx}

        return meta, video

    # not using annotations
    def get_annotation_for_frame(self, item, video_idx, original_shape, allow_zero_box_coordinates=True):
        h, w = original_shape
        df = self.annotations_df[video_idx]
        frame_annotation = self.get_generated_frame_annotations(df, item)

        # boxes = frame_annotation[:, 1:5].astype(np.int)
        # track_idx = frame_annotation[:, 0].astype(np.int)
        # bbox_centers = frame_annotation[:, 7:9].astype(np.int)

        # silence dep warning
        boxes = frame_annotation[:, 1:5].astype(int)
        track_idx = frame_annotation[:, 0].astype(int)
        bbox_centers = frame_annotation[:, 7:9].astype(int)

        if allow_zero_box_coordinates:
            inside_boxes_idx = [b for b, box in enumerate(boxes)
                                if (box[0] >= 0 and box[2] < w) and (box[1] >= 0 and box[3] < h)]
        else:
            inside_boxes_idx = [b for b, box in enumerate(boxes)
                                if (box[0] > 0 and box[2] < w) and (box[1] > 0 and box[3] < h)]

        boxes = boxes[inside_boxes_idx]
        track_idx = track_idx[inside_boxes_idx]
        bbox_centers = bbox_centers[inside_boxes_idx]

        labels = ['object'] * boxes.shape[0]

        return bbox_centers, boxes, track_idx, labels


def setup_multiple_frame_only_datasets_core(
        cfg, video_classes_to_use, video_numbers_to_use, num_videos, multiple_videos,
        df, df_target, use_common_transforms=True):
    datasets = []
    for idx, v_clz in enumerate(video_classes_to_use):
        for v_num in video_numbers_to_use[idx]:
            logger.info(f"Setting up frames only dataset:  {v_clz} - {v_num}")
            condition = (df.CLASS == v_clz) & (df.NUMBER == v_num)
            h, w = df[condition].RESCALED_SHAPE.values.item()
            pad_values = df[condition].PAD_VALUES.values.item()

            target_h, target_w = df_target[condition].RESCALED_SHAPE.values.item()
            target_pad_values = df_target[condition].PAD_VALUES.values.item()

            transform = A.Compose([A.Resize(height=target_h, width=target_w)])
            rgb_transform_fn = None
            rgb_plot_transform = None
            if use_common_transforms:
                common_transform = None
            else:
                common_transform = None

            datasets.append(
                SDDFrameOnlyDataset(
                    root=cfg.root, video_label=getattr(SDDVideoClasses, v_clz),
                    num_videos=num_videos, transform=transform if cfg.data_augmentation else None,
                    num_workers=cfg.dataset_workers, scale=cfg.scale_factor,
                    video_number_to_use=v_num,
                    multiple_videos=multiple_videos,
                    use_generated=cfg.use_generated_dataset,
                    sigma=cfg.sigma,
                    plot=cfg.plot_samples,
                    desired_size=cfg.desired_size,
                    heatmap_shape=cfg.heatmap_shape,
                    return_combined_heatmaps=cfg.return_combined_heatmaps,
                    seg_map_objectness_threshold=cfg.seg_map_objectness_threshold,
                    meta_label=getattr(SDDVideoDatasets, v_clz),
                    heatmap_region_limit_threshold=cfg.heatmap_region_limit_threshold,
                    downscale_only_target_maps=cfg.downscale_only_target_maps,
                    rgb_transform=rgb_transform_fn,
                    rgb_new_shape=(h, w),
                    rgb_pad_value=pad_values,
                    target_pad_value=target_pad_values,
                    rgb_plot_transform=rgb_plot_transform,
                    common_transform=common_transform,
                    using_replay_compose=False,
                    manual_annotation_processing=cfg.manual_annotation_processing,
                    frame_rate=cfg.frame_rate,
                    config=cfg,
                    frames_per_clip=cfg.video_based.frames_per_clip if cfg.video_based.enabled else 1
                ))
    return ConcatDataset(datasets)


def frames_only_collate_fn(batch):
    rgb_img_list, meta_list = [], []
    for batch_item in batch:
        rgb, meta = batch_item
        rgb_img_list.append(rgb)
        meta_list.append(meta)

    rgb_img_list = torch.cat(rgb_img_list)

    return rgb_img_list, meta_list


def get_all_matplotlib_colors():
    clrs = dict(colors.BASE_COLORS, **colors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(colors.rgb_to_hsv(colors.to_rgba(color)[:3])), name)
                    for name, color in clrs.items())
    sorted_names = [name for hsv, name in by_hsv]
    return sorted_names


class Track(object):
    def __init__(self, idx: int, frames: List[int], locations: List, inactive: int = 0):
        super(Track, self).__init__()
        self.idx = idx
        self.frames = frames
        self.locations = locations
        self.inactive = inactive

    def __eq__(self, other):
        return self.idx == other.idx

    def __repr__(self):
        return f"Track ID: {self.idx}" \
               f"\n{'Active' if self.inactive == 0 else ('Inactive since' + str(self.inactive) + 'frames')}" \
               f"\nFrames: {self.frames}" \
               f"\nTrack Positions: {self.locations}\n\n"

    def __lt__(self, other):
        return len(self.frames) < len(other.frames)


class Tracks(object):
    def __init__(self, tracks: List[Track]):
        self.tracks = tracks

    @classmethod
    def init_with_empty_tracks(cls):
        return Tracks([])
