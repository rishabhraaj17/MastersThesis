import math
import os
from typing import Optional, Callable, Tuple, Sequence

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations import normalize_bboxes, denormalize_bboxes
from albumentations.augmentations.functional import bbox_hflip, keypoint_hflip, bbox_vflip, keypoint_vflip
from omegaconf import ListConfig, DictConfig
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from utils import generate_position_map, plot_samples, scale_annotations
from patch_utils import quick_viz
from unsupervised_tp_0.dataset import sort_list

# PAD_MODE = 'constant'
PAD_MODE = 'replicate'


class SDDFrameAndAnnotationDataset(Dataset):
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

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, item):
        if self.frames_per_clip > 1:
            class_maps, distribution_map, heat_mask, meta, position_map, video = self._get_item_core_temporal(item)
        else:
            class_maps, distribution_map, heat_mask, meta, position_map, video = self._get_item_core(item)

        return video, heat_mask, position_map, distribution_map, class_maps, meta

    def _get_item_core(self, item):
        video, audio, info, video_idx = self.video_clips.get_clip(item)
        video = video.permute(0, 3, 1, 2)
        original_shape = new_shape = downscale_shape = (video.shape[-2], video.shape[-1])

        fps = info['video_fps']
        item = int(math.floor(item * (float(self.video_clips.video_fps[video_idx]) / fps)))
        # item = max(0, (item - 1) + (item * (int(round(self.video_clips.video_fps[video_idx]) // fps) - 1)))

        bbox_centers, boxes, track_idx, class_labels = self.get_annotation_for_frame(item, video_idx, original_shape)

        while bbox_centers.size == 0 and boxes.size == 0:
            random_frame_num = np.random.choice(len(self), 1, replace=False).item()

            video, audio, info, video_idx = self.video_clips.get_clip(random_frame_num)
            video = video.permute(0, 3, 1, 2)

            item = random_frame_num
            item = int(math.floor(item * (float(self.video_clips.video_fps[video_idx]) / fps)))
            # item = max(0, (item - 1) + (item * (int(round(self.video_clips.video_fps[video_idx]) // fps) - 1)))

            bbox_centers, boxes, track_idx, class_labels = self.get_annotation_for_frame(item, video_idx,
                                                                                         original_shape)

        video = video.float() / 255.0
        if self.transform is not None:
            out = self.transform(image=video.squeeze(0).permute(1, 2, 0).numpy(), keypoints=bbox_centers,
                                 bboxes=boxes, class_labels=class_labels)
            img = out['image']
            target_boxes = np.stack(out['bboxes'])
            target_bbox_centers = np.stack(out['keypoints'])

            # img = pad(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0), self.target_pad_value)# , mode='replicate')
            # downscale_shape = (img.shape[-2], img.shape[-1])
            downscale_shape = (img.shape[0], img.shape[1])

        heat_mask = torch.from_numpy(
            generate_position_map(list(downscale_shape), target_bbox_centers, sigma=self.sigma,
                                  heatmap_shape=self.heatmap_shape,
                                  return_combined=self.return_combined_heatmaps, hw_mode=True))
        heat_mask = pad(
            heat_mask.unsqueeze(0).unsqueeze(0),
            self.target_pad_value, mode='constant').squeeze(0).squeeze(0)

        key_points = torch.round(torch.from_numpy(target_bbox_centers)).long()
        position_map = heat_mask.clone().clamp(min=0, max=1).int().float()
        class_maps = heat_mask.clone()
        class_maps = torch.where(class_maps > self.seg_map_objectness_threshold, 1.0, 0.0)

        distribution_map = torch.zeros(size=(heat_mask.shape[-2], heat_mask.shape[-1], 3))
        # not in use - remove later
        # variance_list = torch.tensor([self.sigma] * key_points.shape[0])
        # distribution_map[key_points[:, 1], key_points[:, 0]] = torch.stack(
        #     (key_points[:, 1], key_points[:, 0], variance_list)).t().float()
        distribution_map = distribution_map.permute(2, 0, 1)

        heat_mask = heat_mask.to(dtype=torch.float64)
        heat_mask = torch.where(heat_mask > self.heatmap_region_limit_threshold, heat_mask, 0.0)
        heat_mask = heat_mask.float()

        if self.rgb_transform is not None:
            out = self.rgb_transform(image=video.squeeze(0).permute(1, 2, 0).numpy(), keypoints=bbox_centers,
                                     bboxes=boxes, class_labels=class_labels)
            img = out['image']
            rgb_boxes = np.stack(out['bboxes'])
            rgb_bbox_centers = np.stack(out['keypoints'])

            video = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        if self.common_transform is not None:
            # inside_boxes_idx = [b for b, box in enumerate(rgb_boxes)
            #                     if (box[0] > 0 and box[2] < video.shape[-1])
            #                     and (box[1] > 0 and box[3] < video.shape[-2])]
            # filter using smaller image - correct later
            inside_boxes_idx = [b for b, box in enumerate(target_boxes)
                                if (box[0] > 0 and box[2] < heat_mask.shape[-1])
                                and (box[1] > 0 and box[3] < heat_mask.shape[-2])]
            rgb_boxes = rgb_boxes[inside_boxes_idx]
            rgb_bbox_centers = rgb_bbox_centers[inside_boxes_idx]

            target_boxes = target_boxes[inside_boxes_idx]
            target_bbox_centers = target_bbox_centers[inside_boxes_idx]
            class_labels = ['object'] * rgb_boxes.shape[0]

            if self.using_replay_compose:
                out = self.common_transform(image=video.squeeze(0).permute(1, 2, 0).numpy(), keypoints=rgb_bbox_centers,
                                            bboxes=rgb_boxes, class_labels=class_labels)
                img = out['image']
                video = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                rgb_boxes = np.stack(out['bboxes'])
                rgb_bbox_centers = np.stack(out['keypoints'])

                out_mask = A.ReplayCompose.replay(out['replay'], image=heat_mask.numpy(), keypoints=target_bbox_centers,
                                                  bboxes=target_boxes, class_labels=class_labels)
                img = out_mask['image']
                heat_mask = torch.from_numpy(img)

                if not self.manual_annotation_processing:
                    target_boxes = np.stack(out_mask['bboxes'])
                    target_bbox_centers = np.stack(out_mask['keypoints'])
                else:
                    if out['replay']['applied']:
                        kp_extra = np.zeros((target_bbox_centers.shape[0], 2))
                        target_bbox_centers = np.hstack((target_bbox_centers, kp_extra))

                        target_boxes = normalize_bboxes(
                            target_boxes, rows=heat_mask.shape[0], cols=heat_mask.shape[1])

                        all_transforms = {k: v for k, v in out['replay'].items()
                                          if k not in ["__class_fullname__", "applied", "params"]}
                        for transform in all_transforms['transforms']:
                            if transform['applied'] \
                                    and transform['__class_fullname__'] != 'albumentations.augmentations.' \
                                                                           'transforms.' \
                                                                           'RandomBrightnessContrast':
                                class_name = str.split(transform['__class_fullname__'], '.')[-1]
                                if class_name == 'HorizontalFlip':
                                    target_boxes = [bbox_hflip(box, rows=heat_mask.shape[0], cols=heat_mask.shape[1])
                                                    for box in target_boxes]
                                    target_bbox_centers = [keypoint_hflip(
                                        kp, rows=heat_mask.shape[0], cols=heat_mask.shape[1]) for kp in
                                        target_bbox_centers]
                                elif class_name == 'VerticalFlip':
                                    target_boxes = [bbox_vflip(box, rows=heat_mask.shape[0], cols=heat_mask.shape[1])
                                                    for box in target_boxes]
                                    target_bbox_centers = [keypoint_vflip(
                                        kp, rows=heat_mask.shape[0], cols=heat_mask.shape[1]) for kp in
                                        target_bbox_centers]
                                else:
                                    continue  # raise NotImplementedError

                        target_boxes = denormalize_bboxes(
                            target_boxes, rows=heat_mask.shape[0], cols=heat_mask.shape[1])

                        target_boxes = np.stack(target_boxes)
                        target_bbox_centers = np.stack(target_bbox_centers)
            else:
                out = self.common_transform(image=video.squeeze(0).permute(1, 2, 0).numpy(),
                                            image0=heat_mask.numpy(),
                                            keypoints=rgb_bbox_centers,
                                            keypoints0=target_bbox_centers,
                                            bboxes=rgb_boxes,
                                            bboxes0=target_boxes,
                                            class_labels=class_labels)

                video = torch.from_numpy(out['image']).permute(2, 0, 1).unsqueeze(0)
                rgb_boxes = np.stack(out['bboxes'])
                rgb_bbox_centers = np.stack(out['keypoints'])

                heat_mask = torch.from_numpy(out['image0'])
                target_boxes = np.stack(out['bboxes0'])
                target_bbox_centers = np.stack(out['keypoints0'])

        if self.plot and self.rgb_plot_transform is not None:
            out = self.rgb_plot_transform(image=video.squeeze(0).permute(1, 2, 0).numpy(),
                                          keypoints=rgb_bbox_centers,
                                          bboxes=rgb_boxes, class_labels=class_labels)
            plot_samples(img=out['image'], mask=heat_mask, boxes=target_boxes,
                         box_centers=target_bbox_centers, rgb_boxes=np.stack(out['bboxes']),
                         rgb_box_centers=np.stack(out['keypoints']),
                         plot_boxes=True, additional_text=f'Frame Number: {item} | Video Idx: {video_idx}')
        elif self.plot:
            plot_samples(img=video.squeeze().permute(1, 2, 0), mask=heat_mask, boxes=target_boxes,
                         box_centers=target_bbox_centers, rgb_boxes=rgb_boxes, rgb_box_centers=rgb_bbox_centers,
                         plot_boxes=True, additional_text=f'Frame Number: {item} | Video Idx: {video_idx}')

        pre_padded_video = video.clone()
        video = pad(video, self.rgb_pad_value, mode=PAD_MODE)
        new_shape = (video.shape[-2], video.shape[-1])
        meta = {'boxes': target_boxes, 'bbox_centers': target_bbox_centers,
                'rgb_boxes': rgb_boxes, 'rgb_bbox_centers': rgb_bbox_centers,
                'pre_pad_rgb': pre_padded_video,
                'track_idx': track_idx, 'item': item,
                'original_shape': original_shape, 'new_shape': new_shape,
                'downscale_shape': downscale_shape,
                'video_idx': video_idx}

        return class_maps, distribution_map, heat_mask, meta, position_map, video

    def _get_item_core_temporal(self, item):
        annotations = None
        v_idx, _ = self.video_clips.get_clip_location(item)
        while self.annotations_df[v_idx].frame_number.max() < \
                self.video_clips.resampling_idxs[v_idx][item].max().item() or annotations is None:
            item = np.random.choice(len(self), 1, replace=False).item()

            video, audio, info, video_idx = self.video_clips.get_clip(item)
            video = video.permute(0, 3, 1, 2)
            original_shape = new_shape = downscale_shape = (video.shape[-2], video.shape[-1])

            annotations = [self.get_annotation_for_frame(i.item(), v_idx, original_shape)
                           for i in self.video_clips.resampling_idxs[v_idx][item]]
            for annotation in annotations:
                bbox_centers, boxes, track_idx, class_labels = annotation
                if bbox_centers.size == 0 or boxes.size == 0 or track_idx.size == 0:
                    annotations = None
                    break

        # video, audio, info, video_idx = self.video_clips.get_clip(item)

        # video = video.permute(0, 3, 1, 2)
        # original_shape = new_shape = downscale_shape = (video.shape[-2], video.shape[-1])

        # annotations = [self.get_annotation_for_frame(i.item(), video_idx, original_shape)
        #                for i in self.video_clips.resampling_idxs[video_idx][item]]

        video = video.float() / 255.0
        if self.transform is not None:
            rgb_images, target_boxes, target_bbox_centers, downscale_shape = [], [], [], []
            for a_idx, annotation in enumerate(annotations):
                bbox_centers, boxes, track_idx, class_labels = annotation
                out = self.transform(image=video[a_idx].permute(1, 2, 0).numpy(), keypoints=bbox_centers,
                                     bboxes=boxes, class_labels=class_labels)
                img = out['image']
                target_boxes.append(np.stack(out['bboxes']))
                target_bbox_centers.append(np.stack(out['keypoints']))

                downscale_shape.append((img.shape[0], img.shape[1]))
                rgb_images.append(img)

        heat_mask = [torch.from_numpy(
            generate_position_map(list(d_shape), t_b_c, sigma=self.sigma,
                                  heatmap_shape=self.heatmap_shape,
                                  return_combined=self.return_combined_heatmaps, hw_mode=True))
            for t_b_c, d_shape in zip(target_bbox_centers, downscale_shape)]

        heat_mask = [pad(
            h.unsqueeze(0).unsqueeze(0),
            self.target_pad_value, mode='constant').squeeze(0).squeeze(0) for h in heat_mask]

        video = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in rgb_images])
        heat_mask = torch.stack(heat_mask).unsqueeze(1)

        if self.plot:
            for a_idx, (bbox_centers, boxes, vid, mask, f_num) in \
                    enumerate(zip(target_bbox_centers, target_boxes, video, heat_mask,
                                  self.video_clips.resampling_idxs[video_idx][item])):
                plot_samples(img=vid.permute(1, 2, 0), mask=mask.squeeze(0), boxes=boxes,
                             box_centers=bbox_centers, rgb_boxes=boxes, rgb_box_centers=bbox_centers,
                             plot_boxes=True, additional_text=f'Frame Number: {f_num} | Video Idx: {video_idx}')

        pre_padded_video = video.clone()
        video = pad(video, self.rgb_pad_value, mode=PAD_MODE)
        new_shape = (video.shape[-2], video.shape[-1])
        meta = {'boxes': target_boxes, 'bbox_centers': target_bbox_centers,
                'rgb_boxes': target_boxes, 'rgb_bbox_centers': target_bbox_centers,
                'pre_pad_rgb': pre_padded_video,
                'track_idx': track_idx, 'item': item,
                'original_shape': original_shape, 'new_shape': new_shape,
                'downscale_shape': downscale_shape,
                'video_idx': video_idx}

        return torch.zeros_like(heat_mask), torch.zeros_like(heat_mask), heat_mask,\
               meta, torch.zeros_like(heat_mask), video

    def get_annotation_for_frame(self, item, video_idx, original_shape):
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

        inside_boxes_idx = [b for b, box in enumerate(boxes)
                            if (box[0] > 0 and box[2] < w) and (box[1] > 0 and box[3] < h)]

        boxes = boxes[inside_boxes_idx]
        track_idx = track_idx[inside_boxes_idx]
        bbox_centers = bbox_centers[inside_boxes_idx]

        labels = ['object'] * boxes.shape[0]

        return bbox_centers, boxes, track_idx, labels
