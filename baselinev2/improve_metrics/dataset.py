import os
from typing import Optional, Any, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from omegaconf import ListConfig
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from average_image.utils import SDDMeta
from baselinev2.config import BASE_PATH
from baselinev2.improve_metrics.crop_utils import patches_and_labels
from unsupervised_tp_0.dataset import resize_frames, sort_list


def people_collate_fn(batch):
    gt_patches, fp_patches = [], []
    gt_labels, fp_labels = [], []
    for ele in batch:
        gt, fp = ele
        gt_patches.append(gt['patches'])
        fp_patches.append(fp['patches'])

        gt_labels.append(gt['labels'])
        fp_labels.append(fp['labels'])

    gt_patches = torch.cat(gt_patches, dim=0)
    fp_patches = torch.cat(fp_patches, dim=0)

    gt_labels = torch.cat(gt_labels)
    fp_labels = torch.cat(fp_labels)

    return [{'patches': gt_patches, 'labels': gt_labels}, {'patches': fp_patches, 'labels': fp_labels}]


class SDDDatasetV0(Dataset):
    def __init__(self, root: str, video_label: SDDVideoClasses, frames_per_clip: int, num_videos=None, step_factor=None,
                 step_between_clips: int = 1, frame_rate: Optional[float] = None, fold: int = 1, train: bool = True,
                 transform: Any = None, _precomputed_metadata: bool = None, num_workers: int = 1, _video_width: int = 0,
                 _video_height: int = 0, _video_min_dimension: int = 0, _audio_samples: int = 0, scale: float = 1.0,
                 single_track_mode: bool = False, track_id: int = 0, video_number_to_use: int = 0,
                 multiple_videos: bool = False, use_generated: bool = False):
        _mid_path = video_label.value
        _annotation_decider = "generated_annotations/" if use_generated else "annotations/"
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
        if multiple_videos and num_videos == -1:
            annotation_path = self.annotation_list
            self.annotations_df = [self._read_annotation_file(p, self.use_generated) for p in annotation_path]

            video_list_subset = self.video_list
            video_list_idx_subset = self.video_list_idx

            ref_image_path = [os.path.split(p)[0] + '/reference.jpg' for p in annotation_path]
            ref_image = [torchvision.io.read_image(r) for r in ref_image_path]
            self.original_shape = [[r.shape[1], r.shape[2]] for r in ref_image]
        if multiple_videos and (isinstance(num_videos, list) or isinstance(num_videos, ListConfig)):
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
        # self.train_indices = self.video_list_idx[:-1]
        # self.val_indices = [self.video_list_idx[-1]]
        # self.indices = self._select_fold(video_list, annotation_path, fold, train)
        # self.video_clips = video_clips.subset(self.indices)
        self.video_clips = video_clips
        self.transform = transform

        meta_file = 'H_SDD.txt'
        self.sdd_meta = SDDMeta(root + meta_file)

        self.scale = scale
        # self.single_track_mode = single_track_mode
        # self.track_id = track_id

        self.new_scale = None
        self.use_generated = use_generated

        # self.video_frames = video_frames
        # self.selected_frames = [i for i in range(self.video_frames.shape[0]) if not i % step]
        # self.video_frames = self.video_frames[self.selected_frames]
        # self.annotation_path = annotations_path

    @property
    def metadata(self):
        return self.video_clips.metadata

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
        # Dont read annotation here as stacking is not possible when number of objects differ
        # if self.single_track_mode:
        #     label = get_frame_by_track_annotations(self.annotations_df, item, track_id=self.track_id)
        # else:
        #     label = get_frame_annotations(self.annotations_df, item)
        video = video.permute(0, 3, 1, 2)

        # centers = None
        label = None
        new_scale = None
        original_shape = None
        # track_ids = None
        if self.transform is not None:
            # video, label, centers = self.transform(video, label, scale=self.scale)
            video, original_shape, new_scale = self.transform(video, label, scale=self.scale)
            if self.original_shape is None or self.new_scale is None:
                self.original_shape = original_shape
                self.new_scale = new_scale

        return video, item, video_idx


class PatchesDataset(SDDDatasetV0):
    def __init__(self, root: str, video_label: SDDVideoClasses, frames_per_clip: int, num_videos=None, step_factor=None,
                 step_between_clips: int = 1, frame_rate: Optional[float] = None, fold: int = 1, train: bool = True,
                 transform: Any = None, _precomputed_metadata: bool = None, num_workers: int = 1, _video_width: int = 0,
                 _video_height: int = 0, _video_min_dimension: int = 0, _audio_samples: int = 0, scale: float = 1.0,
                 single_track_mode: bool = False, track_id: int = 0, video_number_to_use: int = 0,
                 multiple_videos: bool = False, bounding_box_size: Union[int, Tuple[int]] = 50,
                 num_patches: Optional[int] = None, use_generated: bool = False,
                 radius_elimination: Optional[int] = 100, merge_annotations: bool = False, plot: bool = False):
        super().__init__(root, video_label, frames_per_clip, num_videos, step_factor, step_between_clips, frame_rate,
                         fold, train, transform, _precomputed_metadata, num_workers, _video_width, _video_height,
                         _video_min_dimension, _audio_samples, scale, single_track_mode, track_id, video_number_to_use,
                         multiple_videos, use_generated=use_generated)
        self.bounding_box_size = bounding_box_size
        self.num_patches = num_patches
        self.radius_elimination = radius_elimination
        self.merge_annotations = merge_annotations
        self.multiple_videos = multiple_videos
        self.num_videos = num_videos
        self.plot = plot

        if merge_annotations and multiple_videos and num_videos == -1:
            frame_counts = [d.frame.max() for d in self.annotations_df]
            frame_counts_cumsum = np.cumsum(frame_counts)

            frame_adjusted_dfs = [self.annotations_df[0]]
            for idx in range(1, len(self.annotations_df)):
                temp = self.annotations_df[idx]
                temp.frame += frame_counts_cumsum[idx - 1] + 1
                frame_adjusted_dfs.append(temp)

            self.merged_annotations = pd.concat(frame_adjusted_dfs)

    def __getitem__(self, item):
        frames, frame_numbers, video_idx = super(PatchesDataset, self).__getitem__(item=item)
        gt_patches_and_labels, fp_patches_and_labels = patches_and_labels(
            image=frames.squeeze(0),
            bounding_box_size=self.bounding_box_size,
            annotations=self.merged_annotations
            if self.merge_annotations and self.multiple_videos and self.num_videos == -1
            else self.annotations_df[video_idx],
            frame_number=frame_numbers,
            num_patches=self.num_patches,
            new_shape=self.new_scale,
            use_generated=self.use_generated, plot=self.plot,
            radius_elimination=self.radius_elimination)
        return gt_patches_and_labels, fp_patches_and_labels


if __name__ == '__main__':
    video_class = SDDVideoClasses.LITTLE
    video_number = 0
    n_workers = 12

    dataset = PatchesDataset(root=BASE_PATH, video_label=video_class, frames_per_clip=1,
                             num_workers=n_workers, num_videos=-1, video_number_to_use=video_number,
                             step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                             single_track_mode=False, track_id=5, multiple_videos=True,
                             use_generated=False, plot=True, merge_annotations=True)
    loader = DataLoader(dataset, batch_size=8, num_workers=n_workers, shuffle=False, collate_fn=people_collate_fn)
    for data in loader:
        print()

    # import shutil
    # video_clazzes = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
    #                  SDDVideoClasses.GATES,
    #                  SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS, SDDVideoClasses.QUAD]
    # video_numbers = [[i for i in range(7)], [i for i in range(4)], [i for i in range(5)],
    #                  [i for i in range(9)], [i for i in range(15)], [i for i in range(4)],
    #                  [i for i in range(12)], [i for i in range(4)]]
    # d_root = '../Plots/baseline_v2/v0/'
    # gen_d_root = '../Datasets/SDD/generated_annotations/'
    # for idx, v_clz in tqdm(enumerate(video_clazzes)):
    #     for v_num in video_numbers[idx]:
    #         d_path = f'{d_root}{v_clz.value}{v_num}/csv_annotation/generated_annotations.csv'
    #         gen_d_path = f'{gen_d_root}{v_clz.value}/video{v_num}/'
    #         current_files = os.listdir(gen_d_path)
    #         current_files.remove('reference.jpg')
    #         for f in current_files:
    #             os.remove(os.path.join(gen_d_path, f))
    #         shutil.copyfile(d_path, gen_d_path + 'generated_annotations.csv')

