from typing import Optional, Any, List, Tuple, Union

import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

from average_image.constants import SDDVideoClasses
from bbox_utils import get_frame_annotations, scale_annotations, get_frame_by_track_annotations
from utils import SDDMeta, plot_with_centers, object_of_interest_mask

import warnings


def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


def resize_frames(frame, frame_annotation, size: Optional[Union[Tuple, int]] = None, scale: Optional[float] = None):
    frame = frame.float() / 255.0
    original_shape = frame.shape[2], frame.shape[3]
    if size is not None:
        frame = F.interpolate(frame, size=size, mode='bilinear', align_corners=False,
                              recompute_scale_factor=False)
    if scale is not None:
        frame = F.interpolate(frame, scale_factor=scale, mode='bilinear', align_corners=False,
                              recompute_scale_factor=False)
    new_shape = frame.shape[2], frame.shape[3]
    # track_id = None
    frame_annotation, frame_centers = scale_annotations(frame_annotation, original_scale=original_shape,
                                                        new_scale=new_shape, return_track_id=False,
                                                        tracks_with_annotations=True)
    return frame, frame_annotation, frame_centers


class SDDDatasetBuilder(VisionDataset):

    def __init__(self, root: str, video_label: SDDVideoClasses, frames_per_clip: int, step_between_clips: int = 1,
                 frame_rate: Optional[float] = None, fold: int = 1, train: bool = True, transform: Any = None,
                 _precomputed_metadata: bool = None, num_workers: int = 1, _video_width: int = 0, _video_height: int = 0
                 , _video_min_dimension: int = 0, _audio_samples: int = 0):
        super(SDDDatasetBuilder, self).__init__(root=root)
        _mid_path = video_label.value
        video_path = root + "videos/" + _mid_path
        annotation_path = root + "annotations/" + _mid_path
        video_extensions = ('mov',)
        annotation_extension = ('csv',)

        self.train = train
        self.fold = fold

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
        video_clips = VideoClips(video_list,
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
        self.train_indices = video_list_idx[:-1]
        self.val_indices = [video_list_idx[-1]]
        # self.indices = self._select_fold(video_list, annotation_path, fold, train)
        # self.video_clips = video_clips.subset(self.indices)
        self.video_clips = video_clips
        self.transform = transform

        meta_file = 'H_SDD.txt'
        self.sdd_meta = SDDMeta(root + meta_file)

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return NotImplemented()

    def __getitem__(self, idx):
        return NotImplemented()


class SDDBaseDataset(Dataset):
    def __init__(self, video_clips, samples, transform, indices):
        super(SDDBaseDataset, self).__init__()
        self.video_clips = video_clips.subset(indices)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, item):
        video, audio, info, video_idx = self.video_clips.get_clip(item)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video.permute(0, 3, 1, 2), audio, label


class SDDTrainDataset(SDDBaseDataset):
    def __init__(self, video_clips, samples, transform, indices):
        super(SDDTrainDataset, self).__init__(video_clips, samples, transform, indices)


class SDDValidationDataset(SDDBaseDataset):
    def __init__(self, video_clips, samples, transform, indices):
        super(SDDValidationDataset, self).__init__(video_clips, samples, transform, indices)


class SDDSimpleDataset(Dataset):
    def __init__(self, root: str, video_label: SDDVideoClasses, frames_per_clip: int, num_videos=None, step_factor=None,
                 step_between_clips: int = 1, frame_rate: Optional[float] = None, fold: int = 1, train: bool = True,
                 transform: Any = None, _precomputed_metadata: bool = None, num_workers: int = 1, _video_width: int = 0,
                 _video_height: int = 0, _video_min_dimension: int = 0, _audio_samples: int = 0, scale: float = 1.0,
                 single_track_mode: bool = False, track_id: int = 0):
        _mid_path = video_label.value
        video_path = root + "videos/" + _mid_path
        annotation_path = root + "annotations/" + _mid_path
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

        # restricted to number of videos
        video_list_subset = self.video_list[:num_videos]
        video_list_idx_subset = self.video_list_idx[:num_videos]

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

        annotation_path = self.annotation_list[video_list_idx_subset[0]]
        self.annotations_df = self._read_annotation_file(annotation_path)
        # todo: make it work for more than one video

        self.scale = scale
        self.single_track_mode = single_track_mode
        self.track_id = track_id

        # self.video_frames = video_frames
        # self.selected_frames = [i for i in range(self.video_frames.shape[0]) if not i % step]
        # self.video_frames = self.video_frames[self.selected_frames]
        # self.annotation_path = annotations_path

    @property
    def metadata(self):
        return self.video_clips.metadata

    @staticmethod
    def _read_annotation_file(path):
        dff = pd.read_csv(path)
        dff = dff.drop(dff.columns[[0]], axis=1)
        return dff

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, item):
        video, audio, info, video_idx = self.video_clips.get_clip(item)
        if self.single_track_mode:
            label = get_frame_by_track_annotations(self.annotations_df, item, track_id=self.track_id)
        else:
            label = get_frame_annotations(self.annotations_df, item)
        video = video.permute(0, 3, 1, 2)

        centers = None
        track_ids = None
        if self.transform is not None:
            video, label, centers = self.transform(video, label, scale=self.scale)

        return video, label, centers


class FeaturesDataset(Dataset):
    def __init__(self, x, y):
        super(FeaturesDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x) or len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class SDDDataset(VisionDataset):

    def __init__(self, root: str, video_label: SDDVideoClasses, frames_per_clip: int, step_between_clips: int = 1,
                 frame_rate: Optional[float] = None, fold: int = 1, train: bool = True, transform: Any = None,
                 _precomputed_metadata: bool = None, num_workers: int = 1, _video_width: int = 0, _video_height: int = 0
                 , _video_min_dimension: int = 0, _audio_samples: int = 0):
        super(SDDDataset, self).__init__(root=root)
        _mid_path = video_label.value
        video_path = root + "videos/" + _mid_path
        annotation_path = root + "annotations/" + _mid_path
        extensions = ('mov',)

        self.train = train
        self.fold = fold

        classes = list(sorted(list_dir(video_path),
                              key=lambda x: int(x[-1]) if len(x) == 6 else int(x[-2:])))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(directory=video_path, class_to_idx=class_to_idx,
                                    extensions=('mov',))
        video_list = [x[0] for x in self.samples]
        video_list_idx = [x[1] for x in self.samples]
        video_clips = VideoClips(video_list,
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
        self.train_indices = video_list_idx[:-1]
        self.val_indices = video_list_idx[-1]
        # self.indices = self._select_fold(video_list, annotation_path, fold, train)
        # self.video_clips = video_clips.subset(self.indices)
        if train:
            self.video_clips = video_clips.subset(self.train_indices)
        else:
            self.video_clips = video_clips.subset(self.val_indices)
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_path = "../Datasets/SDD/"
        vid_label = SDDVideoClasses.QUAD

        # sdd = SDDDatasetBuilder(root=base_path, video_label=vid_label, frames_per_clip=3, num_workers=0)
        # sdd_train = SDDTrainDataset(sdd.video_clips, sdd.samples, sdd.transform, sdd.train_indices)
        # sdd_val = SDDValidationDataset(sdd.video_clips, sdd.samples, sdd.transform, sdd.val_indices)
        # print(sdd_train.__getitem__(0))

        idx = 0
        # sdd = SDDDatasetBuilder(root=base_path, video_label=vid_label, frames_per_clip=3, num_workers=8)
        # base_dataset = SDDBaseDataset(sdd.video_clips, sdd.samples, sdd.transform, [0])  # sdd.train_indices)
        # frames = torch.randn((300, 3, 720, 640))
        sdd_simple = SDDSimpleDataset(root=base_path, video_label=vid_label, frames_per_clip=1, num_workers=8,
                                      num_videos=1,
                                      step_between_clips=1, transform=resize_frames, scale=0.5, frame_rate=30,
                                      single_track_mode=True)
        sdd_loader = torch.utils.data.DataLoader(sdd_simple, 4)
        sdd_itr = iter(sdd_loader)
        a, bbox, centers = next(sdd_itr)
        a = a.squeeze()
        print(a.size())

        # plot_with_centers(a, bbox, centers)
        object_of_interest_mask(a[0].numpy(), bbox[0])

        # from torchvision.utils import make_grid, save_image
        # save_image(a, 'test_img.png', 2, padding=5)

        # a, _, _ = next(sdd_itr)
        # a = a.squeeze()
        # print(a.size())
