from typing import Optional, Any

import torch
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

from average_image.constants import SDDVideoClasses


class SDDDatasetBuilder(VisionDataset):

    def __init__(self, root: str, video_label: SDDVideoClasses, frames_per_clip: int, step_between_clips: int = 1,
                 frame_rate: Optional[float] = None, fold: int = 1, train: bool = True, transform: Any = None,
                 _precomputed_metadata: bool = None, num_workers: int = 1, _video_width: int = 0, _video_height: int = 0
                 , _video_min_dimension: int = 0, _audio_samples: int = 0):
        super(SDDDatasetBuilder, self).__init__(root=root)
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
        self.val_indices = [video_list_idx[-1]]
        # self.indices = self._select_fold(video_list, annotation_path, fold, train)
        # self.video_clips = video_clips.subset(self.indices)
        self.video_clips = video_clips
        self.transform = transform

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
    def __init__(self, video_frames, fps, step_factor):
        step = fps * step_factor

        self.video_frames = video_frames
        self.selected_frames = [i for i in range(self.video_frames.shape[0]) if not i % step]
        self.video_frames = self.video_frames[self.selected_frames]

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, item):
        return self.video_frames[item]


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
    base_path = "../Datasets/SDD/"
    vid_label = SDDVideoClasses.LITTLE

    # sdd = SDDDatasetBuilder(root=base_path, video_label=vid_label, frames_per_clip=3, num_workers=0)
    # sdd_train = SDDTrainDataset(sdd.video_clips, sdd.samples, sdd.transform, sdd.train_indices)
    # sdd_val = SDDValidationDataset(sdd.video_clips, sdd.samples, sdd.transform, sdd.val_indices)
    # print(sdd_train.__getitem__(0))

    frames = torch.randn((300, 3, 720, 640))
    sdd = SDDSimpleDataset(frames, 30, 0.4)
    sdd_loader = torch.utils.data.DataLoader(sdd, 10)
    a = next(iter(sdd_loader))
    print(a.size())
