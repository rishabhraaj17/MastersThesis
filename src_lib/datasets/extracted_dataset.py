from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import ConcatenateDataset

DATASET_ROOT = '../../Datasets/SDD_Features/'
EXTRACTED_DATASET_ROOT = '../../Datasets/SDD/pm_extracted_annotations/'
META_PATH = '../../Datasets/SDD/H_SDD.txt'


def extracted_collate(batch):
    in_dxdy, in_xy = [], []
    gt_dxdy, gt_xy = [], []
    in_frame_numbers, gt_frame_numbers = [], []
    in_track_ids, gt_track_ids = [], []
    non_linear_ped, loss_mask = [], []
    seq_start_end, ratio = [], []
    dataset_idx = []

    for b in batch:
        dataset_idx.append(b[1])
        b = b[0]
        in_xy.append(b['in_xy'])
        in_dxdy.append(b['in_dxdy'])
        gt_xy.append(b['gt_xy'])
        gt_dxdy.append(b['gt_dxdy'])
        in_frame_numbers.append(b['in_frames'])
        gt_frame_numbers.append(b['gt_frames'])
        in_track_ids.append(b['in_tracks'])
        gt_track_ids.append(b['gt_tracks'])
        non_linear_ped.append(b['non_linear_ped'])
        loss_mask.append(b['loss_mask'])
        seq_start_end.append(b['seq_start_end'])
        ratio.append(b['ratio'])

    return {
        'in_xy': torch.stack(in_xy).transpose(1, 0),
        'in_dxdy': torch.stack(in_dxdy).transpose(1, 0),
        'gt_xy': torch.stack(gt_xy).transpose(1, 0),
        'gt_dxdy': torch.stack(gt_dxdy).transpose(1, 0),
        'in_frames': torch.stack(in_frame_numbers).transpose(1, 0),
        'gt_frames': torch.stack(gt_frame_numbers).transpose(1, 0),
        'in_tracks': torch.stack(in_track_ids).transpose(1, 0),
        'gt_tracks': torch.stack(gt_track_ids).transpose(1, 0),
        'non_linear_ped': non_linear_ped,
        'loss_mask': loss_mask,
        'seq_start_end': seq_start_end,
        'ratio': torch.tensor(ratio),
        'dataset_idx': torch.LongTensor(dataset_idx)
    }


class BaselineGTDataset(Dataset):
    def __init__(self, video_class: SDDVideoClasses, video_number: int, split: NetworkMode,
                 meta_label: SDDVideoDatasets, root: str = DATASET_ROOT,
                 observation_length: int = 8, prediction_length: int = 12,
                 relative_velocities: bool = False, split_name: str = 'splits_v1', meta: str = META_PATH,
                 return_as_dict: bool = True):
        super(BaselineGTDataset, self).__init__()
        self.meta = SDDMeta(meta)
        try:
            self.ratio = float(self.meta.get_meta(meta_label, video_number)[0]['Ratio'].to_numpy()[0])
        except IndexError:
            # Homography not known!
            self.ratio = 1
        path_to_dataset = f'{root}{video_class.value}/video{video_number}/{split_name}/'

        self.tracks = np.load(f'{path_to_dataset}{split.value}_tracks.npy', allow_pickle=True, mmap_mode='r+')
        self.relative_distances = np.load(f'{path_to_dataset}{split.value}_distances.npy', allow_pickle=True,
                                          mmap_mode='r+')

        self.prediction_length = prediction_length
        self.observation_length = observation_length
        self.relative_velocities = relative_velocities

        self.video_class = video_class
        self.video_number = video_number
        self.meta_label = meta_label
        self.split = split
        self.return_as_dict = return_as_dict

    def __len__(self):
        return len(self.relative_distances)

    def __getitem__(self, item):
        tracks, relative_distances = torch.from_numpy(self.tracks[item]), \
                                     torch.from_numpy(self.relative_distances[item])
        in_xy = tracks[..., :self.observation_length, -2:]
        gt_xy = tracks[..., self.observation_length:, -2:]
        in_velocities = relative_distances[..., :self.observation_length - 1, :] / 0.4 \
            if self.relative_velocities else relative_distances[..., :self.observation_length - 1, :]
        gt_velocities = relative_distances[..., self.observation_length - 1:, :] / 0.4 \
            if self.relative_velocities else relative_distances[..., self.observation_length - 1:, :]
        in_track_ids = tracks[..., :self.observation_length, 0].int()
        gt_track_ids = tracks[..., self.observation_length:, 0].int()
        in_frame_numbers = tracks[..., :self.observation_length, 5].int()
        gt_frame_numbers = tracks[..., self.observation_length:, 5].int()

        if self.return_as_dict:
            return {
                'in_xy': in_xy, 'in_dxdy': in_velocities,
                'gt_xy': gt_xy, 'gt_dxdy': gt_velocities,
                'in_frames': in_frame_numbers, 'gt_frames': gt_frame_numbers,
                'in_tracks': in_track_ids, 'gt_tracks': gt_track_ids,
                'non_linear_ped': [], 'loss_mask': [],
                'seq_start_end': [], 'ratio': self.ratio
            }
        return in_xy, gt_xy, in_velocities, gt_velocities, in_track_ids, gt_track_ids, in_frame_numbers, \
               gt_frame_numbers, self.ratio


class BaselineExtractedDataset(Dataset):
    def __init__(self, video_class: SDDVideoClasses, video_number: int, split: NetworkMode,
                 meta_label: SDDVideoDatasets, root: str = EXTRACTED_DATASET_ROOT,
                 observation_length: int = 8, prediction_length: int = 12, relative_velocities: bool = False,
                 split_name: str = 'v0', meta: str = META_PATH, return_as_dict: bool = True):
        super(BaselineExtractedDataset, self).__init__()
        self.meta = SDDMeta(meta)
        try:
            self.ratio = float(self.meta.get_meta(meta_label, video_number)[0]['Ratio'].to_numpy()[0])
        except IndexError:
            # Homography not known!
            self.ratio = 1
        path_to_dataset = f'{root}{video_class.value}/video{video_number}/{split_name}/'

        self.tracks = np.load(f'{path_to_dataset}{split.value}_tracks.npy', allow_pickle=True, mmap_mode='r+')
        self.relative_distances = np.load(f'{path_to_dataset}{split.value}_distances.npy', allow_pickle=True,
                                          mmap_mode='r+')

        self.prediction_length = prediction_length
        self.observation_length = observation_length
        self.relative_velocities = relative_velocities

        self.video_class = video_class
        self.video_number = video_number
        self.meta_label = meta_label
        self.split = split
        self.return_as_dict = return_as_dict

    def __len__(self):
        return len(self.relative_distances)

    def __getitem__(self, item):
        tracks, relative_distances = torch.from_numpy(self.tracks[item]), \
                                     torch.from_numpy(self.relative_distances[item])
        in_xy = tracks[:self.observation_length, 2:]
        gt_xy = tracks[self.observation_length:, 2:]
        in_velocities = relative_distances[:self.observation_length - 1, :] / 0.4 \
            if self.relative_velocities else relative_distances[:self.observation_length - 1, :]
        gt_velocities = relative_distances[self.observation_length - 1:, :] / 0.4 \
            if self.relative_velocities else relative_distances[self.observation_length - 1:, :]
        in_track_ids = tracks[:self.observation_length, 1].int()
        gt_track_ids = tracks[self.observation_length:, 1].int()
        in_frame_numbers = tracks[:self.observation_length, 0].int()
        gt_frame_numbers = tracks[self.observation_length:, 0].int()

        if self.return_as_dict:
            return {
                'in_xy': in_xy, 'in_dxdy': in_velocities,
                'gt_xy': gt_xy, 'gt_dxdy': gt_velocities,
                'in_frames': in_frame_numbers, 'gt_frames': gt_frame_numbers,
                'in_tracks': in_track_ids, 'gt_tracks': gt_track_ids,
                'non_linear_ped': [], 'loss_mask': [],
                'seq_start_end': [], 'ratio': self.ratio
            }
        return in_xy, gt_xy, in_velocities, gt_velocities, in_track_ids, gt_track_ids, in_frame_numbers, \
               gt_frame_numbers, self.ratio


def get_dataset(video_clazz: SDDVideoClasses, video_number: int, mode: NetworkMode, meta_label: SDDVideoDatasets,
                get_generated: bool = False, split_name: str = 'splits_v1', return_as_dict=True, meta_path=META_PATH,
                root=None):
    return BaselineExtractedDataset(
        video_clazz, video_number, mode, meta_label=meta_label, split_name='v0',
        return_as_dict=return_as_dict, meta=meta_path
        , root=EXTRACTED_DATASET_ROOT if root is None else root) \
        if get_generated \
        else BaselineGTDataset(video_clazz, video_number, mode, meta_label=meta_label, split_name=split_name,
                               return_as_dict=return_as_dict, meta=meta_path,
                               root=DATASET_ROOT if root is None else root)


def get_multiple_datasets(
        video_classes: Sequence[SDDVideoClasses], video_numbers: Sequence[Sequence[int]], mode: NetworkMode,
        meta_label: Sequence[SDDVideoDatasets], get_generated: bool = False,
        split_name: str = 'splits_v1', return_as_dict=True, meta_path=META_PATH,
        root=None):
    datasets = []
    for idx, (v_clz, m) in enumerate(zip(video_classes, meta_label)):
        for v_num in video_numbers[idx]:
            datasets.append(
                get_dataset(video_clazz=v_clz, video_number=v_num, mode=mode, meta_label=m,
                            get_generated=get_generated, split_name=split_name, return_as_dict=return_as_dict,
                            meta_path=meta_path, root=root)
            )
    return datasets


def get_train_datasets(
        video_classes: Sequence[SDDVideoClasses], video_numbers: Sequence[Sequence[int]],
        meta_label: Sequence[SDDVideoDatasets], get_generated: bool = False,
        split_name: str = 'splits_v1', return_as_dict=True, meta_path=META_PATH,
        root=None):
    return ConcatenateDataset(get_multiple_datasets(
        video_classes=video_classes, video_numbers=video_numbers, mode=NetworkMode.TRAIN, meta_label=meta_label,
        get_generated=get_generated, split_name=split_name, return_as_dict=return_as_dict,
        meta_path=meta_path, root=root))


def get_val_datasets(
        video_classes: Sequence[SDDVideoClasses], video_numbers: Sequence[Sequence[int]],
        meta_label: Sequence[SDDVideoDatasets], get_generated: bool = False,
        split_name: str = 'splits_v1', return_as_dict=True, meta_path=META_PATH,
        root=None):
    return ConcatenateDataset(get_multiple_datasets(
        video_classes=video_classes, video_numbers=video_numbers, mode=NetworkMode.VALIDATION, meta_label=meta_label,
        get_generated=get_generated, split_name=split_name, return_as_dict=return_as_dict,
        meta_path=meta_path, root=root))


def get_test_datasets(
        video_classes: Sequence[SDDVideoClasses], video_numbers: Sequence[Sequence[int]],
        meta_label: Sequence[SDDVideoDatasets], get_generated: bool = False,
        split_name: str = 'splits_v1', return_as_dict=True, meta_path=META_PATH,
        root=None):
    return ConcatenateDataset(get_multiple_datasets(
        video_classes=video_classes, video_numbers=video_numbers, mode=NetworkMode.TEST, meta_label=meta_label,
        get_generated=get_generated, split_name=split_name, return_as_dict=return_as_dict,
        meta_path=meta_path, root=root))


def get_train_and_val_datasets(
        video_classes: Sequence[SDDVideoClasses], video_numbers: Sequence[Sequence[int]],
        meta_label: Sequence[SDDVideoDatasets], get_generated: bool = False,
        split_name: str = 'splits_v1', return_as_dict=True, meta_path=META_PATH,
        root=None):
    train_datasets = get_train_datasets(
        video_classes=video_classes, video_numbers=video_numbers, meta_label=meta_label, get_generated=get_generated,
        split_name=split_name, return_as_dict=return_as_dict, meta_path=meta_path, root=root
    )
    val_datasets = get_val_datasets(
        video_classes=video_classes, video_numbers=video_numbers, meta_label=meta_label, get_generated=get_generated,
        split_name=split_name, return_as_dict=return_as_dict, meta_path=meta_path, root=root
    )
    return train_datasets, val_datasets


if __name__ == '__main__':
    dset, vset = get_train_and_val_datasets([SDDVideoClasses.DEATH_CIRCLE], video_numbers=[[0, 1, 2, 3, 4]],
                                            meta_label=[SDDVideoDatasets.DEATH_CIRCLE], get_generated=False)
    loader = DataLoader(dset, batch_size=4, collate_fn=extracted_collate)
    for d in loader:
        print()
