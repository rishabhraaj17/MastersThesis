import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import SAVE_BASE_PATH, \
    DATASET_META, \
    GENERATED_DATASET_ROOT, BASE_PATH
from baselinev2.constants import NetworkMode
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.plot_utils import plot_trajectories_with_frame
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.nn.dataset')


class BaselineDataset(Dataset):
    def __init__(self, video_class: SDDVideoClasses, video_number: int, split: NetworkMode,
                 meta_label: SDDVideoDatasets, root: str = SAVE_BASE_PATH,
                 observation_length: int = 8, prediction_length: int = 12):
        super(BaselineDataset, self).__init__()
        try:
            self.ratio = float(DATASET_META.get_meta(meta_label, video_number)[0]['Ratio'].to_numpy()[0])
        except IndexError:
            # Homography not known!
            self.ratio = 1
        path_to_dataset = f'{root}{video_class.value}/video{video_number}/splits/'

        self.tracks = np.load(f'{path_to_dataset}{split.value}_tracks.npy', allow_pickle=True, mmap_mode='r+')
        self.relative_distances = np.load(f'{path_to_dataset}{split.value}_distances.npy', allow_pickle=True,
                                          mmap_mode='r+')

        self.prediction_length = prediction_length
        self.observation_length = observation_length

    def __len__(self):
        return len(self.relative_distances)

    def __getitem__(self, item):
        tracks, relative_distances = torch.from_numpy(self.tracks[item]), \
                                     torch.from_numpy(self.relative_distances[item])
        in_xy = tracks[..., :self.observation_length, -2:]
        gt_xy = tracks[..., self.observation_length:, -2:]
        in_velocities = relative_distances[..., :self.observation_length - 1, :] / 0.4
        gt_velocities = relative_distances[..., self.observation_length:, :] / 0.4
        in_track_ids = tracks[..., :self.observation_length, 0].int()
        gt_track_ids = tracks[..., self.observation_length:, 0].int()
        in_frame_numbers = tracks[..., :self.observation_length, 5].int()
        gt_frame_numbers = tracks[..., self.observation_length:, 5].int()

        return in_xy, gt_xy, in_velocities, gt_velocities, in_track_ids, gt_track_ids, in_frame_numbers, \
               gt_frame_numbers, self.ratio


class BaselineGeneratedDataset(Dataset):
    def __init__(self, video_class: SDDVideoClasses, video_number: int, split: NetworkMode,
                 meta_label: SDDVideoDatasets, root: str = GENERATED_DATASET_ROOT,
                 observation_length: int = 8, prediction_length: int = 12):
        super(BaselineGeneratedDataset, self).__init__()
        try:
            self.ratio = float(DATASET_META.get_meta(meta_label, video_number)[0]['Ratio'].to_numpy()[0])
        except IndexError:
            # Homography not known!
            self.ratio = 1
        path_to_dataset = f'{root}{video_class.value}{video_number}/splits/'

        self.tracks = np.load(f'{path_to_dataset}{split.value}_tracks.npy', allow_pickle=True, mmap_mode='r+')
        self.relative_distances = np.load(f'{path_to_dataset}{split.value}_distances.npy', allow_pickle=True,
                                          mmap_mode='r+')

        self.prediction_length = prediction_length
        self.observation_length = observation_length

    def __len__(self):
        return len(self.relative_distances)

    def __getitem__(self, item):
        tracks, relative_distances = torch.from_numpy(self.tracks[item]), \
                                     torch.from_numpy(self.relative_distances[item])
        in_xy = tracks[:self.observation_length, 6:8]
        gt_xy = tracks[self.observation_length:, 6:8]
        in_velocities = relative_distances[:self.observation_length - 1, :] / 0.4
        gt_velocities = relative_distances[self.observation_length:, :] / 0.4
        in_track_ids = tracks[:self.observation_length, 0].int()
        gt_track_ids = tracks[self.observation_length:, 0].int()
        in_frame_numbers = tracks[:self.observation_length, 5].int()
        gt_frame_numbers = tracks[self.observation_length:, 5].int()
        mapped_in_xy = tracks[:self.observation_length, -2:]
        mapped_gt_xy = tracks[self.observation_length:, -2:]

        return in_xy, gt_xy, in_velocities, gt_velocities, in_track_ids, gt_track_ids, in_frame_numbers, \
               gt_frame_numbers, mapped_in_xy, mapped_gt_xy, self.ratio


if __name__ == '__main__':
    video_class = SDDVideoClasses.LITTLE
    video_num = 3
    meta_label = SDDVideoDatasets.LITTLE
    d1 = BaselineGeneratedDataset(video_class, video_num, NetworkMode.TRAIN, meta_label=meta_label)
    # d2 = BaselineDataset(SDDVideoClasses.COUPA, 0, NetworkMode.TRAIN, meta_label=SDDVideoDatasets.COUPA)
    # d = ConcatDataset([d1, d2])
    d = ConcatDataset([d1])
    loader = DataLoader(d, batch_size=32, shuffle=True)
    # iterator = iter(loader)
    for data in loader:
        # in_xy, gt_xy, _, _, in_track_ids, _, in_frame_numbers, _, _ = data  # next(iterator)
        in_xy, gt_xy, _, _, in_track_ids, _, in_frame_numbers, _, _, mapped_gt_xy, _ = data  # next(iterator)
        frame_num = in_frame_numbers[0, 0].item()
        track_id = in_track_ids[0, 0].item()
        obs_trajectory = np.stack(in_xy[0].cpu().numpy())
        true_trajectory = np.stack(gt_xy[0].cpu().numpy())
        pred_trajectory = np.stack(mapped_gt_xy[0].cpu().numpy()) \
            if not torch.isnan(mapped_gt_xy[0]).any().item() else np.zeros((0, 2))
        # current_frame = extract_frame_from_video(VIDEO_PATH, frame_number=frame_num)
        current_frame = extract_frame_from_video(f'{BASE_PATH}videos/{video_class.value}/video{video_num}/video.mov',
                                                 frame_number=frame_num)
        plot_trajectories_with_frame(current_frame, obs_trajectory, true_trajectory, pred_trajectory,
                                     frame_number=frame_num, track_id=track_id)
        print()

