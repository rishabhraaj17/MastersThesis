# Simple clustering from simple NN
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from feature_extractor import MOG2
from constants import FeaturesMode, SDDVideoClasses
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames, FeaturesDataset
from utils import BasicTrainData

initialize_logging()
logger = get_logger(__name__)


def features_collate_fn(batch):
    imgs = torch.zeros(size=(0, batch[0][0].shape[1], batch[0][0].shape[2], batch[0][0].shape[3]))
    # bbox = torch.zeros(size=(0, 5))
    bbox = np.zeros(shape=(0, 5))
    # center = torch.zeros(size=(0, 2))
    center = np.zeros(shape=(0, 2))

    for sample in batch:
        imgs = torch.cat((imgs, sample[0]))
        bbox = np.concatenate((bbox, sample[1]))
        center = np.concatenate((center, sample[2]))
    return imgs, bbox, center


class SimpleModel(pl.LightningModule):
    def __init__(self, original_frame_shape=None, num_frames_to_build_bg_sub_model=12, lr=1e-5,
                 mode=FeaturesMode.UV):
        super(SimpleModel, self).__init__()
        if mode == FeaturesMode.UV:
            in_features = 2
        elif mode == FeaturesMode.XYUV:
            in_features = 4
        else:
            raise Exception
        self.layers = nn.Sequential(nn.Linear(in_features, 8),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(8, 16),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(16, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 64),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(64, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 16),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(16, 8),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(8, in_features))
        self.lr = lr
        self.original_frame_shape = original_frame_shape
        self.num_frames_to_build_bg_sub_model = num_frames_to_build_bg_sub_model
        self.mode = mode

    def forward(self, x):
        # features, cluster_centers, centers = self.preprocess_data(x, self.mode)
        return self.layers(x)

    def _one_step(self, batch):
        # features, cluster_centers, centers = self.preprocess_data(batch, self.mode)
        features1, features2 = batch
        features1, features2 = features1.squeeze().float(), features2.squeeze().float()
        pred = self(features1[:, 2:])
        current_center = self._calculate_mean(features1[:, :2])
        pred_displacement = self._calculate_mean(pred)
        pred_center = current_center + pred_displacement
        # true_center = self._calculate_mean(features2[:, :2])
        # loss = self.center_based_loss(pred_center=pred_center, target_center=true_center)
        loss = self.cluster_center_loss(points=features2[:, :2], pred_center=pred_center)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        tensorboard_logs = {'val_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def preprocess_data(self, data_loader, mode: FeaturesMode, annotation_df, classic_clustering=False):
        # original_shape=None, resize_shape=None):
        x_, y_ = [], []
        for data in tqdm(data_loader):
            # frames, bbox, centers = data  # read frames from loader and frame number and get annotations + centers
            # here
            # frames, bbox, centers = frames.squeeze(), bbox.squeeze(), centers.squeeze()
            frames, frame_numbers = data
            frames = frames.squeeze()
            feature_extractor = MOG2.for_frames()
            features_ = feature_extractor. \
                keyframe_based_clustering_from_frames(frames=frames, n=None, use_last_n_to_build_model=True,
                                                      frames_to_build_model=self.num_frames_to_build_bg_sub_model,
                                                      original_shape=data_loader.dataset.original_shape,
                                                      resized_shape=data_loader.dataset.new_scale,
                                                      classic_clustering=classic_clustering,
                                                      object_of_interest_only=False,
                                                      var_threshold=None, track_ids=None,
                                                      all_object_of_interest_only=True,
                                                      frame_numbers=frame_numbers,
                                                      df=annotation_df)
            features, cluster_centers = None, None
            if mode == FeaturesMode.UV:
                # features, cluster_centers = self._process_features(features_, uv=True,
                #                                                    classic_clustering=classic_clustering)
                features = self._process_complex_features(features_, uv=True)
                x, y = self._extract_trainable_features(features)
                x_ += x
                y_ += y
            if mode == FeaturesMode.XYUV:
                return NotImplemented
                # features, cluster_centers = self._process_features(features_, uv=False,
                #                                                    classic_clustering=classic_clustering)
        return x_, y_

    def _process_features(self, features_dict: dict, uv=True, classic_clustering=False):
        if uv:
            f = 2
        else:
            f = 4
        features = np.zeros(shape=(0, f))
        cluster_centers = np.zeros(shape=(0, 2))
        if classic_clustering:
            for key, value in features_dict.items():
                features = np.concatenate((value['data'], features))
                cluster_centers = np.concatenate((value['cluster_center'][:, :2], cluster_centers))
        else:
            for key, value in features_dict.items():
                features = np.concatenate((value['data'][:, f:], features))
        return features, cluster_centers

    def _process_complex_features(self, features_dict: dict, uv=True):
        if uv:
            f = 2
        else:
            f = 4
        train_data = {}
        per_frame_data = []
        total_frames = len(features_dict)
        for frame in range(total_frames):
            pair_0 = features_dict[frame]
            pair_1 = features_dict[(frame + self.num_frames_to_build_bg_sub_model) % total_frames]
            for i in pair_0:
                for j in pair_1:
                    if i == j:
                        per_frame_data.append(BasicTrainData(frame=frame, track_id=i.track_id,
                                                             pair_0_features=i.features,
                                                             pair_1_features=j.features))
            train_data.update({frame: per_frame_data})
            per_frame_data = []
        return train_data

    def _extract_trainable_features(self, train_data):
        x = []
        y = []
        for key, value in train_data.items():
            for data in value:
                x.append(data.pair_0_features)
                y.append(data.pair_1_features)
        return x, y

    def center_based_loss(self, pred_center, target_center):
        # pred_center = current_center + pred
        return F.mse_loss(pred_center, target_center)

    def cluster_center_loss(self, points, pred_center):
        loss = 0
        for point in points:
            loss += F.mse_loss(pred_center, point)
        return loss

    def _calculate_mean(self, points):
        return points.mean(axis=0)

    def train_dataloader(self):
        train_dataset = FeaturesDataset(train_x, train_y, mode=FeaturesMode.UV, preprocess=False)
        return torch.utils.data.DataLoader(train_dataset, 1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(val_dataset, 1)


if __name__ == '__main__':
    compute_features = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        base_path = "../Datasets/SDD/"
        save_base_path = "../Datasets/SDD_Features/"
        vid_label = SDDVideoClasses.LITTLE
        video_number = 0

        net = SimpleModel(num_frames_to_build_bg_sub_model=12)
        save_path = f'{save_base_path}{vid_label.value}/video{video_number}/'

        train_vid_label = SDDVideoClasses.GATES
        train_vid_num = 5
        train_dataset_path = f'{save_base_path}{train_vid_label.value}/video{train_vid_num}/'

        inference_vid_label = SDDVideoClasses.LITTLE
        inference_vid_num = 0
        inference_dataset_path = f'{save_base_path}{inference_vid_label.value}/video{inference_vid_num}/'

        if compute_features:
            sdd_simple = SDDSimpleDataset(root=base_path, video_label=vid_label, frames_per_clip=1, num_workers=8,
                                          num_videos=1, video_number_to_use=video_number,
                                          step_between_clips=1, transform=resize_frames, scale=0.5, frame_rate=30,
                                          single_track_mode=False, track_id=5, multiple_videos=True)
            sdd_loader = torch.utils.data.DataLoader(sdd_simple, 256)

            logger.info('Computing Features')
            x, y = net.preprocess_data(sdd_loader, FeaturesMode.UV, annotation_df=sdd_simple.annotations_df)

            logger.info(f'Saving the features for video {vid_label.value}, video {video_number}')
            if save_path:
                Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save({'x': x, 'y': y}, save_path+'features.pt')

        logger.info('Setting up DataLoaders')
        train_feats = torch.load(train_dataset_path + 'features.pt')
        train_x, train_y = train_feats['x'], train_feats['y']

        inference_feats = torch.load(inference_dataset_path + 'features.pt')
        inference_x, inference_y = inference_feats['x'], inference_feats['y']

        # Split Val and Test
        split_percent = 0.2
        val_length = int(split_percent * len(inference_x))
        test_length = len(inference_x) - val_length

        inference_dataset = FeaturesDataset(inference_x, inference_y, mode=FeaturesMode.UV, preprocess=False)
        val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset, [val_length, test_length])

        logger.info('Setting up network')

        trainer = pl.Trainer(gpus=1, max_epochs=20)

        logger.info('Starting training')
        trainer.fit(net)
