# Simple clustering from simple NN
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extractor import MOG2
from constants import FeaturesMode, SDDVideoClasses
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames
from utils import BasicTrainData


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
        features, cluster_centers, centers = self.preprocess_data(x, self.mode)
        return self.layers(features)

    def _one_step(self, batch):
        features, cluster_centers, centers = self.preprocess_data(batch, self.mode)
        pred = self(features)
        loss = 'None'  # fixme
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def preprocess_data(self, data, mode: FeaturesMode, classic_clustering=False):
        frames, bbox, centers = data
        frames, bbox, centers = frames.squeeze(), bbox.squeeze(), centers.squeeze()
        feature_extractor = MOG2.for_frames()
        features_ = feature_extractor. \
            keyframe_based_clustering_from_frames(frames=frames, n=None, use_last_n_to_build_model=True,
                                                  frames_to_build_model=self.num_frames_to_build_bg_sub_model,
                                                  original_shape=self.original_frame_shape, annotation=bbox,
                                                  classic_clustering=classic_clustering, object_of_interest_only=False,
                                                  var_threshold=None, track_ids=None,
                                                  all_object_of_interest_only=True)
        features, cluster_centers = None, None
        if mode == FeaturesMode.UV:
            # features, cluster_centers = self._process_features(features_, uv=True,
            #                                                    classic_clustering=classic_clustering)
            features = self._process_complex_features(features_, uv=True)
            x, y = self._extract_trainable_features(features)
        if mode == FeaturesMode.XYUV:
            features, cluster_centers = self._process_features(features_, uv=False,
                                                               classic_clustering=classic_clustering)
        return features, cluster_centers, centers

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


    def center_based_loss(self, pred, current_center):
        pred_center = current_center + pred
        return F.mse_loss(pred_center, current_center)

    def cluster_center_loss(self, points, pred, current_center):
        loss = 0
        pred_center = current_center + pred
        for point in points:
            loss += F.mse_loss(pred_center, point)
        return loss

    def _calculate_center(self, points):
        return points.mean(axis=0)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        base_path = "../Datasets/SDD/"
        vid_label = SDDVideoClasses.LITTLE
        sdd_simple = SDDSimpleDataset(root=base_path, video_label=vid_label, frames_per_clip=1, num_workers=8,
                                      num_videos=1,
                                      step_between_clips=1, transform=resize_frames, scale=0.5, frame_rate=30,
                                      single_track_mode=False, track_id=5)
        sdd_loader = torch.utils.data.DataLoader(sdd_simple, 36)
        net = SimpleModel(num_frames_to_build_bg_sub_model=10)

        trainer = pl.Trainer(max_epochs=5)
        trainer.fit(net, sdd_loader, sdd_loader)
