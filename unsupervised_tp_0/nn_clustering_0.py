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


class AgentFeatures(object):
    def __init__(self, features, track_id, frame_number):
        super(AgentFeatures, self).__init__()
        self.frame_number = frame_number
        self.features = features
        self.track_id = track_id

    def __repr__(self):
        print(f"Frame: {self.frame_number}, Track ID: {self.track_id}, Features: {self.features.shape}")


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
        features_dict = feature_extractor. \
            keyframe_based_clustering_from_frames(frames=frames, n=None, use_last_n_to_build_model=True,
                                                  frames_to_build_model=self.num_frames_to_build_bg_sub_model,
                                                  original_shape=self.original_frame_shape, annotation=bbox,
                                                  classic_clustering=classic_clustering, object_of_interest_only=False,
                                                  var_threshold=None)
        features, cluster_centers = None, None
        if mode == FeaturesMode.UV:
            features, cluster_centers = self._process_features(features_dict, uv=True,
                                                               classic_clustering=classic_clustering)
        if mode == FeaturesMode.XYUV:
            features, cluster_centers = self._process_features(features_dict, uv=False,
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
