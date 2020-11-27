from datetime import datetime
from enum import Enum
import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from average_image.constants import SDDVideoClasses, FeaturesMode, SDDVideoDatasets
from average_image.utils import SDDMeta, compute_ade, compute_fde, plot_bars_if_inside_bbox, \
    plot_trajectory_rnn_compare, plot_trajectory_rnn_compare_side_by_side, is_inside_bbox
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import FeaturesDatasetCenterBased

initialize_logging()
logger = get_logger(__name__)

TIME_STEPS = 20
NUM_WORKERS = 10
BATCH_SIZE = 256
LR = 1e-3
USE_BATCH_NORM = False
GT_BASED = True
CENTER_BASED = True
OF_VERSION = 1
GT_VERSION = 0
OF_EPOCH = None
GT_EPOCH = 88

MANUAL_SEED = 42
torch.manual_seed(MANUAL_SEED)
GENERATOR_SEED = torch.Generator().manual_seed(MANUAL_SEED)

LINEAR_CFG = {
    'encoder': [4, 8, 16],
    'decoder': [32, 16, 8, 4, 2]
}


class NetworkMode(Enum):
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'


def center_dataset_collate(batch):
    center_features_list, frames_batched_list, track_ids_batched_list, bbox_center_x_batched_list = [], [], [], []
    bbox_center_y_batched_list, bbox_x_batched_list, bbox_y_batched_list = [], [], []
    for data in batch:
        center_features = np.zeros(shape=(0, 5, 2))
        frames_batched, track_ids_batched = np.zeros(shape=(0, 1)), np.zeros(shape=(0, 1))
        bbox_center_x_batched, bbox_center_y_batched = np.zeros(shape=(0, 2)), np.zeros(shape=(0, 2))
        bbox_x_batched, bbox_y_batched = np.zeros(shape=(0, 4)), np.zeros(shape=(0, 4))
        for features_center, frames, track_ids, bbox_center_x, bbox_center_y, bbox_x, bbox_y in zip(*data):
            center_features = np.concatenate((center_features, np.expand_dims(features_center, axis=0)))
            frames_batched = np.concatenate((frames_batched, np.expand_dims(frames, axis=(0, 1))))
            track_ids_batched = np.concatenate((track_ids_batched, np.expand_dims(track_ids, axis=(0, 1))))
            bbox_center_x_batched = np.concatenate((bbox_center_x_batched, np.expand_dims(bbox_center_x, axis=0)))
            bbox_center_y_batched = np.concatenate((bbox_center_y_batched, np.expand_dims(bbox_center_y, axis=0)))
            bbox_x_batched = np.concatenate((bbox_x_batched, np.expand_dims(bbox_x, axis=0)))
            bbox_y_batched = np.concatenate((bbox_y_batched, np.expand_dims(bbox_y, axis=0)))
        if center_features.shape[0] == TIME_STEPS:
            center_features_list.append(center_features)
            frames_batched_list.append(frames_batched)
            track_ids_batched_list.append(track_ids_batched)
            bbox_center_x_batched_list.append(bbox_center_x_batched)
            bbox_center_y_batched_list.append(bbox_center_y_batched)
            bbox_x_batched_list.append(bbox_x_batched)
            bbox_y_batched_list.append(bbox_y_batched)

    if len(center_features_list) == 0:
        return None
    center_features_out = np.stack(center_features_list, 1)
    frames_batched_out, track_ids_batched_out = np.stack(frames_batched_list, 1), np.stack(track_ids_batched_list, 1)
    bbox_center_x_batched_out, bbox_center_y_batched_out = np.stack(bbox_center_x_batched_list, 1), \
                                                           np.stack(bbox_center_y_batched_list, 1)
    bbox_x_batched_out, bbox_y_batched_out = np.stack(bbox_x_batched_list, 1), np.stack(bbox_y_batched_list, 1)

    return torch.from_numpy(center_features_out), torch.from_numpy(frames_batched_out), torch.from_numpy(
        track_ids_batched_out), \
           torch.from_numpy(bbox_center_x_batched_out), torch.from_numpy(bbox_center_y_batched_out), \
           torch.from_numpy(bbox_x_batched_out), torch.from_numpy(bbox_y_batched_out)


def make_layers(cfg, batch_norm=False, encoder=True, last_without_activation=True):
    layers = []
    if encoder:
        in_features = 2
    else:
        in_features = 64
    if last_without_activation:
        for v in cfg[:-1]:
            in_features, layers = core_layers_maker(batch_norm, in_features, layers, v)
        layers += [nn.Linear(in_features, cfg[-1])]
    else:
        for v in cfg:
            in_features, layers = core_layers_maker(batch_norm, in_features, layers, v)
    return nn.Sequential(*layers)


def core_layers_maker(batch_norm, in_features, layers, v):
    linear = nn.Linear(in_features, v)
    if batch_norm:
        layers += [linear, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
    else:
        layers += [linear, nn.ReLU(inplace=True)]
    in_features = v
    return in_features, layers


class BaselineSequential(LightningModule):
    """
    Watch 8 time steps, predict next 12
    """

    def __init__(self, meta=None, original_frame_shape=None, num_frames_between_two_time_steps=12, lr=1e-5,
                 mode=FeaturesMode.UV, layers_mode=None, meta_video=None, meta_train_video_number=None,
                 meta_val_video_number=None, time_steps=5, train_dataset=None, val_dataset=None, batch_size=1,
                 num_workers=0, use_batch_norm=False, gt_based=True, center_based=True):
        super(BaselineSequential, self).__init__()

        self.block_1 = make_layers(LINEAR_CFG['encoder'], batch_norm=use_batch_norm, encoder=True,
                                   last_without_activation=False)
        self.rnn_cell = nn.LSTMCell(input_size=16, hidden_size=32)
        self.rnn_cell_1 = nn.LSTMCell(input_size=32, hidden_size=64)
        self.block_2 = make_layers(LINEAR_CFG['decoder'], batch_norm=use_batch_norm, encoder=False,
                                   last_without_activation=True)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.lr = lr
        self.original_frame_shape = original_frame_shape
        self.num_frames_between_two_time_steps = num_frames_between_two_time_steps
        self.mode = mode
        self.meta = meta
        self.train_ratio = float(self.meta.get_meta(meta_video, meta_train_video_number)[0]['Ratio'].to_numpy()[0])
        self.val_ratio = float(self.meta.get_meta(meta_video, meta_val_video_number)[0]['Ratio'].to_numpy()[0])
        self.ts = time_steps
        # self.rnn_mode = True if layers_mode == 'rnn' else False
        self.gt_based = gt_based
        self.center_based = center_based

        self.save_hyperparameters('lr', 'time_steps', 'meta_video', 'batch_size', 'meta_train_video_number',
                                  'meta_val_video_number', 'use_batch_norm', 'gt_based', 'center_based')

    def forward(self, x, cx=None, hx=None, cx_1=None, hx_1=None, stacked=True):
        if stacked:
            block_1 = self.block_1(x)
            hx, cx = self.rnn_cell(block_1, (hx, cx))
            hx = F.relu(hx, inplace=True)
            hx_1, cx_1 = self.rnn_cell_1(hx, (hx_1, cx_1))
            hx_1 = F.relu(hx_1, inplace=True)
            out = self.block_2(hx_1)
        else:
            block_1 = self.block_1(x)
            hx, cx = self.rnn_cell(block_1, (hx, cx))
            out = self.block_2(hx)
        return out, cx, hx, cx_1, hx_1

    def one_step(self, batch):
        features, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch
        b_size = features.shape[1]

        cx, cx_1, hx, hx_1 = self.init_hidden_states(b_size)
        total_loss = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        last_input_velocity, last_pred_center, bad_data = None, None, False
        pred_centers, moved_points_by_true_of_list, actual_points_list = [], [], []
        ade, fde, shifted_points = None, None, None
        steps_to_watch = 8

        criterion = self.setup_loss()

        if len(features) == self.ts:
            for idx_i, i in enumerate(range(self.ts)):
                feat, bb_center1, bb_center2 = features[i].squeeze().float(), bbox_center1[i].float(), \
                                               bbox_center2[i].float()
                center_xy, center_true_uv, center_past_uv, shifted_point, gt_past_velocity = \
                    feat[:, 0, :], feat[:, 1, :], feat[:, 2, :], feat[:, 3, :], feat[:, 4, :]
                if self.gt_based:
                    shifted_points = bb_center2
                if not self.center_based:
                    return NotImplemented

                if idx_i < steps_to_watch:
                    if self.gt_based:
                        last_input_velocity = gt_past_velocity
                        last_pred_center = bb_center1
                    else:
                        last_input_velocity = center_past_uv
                        last_pred_center = center_xy

                    block_2, cx, hx, cx_1, hx_1 = self(last_input_velocity, cx, hx, cx_1, hx_1, True)

                    if not self.gt_based:
                        shifted_points = shifted_point

                    moved_points_by_true_of_list.append(shifted_points)
                    pred_center = last_pred_center + (block_2 * 0.4)
                    if idx_i == steps_to_watch - 1:
                        last_input_velocity = block_2
                        last_pred_center = pred_center
                else:
                    block_2, cx, hx, cx_1, hx_1 = self(last_input_velocity, cx, hx, cx_1, hx_1, True)

                    if not self.gt_based:
                        shifted_points = shifted_point

                    moved_points_by_true_of_list.append(shifted_points)
                    pred_center = last_pred_center + (block_2 * 0.4)
                    last_input_velocity = block_2
                    last_pred_center = pred_center

                pred_centers.append(pred_center)
                actual_points_list.append(bb_center2)

                total_loss += criterion(shifted_points, pred_center=pred_center)

            predicted_points = [p.detach().cpu().numpy() for p in pred_centers]
            true_points = [p.detach().cpu().numpy() for p in actual_points_list]

            ade = compute_ade(np.stack(predicted_points), np.stack(true_points)).item()
            fde = compute_fde(np.stack(predicted_points), np.stack(true_points)).item()

        return total_loss / len(features), ade, fde

    def setup_loss(self):
        if self.gt_based and self.center_based:
            return self.cluster_center_based_loss_meters
        if self.gt_based and not self.center_based:
            return self.cluster_all_points_loss_meters
        if not self.gt_based and self.center_based:
            return self.cluster_center_based_loss_meters
        if not self.gt_based and self.center_based:
            return self.cluster_all_points_loss_meters

    def init_hidden_states(self, b_size):
        hx, cx = torch.zeros(size=(b_size, 32), device=self.device), torch.zeros(size=(b_size, 32), device=self.device)
        hx_1, cx_1 = torch.zeros(size=(b_size, 64), device=self.device), torch.zeros(size=(b_size, 64),
                                                                                     device=self.device)
        torch.nn.init.xavier_normal_(hx)
        torch.nn.init.xavier_normal_(cx)
        torch.nn.init.xavier_normal_(hx_1)
        torch.nn.init.xavier_normal_(cx_1)

        return cx, cx_1, hx, hx_1

    def cluster_all_points_loss_meters(self, points, pred_center):
        if self.training:
            to_m = self.train_ratio
        else:
            to_m = self.val_ratio
        loss = 0
        for point in points:
            loss += self.l2_norm(pred_center, point) * to_m
        return loss / len(points)

    def cluster_center_based_loss_meters(self, gt_center, pred_center):
        if self.training:
            to_m = self.train_ratio
        else:
            to_m = self.val_ratio
        loss = self.l2_norm(pred_center, gt_center) * to_m
        return loss

    @staticmethod
    def l2_norm(point1, point2):
        return torch.norm(point1 - point2, p=2)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=center_dataset_collate,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size * 2, collate_fn=center_dataset_collate,
                          num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        loss, ade, fde = self.one_step(batch)
        if ade is None or fde is None:
            tensorboard_logs = {'train_loss': loss, 'train/ade': 0, 'train/fde': 0}
        else:
            tensorboard_logs = {'train_loss': loss, 'train/ade': ade * self.train_ratio,
                                'train/fde': fde * self.train_ratio}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, ade, fde = self.one_step(batch)
        if ade is None or fde is None:
            tensorboard_logs = {'val_loss': loss, 'val/ade': 0, 'val/fde': 0}
        else:
            tensorboard_logs = {'val_loss': loss, 'val/ade': ade * self.val_ratio, 'val/fde': fde * self.val_ratio}

        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(opt, patience=10, verbose=True, factor=0.1, cooldown=2),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }]
        return [opt], schedulers


class BaselineSequentialV0(BaselineSequential):
    """
    Watch 1 time step, predict next N
    """

    def __init__(self, meta=None, original_frame_shape=None, num_frames_between_two_time_steps=12, lr=1e-5,
                 mode=FeaturesMode.UV, layers_mode=None, meta_video=None, meta_train_video_number=None,
                 meta_val_video_number=None, time_steps=5, train_dataset=None, val_dataset=None, batch_size=1,
                 num_workers=0, use_batch_norm=False, gt_based=True, center_based=True):
        super(BaselineSequentialV0, self).__init__(meta=meta, original_frame_shape=original_frame_shape,
                                                   num_frames_between_two_time_steps=num_frames_between_two_time_steps,
                                                   lr=lr, mode=mode, layers_mode=layers_mode, meta_video=meta_video,
                                                   meta_train_video_number=meta_train_video_number,
                                                   meta_val_video_number=meta_val_video_number, time_steps=time_steps,
                                                   train_dataset=train_dataset, val_dataset=val_dataset,
                                                   batch_size=batch_size, num_workers=num_workers,
                                                   use_batch_norm=use_batch_norm, gt_based=gt_based,
                                                   center_based=center_based)

    def one_step(self, batch):
        features, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch
        b_size = features.shape[1]

        cx, cx_1, hx, hx_1 = self.init_hidden_states(b_size)
        total_loss = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        last_input_velocity, last_pred_center, bad_data = None, None, False
        pred_centers, moved_points_by_true_of_list, actual_points_list = [], [], []
        ade, fde, shifted_points = None, None, None
        steps_to_watch = 8

        criterion = self.setup_loss()

        if len(features) == self.ts:
            for idx_i, i in enumerate(range(self.ts)):
                feat, bb_center1, bb_center2 = features[i].squeeze().float(), bbox_center1[i].float(), \
                                               bbox_center2[i].float()
                center_xy, center_true_uv, center_past_uv, shifted_point, gt_past_velocity = \
                    feat[:, 0, :], feat[:, 1, :], feat[:, 2, :], feat[:, 3, :], feat[:, 4, :]
                if self.gt_based:
                    shifted_points = bb_center2
                if not self.center_based:
                    return NotImplemented

                if idx_i == 0:
                    if self.gt_based:
                        last_input_velocity = gt_past_velocity
                        last_pred_center = bb_center1
                    else:
                        last_input_velocity = center_past_uv
                        last_pred_center = center_xy

                block_2, cx, hx, cx_1, hx_1 = self(last_input_velocity, cx, hx, cx_1, hx_1, True)

                if not self.gt_based:
                    shifted_points = shifted_point

                moved_points_by_true_of_list.append(shifted_points)
                pred_center = last_pred_center + (block_2 * 0.4)
                last_input_velocity = block_2
                last_pred_center = pred_center
                pred_centers.append(pred_center)
                actual_points_list.append(bb_center2)

                total_loss += criterion(shifted_points, pred_center=pred_center)

            predicted_points = [p.detach().cpu().numpy() for p in pred_centers]
            true_points = [p.detach().cpu().numpy() for p in actual_points_list]

            ade = compute_ade(np.stack(predicted_points), np.stack(true_points)).item()
            fde = compute_fde(np.stack(predicted_points), np.stack(true_points)).item()

        return total_loss / len(features), ade, fde


USE_NETWORK_V0 = False
if USE_NETWORK_V0:
    NETWORK = BaselineSequentialV0
else:
    NETWORK = BaselineSequential


def split_dataset(features, split_percent=0.2):
    inference_length = int(split_percent * len(features))
    train_length = len(features) - inference_length
    val_length = int(split_percent * inference_length)
    test_length = inference_length - val_length
    return inference_length, test_length, train_length, val_length


def prepare_datasets(features, split_percent=0.2):
    x, y, frames, track_ids, center_x, center_y, bbox_x, bbox_y, center_data = \
        features['x'], features['y'], features['frames'], features['track_ids'], features['bbox_center_x'], \
        features['bbox_center_y'], features['bbox_x'], features['bbox_y'], features['center_based']

    dataset = FeaturesDatasetCenterBased(x, y, frames=frames, track_ids=track_ids,
                                         mode=FeaturesMode.UV, preprocess=False,
                                         bbox_center_x=center_x,
                                         bbox_center_y=center_y,
                                         bbox_x=bbox_x,
                                         bbox_y=bbox_y,
                                         features_center=center_data)

    # Split Train and Test
    inference_length, test_length, train_length, val_length = split_dataset(features=x, split_percent=split_percent)

    train_dataset, inference_dataset = random_split(dataset, [train_length, inference_length], generator=GENERATOR_SEED)
    val_dataset, test_dataset = random_split(inference_dataset, [val_length, test_length], generator=GENERATOR_SEED)

    return train_dataset, val_dataset, test_dataset


def train(meta, num_frames_between_two_time_steps, meta_video, meta_train_video_number, meta_val_video_number, lr,
          train_dataset, val_dataset, time_steps, batch_size, num_workers, use_batch_norm, gt_based, center_based):
    model = NETWORK(meta=meta, num_frames_between_two_time_steps=num_frames_between_two_time_steps,
                    lr=lr, meta_video=meta_video, meta_train_video_number=meta_train_video_number,
                    meta_val_video_number=meta_val_video_number, train_dataset=train_dataset,
                    val_dataset=val_dataset, time_steps=time_steps, batch_size=batch_size,
                    num_workers=num_workers, use_batch_norm=use_batch_norm, gt_based=gt_based,
                    center_based=center_based)
    logger.info('Setting up network')
    trainer = pl.Trainer(gpus=1, max_epochs=400)
    logger.info('Initiating Training')
    trainer.fit(model)


def inference(meta, num_frames_between_two_time_steps, meta_video, meta_train_video_number, meta_val_video_number, lr,
              train_dataset, val_dataset, test_dataset, time_steps, batch_size, num_workers, of_model_version,
              of_model_epoch, gt_model_version, gt_model_epoch, use_batch_norm, center_based, device, dataset_video,
              loader_to_use_for_inference):
    train_loader = DataLoader(train_dataset, batch_size, collate_fn=center_dataset_collate, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=center_dataset_collate, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=center_dataset_collate, num_workers=num_workers)

    gt_model, of_model = inference_model_setup(batch_size, center_based, gt_model_epoch, gt_model_version, lr, meta,
                                               meta_train_video_number, meta_val_video_number, meta_video,
                                               num_frames_between_two_time_steps, num_workers, of_model_epoch,
                                               of_model_version, time_steps, train_dataset, use_batch_norm, val_dataset)

    ade_dataset_of, fde_dataset_of, ade_dataset_gt, fde_dataset_gt = [], [], [], []
    ade_dataset_linear, fde_dataset_linear = [], []
    ade_dataset_gt_of_gt, fde_dataset_gt_of_gt = [], []
    is_gt_based_shifted_points_inside_list, is_of_based_shifted_points_center_inside_list = [], []
    is_pred_center_inside_list, is_pred_center_gt_inside_list = [], []
    plot_results = False
    steps_to_watch = 8

    batches_processed = 0
    img_save_path = f'../Plots/T={time_steps}_{datetime.now()}'
    if img_save_path:
        Path(img_save_path).mkdir(parents=True, exist_ok=True)

    if USE_NETWORK_V0:
        inference_core_method = inference_core_v0
    else:
        inference_core_method = inference_core

    if loader_to_use_for_inference == NetworkMode.TRAIN:
        inference_core_method(ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear, ade_dataset_of,
                              batches_processed,
                              dataset_video, device, fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear,
                              fde_dataset_of,
                              gt_model, img_save_path, is_gt_based_shifted_points_inside_list,
                              is_of_based_shifted_points_center_inside_list, is_pred_center_gt_inside_list,
                              is_pred_center_inside_list, meta_val_video_number, of_model, plot_results, steps_to_watch,
                              time_steps, train_loader)
    if loader_to_use_for_inference == NetworkMode.VALIDATION:
        inference_core_method(ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear, ade_dataset_of,
                              batches_processed,
                              dataset_video, device, fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear,
                              fde_dataset_of,
                              gt_model, img_save_path, is_gt_based_shifted_points_inside_list,
                              is_of_based_shifted_points_center_inside_list, is_pred_center_gt_inside_list,
                              is_pred_center_inside_list, meta_val_video_number, of_model, plot_results, steps_to_watch,
                              time_steps, val_loader)
    if loader_to_use_for_inference == NetworkMode.TEST:
        inference_core_method(ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear, ade_dataset_of,
                              batches_processed,
                              dataset_video, device, fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear,
                              fde_dataset_of,
                              gt_model, img_save_path, is_gt_based_shifted_points_inside_list,
                              is_of_based_shifted_points_center_inside_list, is_pred_center_gt_inside_list,
                              is_pred_center_inside_list, meta_val_video_number, of_model, plot_results, steps_to_watch,
                              time_steps, test_loader)


def inference_core(ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear, ade_dataset_of, batches_processed,
                   dataset_video, device, fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear, fde_dataset_of,
                   gt_model, img_save_path, is_gt_based_shifted_points_inside_list,
                   is_of_based_shifted_points_center_inside_list, is_pred_center_gt_inside_list,
                   is_pred_center_inside_list, meta_val_video_number, of_model, plot_results, steps_to_watch,
                   time_steps, loader):
    for sav_i, batch_inference in enumerate(tqdm(loader)):
        try:
            features, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch_inference
            batches_processed += 1
        except TypeError:
            continue
        b_size = features.shape[1]

        cx_of, cx_1_of, cx_1_gt, cx_gt, hx_of, hx_1_of, hx_1_gt, hx_gt = inference_init_weights(b_size, device)

        frame_nums = [f.item() for f in frame_]
        img_frames, linear_pred_centers, pred_centers, pred_centers_gt = [], [], [], []
        gt_based_shifted_points_list, of_based_shifted_points_list, actual_points_list = [], [], []
        last_input_velocity_of, last_input_velocity_gt, last_pred_center_of, last_pred_center_gt, last_true_center = \
            None, None, None, None, None
        cap_count = 0

        first_ts_velocity_linear, linear_pred_center = None, None

        if len(features) == time_steps:
            if plot_results:
                cap = cv.VideoCapture(f'{base_path}videos/{dataset_video.value}/video{meta_val_video_number}/video.mov')

                while 1:
                    ret, video_frame = cap.read()
                    if cap_count in frame_nums:
                        img_frames.append(video_frame)
                        break
                    cap_count += 1

            for i in range(time_steps):
                feat, bb_center1, bb_center2, bb1, bb2 = features[i].squeeze().float(), bbox_center1[i].float(), \
                                                         bbox_center2[i].float(), bbox1[i].float(), bbox2[i].float()
                center_xy, center_true_uv, center_past_uv, shifted_point, gt_past_velocity = \
                    feat[0, :], feat[1, :], feat[2, :], feat[3, :], feat[4, :]
                # gt_based_shifted_points = feat2[:, :2]  # for gt based (all points)
                gt_based_shifted_points = bb_center2  # for bbox_center gt based

                if i == 0:
                    linear_pred_center = bb_center1
                    first_ts_velocity_linear = gt_past_velocity

                if i < steps_to_watch:
                    last_input_velocity_of = center_past_uv.unsqueeze(0)
                    last_pred_center_of = center_xy
                    last_input_velocity_gt = gt_past_velocity.unsqueeze(0)
                    last_pred_center_gt = bb_center1

                    last_true_center = bb_center1

                    with torch.no_grad():
                        block_2_of, cx_of, hx_of, cx_1_of, hx_1_of = of_model(last_input_velocity_of, cx_of, hx_of,
                                                                              cx_1_of, hx_1_of, True)
                        block_2_gt, cx_gt, hx_gt, cx_1_gt, hx_1_gt = gt_model(last_input_velocity_gt, cx_gt,
                                                                              hx_gt, cx_1_gt, hx_1_gt, True)

                    gt_based_shifted_points_list.append(gt_based_shifted_points)
                    of_based_shifted_points = shifted_point
                    of_based_shifted_points_list.append(of_based_shifted_points)

                    pred_center_of = last_pred_center_of + (block_2_of * 0.4)
                    pred_center_gt = last_pred_center_gt + (block_2_gt * 0.4)

                    if i == steps_to_watch - 1:
                        last_input_velocity_of = block_2_of
                        last_input_velocity_gt = block_2_gt
                        last_pred_center_of = pred_center_of
                        last_pred_center_gt = pred_center_gt
                else:
                    with torch.no_grad():
                        block_2_of, cx_of, hx_of, cx_1_of, hx_1_of = of_model(last_input_velocity_of, cx_of, hx_of,
                                                                              cx_1_of, hx_1_of, True)
                        block_2_gt, cx_gt, hx_gt, cx_1_gt, hx_1_gt = gt_model(last_input_velocity_gt, cx_gt,
                                                                              hx_gt, cx_1_gt, hx_1_gt, True)

                    gt_based_shifted_points_list.append(gt_based_shifted_points)
                    of_based_shifted_points = shifted_point
                    of_based_shifted_points_list.append(of_based_shifted_points)

                    pred_center_of = last_pred_center_of + (block_2_of * 0.4)
                    pred_center_gt = last_pred_center_gt + (block_2_gt * 0.4)

                    last_input_velocity_of = block_2_of
                    last_input_velocity_gt = block_2_gt
                    last_pred_center_of = pred_center_of
                    last_pred_center_gt = pred_center_gt

                linear_pred_center += first_ts_velocity_linear * 0.4  # * 0.4  # d = v * t

                # last_true_center = moved_points_by_true_of_center
                pred_centers.append(pred_center_of.squeeze(0))
                pred_centers_gt.append(pred_center_gt.squeeze(0))
                linear_pred_centers.append(linear_pred_center)

                # Remove mean for per time step plot
                actual_points_list.append(bb_center1.squeeze().detach().numpy())

                is_gt_based_shifted_points_inside = is_inside_bbox(point=gt_based_shifted_points.squeeze(),
                                                                   bbox=bb2.squeeze()).item()
                is_of_based_shifted_points_center_inside = is_inside_bbox(
                    # point=of_based_shifted_points_center.squeeze(),
                    point=of_based_shifted_points.squeeze(),
                    bbox=bb2.squeeze()).item()
                is_pred_center_inside = is_inside_bbox(point=pred_center_of.squeeze(),
                                                       bbox=bb2.squeeze()).item()
                is_pred_center_gt_inside = is_inside_bbox(point=pred_center_gt.squeeze(),
                                                          bbox=bb2.squeeze()).item()

                is_gt_based_shifted_points_inside_list.append(is_gt_based_shifted_points_inside)
                is_of_based_shifted_points_center_inside_list.append(
                    is_of_based_shifted_points_center_inside)
                is_pred_center_inside_list.append(is_pred_center_inside)
                is_pred_center_gt_inside_list.append(is_pred_center_gt_inside)

            inference_results_post_processing(actual_points_list, ade_dataset_gt, ade_dataset_gt_of_gt,
                                              ade_dataset_linear, ade_dataset_of, fde_dataset_gt, fde_dataset_gt_of_gt,
                                              fde_dataset_linear, fde_dataset_of, gt_based_shifted_points_list,
                                              gt_model, img_frames, img_save_path, linear_pred_centers,
                                              of_based_shifted_points_list, plot_results, pred_centers, pred_centers_gt,
                                              sav_i)
    plot_bars_if_inside_bbox([is_gt_based_shifted_points_inside_list,
                              is_of_based_shifted_points_center_inside_list,
                              is_pred_center_inside_list,
                              is_pred_center_gt_inside_list])
    inference_log_stdout(ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear, ade_dataset_of, batches_processed,
                         fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear, fde_dataset_of, gt_model, of_model)


def inference_core_v0(ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear, ade_dataset_of, batches_processed,
                      dataset_video, device, fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear, fde_dataset_of,
                      gt_model, img_save_path, is_gt_based_shifted_points_inside_list,
                      is_of_based_shifted_points_center_inside_list, is_pred_center_gt_inside_list,
                      is_pred_center_inside_list, meta_val_video_number, of_model, plot_results, steps_to_watch,
                      time_steps, loader):
    for sav_i, batch_inference in enumerate(tqdm(loader)):
        try:
            features, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch_inference
            batches_processed += 1
        except TypeError:
            continue
        b_size = features.shape[1]

        cx_of, cx_1_of, cx_1_gt, cx_gt, hx_of, hx_1_of, hx_1_gt, hx_gt = inference_init_weights(b_size, device)

        frame_nums = [f.item() for f in frame_]
        img_frames, linear_pred_centers, pred_centers, pred_centers_gt = [], [], [], []
        gt_based_shifted_points_list, of_based_shifted_points_list, actual_points_list = [], [], []
        last_input_velocity_of, last_input_velocity_gt, last_pred_center_of, last_pred_center_gt, last_true_center = \
            None, None, None, None, None
        cap_count = 0

        first_ts_velocity_linear, linear_pred_center = None, None

        if len(features) == time_steps:
            if plot_results:
                cap = cv.VideoCapture(f'{base_path}videos/{dataset_video.value}/video{meta_val_video_number}/video.mov')

                while 1:
                    ret, video_frame = cap.read()
                    if cap_count in frame_nums:
                        img_frames.append(video_frame)
                        break
                    cap_count += 1

            for i in range(time_steps):
                feat, bb_center1, bb_center2, bb1, bb2 = features[i].squeeze().float(), bbox_center1[i].float(), \
                                                         bbox_center2[i].float(), bbox1[i].float(), bbox2[i].float()
                center_xy, center_true_uv, center_past_uv, shifted_point, gt_past_velocity = \
                    feat[0, :], feat[1, :], feat[2, :], feat[3, :], feat[4, :]
                # gt_based_shifted_points = feat2[:, :2]  # for gt based (all points)
                gt_based_shifted_points = bb_center2  # for bbox_center gt based

                if i == 0:
                    last_input_velocity_of = center_past_uv.unsqueeze(0)
                    last_pred_center_of = center_xy
                    last_input_velocity_gt = gt_past_velocity.unsqueeze(0)
                    last_pred_center_gt = bb_center1
                    linear_pred_center = bb_center1
                    first_ts_velocity_linear = gt_past_velocity
                    last_true_center = bb_center1

                with torch.no_grad():
                    block_2_of, cx_of, hx_of, cx_1_of, hx_1_of = of_model(last_input_velocity_of, cx_of, hx_of,
                                                                          cx_1_of, hx_1_of, True)
                    block_2_gt, cx_gt, hx_gt, cx_1_gt, hx_1_gt = gt_model(last_input_velocity_gt, cx_gt,
                                                                          hx_gt, cx_1_gt, hx_1_gt, True)

                gt_based_shifted_points_list.append(gt_based_shifted_points)
                of_based_shifted_points = shifted_point
                of_based_shifted_points_list.append(of_based_shifted_points)

                pred_center_of = last_pred_center_of + (block_2_of * 0.4)
                pred_center_gt = last_pred_center_gt + (block_2_gt * 0.4)


                last_input_velocity_of = block_2_of
                last_input_velocity_gt = block_2_gt
                last_pred_center_of = pred_center_of
                last_pred_center_gt = pred_center_gt

                linear_pred_center += first_ts_velocity_linear * 0.4  # * 0.4  # d = v * t

                # last_true_center = moved_points_by_true_of_center
                pred_centers.append(pred_center_of.squeeze(0))
                pred_centers_gt.append(pred_center_gt.squeeze(0))
                linear_pred_centers.append(linear_pred_center)

                # Remove mean for per time step plot
                actual_points_list.append(bb_center1.squeeze().detach().numpy())

                is_gt_based_shifted_points_inside = is_inside_bbox(point=gt_based_shifted_points.squeeze(),
                                                                   bbox=bb2.squeeze()).item()
                is_of_based_shifted_points_center_inside = is_inside_bbox(
                    # point=of_based_shifted_points_center.squeeze(),
                    point=of_based_shifted_points.squeeze(),
                    bbox=bb2.squeeze()).item()
                is_pred_center_inside = is_inside_bbox(point=pred_center_of.squeeze(),
                                                       bbox=bb2.squeeze()).item()
                is_pred_center_gt_inside = is_inside_bbox(point=pred_center_gt.squeeze(),
                                                          bbox=bb2.squeeze()).item()

                is_gt_based_shifted_points_inside_list.append(is_gt_based_shifted_points_inside)
                is_of_based_shifted_points_center_inside_list.append(
                    is_of_based_shifted_points_center_inside)
                is_pred_center_inside_list.append(is_pred_center_inside)
                is_pred_center_gt_inside_list.append(is_pred_center_gt_inside)

            inference_results_post_processing(actual_points_list, ade_dataset_gt, ade_dataset_gt_of_gt,
                                              ade_dataset_linear, ade_dataset_of, fde_dataset_gt, fde_dataset_gt_of_gt,
                                              fde_dataset_linear, fde_dataset_of, gt_based_shifted_points_list,
                                              gt_model, img_frames, img_save_path, linear_pred_centers,
                                              of_based_shifted_points_list, plot_results, pred_centers, pred_centers_gt,
                                              sav_i)
    plot_bars_if_inside_bbox([is_gt_based_shifted_points_inside_list,
                              is_of_based_shifted_points_center_inside_list,
                              is_pred_center_inside_list,
                              is_pred_center_gt_inside_list])
    inference_log_stdout(ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear, ade_dataset_of, batches_processed,
                         fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear, fde_dataset_of, gt_model, of_model)


def inference_results_post_processing(actual_points_list, ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear,
                                      ade_dataset_of, fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear,
                                      fde_dataset_of, gt_based_shifted_points_list, gt_model, img_frames, img_save_path,
                                      linear_pred_centers, of_based_shifted_points_list, plot_results, pred_centers,
                                      pred_centers_gt, sav_i):
    predicted_points = [p.detach().numpy() for p in pred_centers]
    predicted_points_gt = [p.detach().numpy() for p in pred_centers_gt]
    linear_predicted_points = [p.detach().numpy() for p in linear_pred_centers]

    # true_points = [p.detach().mean(dim=0).numpy() for p in gt_based_shifted_points_list]
    # true_points_of = [p.detach().mean(dim=0).numpy() for p in of_based_shifted_points_list]
    true_points = [p.detach().numpy() for p in gt_based_shifted_points_list]
    true_points_of = [p.detach().numpy() for p in of_based_shifted_points_list]

    l2_points = {idx: np.linalg.norm(i - j, 2)
                 for idx, (i, j) in enumerate(zip(true_points_of, predicted_points))}
    l2_points_gt = {idx: np.linalg.norm(i - j, 2)
                    for idx, (i, j) in enumerate(zip(true_points, predicted_points_gt))}
    l2_linear = {idx: np.linalg.norm(i - j, 2)
                 for idx, (i, j) in enumerate(zip(true_points_of, linear_predicted_points))}

    # ade_of = compute_ade(np.stack(predicted_points), np.stack(true_points_of))
    # fde_of = compute_fde(np.stack(predicted_points), np.stack(true_points_of))
    ade_of = compute_ade(np.stack(predicted_points), np.stack(true_points).squeeze())
    fde_of = compute_fde(np.stack(predicted_points), np.stack(true_points).squeeze())
    ade_dataset_of.append(ade_of.item())
    fde_dataset_of.append(fde_of.item())

    ade_gt = compute_ade(np.stack(predicted_points_gt), np.stack(true_points).squeeze())
    fde_gt = compute_fde(np.stack(predicted_points_gt), np.stack(true_points).squeeze())
    ade_dataset_gt.append(ade_gt.item())
    fde_dataset_gt.append(fde_gt.item())

    ade_gt_of_gt = compute_ade(np.stack(true_points_of), np.stack(true_points).squeeze())
    fde_gt_of_gt = compute_fde(np.stack(true_points_of), np.stack(true_points).squeeze())
    ade_dataset_gt_of_gt.append(ade_gt_of_gt.item())
    fde_dataset_gt_of_gt.append(fde_gt_of_gt.item())

    ade_linear = compute_ade(np.stack(linear_predicted_points), np.stack(true_points).squeeze())
    fde_linear = compute_fde(np.stack(linear_predicted_points), np.stack(true_points).squeeze())
    ade_dataset_linear.append(ade_linear.item())
    fde_dataset_linear.append(fde_linear.item())

    # plot
    if plot_results:
        inference_plot_images(actual_points_list, gt_model, img_frames, img_save_path, l2_points, l2_points_gt,
                              predicted_points, predicted_points_gt, sav_i, true_points, true_points_of)


def inference_plot_images(actual_points_list, gt_model, img_frames, img_save_path, l2_points, l2_points_gt,
                          predicted_points, predicted_points_gt, sav_i, true_points, true_points_of):
    plot_trajectory_rnn_compare(predicted_points=np.stack(predicted_points),
                                predicted_points_gt=np.stack(predicted_points_gt),
                                true_points=np.stack(true_points).squeeze(),
                                true_points_of=np.stack(true_points_of),
                                of_l2=l2_points,
                                gt_l2=l2_points_gt,
                                actual_points=actual_points_list,
                                imgs=img_frames, gt=True,
                                m_ratio=gt_model.val_ratio,
                                show=False,
                                save_path=f'{img_save_path}compare_rgb_{sav_i}')
    plot_trajectory_rnn_compare(predicted_points=np.stack(predicted_points),
                                predicted_points_gt=np.stack(predicted_points_gt),
                                true_points=np.stack(true_points).squeeze(),
                                true_points_of=np.stack(true_points_of),
                                of_l2=l2_points,
                                gt_l2=l2_points_gt,
                                actual_points=actual_points_list,
                                imgs=None, gt=True,
                                m_ratio=gt_model.val_ratio,
                                show=False,
                                save_path=f'{img_save_path}compare_traj_{sav_i}')
    plot_trajectory_rnn_compare_side_by_side(predicted_points=np.stack(predicted_points),
                                             predicted_points_gt=np.stack(predicted_points_gt),
                                             true_points=np.stack(true_points).squeeze(),
                                             true_points_of=np.stack(true_points_of),
                                             of_l2=l2_points,
                                             gt_l2=l2_points_gt,
                                             actual_points=actual_points_list,
                                             imgs=img_frames, gt=True,
                                             m_ratio=gt_model.val_ratio,
                                             show=False,
                                             save_path=
                                             f'{img_save_path}compare_side_by_side_rgb_{sav_i}')


def inference_log_stdout(ade_dataset_gt, ade_dataset_gt_of_gt, ade_dataset_linear, ade_dataset_of, batches_processed,
                         fde_dataset_gt, fde_dataset_gt_of_gt, fde_dataset_linear, fde_dataset_of, gt_model, of_model):
    logger.info('Based on Optical Flow')
    logger.info(f'OF ADE: {np.array(ade_dataset_of).mean()}')
    logger.info(f'OF FDE: {np.array(fde_dataset_of).mean()}')
    logger.info(f'[m]OF ADE: {np.array(ade_dataset_of).mean() * of_model.val_ratio}')
    logger.info(f'[m]OF FDE: {np.array(fde_dataset_of).mean() * of_model.val_ratio}')

    logger.info('Based on GT Flow')
    logger.info(f'GT ADE: {np.array(ade_dataset_gt).mean()}')
    logger.info(f'GT FDE: {np.array(fde_dataset_gt).mean()}')
    logger.info(f'[m]GT ADE: {np.array(ade_dataset_gt).mean() * gt_model.val_ratio}')
    logger.info(f'[m]GT FDE: {np.array(fde_dataset_gt).mean() * gt_model.val_ratio}')

    logger.info('Between OF motion and GT Flow')
    logger.info(f'GT ADE: {np.array(ade_dataset_gt_of_gt).mean()}')
    logger.info(f'GT FDE: {np.array(fde_dataset_gt_of_gt).mean()}')
    logger.info(f'[m]GT ADE: {np.array(ade_dataset_gt_of_gt).mean() * gt_model.val_ratio}')
    logger.info(f'[m]GT FDE: {np.array(fde_dataset_gt_of_gt).mean() * gt_model.val_ratio}')

    logger.info('Linear Flow')
    logger.info(f'Linear ADE: {np.array(ade_dataset_linear).mean()}')
    logger.info(f'Linear FDE: {np.array(fde_dataset_linear).mean()}')
    logger.info(f'[m]Linear ADE: {np.array(ade_dataset_linear).mean() * of_model.val_ratio}')
    logger.info(f'[m]Linear FDE: {np.array(fde_dataset_linear).mean() * of_model.val_ratio}')

    logger.info(f'{batches_processed} batches processed!')


def inference_init_weights(b_size, device):
    hx, cx = torch.zeros(size=(b_size, 32), device=device), torch.zeros(size=(b_size, 32), device=device)
    hx_1, cx_1 = torch.zeros(size=(b_size, 64), device=device), torch.zeros(size=(b_size, 64), device=device)
    torch.nn.init.xavier_normal_(hx)
    torch.nn.init.xavier_normal_(cx)
    torch.nn.init.xavier_normal_(hx_1)
    torch.nn.init.xavier_normal_(cx_1)

    hx_gt, cx_gt = torch.zeros(size=(b_size, 32), device=device), torch.zeros(size=(b_size, 32), device=device)
    hx_1_gt, cx_1_gt = torch.zeros(size=(b_size, 64), device=device), torch.zeros(size=(b_size, 64), device=device)

    torch.nn.init.xavier_normal_(hx_gt)
    torch.nn.init.xavier_normal_(cx_gt)
    torch.nn.init.xavier_normal_(hx_1_gt)
    torch.nn.init.xavier_normal_(cx_1_gt)
    return cx, cx_1, cx_1_gt, cx_gt, hx, hx_1, hx_1_gt, hx_gt


def inference_model_setup(batch_size, center_based, gt_model_epoch, gt_model_version, lr, meta, meta_train_video_number,
                          meta_val_video_number, meta_video, num_frames_between_two_time_steps, num_workers,
                          of_model_epoch, of_model_version, time_steps, train_dataset, use_batch_norm, val_dataset):
    of_model = NETWORK(meta=meta, num_frames_between_two_time_steps=num_frames_between_two_time_steps,
                       lr=lr, meta_video=meta_video, meta_train_video_number=meta_train_video_number,
                       meta_val_video_number=meta_val_video_number, train_dataset=train_dataset,
                       val_dataset=val_dataset, time_steps=time_steps, batch_size=batch_size,
                       num_workers=num_workers, use_batch_norm=use_batch_norm, gt_based=False,
                       center_based=center_based)
    gt_model = NETWORK(meta=meta, num_frames_between_two_time_steps=num_frames_between_two_time_steps,
                       lr=lr, meta_video=meta_video, meta_train_video_number=meta_train_video_number,
                       meta_val_video_number=meta_val_video_number, train_dataset=train_dataset,
                       val_dataset=val_dataset, time_steps=time_steps, batch_size=batch_size,
                       num_workers=num_workers, use_batch_norm=use_batch_norm, gt_based=True,
                       center_based=center_based)
    of_model.load_state_dict(torch.load(f'lightning_logs/version_{of_model_version}/checkpoints/'
                                        f'epoch={of_model_epoch}.ckpt')['state_dict'])
    gt_model.load_state_dict(torch.load(f'lightning_logs/version_{gt_model_version}/checkpoints/'
                                        f'epoch={gt_model_epoch}.ckpt')['state_dict'])
    of_model.eval()
    gt_model.eval()
    return gt_model, of_model


def main(do_train, features_load_path, meta, meta_video, meta_train_video_number, meta_val_video_number, time_steps, lr,
         batch_size, num_workers, of_model_version, of_model_epoch, gt_model_version, gt_model_epoch, use_batch_norm,
         center_based, gt_based, dataset_video, loader_to_use_for_inference):
    logger.info('Setting up DataLoaders')
    feats = torch.load(features_load_path)

    train_set, val_set, test_set = prepare_datasets(features=feats)

    if do_train:
        kwargs_dict = {
            'meta': meta,
            'num_frames_between_two_time_steps': 12,
            'meta_video': meta_video,
            'meta_train_video_number': meta_train_video_number,
            'meta_val_video_number': meta_val_video_number,
            'lr': lr,
            'train_dataset': train_set,
            'val_dataset': val_set,
            'time_steps': time_steps,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'use_batch_norm': use_batch_norm,
            'gt_based': gt_based,
            'center_based': center_based
        }
        train(**kwargs_dict)
    else:
        kwargs_dict = {
            'meta': meta,
            'num_frames_between_two_time_steps': 12,
            'meta_video': meta_video,
            'meta_train_video_number': meta_train_video_number,
            'meta_val_video_number': meta_val_video_number,
            'lr': lr,
            'train_dataset': train_set,
            'val_dataset': val_set,
            'test_dataset': test_set,
            'time_steps': time_steps,
            'batch_size': 1,
            'num_workers': 0,
            'use_batch_norm': use_batch_norm,
            'center_based': center_based,
            'device': 'cpu',
            'of_model_version': of_model_version,
            'of_model_epoch': of_model_epoch,
            'gt_model_version': gt_model_version,
            'gt_model_epoch': gt_model_epoch,
            'dataset_video': dataset_video,
            'loader_to_use_for_inference': loader_to_use_for_inference
        }
        inference(**kwargs_dict)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # meta_path = '/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/H_SDD.txt'
        # base_path = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/"
        # save_base_path = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD_Features/"

        train_mode = True

        meta_path = '../Datasets/SDD/H_SDD.txt'
        base_path = "../Datasets/SDD/"
        save_base_path = "../Datasets/SDD_Features/"

        video_label = SDDVideoClasses.LITTLE
        video_number = 3

        save_path = f'{save_base_path}{video_label.value}/video{video_number}/'

        dataset_meta = SDDMeta(meta_path)
        dataset_meta_video = SDDVideoDatasets.LITTLE

        file_name = f'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_' \
                    f'center_based_gt_velocity_t{TIME_STEPS}.pt'

        USE_NETWORK_V0 = False
        main(do_train=train_mode, features_load_path=save_path + file_name, meta=dataset_meta,
             meta_video=dataset_meta_video, meta_train_video_number=video_number, meta_val_video_number=video_number,
             time_steps=TIME_STEPS, lr=LR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, of_model_version=OF_VERSION,
             of_model_epoch=OF_EPOCH, gt_model_version=GT_VERSION, gt_model_epoch=GT_EPOCH,
             use_batch_norm=USE_BATCH_NORM, center_based=CENTER_BASED, gt_based=GT_BASED, dataset_video=video_label,
             loader_to_use_for_inference=NetworkMode.VALIDATION)
