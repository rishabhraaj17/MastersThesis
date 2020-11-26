import warnings

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader

from average_image.constants import SDDVideoClasses, FeaturesMode, SDDVideoDatasets
from average_image.utils import SDDMeta, compute_ade, compute_fde
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import FeaturesDatasetCenterBased

initialize_logging()
logger = get_logger(__name__)

TIME_STEPS = 20
NUM_WORKERS = 10
BATCH_SIZE = 256
LR = 1e-3
MANUAL_SEED = 42
torch.manual_seed(MANUAL_SEED)
GENERATOR_SEED = torch.Generator().manual_seed(MANUAL_SEED)

LINEAR_CFG = {
    'encoder': [4, 8, 16],
    'decoder': [32, 16, 8, 4, 2]
}


def center_dataset_collate(batch):
    center_features_list, frames_batched_list, track_ids_batched_list, bbox_center_x_batched_list, \
    bbox_center_y_batched_list, bbox_x_batched_list, bbox_y_batched_list = [], [], [], [], [], [], []
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
                                  'meta_val_video_number')

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
    model = BaselineSequential(meta=meta, num_frames_between_two_time_steps=num_frames_between_two_time_steps,
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
              of_model_epoch, gt_model_version, gt_model_epoch):
    pass


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # meta_path = '/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/H_SDD.txt'
        # base_path = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/"
        # save_base_path = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD_Features/"

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

        logger.info('Setting up DataLoaders')
        feats = torch.load(save_path + file_name)

        train_set, val_set, test_set = prepare_datasets(features=feats)

        kwargs_dict = {
            'meta': dataset_meta,
            'num_frames_between_two_time_steps': 12,
            'meta_video': dataset_meta_video,
            'meta_train_video_number': video_number,
            'meta_val_video_number': video_number,
            'lr': LR,
            'train_dataset': train_set,
            'val_dataset': val_set,
            'time_steps': TIME_STEPS,
            'batch_size': BATCH_SIZE,
            'num_workers': NUM_WORKERS,
            'use_batch_norm': False,
            'gt_based': False,
            'center_based': True
        }

        train(**kwargs_dict)
