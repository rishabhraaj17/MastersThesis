import torch
import numpy as np
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from average_image.utils import compute_ade, compute_fde
from baselinev2.config import MANUAL_SEED, LINEAR_CFG, DATASET_META, META_LABEL, VIDEO_NUMBER
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.nn.models')

torch.manual_seed(MANUAL_SEED)


def make_layers(cfg, batch_norm=False, encoder=True, last_without_activation=True):
    layers = []
    if encoder:
        in_features = 2
    else:
        in_features = 64
    if last_without_activation:
        for v in cfg[:-1]:
            in_features, layers = core_linear_layers_maker(batch_norm, in_features, layers, v)
        layers += [nn.Linear(in_features, cfg[-1])]
    else:
        for v in cfg:
            in_features, layers = core_linear_layers_maker(batch_norm, in_features, layers, v)
    return nn.Sequential(*layers)


def core_linear_layers_maker(batch_norm, in_features, layers, v):
    linear = nn.Linear(in_features, v)
    if batch_norm:
        layers += [linear, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
    else:
        layers += [linear, nn.ReLU(inplace=True)]
    in_features = v
    return in_features, layers


class BaselineEncoder(nn.Module):
    def __init__(self):
        super(BaselineEncoder, self).__init__()

    def forward(self, x):
        pass


class BaselineDecoder(nn.Module):
    def __init__(self):
        super(BaselineDecoder, self).__init__()

    def forward(self, x):
        pass


class BaselineRNN(LightningModule):
    """
    Watch 8 time steps, predict next 12 (learn 8 - supervised, 12 - unsupervised)
    """

    def __init__(self, meta=None, original_frame_shape=None, num_frames_between_two_time_steps=12, lr=1e-5,
                 meta_video=None, meta_train_video_number=None, meta_val_video_number=None, time_steps=5,
                 train_dataset=None, val_dataset=None, batch_size=1, num_workers=0, use_batch_norm=False):
        super(BaselineRNN, self).__init__()

        self.block_1 = make_layers(LINEAR_CFG['encoder'], batch_norm=use_batch_norm, encoder=True,
                                   last_without_activation=False)
        self.encoder = nn.LSTMCell(input_size=16, hidden_size=32)
        self.decoder = nn.LSTMCell(input_size=32, hidden_size=64)
        self.block_2 = make_layers(LINEAR_CFG['decoder'], batch_norm=use_batch_norm, encoder=False,
                                   last_without_activation=True)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.lr = lr

        self.val_dataset = val_dataset
        self.train_dataset = train_dataset

        self.original_frame_shape = original_frame_shape
        self.num_frames_between_two_time_steps = num_frames_between_two_time_steps

        self.meta = meta
        self.train_ratio = float(self.meta.get_meta(meta_video, meta_train_video_number)[0]['Ratio'].to_numpy()[0])
        self.val_ratio = float(self.meta.get_meta(meta_video, meta_val_video_number)[0]['Ratio'].to_numpy()[0])

        self.ts = time_steps

        self.save_hyperparameters('lr', 'time_steps', 'meta_video', 'batch_size', 'meta_train_video_number',
                                  'meta_val_video_number', 'use_batch_norm')

    def forward(self, x):
        return NotImplemented

    def one_step(self, batch):
        tracks, relative_distances = batch
        # last 2 values are bounding box centers
        agents_position = tracks[..., -2:]
        relative_velocities = relative_distances / 0.4
        print()

        return NotImplemented

    def init_hidden_states(self, b_size):
        hx, cx = torch.zeros(size=(b_size, 32), device=self.device), torch.zeros(size=(b_size, 32), device=self.device)
        hx_1, cx_1 = torch.zeros(size=(b_size, 64), device=self.device), torch.zeros(size=(b_size, 64),
                                                                                     device=self.device)
        torch.nn.init.xavier_normal_(hx)
        torch.nn.init.xavier_normal_(cx)
        torch.nn.init.xavier_normal_(hx_1)
        torch.nn.init.xavier_normal_(cx_1)

        return cx, cx_1, hx, hx_1

    def center_based_loss_meters(self, gt_center, pred_center):
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
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=None,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size * 2, collate_fn=None,
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


if __name__ == '__main__':
    dummy_tracks_in = torch.randn(size=(20, 64, 12))
    dummy_rel_distance_in = torch.randn(size=(19, 64, 2))
    # encoder_linear = nn.Sequential(nn.Linear(2, 16),
    #                                nn.ReLU(),
    #                                nn.Linear(16, 32),
    #                                nn.ReLU())
    # encoder_lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, bias=True)
    # o = encoder_linear(dummy_tracks_in[:8, :, -2:].view(-1, 2))
    # o = encoder_lstm(o.view(8, 64, -1))
    m = BaselineRNN(meta=DATASET_META, meta_video=META_LABEL, meta_train_video_number=VIDEO_NUMBER,
                    meta_val_video_number=VIDEO_NUMBER)
    o = m.one_step([dummy_tracks_in, dummy_rel_distance_in])
    print(o.shape)