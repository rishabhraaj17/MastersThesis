from typing import Optional

import torch
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import compute_ade, compute_fde
from baselinev2.config import MANUAL_SEED, LINEAR_CFG, SDD_VIDEO_CLASSES_LIST_FOR_NN, SDD_PER_CLASS_VIDEOS_LIST_FOR_NN, \
    SDD_VIDEO_META_CLASSES_LIST_FOR_NN, NUM_WORKERS, BATCH_SIZE, LR, USE_BATCH_NORM, NUM_EPOCHS, OVERFIT
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import BaselineDataset, BaselineGeneratedDataset
from baselinev2.plot_utils import plot_trajectories
from baselinev2.stochastic.losses import cal_ade, cal_fde, cal_ade_fde_stochastic
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.nn.models')

torch.manual_seed(MANUAL_SEED)


def make_layers(cfg, batch_norm=False, encoder=True, last_without_activation=True, dropout=None,
                decoder_in_dim=LINEAR_CFG['lstm_encoder']):
    layers = []
    if encoder:
        in_features = 2
    else:
        in_features = decoder_in_dim
    if last_without_activation:
        for v in cfg[:-1]:
            in_features, layers = core_linear_layers_maker(batch_norm, in_features, layers, v, dropout)
        layers += [nn.Linear(in_features, cfg[-1])]
    else:
        for v in cfg:
            in_features, layers = core_linear_layers_maker(batch_norm, in_features, layers, v, dropout)
    return nn.Sequential(*layers)


def core_linear_layers_maker(batch_norm, in_features, layers, v, dropout):
    linear = nn.Linear(in_features, v)
    if batch_norm and dropout is not None:
        layers += [linear, nn.BatchNorm1d(v), nn.ReLU(inplace=True), nn.Dropout(dropout)]
    elif batch_norm:
        layers += [linear, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
    elif dropout is not None:
        layers += [linear, nn.ReLU(inplace=True), nn.Dropout(dropout)]
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
    Watch 8 time steps, predict next 12
    """

    def __init__(self, arch_config=LINEAR_CFG, original_frame_shape=None, prediction_length=12, lr=1e-5, time_steps=5,
                 train_dataset=None, val_dataset=None, batch_size=1, num_workers=0, use_batch_norm=False,
                 lstm_num_layers: int = 1, overfit_mode: bool = False, shuffle: bool = False, pin_memory: bool = True,
                 two_losses: bool = False):
        super(BaselineRNN, self).__init__()
        
        self.arch_config = arch_config

        self.pre_encoder = make_layers(arch_config['encoder'], batch_norm=use_batch_norm, encoder=True,
                                       last_without_activation=False)
        self.encoder = nn.LSTM(input_size=arch_config['lstm_in'], hidden_size=arch_config['lstm_encoder'],
                               num_layers=lstm_num_layers, bias=True)
        self.pre_decoder = make_layers(arch_config['encoder'], batch_norm=use_batch_norm, encoder=True,
                                       last_without_activation=False)
        self.decoder = nn.LSTMCell(input_size=arch_config['lstm_in'], hidden_size=arch_config['lstm_encoder'])
        self.post_decoder = make_layers(arch_config['decoder'], batch_norm=use_batch_norm, encoder=False,
                                        last_without_activation=True)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.lr = lr

        self.val_dataset = val_dataset
        self.train_dataset = train_dataset

        self.lstm_num_layers = lstm_num_layers
        self.overfit_mode = overfit_mode
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.original_frame_shape = original_frame_shape
        self.prediction_length = prediction_length

        self.two_losses = two_losses
        self.ts = time_steps

        self.save_hyperparameters('lr', 'time_steps', 'batch_size', 'use_batch_norm', 'overfit_mode', 'shuffle')

    def forward(self, x):
        if self.two_losses:
            return self.forward_two_losses(x)
        else:
            return self.forward_one_loss(x)

    def forward_two_losses(self, x):
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = x

        total_loss = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        total_loss_velocity = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        predicted_xy, true_xy = [], []

        # Encoder
        b, seq_len = in_uv.size(0), in_uv.size(1)
        h0, c0 = self.init_hidden_states(b_size=b)
        out = self.pre_encoder(in_uv.view(-1, 2))
        out = F.relu(out.view(seq_len, b, -1))
        out, (h_enc, c_enc) = self.encoder(out, (h0, c0))
        # Decoder
        # Last (x,y) and (u,v) position at T=8
        last_xy = in_xy[:, -1, ...]
        last_uv = in_uv[:, -1, ...]

        h_dec, c_dec = h_enc.squeeze(0), c_enc.squeeze(0)
        for gt_pred_xy, gt_pred_uv in zip(gt_xy.permute(1, 0, 2), gt_uv.permute(1, 0, 2)):
            out = self.pre_decoder(last_uv)
            h_dec, c_dec = self.decoder(out, (h_dec, c_dec))
            pred_uv = self.post_decoder(F.relu(h_dec))
            out = last_xy + (pred_uv * 0.4)

            total_loss += self.center_based_loss_meters(gt_center=gt_pred_xy, pred_center=out, ratio=ratio[0].item())
            total_loss_velocity += self.center_based_loss_meters(gt_center=gt_pred_uv, pred_center=pred_uv, ratio=1)

            predicted_xy.append(out.detach().cpu().numpy())
            true_xy.append(gt_pred_xy.detach().cpu().numpy())

            last_xy = out
            last_uv = pred_uv

        ade = compute_ade(np.stack(predicted_xy), np.stack(true_xy)).item()
        fde = compute_fde(np.stack(predicted_xy), np.stack(true_xy)).item()

        return total_loss / self.prediction_length, total_loss_velocity / self.prediction_length, ade, fde, \
               ratio[0].item(), np.stack(predicted_xy)

    def forward_one_loss(self, x):
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = x

        total_loss = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        predicted_xy, true_xy = [], []

        # Encoder
        b, seq_len = in_uv.size(0), in_uv.size(1)
        h0, c0 = self.init_hidden_states(b_size=b)
        out = self.pre_encoder(in_uv.view(-1, 2))
        out = F.relu(out.view(seq_len, b, -1))
        out, (h_enc, c_enc) = self.encoder(out, (h0, c0))
        # Decoder
        # Last (x,y) and (u,v) position at T=8
        last_xy = in_xy[:, -1, ...]
        last_uv = in_uv[:, -1, ...]

        h_dec, c_dec = h_enc.squeeze(0), c_enc.squeeze(0)
        for gt_pred_xy in gt_xy.permute(1, 0, 2):
            out = self.pre_decoder(last_uv)
            h_dec, c_dec = self.decoder(out, (h_dec, c_dec))
            pred_uv = self.post_decoder(F.relu(h_dec))
            out = last_xy + (pred_uv * 0.4)
            total_loss += self.center_based_loss_meters(gt_center=gt_pred_xy, pred_center=out, ratio=ratio[0].item())

            predicted_xy.append(out.detach().cpu().numpy())
            true_xy.append(gt_pred_xy.detach().cpu().numpy())

            last_xy = out
            last_uv = pred_uv

        ade = compute_ade(np.stack(predicted_xy), np.stack(true_xy)).item()
        fde = compute_fde(np.stack(predicted_xy), np.stack(true_xy)).item()

        return total_loss / self.prediction_length, ade, fde, ratio[0].item(), np.stack(predicted_xy)

    def one_step(self, batch):
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = batch

        total_loss = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        predicted_xy, true_xy = [], []

        # Encoder
        b, seq_len = in_uv.size(0), in_uv.size(1)
        h0, c0 = self.init_hidden_states(b_size=b)
        out = self.pre_encoder(in_uv.view(-1, 2))
        out = F.relu(out.view(seq_len, b, -1))
        out, (h_enc, c_enc) = self.encoder(out, (h0, c0))
        # Decoder
        # Last (x,y) and (u,v) position at T=8
        last_xy = in_xy[:, -1, ...]
        last_uv = in_uv[:, -1, ...]

        h_dec, c_dec = h_enc.squeeze(0), c_enc.squeeze(0)
        for gt_pred_xy in gt_xy.permute(1, 0, 2):
            out = self.pre_decoder(last_uv)
            h_dec, c_dec = self.decoder(out, (h_dec, c_dec))
            pred_uv = self.post_decoder(F.relu(h_dec))
            out = last_xy + (pred_uv * 0.4)
            total_loss += self.center_based_loss_meters(gt_center=gt_pred_xy, pred_center=out, ratio=ratio[0].item())

            predicted_xy.append(out.detach().cpu().numpy())
            true_xy.append(gt_pred_xy.detach().cpu().numpy())

            last_xy = out
            last_uv = pred_uv

        ade = compute_ade(np.stack(predicted_xy), np.stack(true_xy)).item()
        fde = compute_fde(np.stack(predicted_xy), np.stack(true_xy)).item()

        return total_loss / self.prediction_length, ade, fde, ratio[0].item()

    def init_hidden_states(self, b_size):
        hx, cx = torch.zeros(
            size=(self.lstm_num_layers, b_size, self.arch_config['lstm_encoder']), device=self.device), \
                 torch.zeros(size=(self.lstm_num_layers, b_size, self.arch_config['lstm_encoder']), device=self.device)
        torch.nn.init.xavier_normal_(hx)
        torch.nn.init.xavier_normal_(cx)

        return hx, cx

    def center_based_loss_meters(self, gt_center, pred_center, ratio):
        loss = self.l2_norm(pred_center, gt_center) * ratio
        return loss

    @staticmethod
    def plot_one_trajectory(in_xy, true_xy, predicted_xy, idx, frame_number=0):
        obs_trajectory = np.stack(in_xy[idx].cpu().numpy())
        true_trajectory = np.stack([t[idx] for t in true_xy])
        pred_trajectory = np.stack([t[idx] for t in predicted_xy])

        plot_trajectories(obs_trajectory=obs_trajectory, gt_trajectory=true_trajectory, pred_trajectory=pred_trajectory,
                          frame_number=frame_number, track_id=idx)

    @staticmethod
    def l2_norm(point1, point2):
        return torch.linalg.norm(point1 - point2, ord=2, dim=-1).mean()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=None,
                          num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size * 2, collate_fn=None,
                          num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)

    def training_step(self, batch, batch_idx):
        loss, ade, fde, ratio = self.one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/ade', ade * ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/fde', fde * ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, ade, fde, ratio = self.one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/ade', ade * ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/fde', fde * ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(opt, patience=15, verbose=True, factor=0.2, cooldown=2),
                'monitor': 'val_loss_epoch',
                'interval': 'epoch',
                'frequency': 1
            }]
        return [opt], schedulers


class BaselineRNNStacked(BaselineRNN):
    def __init__(self, arch_config=LINEAR_CFG, original_frame_shape=None, prediction_length=12, lr=1e-5, time_steps=5,
                 train_dataset=None, val_dataset=None, batch_size=1, num_workers=0, use_batch_norm=False,
                 encoder_lstm_num_layers: int = 1, overfit_mode: bool = False, shuffle: bool = False,
                 pin_memory: bool = True, decoder_lstm_num_layers: int = 1, return_pred: bool = False,
                 generated_dataset: bool = False, relative_velocities: bool = False):
        super(BaselineRNNStacked, self).__init__(
            arch_config=arch_config,
            original_frame_shape=original_frame_shape, prediction_length=prediction_length, lr=lr,
            time_steps=time_steps, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size,
            num_workers=num_workers, use_batch_norm=use_batch_norm, lstm_num_layers=encoder_lstm_num_layers,
            overfit_mode=overfit_mode, shuffle=shuffle, pin_memory=pin_memory)

        self.decoder_lstm_num_layers = decoder_lstm_num_layers

        self.pre_encoder = make_layers(arch_config['encoder'], batch_norm=use_batch_norm, encoder=True,
                                       last_without_activation=False)
        self.encoder = nn.LSTM(input_size=arch_config['lstm_in'], hidden_size=arch_config['lstm_encoder'],
                               num_layers=encoder_lstm_num_layers, bias=True)
        self.pre_decoder = make_layers(arch_config['encoder'], batch_norm=use_batch_norm, encoder=True,
                                       last_without_activation=False)
        self.decoder = nn.LSTMCell(input_size=arch_config['lstm_in'], hidden_size=arch_config['lstm_encoder'])

        if decoder_lstm_num_layers > 1:
            self.decoder_extra_layers = nn.ModuleList(
                [nn.LSTMCell(input_size=arch_config['lstm_encoder'], hidden_size=arch_config['lstm_encoder'])
                 for _ in range(decoder_lstm_num_layers)])

        self.post_decoder = make_layers(arch_config['decoder'], batch_norm=use_batch_norm, encoder=False,
                                        last_without_activation=True)
        self.return_pred = return_pred
        self.generated_dataset = generated_dataset
        self.relative_velocities = relative_velocities

        self.save_hyperparameters('lr', 'generated_dataset', 'batch_size', 'use_batch_norm', 'overfit_mode', 'shuffle',
                                  'relative_velocities')

    def forward(self, batch):
        if self.generated_dataset:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, \
            mapped_in_xy, mapped_gt_xy, ratio = batch
        else:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = batch

        total_loss = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        predicted_xy, true_xy = [], []
        hidden_states, cell_states = None, None

        # Encoder
        b, seq_len = in_uv.size(0), in_uv.size(1)
        if self.decoder_lstm_num_layers > 1:
            hidden_states, cell_states = self.init_hidden_states(b_size=b)
            h0, c0 = hidden_states[0], cell_states[0]
        else:
            h0, c0 = self.init_hidden_states(b_size=b)

        out = self.pre_encoder(in_uv.view(-1, 2))
        # out = F.relu(out.view(seq_len, b, -1))  # covered in pre-encoder
        out = out.view(seq_len, b, -1)
        out, (h_enc, c_enc) = self.encoder(out, (h0, c0))

        # Decoder
        # Last (x,y) and (u,v) position at T=8
        last_xy = in_xy[:, -1, ...]
        last_uv = in_uv[:, -1, ...]

        # h_dec, c_dec = h_enc[-1, ...], c_enc[-1, ...]
        h_dec, c_dec = h_enc[-1, ...], torch.zeros_like(c_enc[-1, ...])
        torch.nn.init.xavier_normal_(c_dec)

        for gt_pred_xy in gt_xy.permute(1, 0, 2):
            out = self.pre_decoder(last_uv)
            h_dec, c_dec = self.decoder(out, (h_dec, c_dec))

            if self.decoder_lstm_num_layers > 1:
                for d_idx, extra_decoder in enumerate(self.decoder_extra_layers):
                    # h_dec, c_dec = extra_decoder(F.relu(h_dec), (hidden_states[d_idx + 1], cell_states[d_idx + 1]))
                    h_dec, c_dec = extra_decoder(h_dec, (hidden_states[d_idx + 1], cell_states[d_idx + 1]))

            # pred_uv = self.post_decoder(F.relu(h_dec))
            pred_uv = self.post_decoder(h_dec)
            out = last_xy + (pred_uv * 0.4) if self.relative_velocities else last_xy + pred_uv
            total_loss += self.center_based_loss_meters(gt_center=gt_pred_xy, pred_center=out, ratio=ratio[0].item())

            predicted_xy.append(out.detach().cpu().numpy())
            true_xy.append(gt_pred_xy.detach().cpu().numpy())

            last_xy = out
            last_uv = pred_uv

        ade = compute_ade(np.stack(predicted_xy), np.stack(true_xy)).item()
        fde = compute_fde(np.stack(predicted_xy), np.stack(true_xy)).item()

        ade *= ratio[0].item()
        fde *= ratio[0].item()

        if self.return_pred:
            return total_loss / self.prediction_length, ade, fde, ratio[0].item(), np.stack(predicted_xy)

        return total_loss / self.prediction_length, ade, fde, ratio[0].item()

    def one_step(self, batch):
        return self(batch)

    def init_hidden_states(self, b_size):
        hx = torch.zeros(size=(self.lstm_num_layers, b_size, self.arch_config['lstm_encoder']), device=self.device)
        cx = torch.zeros(size=(self.lstm_num_layers, b_size, self.arch_config['lstm_encoder']), device=self.device)
        torch.nn.init.xavier_normal_(hx)
        torch.nn.init.xavier_normal_(cx)

        if self.decoder_lstm_num_layers > 1:
            hidden_states, cell_states = [hx], [cx]
            for _ in range(self.decoder_lstm_num_layers):
                h = torch.zeros(
                    size=(b_size, self.arch_config['lstm_encoder']), device=self.device)
                c = torch.zeros(
                    size=(b_size, self.arch_config['lstm_encoder']), device=self.device)
                torch.nn.init.xavier_normal_(h)
                torch.nn.init.xavier_normal_(c)
                hidden_states.append(h)
                cell_states.append(c)
            return hidden_states, cell_states
        else:
            return hx, cx


class BaselineRNNStackedSimple(BaselineRNN):
    def __init__(self, arch_config=LINEAR_CFG, original_frame_shape=None, prediction_length=12, lr=1e-5, time_steps=5,
                 train_dataset=None, val_dataset=None, batch_size=1, num_workers=0, use_batch_norm=False,
                 encoder_lstm_num_layers: int = 1, overfit_mode: bool = False, shuffle: bool = False,
                 pin_memory: bool = True, decoder_lstm_num_layers: int = 1, return_pred: bool = True,
                 generated_dataset: bool = False, relative_velocities: bool = False, dropout: Optional[float] = None,
                 rnn_dropout: float = 0, use_gru: bool = False, learn_hidden_states: bool = False,
                 feed_model_distances_in_meters: bool = False):
        super(BaselineRNNStackedSimple, self).__init__(
            arch_config=arch_config,
            original_frame_shape=original_frame_shape, prediction_length=prediction_length, lr=lr,
            time_steps=time_steps, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size,
            num_workers=num_workers, use_batch_norm=use_batch_norm, lstm_num_layers=encoder_lstm_num_layers,
            overfit_mode=overfit_mode, shuffle=shuffle, pin_memory=pin_memory)

        if learn_hidden_states:
            data = torch.zeros((encoder_lstm_num_layers, batch_size, arch_config['lstm_encoder']))
            torch.nn.init.xavier_normal_(data)

        self.pre_encoder = make_layers(arch_config['encoder'], batch_norm=use_batch_norm, encoder=True,
                                       last_without_activation=False, dropout=dropout)
        if use_gru:
            self.encoder = nn.GRU(input_size=arch_config['lstm_in'], hidden_size=arch_config['lstm_encoder'],
                                  num_layers=encoder_lstm_num_layers, bias=True, dropout=rnn_dropout)
            if learn_hidden_states:
                self.encoder_hidden = nn.Parameter(data.clone())
        else:
            self.encoder = nn.LSTM(input_size=arch_config['lstm_in'], hidden_size=arch_config['lstm_encoder'],
                                   num_layers=encoder_lstm_num_layers, bias=True, dropout=rnn_dropout)
            if learn_hidden_states:
                self.encoder_hidden = nn.Parameter(data.clone())
                self.encoder_cell_state = nn.Parameter(data.clone())

        self.pre_decoder = make_layers(arch_config['encoder'], batch_norm=use_batch_norm, encoder=True,
                                       last_without_activation=False, dropout=dropout)

        if use_gru:
            self.decoder = nn.GRU(input_size=arch_config['lstm_in'], hidden_size=arch_config['lstm_encoder'],
                                  num_layers=decoder_lstm_num_layers, bias=True, dropout=rnn_dropout)
        else:
            self.decoder = nn.LSTM(input_size=arch_config['lstm_in'], hidden_size=arch_config['lstm_encoder'],
                                   num_layers=decoder_lstm_num_layers, bias=True, dropout=rnn_dropout)
            if learn_hidden_states:
                self.decoder_cell_state = nn.Parameter(data.clone())

        self.post_decoder = make_layers(arch_config['decoder'], batch_norm=use_batch_norm, encoder=False,
                                        last_without_activation=True, dropout=dropout)
        self.return_pred = return_pred
        self.generated_dataset = generated_dataset
        self.relative_velocities = relative_velocities
        self.decoder_lstm_num_layers = decoder_lstm_num_layers
        self.use_gru = use_gru
        self.learn_hidden_states = learn_hidden_states

        self.feed_model_distances_in_meters = feed_model_distances_in_meters

        self.save_hyperparameters('lr', 'generated_dataset', 'batch_size', 'use_batch_norm', 'overfit_mode', 'shuffle',
                                  'relative_velocities', 'dropout', 'rnn_dropout', 'use_gru', 'learn_hidden_states',
                                  'encoder_lstm_num_layers', 'decoder_lstm_num_layers', 'generated_dataset',
                                  'feed_model_distances_in_meters')

    def init_hidden_states(self, b_size):
        hx = torch.zeros(size=(self.lstm_num_layers, b_size, self.arch_config['lstm_encoder']), device=self.device)
        cx = torch.zeros(size=(self.lstm_num_layers, b_size, self.arch_config['lstm_encoder']), device=self.device)
        dec_cx = torch.zeros(size=(self.decoder_lstm_num_layers, b_size, self.arch_config['lstm_encoder']),
                             device=self.device)
        torch.nn.init.xavier_normal_(hx)
        torch.nn.init.xavier_normal_(cx)
        torch.nn.init.xavier_normal_(dec_cx)

        return hx, cx, dec_cx

    def forward(self, x):
        return self.forward_gru(x) if self.use_gru else self.forward_lstm(x)

    def forward_gru(self, batch):
        if self.generated_dataset:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, \
            mapped_in_xy, mapped_gt_xy, ratio = batch
        else:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = batch

        if self.feed_model_distances_in_meters:
            # in_uv_pixels = in_uv.clone()
            in_uv *= ratio[0].item()

        predicted_xy, true_xy, predicted_xy_for_loss = [], [], []

        # Encoder
        b, seq_len = in_uv.size(0), in_uv.size(1)
        if self.learn_hidden_states:
            h0 = self.encoder_hidden
        else:
            h0, _, _ = self.init_hidden_states(b_size=b)

        out = self.pre_encoder(in_uv.view(-1, 2))
        out = out.view(seq_len, b, -1)
        out, h_enc = self.encoder(out, h0)

        # Decoder
        # Last (x,y) and (u,v) position at T=8
        last_xy = in_xy[:, -1, ...]
        last_uv = in_uv[:, -1, ...]

        h_dec = h_enc  # [-1, ...]  # , torch.zeros_like(c_enc[-1, ...])

        for gt_pred_xy in gt_xy.permute(1, 0, 2):
            out = self.pre_decoder(last_uv)
            out, h_dec = self.decoder(out.unsqueeze(0), h_dec)

            pred_uv = self.post_decoder(out.squeeze(0))
            if self.feed_model_distances_in_meters:
                pred_uv /= ratio[0].item()
            out = last_xy + (pred_uv * 0.4) if self.relative_velocities else last_xy + pred_uv

            predicted_xy_for_loss.append(out)
            predicted_xy.append(out.detach().cpu().numpy())
            true_xy.append(gt_pred_xy.detach().cpu().numpy())

            last_xy = out
            if self.feed_model_distances_in_meters:
                pred_uv *= ratio[0].item()
            last_uv = pred_uv

        ade = compute_ade(np.stack(predicted_xy, axis=1), np.stack(true_xy, axis=1)).item()
        fde = compute_fde(np.stack(predicted_xy, axis=1), np.stack(true_xy, axis=1),
                          batched_v2=True, batched=False).item()

        ade *= ratio[0].item()
        fde *= ratio[0].item()

        loss = torch.linalg.norm(gt_xy -
                                 torch.stack(predicted_xy_for_loss, dim=1),
                                 ord=2, dim=-1).mean() * ratio[0].item()

        if self.return_pred:
            return loss, ade, fde, ratio[0].item(), np.stack(predicted_xy)

        return loss, ade, fde, ratio[0].item()

    def forward_lstm(self, batch):
        if self.generated_dataset:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, \
            mapped_in_xy, mapped_gt_xy, ratio = batch
        else:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = batch

        if self.feed_model_distances_in_meters:
            # in_uv_pixels = in_uv.clone()
            in_uv *= ratio[0].item()

        predicted_xy, true_xy, predicted_xy_for_loss = [], [], []

        # Encoder
        b, seq_len = in_uv.size(0), in_uv.size(1)
        if self.learn_hidden_states:
            h0, c0, c_dec = self.encoder_hidden, self.encoder_cell_state, self.decoder_cell_state
        else:
            h0, c0, c_dec = self.init_hidden_states(b_size=b)

        out = self.pre_encoder(in_uv.view(-1, 2))
        out = out.view(seq_len, b, -1)
        out, (h_enc, c_enc) = self.encoder(out, (h0, c0))

        # Decoder
        # Last (x,y) and (u,v) position at T=8
        last_xy = in_xy[:, -1, ...]
        last_uv = in_uv[:, -1, ...]

        h_dec = h_enc  # [-1, ...]  # , torch.zeros_like(c_enc[-1, ...])

        for gt_pred_xy in gt_xy.permute(1, 0, 2):
            out = self.pre_decoder(last_uv)
            out, (h_dec, c_dec) = self.decoder(out.unsqueeze(0), (h_dec, c_dec))

            pred_uv = self.post_decoder(out.squeeze(0))
            if self.feed_model_distances_in_meters:
                pred_uv /= ratio[0].item()
            out = last_xy + (pred_uv * 0.4) if self.relative_velocities else last_xy + pred_uv

            predicted_xy_for_loss.append(out)
            predicted_xy.append(out.detach().cpu().numpy())
            true_xy.append(gt_pred_xy.detach().cpu().numpy())

            last_xy = out
            if self.feed_model_distances_in_meters:
                pred_uv *= ratio[0].item()
            last_uv = pred_uv

        ade = compute_ade(np.stack(predicted_xy, axis=1), np.stack(true_xy, axis=1)).item()
        fde = compute_fde(np.stack(predicted_xy, axis=1), np.stack(true_xy, axis=1),
                          batched_v2=True, batched=False).item()

        ade *= ratio[0].item()
        fde *= ratio[0].item()

        loss = torch.linalg.norm(gt_xy -
                                 torch.stack(predicted_xy_for_loss, dim=1),
                                 ord=2, dim=-1).mean() * ratio[0].item()

        if self.return_pred:
            return loss, ade, fde, ratio[0].item(), np.stack(predicted_xy)

        return loss, ade, fde, ratio[0].item()

    def one_step(self, batch):
        return self(batch)


class ConstantLinearBaseline(object):
    def __init__(self, xy=None, uv=None, prediction_length=12, relative_velocities=False):
        super(ConstantLinearBaseline, self).__init__()
        self.xy = xy
        self.uv = uv
        self.prediction_length = prediction_length

        self.trajectories = []

        if relative_velocities:
            self.uv *= 0.4

    def __call__(self, *args, **kwargs):
        if self.xy is None or self.uv is None:
            logger.error('xy or uv is missing!')
            raise RuntimeError()
        last_xy = self.xy
        for idx in range(self.prediction_length):
            new_xy = last_xy + self.uv
            self.trajectories.append(new_xy.tolist())
            last_xy = new_xy

        return np.stack(self.trajectories, axis=1)

    def eval(self, obs_trajectory, obs_distances, gt_trajectory, ratio):
        self.reset(xy=obs_trajectory[:, -1, ...], uv=obs_distances[:, -1, ...])

        pred_trajectory = self()

        ade = compute_ade(np.stack(pred_trajectory), np.stack(gt_trajectory)).item() * ratio[0]
        fde = compute_fde(np.stack(pred_trajectory), np.stack(gt_trajectory),
                          batched_v2=True, batched=False).item() * ratio[0]

        return pred_trajectory, ade, fde

    def reset(self, xy=None, uv=None):
        self.xy = xy if xy is not None else self.xy
        self.uv = uv if uv is not None else self.uv

        self.trajectories = []


class ConstantLinearBaselineV2(ConstantLinearBaseline):
    def __init__(self, xy=None, uv=None, prediction_length=12, relative_velocities=False, foreign_dataset=False):
        super(ConstantLinearBaselineV2, self).__init__(
            xy=xy, uv=uv, prediction_length=prediction_length, relative_velocities=relative_velocities)
        self.foreign_dataset = foreign_dataset

    def eval(self, obs_trajectory, obs_distances, gt_trajectory, ratio, batch_size):
        self.reset(xy=obs_trajectory[:, -1, ...], uv=obs_distances[:, -1, ...])

        pred_trajectory = self()

        pred_trajectory = torch.from_numpy(pred_trajectory).transpose(0, 1)
        gt_trajectory = torch.from_numpy(gt_trajectory).transpose(0, 1)

        gt_trajectory = gt_trajectory.view(gt_trajectory.shape[0], -1, batch_size, gt_trajectory.shape[-1])
        pred_trajectory = pred_trajectory.view(pred_trajectory.shape[0], -1, batch_size, pred_trajectory.shape[-1])

        ade, fde, best_idx = cal_ade_fde_stochastic(gt_trajectory, pred_trajectory)

        ade = ade.squeeze()
        fde = fde.squeeze()

        if not self.foreign_dataset:
            ade *= ratio.squeeze().cpu()
            fde *= ratio.squeeze().cpu()

        return pred_trajectory, ade, fde


if __name__ == '__main__':
    if OVERFIT:
        overfit_batches = 2
    else:
        overfit_batches = 0.0

    video_number: int = 3

    # dataset_train = BaselineDataset(SDDVideoClasses.LITTLE, video_number, NetworkMode.TRAIN,
    #                                 meta_label=SDDVideoDatasets.LITTLE)
    # dataset_val = BaselineDataset(SDDVideoClasses.LITTLE, video_number, NetworkMode.VALIDATION,
    #                               meta_label=SDDVideoDatasets.LITTLE)
    dataset_train = BaselineGeneratedDataset(SDDVideoClasses.LITTLE, video_number, NetworkMode.TRAIN,
                                             meta_label=SDDVideoDatasets.LITTLE)
    dataset_val = BaselineGeneratedDataset(SDDVideoClasses.LITTLE, video_number, NetworkMode.VALIDATION,
                                           meta_label=SDDVideoDatasets.LITTLE)

    m = BaselineRNNStackedSimple(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=BATCH_SIZE,
                                 num_workers=0, lr=LR, use_batch_norm=USE_BATCH_NORM, overfit_mode=OVERFIT,
                                 shuffle=True, pin_memory=True, generated_dataset=True, dropout=None, rnn_dropout=0,
                                 encoder_lstm_num_layers=1, return_pred=False)

    trainer = Trainer(gpus=1, max_epochs=NUM_EPOCHS)
    trainer.fit(model=m)
