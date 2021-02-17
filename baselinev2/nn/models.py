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
from baselinev2.nn.dataset import BaselineDataset
from baselinev2.plot_utils import plot_trajectories
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.nn.models')

torch.manual_seed(MANUAL_SEED)


def make_layers(cfg, batch_norm=False, encoder=True, last_without_activation=True,
                decoder_in_dim=LINEAR_CFG['lstm_encoder']):
    layers = []
    if encoder:
        in_features = 2
    else:
        in_features = decoder_in_dim
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
    Watch 8 time steps, predict next 12
    """

    def __init__(self, original_frame_shape=None, prediction_length=12, lr=1e-5, time_steps=5,
                 train_dataset=None, val_dataset=None, batch_size=1, num_workers=0, use_batch_norm=False,
                 lstm_num_layers: int = 1, overfit_mode: bool = False, shuffle: bool = False, pin_memory: bool = True):
        super(BaselineRNN, self).__init__()

        self.pre_encoder = make_layers(LINEAR_CFG['encoder'], batch_norm=use_batch_norm, encoder=True,
                                       last_without_activation=False)
        self.encoder = nn.LSTM(input_size=LINEAR_CFG['lstm_in'], hidden_size=LINEAR_CFG['lstm_encoder'],
                               num_layers=lstm_num_layers, bias=True)
        self.pre_decoder = make_layers(LINEAR_CFG['encoder'], batch_norm=use_batch_norm, encoder=True,
                                       last_without_activation=False)
        self.decoder = nn.LSTMCell(input_size=LINEAR_CFG['lstm_in'], hidden_size=LINEAR_CFG['lstm_encoder'])
        self.post_decoder = make_layers(LINEAR_CFG['decoder'], batch_norm=use_batch_norm, encoder=False,
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

        self.ts = time_steps

        self.save_hyperparameters('lr', 'time_steps', 'batch_size', 'use_batch_norm', 'overfit_mode', 'shuffle')

    def forward(self, x, two_losses=False):
        if two_losses:
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
        hx, cx = torch.zeros(size=(self.lstm_num_layers, b_size, LINEAR_CFG['lstm_encoder']), device=self.device), \
                 torch.zeros(size=(self.lstm_num_layers, b_size, LINEAR_CFG['lstm_encoder']), device=self.device)
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
        return torch.norm(point1 - point2, p=2)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=None,
                          num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size * 2, collate_fn=None,
                          num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory)

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
                'scheduler': ReduceLROnPlateau(opt, patience=15, verbose=True, factor=0.1, cooldown=2),
                'monitor': 'val_loss_epoch',
                'interval': 'epoch',
                'frequency': 1
            }]
        return [opt], schedulers


if __name__ == '__main__':
    if OVERFIT:
        overfit_batches = 2
    else:
        overfit_batches = 0.0

    video_number: int = 3

    # train_datasets, val_datasets = [], []
    # for v_idx, (video_class, meta) in enumerate(zip(SDD_VIDEO_CLASSES_LIST_FOR_NN, SDD_VIDEO_META_CLASSES_LIST_FOR_NN)):
    #     for video_number in SDD_PER_CLASS_VIDEOS_LIST_FOR_NN[v_idx]:
    #         train_datasets.append(BaselineDataset(video_class=video_class, video_number=video_number,
    #                                               split=NetworkMode.TRAIN, meta_label=meta))
    #         val_datasets.append(BaselineDataset(video_class=video_class, video_number=video_number,
    #                                             split=NetworkMode.VALIDATION, meta_label=meta))
    # dataset_train = ConcatDataset(datasets=train_datasets)
    # dataset_val = ConcatDataset(datasets=val_datasets)

    dataset_train = BaselineDataset(SDDVideoClasses.LITTLE, video_number, NetworkMode.TRAIN,
                                    meta_label=SDDVideoDatasets.LITTLE)
    dataset_val = BaselineDataset(SDDVideoClasses.LITTLE, video_number, NetworkMode.VALIDATION,
                                  meta_label=SDDVideoDatasets.LITTLE)

    m = BaselineRNN(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS, lr=LR, use_batch_norm=USE_BATCH_NORM, overfit_mode=OVERFIT,
                    shuffle=True, pin_memory=True)

    trainer = Trainer(gpus=1, max_epochs=NUM_EPOCHS)
    trainer.fit(model=m)
