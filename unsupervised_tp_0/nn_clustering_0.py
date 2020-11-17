# Simple clustering from simple NN
import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from average_image.feature_extractor import MOG2
from average_image.constants import FeaturesMode, SDDVideoClasses, SDDVideoDatasets
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames, FeaturesDataset, FeaturesDatasetExtra, \
    FeaturesDatasetCenterBased
from average_image.utils import BasicTrainData, BasicTestData, plot_extracted_features_and_verify_flow, \
    plot_points_predicted_and_true, plot_and_compare_points_predicted_and_true, \
    plot_points_predicted_and_true_center_only, SDDMeta, plot_points_predicted_and_true_center_only_rnn, \
    plot_trajectory_rnn, compute_ade, compute_fde, trajectory_length, plot_trajectory_rnn_compare, \
    plot_trajectory_rnn_compare_side_by_side, plot_trajectory_rnn_tb, is_inside_bbox, plot_bars_if_inside_bbox, \
    compute_per_stop_de, plot_track_analysis, plot_violin_plot

initialize_logging()
logger = get_logger(__name__)

torch.manual_seed(42)
FIG_SAVE_EPOCH = 199
COLLATE_TS = 5


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
    def __init__(self, meta=None, original_frame_shape=None, num_frames_to_build_bg_sub_model=12, lr=1e-5,
                 mode=FeaturesMode.UV, layers_mode=None, meta_video=None, meta_train_video_number=None,
                 meta_val_video_number=None, time_steps=5, train_dataset=None, val_dataset=None, batch_size=1,
                 num_workers=0):
        super(SimpleModel, self).__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        if mode == FeaturesMode.UV:
            in_features = 2
        elif mode == FeaturesMode.XYUV:
            in_features = 4
        else:
            raise Exception
        if layers_mode == 'small':
            self.layers = nn.Sequential(nn.Linear(in_features, 8),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(8, 16),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(16, 32),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(32, 16),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(16, 8),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(8, in_features))
        # elif layers_mode == 'rnn':
        #     self.block_1 = nn.Sequential(nn.Linear(in_features, 8),
        #                                  # nn.BatchNorm1d(8),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(8, 16),
        #                                  # nn.BatchNorm1d(16),
        #                                  nn.ReLU(inplace=True))
        #     self.rnn_cell = nn.LSTMCell(input_size=16, hidden_size=32)
        #     self.block_2 = nn.Sequential(nn.ReLU(),
        #                                  nn.Linear(32, 16),
        #                                  # nn.BatchNorm1d(16),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(16, 8),
        #                                  # nn.BatchNorm1d(8),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(8, in_features))
        # elif layers_mode == 'rnn':
        #     self.block_1 = nn.Sequential(nn.Linear(in_features, 4),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(4, 8),
        #                                  nn.ReLU(inplace=True))
        #     self.rnn_cell = nn.LSTMCell(input_size=8, hidden_size=16)
        #     self.rnn_cell_1 = nn.LSTMCell(input_size=16, hidden_size=32)
        #     self.block_2 = nn.Sequential(nn.Linear(32, 16),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(16, 8),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(8, 4),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(4, in_features))
        elif layers_mode == 'rnn':
            self.block_1 = nn.Sequential(nn.Linear(in_features, 4),
                                         # nn.BatchNorm1d(4),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4, 8),
                                         # nn.BatchNorm1d(8),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(8, 16),
                                         # nn.BatchNorm1d(16),
                                         nn.ReLU(inplace=True))
            self.rnn_cell = nn.LSTMCell(input_size=16, hidden_size=32)
            self.rnn_cell_1 = nn.LSTMCell(input_size=32, hidden_size=64)
            self.block_2 = nn.Sequential(nn.Linear(64, 32),
                                         # nn.BatchNorm1d(32),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(32, 16),
                                         # nn.BatchNorm1d(16),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(16, 8),
                                         # nn.BatchNorm1d(8),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(8, 4),
                                         # nn.BatchNorm1d(4),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4, in_features))
        else:
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
        self.meta = meta
        self.train_ratio = float(self.meta.get_meta(meta_video, meta_train_video_number)[0]['Ratio'].to_numpy()[0])
        self.val_ratio = float(self.meta.get_meta(meta_video, meta_val_video_number)[0]['Ratio'].to_numpy()[0])
        self.ts = time_steps
        self.rnn_mode = True if layers_mode == 'rnn' else False

        self.save_hyperparameters('lr', 'time_steps')

    def forward(self, x, cx=None, hx=None, cx_1=None, hx_1=None, one_layer=False):
        if self.rnn_mode and not one_layer:
            block_1 = self.block_1(x)
            hx, cx = self.rnn_cell(block_1, (hx, cx))
            hx = F.relu(hx, inplace=True)
            hx_1, cx_1 = self.rnn_cell_1(hx, (hx_1, cx_1))
            hx_1 = F.relu(hx_1, inplace=True)
            out = self.block_2(hx_1)
            return out, cx, hx, cx_1, hx_1
        elif self.rnn_mode and one_layer:
            block_1 = self.block_1(x)
            hx, cx = self.rnn_cell(block_1, (hx, cx))
            out = self.block_2(hx)
            return out, cx, hx, cx_1, hx_1
        else:
            out = self.layers(x)
            return out

    def _one_step_rnn(self, batch, log_plot=False, of_based=True, gt_all_points_based=False,
                      gt_bbox_center_based=False, center_based_loss=True):
        features1, features2, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch

        # hx, cx = torch.zeros(size=(1, 16), device=self.device), torch.zeros(size=(1, 16), device=self.device)
        # hx_1, cx_1 = torch.zeros(size=(1, 32), device=self.device), torch.zeros(size=(1, 32), device=self.device)
        hx, cx = torch.zeros(size=(1, 32), device=self.device), torch.zeros(size=(1, 32), device=self.device)
        hx_1, cx_1 = torch.zeros(size=(1, 64), device=self.device), torch.zeros(size=(1, 64), device=self.device)

        torch.nn.init.xavier_normal_(hx)
        torch.nn.init.xavier_normal_(cx)
        torch.nn.init.xavier_normal_(hx_1)
        torch.nn.init.xavier_normal_(cx_1)

        total_loss = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        last_input_velocity = None
        last_pred_center = None
        bad_data = False
        pred_centers = []
        moved_points_by_true_of_list = []
        actual_points_list = []
        # all_fig = []
        ade = None
        fde = None
        fig = None
        shifted_points = None

        if of_based and not center_based_loss:
            criterion = self.cluster_center_loss_meters
        elif of_based and center_based_loss:
            criterion = self.cluster_center_points_center_loss_meters
        elif gt_all_points_based:
            criterion = self.cluster_center_loss_meters
        else:
            criterion = self.cluster_center_points_center_loss_meters_bbox_gt

        if len(features1) == self.ts:
            for idx_i, i in enumerate(range(self.ts)):
                feat1, feat2, bb_center1, bb_center2 = features1[i].squeeze().float(), features2[i].squeeze().float(), \
                                                       bbox_center1[i].float(), bbox_center2[i].float()
                try:
                    center_xy, center_true_uv, center_past_uv = self.find_center_point(feat1)
                    if gt_all_points_based:
                        shifted_points = feat2[:, :2]
                    if gt_bbox_center_based:
                        shifted_points = bb_center2
                except IndexError:
                    bad_data = True
                    break

                if idx_i == 0:
                    last_input_velocity = center_past_uv.unsqueeze(0)  # * 0.4  # bug workaround
                    last_pred_center = center_xy

                block_2, cx, hx, cx_1, hx_1 = self(last_input_velocity, cx, hx, cx_1, hx_1, True)

                if of_based:
                    shifted_points = feat1[:, :2] + feat1[:, 2:4]

                moved_points_by_true_of_list.append(shifted_points)
                pred_center = last_pred_center + block_2
                last_input_velocity = block_2
                last_pred_center = pred_center
                pred_centers.append(pred_center.squeeze(0))
                actual_points_list.append(feat1[:, :2].detach().cpu().mean(dim=0).numpy())

                total_loss += criterion(points=shifted_points, pred_center=pred_center.squeeze(0))

            predicted_points = [p.detach().cpu().numpy() for p in pred_centers]
            true_points = [p.detach().cpu().mean(dim=0).numpy() for p in moved_points_by_true_of_list]

            if len(predicted_points) != 0 or len(true_points) != 0:
                ade = compute_ade(np.stack(predicted_points), np.stack(true_points)).item()
                fde = compute_fde(np.stack(predicted_points), np.stack(true_points)).item()

                if log_plot and self.current_epoch % FIG_SAVE_EPOCH == 0:
                    fig = plot_trajectory_rnn_tb(predicted_points=np.stack(predicted_points),
                                                 true_points=np.stack(true_points),
                                                 actual_points=actual_points_list,
                                                 imgs=None, gt=False)
                    # all_fig.append(fig)
        # Fixme: fix dataset, remove hacks
        else:
            total_loss = torch.zeros(size=(5, 5), device=self.device, requires_grad=True).mean()

        if bad_data:
            total_loss = torch.zeros(size=(5, 5), device=self.device, requires_grad=True).mean()

        return total_loss / len(features1), ade, fde, fig

    def _one_step_rnn_center_based(self, batch, log_plot=False, of_based=False, gt_all_points_based=False,
                                   gt_bbox_center_based=True, center_based_loss=False):
        features, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch
        b_size = features.shape[1]

        # hx, cx = torch.zeros(size=(1, 16), device=self.device), torch.zeros(size=(1, 16), device=self.device)
        # hx_1, cx_1 = torch.zeros(size=(1, 32), device=self.device), torch.zeros(size=(1, 32), device=self.device)
        hx, cx = torch.zeros(size=(b_size, 32), device=self.device), \
                 torch.zeros(size=(b_size, 32), device=self.device)
        hx_1, cx_1 = torch.zeros(size=(b_size, 64), device=self.device), \
                     torch.zeros(size=(b_size, 64), device=self.device)

        torch.nn.init.xavier_normal_(hx)
        torch.nn.init.xavier_normal_(cx)
        torch.nn.init.xavier_normal_(hx_1)
        torch.nn.init.xavier_normal_(cx_1)

        total_loss = torch.tensor(data=0, dtype=torch.float32, device=self.device)
        last_input_velocity = None
        last_pred_center = None
        bad_data = False
        pred_centers = []
        moved_points_by_true_of_list = []
        actual_points_list = []
        # all_fig = []
        ade = None
        fde = None
        fig = None
        shifted_points = None

        if of_based and not center_based_loss:
            criterion = self.cluster_center_loss_meters
        elif of_based and center_based_loss:
            criterion = self.cluster_center_points_center_loss_meters
        elif gt_all_points_based:
            criterion = self.cluster_center_loss_meters
        else:
            criterion = self.cluster_center_points_center_loss_meters_bbox_gt

        if len(features) == self.ts:
            for idx_i, i in enumerate(range(self.ts)):
                feat, bb_center1, bb_center2 = features[i].squeeze().float(), bbox_center1[i].float(), \
                                               bbox_center2[i].float()
                try:
                    center_xy, center_true_uv, center_past_uv, shifted_point = feat[:, 0, :], feat[:, 1, :], \
                                                                               feat[:, 2, :], feat[:, 3, :]
                    if gt_all_points_based:
                        return NotImplemented
                    if gt_bbox_center_based:
                        shifted_points = bb_center2
                except IndexError:
                    bad_data = True
                    break

                if idx_i == 0:
                    last_input_velocity = center_past_uv  # * 0.4  # bug workaround
                    last_pred_center = center_xy

                block_2, cx, hx, cx_1, hx_1 = self(last_input_velocity, cx, hx, cx_1, hx_1, False)

                if of_based:
                    shifted_points = shifted_point

                moved_points_by_true_of_list.append(shifted_points)
                pred_center = last_pred_center + block_2
                last_input_velocity = block_2
                last_pred_center = pred_center
                pred_centers.append(pred_center)
                actual_points_list.append(bb_center2)

                total_loss += criterion(points=shifted_points, pred_center=pred_center)

            predicted_points = [p.detach().cpu().numpy() for p in pred_centers]
            # true_points = [p.detach().cpu().numpy() for p in moved_points_by_true_of_list]
            true_points = [p.detach().cpu().numpy() for p in actual_points_list]

            if len(predicted_points) != 0 or len(true_points) != 0:
                ade = compute_ade(np.stack(predicted_points), np.stack(true_points)).item()
                fde = compute_fde(np.stack(predicted_points), np.stack(true_points)).item()

                if log_plot and self.current_epoch % FIG_SAVE_EPOCH == 0:
                    fig = plot_trajectory_rnn_tb(predicted_points=np.stack(predicted_points),
                                                 true_points=np.stack(true_points),
                                                 actual_points=actual_points_list,
                                                 imgs=None, gt=False)
                    # all_fig.append(fig)
        # Fixme: fix dataset, remove hacks
        else:
            total_loss = torch.zeros(size=(5, 5), device=self.device, requires_grad=True).mean()

        if bad_data:
            total_loss = torch.zeros(size=(5, 5), device=self.device, requires_grad=True).mean()

        return total_loss / len(features), ade, fde, fig

    def _one_step(self, batch):
        features1, features2 = batch
        features1, features2 = features1.squeeze().float(), features2.squeeze().float()
        pred = self(features1[:, 4:])
        current_center = self._calculate_mean(features1[:, :2])
        # center of predicted points
        pred_center = self._calculate_mean(pred)
        # move points based on optical flow
        shifted_points = pred + features1[:, 2:]
        # the pred center and actual center should be close
        reg = self.center_based_loss(pred_center, current_center)
        # center should be close to shifted points
        loss = self.cluster_center_loss(points=shifted_points, pred_center=pred_center)
        return loss

    def _one_step_all_points(self, batch):
        features1, features2 = batch
        features1, features2 = features1.squeeze().float(), features2.squeeze().float()
        # input prior velocity
        try:
            pred = self(features1[:, 4:])
        except IndexError:
            return torch.zeros((5, 5), requires_grad=True).float().mean()
        # predicted optical flow moved points
        moved_points_pred_of = features1[:, :2] + pred
        # move points by actual optical flow
        moved_points_by_true_of = features1[:, :2] + features1[:, 2:4]
        # center of predicted points
        pred_center = self._calculate_mean(moved_points_pred_of)
        # center should be close to shifted points
        loss = self.cluster_center_loss(points=moved_points_by_true_of, pred_center=pred_center)
        return loss

    def _one_step_center_only(self, batch):
        features1, features2, frame_, track_id_ = batch
        features1, features2 = features1.squeeze().float(), features2.squeeze().float()
        try:
            # find the center
            center_xy, center_true_uv, center_past_uv = self.find_center_point(features1)
            pred = self(center_past_uv)
        except IndexError:
            return torch.zeros((5, 5), requires_grad=True).float().mean()

        # predicted optical flow moved points
        # moved_points_pred_of = features1[:, :2] + pred
        # move points by actual optical flow
        moved_points_by_true_of = features1[:, :2] + features1[:, 2:4]
        # center of predicted points
        # pred_center = self._calculate_mean(moved_points_pred_of)
        pred_center = center_xy + pred
        # center should be close to shifted points
        loss = self.cluster_center_loss_meters(points=moved_points_by_true_of, pred_center=pred_center)
        return loss

    def find_center_point(self, points):
        xy = points[:, :2]
        mean = self._calculate_mean(xy)
        idx = None
        dist = np.inf
        for p, point in enumerate(xy):
            # d = F.mse_loss(point, mean)
            d = torch.norm(point - mean, p=2)
            if d < dist:
                idx = p
                dist = d
        return points[idx, :2], points[idx, 2:4], points[idx, 4:]

    def training_step(self, batch, batch_idx):
        # loss = self._one_step_center_only(batch)
        # loss = self._one_step_all_points(batch)
        # loss, ade, fde, fig = self._one_step_rnn(batch)
        loss, ade, fde, fig = self._one_step_rnn_center_based(batch)
        # tensorboard_logs = {'train_loss': loss}
        self.logger.experiment.add_scalar('lr', self.lr)
        if ade is None or fde is None:
            tensorboard_logs = {'train_loss': loss, 'train/ade': 0, 'train/fde': 0}
            # self.log('train/loss', loss, on_step=True, on_epoch=True)
            # self.log('train/ade', 0, on_step=True, on_epoch=True)
            # self.log('train/fde', 0, on_step=True, on_epoch=True)
        else:
            tensorboard_logs = {'train_loss': loss, 'train/ade': ade * self.train_ratio,
                                'train/fde': fde * self.train_ratio}
            # self.log('train/loss', loss, on_step=True, on_epoch=True)
            # self.log('train/ade', ade * self.train_ratio, on_step=True, on_epoch=True)
            # self.log('train/fde', fde * self.train_ratio, on_step=True, on_epoch=True)
        # if fig is not None:
        #     self.logger.experiment.add_figure('train/trajectory', fig, self.current_epoch)
        # return {'loss': loss}
        # if self.current_epoch % 49 == 0 and fig is not None:
        #     fig.savefig(f'/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/unsupervised_tp_0/lightning_logs/'
        #                 f'trajectory_plot/fig_epoch_train_{self.current_epoch}.png')
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # loss = self._one_step_center_only(batch)
        # loss = self._one_step_all_points(batch)
        # loss, ade, fde, fig = self._one_step_rnn(batch)
        loss, ade, fde, fig = self._one_step_rnn_center_based(batch)
        if ade is None or fde is None:
            tensorboard_logs = {'val_loss': loss, 'val/ade': 0, 'val/fde': 0}
            # self.log('val/loss', loss, on_step=True, on_epoch=True)
            # self.log('val/ade', 0, on_step=True, on_epoch=True)
            # self.log('val/fde', 0, on_step=True, on_epoch=True)
        else:
            tensorboard_logs = {'val_loss': loss, 'val/ade': ade * self.val_ratio, 'val/fde': fde * self.val_ratio}
            # self.log('val/loss', loss, on_step=True, on_epoch=True)
            # self.log('val/ade', ade * self.val_ratio, on_step=True, on_epoch=True)
            # self.log('val/fde', fde * self.val_ratio, on_step=True, on_epoch=True)
        # if fig is not None:
        #     self.logger.experiment.add_figure('val/trajectory', fig, self.current_epoch)
        # return {'loss': loss}
        # if self.current_epoch % FIG_SAVE_EPOCH == 0 and fig is not None:
        #     fig.savefig(f'/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/unsupervised_tp_0/lightning_logs/'
        #                 f'trajectory_plot/fig_epoch_val_{self.current_epoch}.png')

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
        # scheduler = ReduceLROnPlateau(opt, patience=5, verbose=True)
        return [opt], schedulers

    def preprocess_data(self, data_loader, mode: FeaturesMode, annotation_df, classic_clustering=False,
                        test_mode=False, equal_time_distributed=True, normalized=True,
                        one_step_feature_extraction=False, save_per_part_path=None, resume_idx=None):
        # original_shape=None, resize_shape=None):
        save_per_part_path += 'parts/'
        remaining_frames = None
        remaining_frames_idx = None
        past_12_frames_optical_flow = []
        last_frame_from_last_used_batch = None
        accumulated_features = {}
        x_, y_ = [], []
        for part_idx, data in enumerate(tqdm(data_loader)):
            # frames, bbox, centers = data  # read frames from loader and frame number and get annotations + centers
            # here
            # frames, bbox, centers = frames.squeeze(), bbox.squeeze(), centers.squeeze()
            if resume_idx is not None:
                if part_idx < resume_idx:
                    continue
            frames, frame_numbers = data
            frames = frames.squeeze()
            feature_extractor = MOG2.for_frames()
            features_, remaining_frames, remaining_frames_idx, last_frame_from_last_used_batch, \
            past_12_frames_optical_flow = feature_extractor. \
                keyframe_based_clustering_from_frames_nn(frames=frames, n=30, use_last_n_to_build_model=False,
                                                         frames_to_build_model=self.num_frames_to_build_bg_sub_model,
                                                         original_shape=data_loader.dataset.original_shape,
                                                         resized_shape=data_loader.dataset.new_scale,
                                                         classic_clustering=classic_clustering,
                                                         object_of_interest_only=False,
                                                         var_threshold=None, track_ids=None,
                                                         all_object_of_interest_only=True,
                                                         equal_time_distributed=equal_time_distributed,
                                                         frame_numbers=frame_numbers,
                                                         df=annotation_df,
                                                         return_normalized=False,  # todo: change back to variable
                                                         remaining_frames=remaining_frames,
                                                         remaining_frames_idx=remaining_frames_idx,
                                                         past_12_frames_optical_flow=past_12_frames_optical_flow,
                                                         last_frame_from_last_used_batch=
                                                         last_frame_from_last_used_batch)
            plot_extracted_features_and_verify_flow(features_, frames)
            accumulated_features = {**accumulated_features, **features_}
            if save_per_part_path is not None:
                Path(save_per_part_path).mkdir(parents=True, exist_ok=True)
                f_n = f'time_distributed_dict_with_gt_bbox_centers_and_bbox_part{part_idx}.pt'
                torch.save(accumulated_features, save_per_part_path + f_n)

            if one_step_feature_extraction:
                if mode == FeaturesMode.UV:
                    # features, cluster_centers = self._process_features(features_, uv=True,
                    #                                                    classic_clustering=classic_clustering)
                    features = self._process_complex_features(features_, uv=True, test_mode=test_mode)
                    x, y = self._extract_trainable_features(features)
                    x_ += x
                    y_ += y
                if mode == FeaturesMode.XYUV:
                    return NotImplemented
                features, cluster_centers = self._process_features(features_, uv=False,
                                                                   classic_clustering=classic_clustering)
                return features, cluster_centers
        return accumulated_features

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

    def _process_complex_features(self, features_dict: dict, uv=True, test_mode=False):
        if uv:
            f = 2
        else:
            f = 4
        train_data = {}
        per_frame_data = []
        total_frames = len(features_dict)
        for frame_ in tqdm(range(self.num_frames_to_build_bg_sub_model,
                                 total_frames - self.num_frames_to_build_bg_sub_model)):
            pair_0 = features_dict[frame_]
            pair_1 = features_dict[(frame_ + self.num_frames_to_build_bg_sub_model) % total_frames]
            for i in pair_0:
                for j in pair_1:
                    if i == j:  # and i.track_id == 8:  # bad readability
                        if test_mode:
                            per_frame_data.append(BasicTestData(frame=frame_, track_id=i.track_id,
                                                                pair_0_features=i.features,
                                                                pair_1_features=j.features,
                                                                pair_0_normalize=i.normalize_params,
                                                                pair_1_normalize=j.normalize_params))
                        else:
                            per_frame_data.append(BasicTrainData(frame=frame_, track_id=i.track_id,
                                                                 pair_0_features=i.features,
                                                                 pair_1_features=j.features))
            train_data.update({frame_: per_frame_data})
            per_frame_data = []
        return train_data

    def _process_complex_features_rnn(self, features_dict: dict, test_mode=False, time_steps=5):
        train_data = {}
        per_frame_data = []
        per_batch_data = []
        total_frames = len(features_dict)
        for frame_ in tqdm(range(self.num_frames_to_build_bg_sub_model,
                                 total_frames - self.num_frames_to_build_bg_sub_model * time_steps)):
            for k in range(1, time_steps + 1):
                pair_0 = features_dict[frame_ + self.num_frames_to_build_bg_sub_model * (k - 1)]
                pair_1 = features_dict[(frame_ + self.num_frames_to_build_bg_sub_model * k) % total_frames]
                for i in pair_0:
                    for j in pair_1:
                        if i == j:  # and i.track_id == 8:  # bad readability
                            if test_mode:
                                per_frame_data.append(BasicTestData(frame=frame_, track_id=i.track_id,
                                                                    pair_0_features=i.features,
                                                                    pair_1_features=j.features,
                                                                    pair_0_normalize=i.normalize_params,
                                                                    pair_1_normalize=j.normalize_params))
                            else:
                                per_frame_data.append(
                                    BasicTrainData(frame=frame_ + self.num_frames_to_build_bg_sub_model * (k - 1),
                                                   track_id=i.track_id,
                                                   pair_0_features=i.features,
                                                   pair_1_features=j.features,
                                                   frame_t0=i.frame_number,
                                                   frame_t1=j.frame_number,
                                                   bbox_center_t0=i.bbox_center,
                                                   bbox_center_t1=j.bbox_center,
                                                   bbox_t0=i.bbox,
                                                   bbox_t1=j.bbox))
                                # per_frame_data.append([i.features, j.features])
                per_batch_data.append(per_frame_data)
                per_frame_data = []
            train_data.update({frame_: per_batch_data})
            per_batch_data = []
        return train_data

    def analyze_complex_features_rnn(self, features_dict: dict, test_mode=False, time_steps=5, track_info=None,
                                     ratio=1.0):
        train_data = {}
        per_frame_data = []
        per_batch_data = []
        total_frames = len(features_dict)
        for frame_ in tqdm(range(self.num_frames_to_build_bg_sub_model,
                                 total_frames - self.num_frames_to_build_bg_sub_model * time_steps)):
            for k in range(1, time_steps + 1):
                pair_0 = features_dict[frame_ + self.num_frames_to_build_bg_sub_model * (k - 1)]
                pair_1 = features_dict[(frame_ + self.num_frames_to_build_bg_sub_model * k) % total_frames]
                for i in pair_0:
                    for j in pair_1:
                        if i == j:  # and i.track_id == 8:  # bad readability
                            if test_mode:
                                per_frame_data.append(BasicTestData(frame=frame_, track_id=i.track_id,
                                                                    pair_0_features=i.features,
                                                                    pair_1_features=j.features,
                                                                    pair_0_normalize=i.normalize_params,
                                                                    pair_1_normalize=j.normalize_params))
                            else:
                                per_frame_data.append(
                                    BasicTrainData(frame=frame_ + self.num_frames_to_build_bg_sub_model * (k - 1),
                                                   track_id=i.track_id,
                                                   pair_0_features=i.features,
                                                   pair_1_features=j.features,
                                                   frame_t0=i.frame_number,
                                                   frame_t1=j.frame_number,
                                                   bbox_center_t0=i.bbox_center,
                                                   bbox_center_t1=j.bbox_center,
                                                   bbox_t0=i.bbox,
                                                   bbox_t1=j.bbox))
                                # per_frame_data.append([i.features, j.features])
                per_batch_data.append(per_frame_data)
                per_frame_data = []
            train_data.update({frame_: per_batch_data})
            per_batch_data = []

        x_, y_, frame_info, track_id_info, bbox_center_x, bbox_center_y, bbox_x, bbox_y = \
            self._extract_trainable_features_rnn(train_data)
        of_track_analysis = {}
        of_track_analysis_df = None
        for features_u, features_v, features_f_info, features_t_info, features_b_c_x, features_b_c_y, features_b_x, \
            features_b_y in tqdm(zip(x_, y_, frame_info, track_id_info, bbox_center_x,
                                     bbox_center_y, bbox_x, bbox_y)):
            unique_tracks = np.unique(features_t_info)
            current_track = unique_tracks[0]
            of_inside_bbox_list = []
            of_track_list = []
            gt_track_list = []
            of_ade_list = []
            of_fde_list = []
            of_per_stop_de = []
            for u, v, f_info, t_info, b_c_x, b_c_y, b_x, b_y in zip(features_u, features_v, features_f_info,
                                                                    features_t_info, features_b_c_x, features_b_c_y,
                                                                    features_b_x, features_b_y):
                of_flow = u[:, :2] + u[:, 2:4]
                of_flow_center = of_flow.mean(0)
                of_inside_bbox = is_inside_bbox(of_flow_center, b_y)
                of_inside_bbox_list.append(of_inside_bbox)

                of_track_list.append(of_flow_center)
                gt_track_list.append(b_c_y)

            of_ade = compute_ade(np.stack(of_track_list), np.stack(gt_track_list))
            of_fde = compute_fde(np.stack(of_track_list), np.stack(gt_track_list))
            of_ade_list.append(of_ade.item() * ratio)
            of_fde_list.append(of_fde.item() * ratio)

            per_stop_de = compute_per_stop_de(np.stack(of_track_list), np.stack(gt_track_list))
            of_per_stop_de.append(per_stop_de)

            if len(unique_tracks) == 1:
                d = {'track_id': current_track,
                     'of_inside_bbox_list': of_inside_bbox_list,
                     'ade': of_ade.item() * ratio,
                     'fde': of_fde.item() * ratio,
                     'per_stop_de': [p * ratio for p in per_stop_de]}
                if of_track_analysis_df is None:
                    of_track_analysis_df = pd.DataFrame(data=d)
                else:
                    temp_df = pd.DataFrame(data=d)
                    of_track_analysis_df = of_track_analysis_df.append(temp_df, ignore_index=False)
                # of_track_analysis.update({current_track: {
                #     'of_inside_bbox_list': of_inside_bbox_list,
                #     'ade': of_ade.item() * ratio,
                #     'fde': of_fde.item() * ratio,
                #     'per_stop_de': [p * ratio for p in per_stop_de]}})
            else:
                logger.info(f'Found multiple tracks! - {unique_tracks}')

        return of_track_analysis_df

    def _extract_trainable_features(self, train_data, frame_info=True):
        frame_info = []
        track_id_info = []
        x_ = []
        y_ = []
        for key, value in train_data.items():
            for data in value:
                frame_info.append(key)
                x_.append(data.pair_0_features)
                y_.append(data.pair_1_features)
                track_id_info.append(data.track_id)
        if frame_info:
            return x_, y_, frame_info, track_id_info
        return x_, y_

    def _extract_trainable_features_rnn(self, train_data, return_frame_info=True):
        frame_info = []
        track_id_info = []
        x_ = []
        y_ = []
        bbox_center_x = []
        bbox_center_y = []
        bbox_x = []
        bbox_y = []
        for key, value in tqdm(train_data.items()):
            num_frames = len(value)
            t_0 = value[0]
            t_rest = [value[v] for v in range(1, num_frames)]
            for fr in t_0:
                temp_x = []
                temp_y = []
                temp_f_info = []
                temp_track_info = []
                temp_bbox_center_x = []
                temp_bbox_center_y = []
                temp_bbox_x = []
                temp_bbox_y = []

                temp_f_info.append(fr.frame)
                temp_track_info.append(fr.track_id)
                temp_x.append(fr.pair_0_features)
                temp_y.append(fr.pair_1_features)
                temp_bbox_center_x.append(fr.bbox_center_t0)
                temp_bbox_center_y.append(fr.bbox_center_t1)
                temp_bbox_x.append(fr.bbox_t0)
                temp_bbox_y.append(fr.bbox_t1)
                for t_i in t_rest:
                    for fr_other in t_i:
                        if fr == fr_other:
                            temp_f_info.append(fr_other.frame)
                            temp_track_info.append(fr_other.track_id)
                            temp_x.append(fr_other.pair_0_features)
                            temp_y.append(fr_other.pair_1_features)
                            temp_bbox_center_x.append(fr_other.bbox_center_t0)
                            temp_bbox_center_y.append(fr_other.bbox_center_t1)
                            temp_bbox_x.append(fr_other.bbox_t0)
                            temp_bbox_y.append(fr_other.bbox_t1)
                frame_info.append(temp_f_info)
                track_id_info.append(temp_track_info)
                x_.append(temp_x)
                y_.append(temp_y)
                bbox_center_x.append(temp_bbox_center_x)
                bbox_center_y.append(temp_bbox_center_y)
                bbox_x.append(temp_bbox_x)
                bbox_y.append(temp_bbox_y)

        if return_frame_info:
            return x_, y_, frame_info, track_id_info, bbox_center_x, bbox_center_y, bbox_x, bbox_y
        return x_, y_

    def _extract_test_features(self, train_data):
        # No use
        frame = []
        track_id = []
        x_ = []
        y_ = []
        x_normalize = []
        y_normalize = []
        for key, value in train_data.items():
            for data in value:
                x_.append(data.pair_0_features)
                y_.append(data.pair_1_features)
                x_normalize.append(data.pair_0_normalize)
                y_normalize.append(data.pair_1_normalize)
                frame.append(data.frame)
                track_id.append(data.track_id)
        return x_, y_, x_normalize, y_normalize, track_id, frame

    def center_based_loss(self, pred_center, target_center):
        return F.mse_loss(pred_center, target_center)

    def cluster_center_loss(self, points, pred_center):
        loss = 0
        for point in points:
            loss += F.mse_loss(pred_center, point)
        return loss

    def cluster_center_loss_meters(self, points, pred_center):
        if self.training:
            to_m = self.train_ratio
        else:
            to_m = self.val_ratio
        loss = 0
        for point in points:
            loss += self.l2_norm(pred_center, point) * to_m
        return loss / len(points)

    def cluster_center_points_center_loss_meters(self, points, pred_center):
        if self.training:
            to_m = self.train_ratio
        else:
            to_m = self.val_ratio
        # points_center = points.mean(dim=0)  # fixme: for center_batch based
        points_center = points
        loss = self.l2_norm(pred_center, points_center) * to_m
        return loss

    def cluster_center_points_center_loss_meters_bbox_gt(self, points, pred_center):
        if self.training:
            to_m = self.train_ratio
        else:
            to_m = self.val_ratio
        # points == points_center
        loss = self.l2_norm(pred_center, points) * to_m
        return loss

    def l2_norm(self, point1, point2):
        return torch.norm(point1 - point2, p=2)

    def _calculate_mean(self, points):
        return points.mean(axis=0)

    def train_dataloader(self):
        # train_dataset = FeaturesDataset(train_x, train_y, mode=FeaturesMode.UV, preprocess=False)
        # train_dataset__ = FeaturesDatasetExtra(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
        #                                        mode=FeaturesMode.UV, preprocess=False, bbox_center_x=train_center_x,
        #                                        bbox_center_y=train_center_y)
        return torch.utils.data.DataLoader(self.train_dataset, self.batch_size, collate_fn=center_dataset_collate,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, self.batch_size * 2, collate_fn=center_dataset_collate,
                                           num_workers=self.num_workers)


def get_track_info(annotation_path):
    df = pd.read_csv(annotation_path)
    tracks = df['track_id'].to_numpy()
    track, count = np.unique(tracks, return_counts=True)
    return dict(zip(track, count))


def parse_df_analysis(in_df, save_path=None):
    t_id_list, ade_list, fde_list = [], [], []
    inside_bbox_list, per_stop_de_list, inside_bbox, per_stop_de = [], [], [], []
    inside_bbox_count, outside_bbox_count = [], []
    t_id, ade, fde = None, None, None
    for idx, (index, row) in enumerate(tqdm(in_df.iterrows())):
        if idx == 0:
            t_id = row['track_id']
            ade = row['ade']
            fde = row['fde']
        if row['Unnamed: 0'] == 0:
            if idx != 0:
                t_id_list.append(t_id)
                ade_list.append(ade)
                fde_list.append(fde)
                inside_bbox_list.append(inside_bbox)
                per_stop_de_list.append(per_stop_de)
                inside_bbox_count.append(inside_bbox.count(True))
                outside_bbox_count.append(inside_bbox.count(False))
                if idx % 99 == 0:
                    plot_track_analysis(t_id, ade, fde, inside_bbox, per_stop_de, save_path + 'plots/', idx)
                # plot_track_analysis(t_id, ade, fde, inside_bbox, per_stop_de, save_path+'plots/', idx)
                inside_bbox, per_stop_de = [], []
            t_id = row['track_id']
            ade = row['ade']
            fde = row['fde']
            inside_bbox.append(row['of_inside_bbox_list'])
            per_stop_de.append(row['per_stop_de'])
        else:
            inside_bbox.append(row['of_inside_bbox_list'])
            per_stop_de.append(row['per_stop_de'])
    plot_violin_plot(ade_list, fde_list, save_path)
    in_count = sum(inside_bbox_count)
    out_count = sum(outside_bbox_count)
    print(f'% inside = {(in_count / (in_count + out_count)) * 100}')
    return t_id_list, ade_list, fde_list, inside_bbox_list, per_stop_de_list


def find_center_point(points):
    xy = points[:, :2]
    mean = xy.mean(axis=0)
    idx = None
    dist = np.inf
    for p, point in enumerate(xy):
        d = np.linalg.norm(point - mean, 2)
        if d < dist:
            idx = p
            dist = d
    return points[idx, :2], points[idx, 2:4], points[idx, 4:]


def center_based_dataset(features):
    features_x, features_y, frames, track_ids, center_x, center_y, bbox_x, bbox_y = features['x'], features['y'], \
                                                                                    features['frames'], \
                                                                                    features['track_ids'], features[
                                                                                        'bbox_center_x'], \
                                                                                    features['bbox_center_y'], features[
                                                                                        'bbox_x'], \
                                                                                    features['bbox_y']
    features_center = []
    for feat1 in tqdm(features_x):
        f1_list = []
        for f1 in feat1:
            center_xy, center_true_uv, center_past_uv = find_center_point(f1)
            shifted_points = (f1[:, :2] + f1[:, 2:4]).mean(axis=0)
            f1_list.append(np.vstack((center_xy, center_true_uv, center_past_uv, shifted_points)))
        features_center.append(f1_list)
    save_dict = {'center_based': features_center,
                 'x': features_x,
                 'y': features_y,
                 'frames': frames,
                 'track_ids': track_ids,
                 'bbox_center_x': center_x,
                 'bbox_center_y': center_y,
                 'bbox_x': bbox_x,
                 'bbox_y': bbox_y}
    return save_dict


def center_dataset_collate(batch):
    center_features_list, frames_batched_list, track_ids_batched_list, bbox_center_x_batched_list, \
    bbox_center_y_batched_list, bbox_x_batched_list, bbox_y_batched_list = [], [], [], [], [], [], []
    for data in batch:
        center_features = np.zeros(shape=(0, 4, 2))
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
        if center_features.shape[0] == COLLATE_TS:  # todo: make it dynamic
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


if __name__ == '__main__':
    # 1st stage
    compute_features = False
    # 2nd stage
    velocity_based = False  # feature extraction
    train_velocity_based = False
    inference_velocity_based = False
    graph_debug = False
    overfit = False
    normalized_ = False
    compare_models = False
    compare_of_gt = False
    analysis_mode = False
    train_larger_dataset = True

    time_distributed = True
    resume = False
    single_track = False

    test = False
    inference = False

    meta_path = '../Datasets/SDD/H_SDD.txt'
    # meta_path = '/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/H_SDD.txt'
    meta = SDDMeta(meta_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        base_path = "../Datasets/SDD/"
        # base_path = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/"
        save_base_path = "../Datasets/SDD_Features/"
        # save_base_path = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD_Features/"
        vid_label = SDDVideoClasses.LITTLE
        video_number = 3

        save_path = f'{save_base_path}{vid_label.value}/video{video_number}/'

        train_vid_label = SDDVideoClasses.QUAD
        train_vid_num = 1
        train_dataset_path = f'{save_base_path}{train_vid_label.value}/video{train_vid_num}/'

        inference_vid_label = SDDVideoClasses.QUAD
        inference_vid_num = 0
        inference_dataset_path = f'{save_base_path}{inference_vid_label.value}/video{inference_vid_num}/'

        net = SimpleModel(meta=meta, num_frames_to_build_bg_sub_model=12, layers_mode='small',
                          meta_video=SDDVideoDatasets.QUAD, meta_train_video_number=train_vid_num,
                          meta_val_video_number=inference_vid_num)

        if time_distributed and not velocity_based and not normalized_:
            file_name = 'time_distributed_features_un_normalized.pt'
        elif time_distributed and velocity_based and not normalized_:
            file_name = 'time_distributed_velocity_features_with_frame_track.pt'
        elif time_distributed and normalized_:
            file_name = 'time_distributed_features.pt'
        elif single_track:
            file_name = 'time_distributed_selected_track_features.pt'
        else:
            file_name = 'features.pt'

        if compute_features:
            sdd_simple = SDDSimpleDataset(root=base_path, video_label=vid_label, frames_per_clip=1, num_workers=8,
                                          num_videos=1, video_number_to_use=video_number,
                                          step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                          single_track_mode=False, track_id=5, multiple_videos=False)
            sdd_loader = torch.utils.data.DataLoader(sdd_simple, 16)

            logger.info('Computing Features')
            if test:
                x, y, x_norm_dict, y_norm_dict, track_id, frame = net.preprocess_data(sdd_loader, FeaturesMode.UV,
                                                                                      annotation_df=
                                                                                      sdd_simple.annotations_df,
                                                                                      classic_clustering=False,
                                                                                      test_mode=test,
                                                                                      equal_time_distributed=
                                                                                      time_distributed,
                                                                                      normalized=normalized_)
                features_save_dict = {'x': x, 'y': y, 'x_norm_dict': x_norm_dict, 'y_norm_dict': y_norm_dict,
                                      'track_id': track_id, 'frame': frame}
            else:
                # x, y = net.preprocess_data(sdd_loader, FeaturesMode.UV, annotation_df=sdd_simple.annotations_df,
                #                            classic_clustering=False, test_mode=test,
                #                            equal_time_distributed=time_distributed, normalized=normalized_)
                final_featues_dict = net.preprocess_data(sdd_loader, FeaturesMode.UV,
                                                         annotation_df=sdd_simple.annotations_df,
                                                         classic_clustering=False, test_mode=test,
                                                         equal_time_distributed=time_distributed,
                                                         normalized=normalized_, save_per_part_path=save_path,
                                                         resume_idx=None)
                # features_save_dict = {'x': x, 'y': y}

            logger.info(f'Saving the features for video {vid_label.value}, video {video_number}')
            if save_path:
                Path(save_path).mkdir(parents=True, exist_ok=True)
            # file_name = 'time_distributed_dict_with_gt_bbox_centers.pt'
            file_name = 'time_distributed_dict_with_gt_bbox_centers_and_bbox.pt'
            torch.save(final_featues_dict, save_path + file_name)
        elif analysis_mode:
            save_path = '../Datasets/SDD_Features/quad/video1/analysis/'
            file_name = 'analysis_t10.csv'
            df = pd.read_csv(save_path + file_name)
            data = parse_df_analysis(df, save_path)
        elif velocity_based:
            # save_path = inference_dataset_path
            # logger.info('Setting up DataLoaders')
            # train_feats = torch.load(save_path
            #                          + 'time_distributed_dict_with_gt_bbox_centers_and_bbox.pt')  # time_distributed_dict

            # save_path = '../Datasets/SDD_Features/little/video3/parts/'
            save_path = '../Datasets/SDD_Features/quad/video1/'
            train_feats = torch.load(save_path
                                     + 'time_distributed_dict_with_gt_bbox_centers_and_bbox.pt')  # time_distributed_dict
            # # feats_data = net._process_complex_features(train_feats)
            # # data_x, data_y, data_frame_info, data_track_id_info = net._extract_trainable_features(feats_data,
            # #                                                                                       frame_info=True)
            # # features_save_dict = {'x': data_x, 'y': data_y, 'frames': data_frame_info, 'track_ids': data_track_id_info}
            # # logger.info(f'Saving the features for video {vid_label.value}, video {video_number}')
            # # if save_path:
            # #     Path(save_path).mkdir(parents=True, exist_ok=True)
            # # torch.save(features_save_dict, save_path + file_name)

            # Stage 1 - RNN
            # feats_data = net._process_complex_features_rnn(train_feats, time_steps=10)
            ann_path = '../Datasets/SDD/annotations/quad/video1/annotation_augmented.csv'
            ann = get_track_info(ann_path)
            meter_ratio = float(meta.get_meta(SDDVideoDatasets.LITTLE, 3)[0]['Ratio'].to_numpy()[0])
            feats_data = net.analyze_complex_features_rnn(train_feats, time_steps=10, track_info=ann,
                                                          ratio=meter_ratio)
            save_path = '../Datasets/SDD_Features/quad/video1/analysis/'
            file_name = 'analysis_t10.csv'
            if save_path:
                Path(save_path).mkdir(parents=True, exist_ok=True)
            # torch.save(feats_data, save_path + file_name)
            feats_data.to_csv(save_path + file_name)

            # logger.info(f'Saving the features for video {vid_label.value}, video {video_number}')
            # if save_path:
            #     Path(save_path).mkdir(parents=True, exist_ok=True)
            # torch.save(feats_data, save_path + 'pre_train_rnn_data_with_gt_bbox_centers.pt')  # pre_train_rnn_data

            # Stage 2 - RNN
            # train_feats = torch.load(save_path
            #                          + 'pre_train_rnn_data_with_gt_bbox_centers.pt')  # pre_train_rnn_data
            # data_x, data_y, data_frame_info, data_track_id_info, data_bbox_center_x, data_bbox_center_y, data_bbox_x, \
            # data_bbox_y = \
            #     net._extract_trainable_features_rnn(feats_data,
            #                                         return_frame_info=True)
            # features_save_dict = {'x': data_x, 'y': data_y, 'frames': data_frame_info, 'track_ids': data_track_id_info,
            #                       'bbox_center_x': data_bbox_center_x, 'bbox_center_y': data_bbox_center_y,
            #                       'bbox_x': data_bbox_x, 'bbox_y': data_bbox_y}
            # logger.info(f'Saving the features for video {vid_label.value}, video {video_number}')
            #
            # # time_distributed_velocity_features_with_frame_track_rnn
            # file_name = 'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_t10.pt'
            # if save_path:
            #     Path(save_path).mkdir(parents=True, exist_ok=True)
            # torch.save(features_save_dict, save_path + file_name)

        elif inference_velocity_based:
            # file_name = 'time_distributed_velocity_features_with_frame_track.pt'
            # file_name = 'time_distributed_velocity_features_with_frame_track_rnn.pt'
            file_name = 'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox.pt'
            logger.info('Setting up DataLoaders')
            train_feats = torch.load(train_dataset_path + file_name)
            train_x, train_y, train_frames, train_track_ids, train_center_x, train_center_y, train_bbox_x, \
            train_bbox_y = train_feats['x'], train_feats['y'], train_feats['frames'], train_feats['track_ids'], \
                           train_feats['bbox_center_x'], train_feats['bbox_center_y'], train_feats['bbox_x'], \
                           train_feats['bbox_y']

            inference_feats = torch.load(inference_dataset_path + file_name)
            inference_x, inference_y, inference_frames, inference_track_ids, inference_center_x, inference_center_y, \
            inference_bbox_x, inference_bbox_y = \
                inference_feats['x'], inference_feats['y'], inference_feats['frames'], inference_feats['track_ids'], \
                inference_feats['bbox_center_x'], inference_feats['bbox_center_y'], inference_feats['bbox_x'], \
                inference_feats['bbox_y']

            # Split Val and Test
            split_percent = 0.2
            val_length = int(split_percent * len(inference_x))
            inference_length = len(inference_x) - val_length

            inference_dataset = FeaturesDatasetExtra(inference_x, inference_y, frames=inference_frames,
                                                     track_ids=inference_track_ids, mode=FeaturesMode.UV,
                                                     preprocess=False, bbox_center_x=inference_center_x,
                                                     bbox_center_y=inference_center_y,
                                                     bbox_x=inference_bbox_x,
                                                     bbox_y=inference_bbox_y)
            val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset, [val_length, inference_length],
                                                                      generator=torch.Generator().manual_seed(42))

            logger.info('Setting up network')

            train_dataset = FeaturesDatasetExtra(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
                                                 mode=FeaturesMode.UV, preprocess=False, bbox_center_x=train_center_x,
                                                 bbox_center_y=train_center_y,
                                                 bbox_x=train_bbox_x,
                                                 bbox_y=train_bbox_y)
            train_subset = torch.utils.data.Subset(train_dataset, [i for i in range(10)])

            # overfit - active
            train_loader = torch.utils.data.DataLoader(train_subset, 1)

            val_loader = torch.utils.data.DataLoader(val_dataset, 1)

            # One-Step NN
            # m = SimpleModel.load_from_checkpoint(meta=meta, 'lightning_logs/version_11/checkpoints/epoch=9.ckpt',
            #                                      layers_mode='small',
            #                           meta_video=SDDVideoDatasets.QUAD, meta_train_video_number=train_vid_num,
            #                           meta_val_video_number=inference_vid_num)
            # for batch in tqdm(val_loader):
            #     features1, features2 = batch
            #     features1, features2 = features1.squeeze().float(), features2.squeeze().float()
            #     # find the center
            #     center_xy, center_true_uv, center_past_uv = m.find_center_point(features1)
            #     # input prior velocity
            #     pred = m(center_past_uv)
            #     pred_center = center_xy + pred
            #     # move points by actual optical flow
            #     moved_points_by_true_of = features1[:, :2] + features1[:, 2:4]
            #
            #     # plot
            #     plot_points_predicted_and_true_center_only(predicted_points=pred_center.detach().numpy(),
            #                                                true_points=moved_points_by_true_of.detach().numpy(),
            #                                                actual_points=features1[:, :2].detach().numpy())

            # One-Step NN - all points
            # m = SimpleModel.load_from_checkpoint('lightning_logs/version_12/checkpoints/epoch=20.ckpt',
            #                                      layers_mode='small', meta=meta,
            #                           meta_video=SDDVideoDatasets.QUAD, meta_train_video_number=train_vid_num,
            #                           meta_val_video_number=inference_vid_num)
            # for batch in tqdm(val_loader):
            #     features1, features2, frame_num, track_id = batch
            #     features1, features2 = features1.squeeze().float(), features2.squeeze().float()
            #     pred = m(features1[:, 4:])
            #     moved_points_pred_of = features1[:, :2] + pred
            #     moved_points_by_true_of = features1[:, :2] + features1[:, 2:4]
            #     # plot
            #     plot_points_predicted_and_true(moved_points_pred_of.detach().numpy(),
            #                                    moved_points_by_true_of.detach().numpy(),
            #                                    features1[:, :2].detach().numpy())

            # One-Step NN wih plot
            # m = SimpleModel.load_from_checkpoint('lightning_logs/version_13/checkpoints/epoch=99.ckpt', meta=meta,
            #                                      layers_mode='small',
            #                                      meta_video=SDDVideoDatasets.QUAD,
            #                                      meta_train_video_number=train_vid_num,
            #                                      meta_val_video_number=inference_vid_num)
            # # for infr_i in tqdm(range(10)):
            # for batch in tqdm(val_loader):
            #     features1, features2, frame_num, track_id = batch
            #     features1, features2 = features1.squeeze().float(), features2.squeeze().float()
            #     # go to the frame
            #     cap = cv.VideoCapture(f'{base_path}videos/{vid_label.value}/video{inference_vid_num}/video.mov')
            #     cap_count = 0
            #     frame_img = None
            #     while 1:
            #         ret, video_frame = cap.read()
            #         if cap_count == frame_num.item():
            #             frame_img = video_frame
            #             break
            #         cap_count += 1
            #     # find the center
            #     center_xy, center_true_uv, center_past_uv = m.find_center_point(features1)
            #     # input prior velocity
            #     pred = m(center_past_uv)
            #     pred_center = center_xy + pred
            #     # move points by actual optical flow
            #     moved_points_by_true_of = features1[:, :2] + features1[:, 2:4]
            #
            #     # plot
            #     plot_points_predicted_and_true_center_only(predicted_points=pred_center.detach().numpy(),
            #                                                true_points=moved_points_by_true_of.detach().numpy(),
            #                                                actual_points=features1[:, :2].detach().numpy(),
            #                                                img=frame_img)

            kwargs_dict = {
                'meta': meta,
                'layers_mode': 'rnn',
                'meta_video': SDDVideoDatasets.QUAD,
                'meta_train_video_number': train_vid_num,
                'meta_val_video_number': inference_vid_num,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'time_steps': 5
            }
            of_model = SimpleModel(**kwargs_dict)
            of_model.load_state_dict(torch.load('lightning_logs/version_33/checkpoints/epoch=30.ckpt')['state_dict'])
            # of_model = torch.load('../Plots/overfit_scheduler_b1_of.pt')
            of_model.eval()
            time_steps = 5
            for batch_inference in tqdm(val_loader):
                features1, features2, frame_, track_id, bbox_center1, bbox_center2 = batch_inference
                frame_nums = [f.item() for f in frame_]

                hx, cx = torch.zeros(size=(1, 32)), torch.zeros(size=(1, 32))

                total_loss = torch.tensor(data=0, dtype=torch.float32)
                img_frames = []
                pred_centers = []
                gt_based_shifted_points_list = []
                actual_points_list = []
                last_input_velocity = None
                last_pred_center = None
                last_true_center = None
                cap_count = 0
                if len(features1) == time_steps:
                    cap = cv.VideoCapture(f'{base_path}videos/{vid_label.value}/video{inference_vid_num}/video.mov')

                    while 1:
                        ret, video_frame = cap.read()
                        if cap_count in frame_nums:
                            img_frames.append(video_frame)
                            # if len(img_frames) == len(frame_nums):
                            break
                        cap_count += 1

                    for i in range(time_steps):

                        feat1, feat2, bb_center1, bb_center2 = features1[i].squeeze().float(), features2[
                            i].squeeze().float(), \
                                                               bbox_center1[i].float(), bbox_center2[i].float()
                        try:
                            center_xy, center_true_uv, center_past_uv = of_model.find_center_point(feat1)
                            # moved_points_by_true_of = feat2[:, :2]  # for gt based
                            gt_based_shifted_points = bb_center2  # for bbox_center gt based
                        except IndexError:
                            print('Bad Data')
                            break

                        if i == 0:
                            last_input_velocity = center_past_uv.unsqueeze(0)
                            last_pred_center = center_xy
                            last_true_center = center_xy

                        with torch.no_grad():
                            # block_1 = m.block_1(center_past_uv.unsqueeze(0))
                            # hx, cx = m.rnn_cell(block_1, (hx, cx))
                            # block_2 = m.block_2(hx)
                            # logger.info(f'{i} -> Input Velocity: {last_input_velocity}')
                            # logger.info(f'{i} -> Current Center: {last_pred_center}')
                            block_1 = of_model.block_1(last_input_velocity)
                            hx, cx = of_model.rnn_cell(block_1, (hx, cx))
                            block_2 = of_model.block_2(hx)

                        # moved_points_by_true_of = feat1[:, :2] + feat1[:, 2:4]  # of based
                        # moved_points_by_true_of_center = moved_points_by_true_of.mean(0)  # for of based
                        gt_based_shifted_points_list.append(gt_based_shifted_points)
                        # pred_center = center_xy + block_2
                        pred_center = last_pred_center + block_2
                        # pred_center = last_pred_center + (block_2 * 0.4)  # velocity * time - bad

                        # swapped
                        # pred_center_swapped = torch.zeros_like(pred_center)
                        # pred_center_swapped[:, 0] = pred_center[:, 1]
                        # pred_center_swapped[:, 1] = pred_center[:, 0]

                        last_input_velocity = block_2
                        # logger.info(f'{i} -> True Center: {moved_points_by_true_of_center}')
                        # logger.info(f'{i} -> True Delta Center: {moved_points_by_true_of_center - last_true_center}')
                        # logger.info(f'{i} -> True Delta Distance: '
                        #             f'{torch.norm(moved_points_by_true_of_center - last_true_center, p=2)}')

                        logger.info(f'{i} -> Pred Delta Center: {pred_center - last_pred_center}')
                        logger.info(f'{i} -> Pred Delta Distance: {torch.norm(pred_center - last_pred_center, p=2)}')
                        last_pred_center = pred_center
                        # last_true_center = moved_points_by_true_of_center
                        pred_centers.append(pred_center.squeeze(0))

                        # swapped
                        # logger.info(f'{i} -> Pred Delta Center: {pred_center_swapped - last_pred_center}')
                        # logger.info(f'{i} -> Pred Delta Distance: '
                        #             f'{torch.norm(pred_center_swapped - last_pred_center, p=2)}')
                        # last_pred_center = pred_center_swapped
                        # last_true_center = moved_points_by_true_of_center
                        # pred_centers.append(pred_center_swapped.squeeze(0))

                        # Remove mean for per time step plot
                        actual_points_list.append(feat1[:, :2].detach().mean(dim=0).numpy())
                        logger.info(f'{i} -> Predicted Velocity: {last_input_velocity}')
                        logger.info(f'{i} -> True Velocity: {feat1[:, 2:4].mean(dim=0)}')
                        logger.info(f'*****Next Step*****')
                        # logger.info(f'{i} -> Pred Center: {last_pred_center}')

                    predicted_points = [p.detach().numpy() for p in pred_centers]
                    true_points = [p.detach().mean(dim=0).numpy() for p in gt_based_shifted_points_list]
                    logger.info(f'Pred Trajectory Length: '
                                f'{np.linalg.norm(predicted_points[-1] - predicted_points[0], 2)}')
                    logger.info(f'True Trajectory Length: {np.linalg.norm(true_points[-1] - true_points[0], 2)}')
                    l2_points = {idx: np.linalg.norm(i - j, 2)
                                 for idx, (i, j) in enumerate(zip(true_points, predicted_points))}
                    logger.info(f'L2 corresponding centers: {l2_points}')

                    # plot
                    plot_trajectory_rnn(predicted_points=np.stack(predicted_points),
                                        true_points=np.stack(true_points),
                                        actual_points=actual_points_list,
                                        imgs=img_frames, gt=True)
                    plot_trajectory_rnn(predicted_points=np.stack(predicted_points),
                                        true_points=np.stack(true_points),
                                        actual_points=actual_points_list,
                                        imgs=None, gt=True)
                    # plot_points_predicted_and_true_center_only_rnn(predicted_points=predicted_points,
                    #                                                true_points=true_points,
                    #                                                actual_points=actual_points_list,
                    #                                                imgs=img_frames)
                    # plot_points_predicted_and_true_center_only_rnn(predicted_points=predicted_points,
                    #                                                true_points=true_points,
                    #                                                actual_points=actual_points_list,
                    #                                                imgs=None)

        elif train_velocity_based:
            # file_name = 'time_distributed_velocity_features_with_frame_track_rnn.pt'
            file_name = 'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox.pt'
            logger.info('Setting up DataLoaders')
            train_feats = torch.load(train_dataset_path + file_name)
            train_x, train_y, train_frames, train_track_ids, train_center_x, train_center_y, train_bbox_x, \
            train_bbox_y = train_feats['x'], train_feats['y'], train_feats['frames'], train_feats['track_ids'], \
                           train_feats['bbox_center_x'], train_feats['bbox_center_y'], train_feats['bbox_x'], \
                           train_feats['bbox_y']

            train_dataset = FeaturesDatasetExtra(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
                                                 mode=FeaturesMode.UV, preprocess=False, bbox_center_x=train_center_x,
                                                 bbox_center_y=train_center_y,
                                                 bbox_x=train_bbox_x,
                                                 bbox_y=train_bbox_y)

            inference_feats = torch.load(inference_dataset_path + file_name)
            inference_x, inference_y, inference_frames, inference_track_ids, inference_center_x, inference_center_y, \
            inference_bbox_x, inference_bbox_y = \
                inference_feats['x'], inference_feats['y'], inference_feats['frames'], inference_feats['track_ids'], \
                inference_feats['bbox_center_x'], inference_feats['bbox_center_y'], inference_feats['bbox_x'], \
                inference_feats['bbox_y']

            # Split Val and Test
            split_percent = 0.2
            val_length = int(split_percent * len(inference_x))
            inference_length = len(inference_x) - val_length

            inference_dataset = FeaturesDatasetExtra(inference_x, inference_y, frames=inference_frames,
                                                     track_ids=inference_track_ids, mode=FeaturesMode.UV,
                                                     preprocess=False, bbox_center_x=inference_center_x,
                                                     bbox_center_y=inference_center_y,
                                                     bbox_x=inference_bbox_x,
                                                     bbox_y=inference_bbox_y)

            val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset, [val_length, inference_length],
                                                                      generator=torch.Generator().manual_seed(42))

            net = SimpleModel(meta=meta, num_frames_to_build_bg_sub_model=12, layers_mode='rnn',
                              meta_video=SDDVideoDatasets.QUAD, meta_train_video_number=train_vid_num,
                              meta_val_video_number=inference_vid_num, lr=1e-2, train_dataset=train_dataset,
                              val_dataset=val_dataset, time_steps=5)

            logger.info('Setting up network')

            # from pytorch_lightning import loggers as pl_loggers
            #
            # tb_logger = pl_loggers.TensorBoardLogger('lightning_logs', name='trajectory_plot')
            # checkpoint_callback = ModelCheckpoint(monitor='val/loss')

            if resume:
                resume_path = 'lightning_logs/version_26/checkpoints/epoch=9.ckpt'
                trainer = pl.Trainer(gpus=1, max_epochs=200, resume_from_checkpoint=resume_path)
            else:
                # trainer = pl.Trainer(gpus=1, max_epochs=200, accumulate_grad_batches=16)
                trainer = pl.Trainer(gpus=1, max_epochs=1500, accumulate_grad_batches=16, overfit_batches=64)
                # trainer = pl.Trainer(gpus=1, max_epochs=200)
                # trainer = pl.Trainer(gpus=1, max_epochs=200, accumulate_grad_batches=128, logger=tb_logger,
                #                      log_every_n_steps=1000, flush_logs_every_n_steps=2000, checkpoint_callback=
                #                      checkpoint_callback)

            logger.info('Starting training')
            trainer.fit(net)
        elif train_larger_dataset:
            t_steps = COLLATE_TS
            extract_features = False
            inference = True
            if extract_features:
                # process for dataset
                file_name = '../Datasets/SDD_Features/little/video3/time_distributed_dict_with_gt_bbox_centers_and_bbox.pt'
                train_feats = torch.load(file_name)
                feats_data = net._process_complex_features_rnn(train_feats, time_steps=10)
                # logger.info(f'Saving the features for video {vid_label.value}, video {video_number}')
                # # if save_path:
                # #     Path(save_path).mkdir(parents=True, exist_ok=True)
                # # torch.save(feats_data, save_path + 'pre_train_rnn_data_with_gt_bbox_centers.pt')  # pre_train_rnn_data
                #
                # # Stage 2 - RNN
                # # train_feats = torch.load(save_path
                # #                          + 'pre_train_rnn_data_with_gt_bbox_centers.pt')  # pre_train_rnn_data
                data_x, data_y, data_frame_info, data_track_id_info, data_bbox_center_x, data_bbox_center_y, data_bbox_x, \
                data_bbox_y = \
                    net._extract_trainable_features_rnn(feats_data,
                                                        return_frame_info=True)
                features_save_dict = {'x': data_x, 'y': data_y, 'frames': data_frame_info,
                                      'track_ids': data_track_id_info,
                                      'bbox_center_x': data_bbox_center_x, 'bbox_center_y': data_bbox_center_y,
                                      'bbox_x': data_bbox_x, 'bbox_y': data_bbox_y}
                # logger.info(f'Saving the features for video {vid_label.value}, video {video_number}')
                #
                # Center based preprocessing
                # # time_distributed_velocity_features_with_frame_track_rnn
                # uncomment below this for usual training
                # file_name = 'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_t5.pt'
                # just to save
                # # if save_path:
                # #     Path(save_path).mkdir(parents=True, exist_ok=True)
                # # torch.save(features_save_dict, save_path + file_name)

                # logger.info('Setting up DataLoaders')
                # train_feats = torch.load(save_path + file_name)
                # train_feats = center_based_dataset(train_feats)
                train_feats = center_based_dataset(features_save_dict)
                file_name = 'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_' \
                            'center_based_t10.pt'
                # if save_path:
                #     Path(save_path).mkdir(parents=True, exist_ok=True)
                # torch.save(train_feats, save_path + file_name)

            file_name = f'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_' \
                        f'center_based_t{t_steps}.pt'

            logger.info('Setting up DataLoaders')
            train_feats = torch.load(save_path + file_name)
            # train_x, train_y, train_frames, train_track_ids, train_center_x, train_center_y, train_bbox_x, \
            # train_bbox_y = train_feats['x'], train_feats['y'], train_feats['frames'], train_feats['track_ids'], \
            #                train_feats['bbox_center_x'], train_feats['bbox_center_y'], train_feats['bbox_x'], \
            #                train_feats['bbox_y']
            #
            # total_dataset = FeaturesDatasetExtra(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
            #                                      mode=FeaturesMode.UV, preprocess=False, bbox_center_x=train_center_x,
            #                                      bbox_center_y=train_center_y,
            #                                      bbox_x=train_bbox_x,
            #                                      bbox_y=train_bbox_y)
            #

            train_x, train_y, train_frames, train_track_ids, train_center_x, train_center_y, train_bbox_x, \
            train_bbox_y, train_center_data = train_feats['x'], train_feats['y'], train_feats['frames'], \
                                              train_feats['track_ids'], train_feats['bbox_center_x'], \
                                              train_feats['bbox_center_y'], train_feats['bbox_x'], \
                                              train_feats['bbox_y'], train_feats['center_based']

            total_dataset = FeaturesDatasetCenterBased(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
                                                       mode=FeaturesMode.UV, preprocess=False,
                                                       bbox_center_x=train_center_x,
                                                       bbox_center_y=train_center_y,
                                                       bbox_x=train_bbox_x,
                                                       bbox_y=train_bbox_y,
                                                       features_center=train_center_data)

            # Split Train and Test
            split_percent = 0.2
            inference_length = int(split_percent * len(train_x))
            train_length = len(train_x) - inference_length

            val_length = int(split_percent * inference_length)
            test_length = inference_length - val_length

            train_dataset, inference_dataset = torch.utils.data.random_split(total_dataset,
                                                                             [train_length, inference_length],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 42))
            val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset,
                                                                      [val_length, test_length],
                                                                      generator=torch.Generator().manual_seed(42))
            if inference:
                batch_size = 1
                num_workers = 0
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, collate_fn=center_dataset_collate,
                                                           num_workers=num_workers)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, collate_fn=center_dataset_collate,
                                                         num_workers=num_workers)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, collate_fn=center_dataset_collate,
                                                          num_workers=num_workers)
                device = 'cpu'
                kwargs_dict = {
                    'meta': meta,
                    'num_frames_to_build_bg_sub_model': 12,
                    'layers_mode': 'rnn',
                    'meta_video': SDDVideoDatasets.LITTLE,
                    'meta_train_video_number': 3,
                    'meta_val_video_number': 3,
                    'lr': 1e-3,
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset,
                    'time_steps': t_steps,
                    'batch_size': batch_size,
                    'num_workers': num_workers
                }

                # # T=5
                # of_model_version = 60
                # of_model_epoch = 124
                #
                # gt_model_version = 68
                # gt_model_epoch = 13

                # # T=5 w/o batchnorm
                of_model_version = 65
                of_model_epoch = 18

                gt_model_version = 64
                gt_model_epoch = 52

                # # T=10 - batchnorm
                # of_model_version = 61
                # of_model_epoch = 134
                #
                # gt_model_version = 62
                # gt_model_epoch = 275

                # # T=10 - w/o batchnorm
                # of_model_version = 66
                # of_model_epoch = 201
                #
                # gt_model_version = 67
                # gt_model_epoch = 6

                of_model = SimpleModel(**kwargs_dict)
                of_model.load_state_dict(torch.load(f'lightning_logs/version_{str(of_model_version)}/checkpoints/'
                                                    f'epoch={str(of_model_epoch)}.ckpt')['state_dict'])
                of_model.to(device)
                of_model.eval()

                gt_model = SimpleModel(**kwargs_dict)
                gt_model.load_state_dict(torch.load(f'lightning_logs/version_{str(gt_model_version)}/checkpoints/'
                                                    f'epoch={str(gt_model_epoch)}.ckpt')['state_dict'])
                gt_model.to(device)
                gt_model.eval()
                time_steps = kwargs_dict['time_steps']

                ade_dataset_of = []
                fde_dataset_of = []

                ade_dataset_gt = []
                fde_dataset_gt = []

                ade_dataset_gt_of_gt = []
                fde_dataset_gt_of_gt = []

                ade_dataset_linear = []
                fde_dataset_linear = []

                is_gt_based_shifted_points_inside_list = []
                is_of_based_shifted_points_center_inside_list = []
                is_pred_center_inside_list = []
                is_pred_center_gt_inside_list = []

                bad_data = False
                batches_processed = 0
                base_save_path = f'/home/rishabh/Thesis/imgs/t={t_steps}_batchnorm_stacked_of_vs_gt/'
                inference_vid_num = 3

                for sav_i, batch_inference in enumerate(tqdm(train_loader)):
                    try:
                        features, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch_inference
                        batches_processed += 1
                    except TypeError:
                        continue
                    b_size = features.shape[1]

                    hx, cx = torch.zeros(size=(b_size, 32), device=device), \
                             torch.zeros(size=(b_size, 32), device=device)
                    hx_1, cx_1 = torch.zeros(size=(b_size, 64), device=device), \
                                 torch.zeros(size=(b_size, 64), device=device)

                    torch.nn.init.xavier_normal_(hx)
                    torch.nn.init.xavier_normal_(cx)
                    torch.nn.init.xavier_normal_(hx_1)
                    torch.nn.init.xavier_normal_(cx_1)

                    hx_gt, cx_gt = torch.zeros(size=(b_size, 32), device=device), \
                                   torch.zeros(size=(b_size, 32), device=device)
                    hx_1_gt, cx_1_gt = torch.zeros(size=(b_size, 64), device=device), \
                                       torch.zeros(size=(b_size, 64), device=device)

                    torch.nn.init.xavier_normal_(hx_gt)
                    torch.nn.init.xavier_normal_(cx_gt)
                    torch.nn.init.xavier_normal_(hx_1_gt)
                    torch.nn.init.xavier_normal_(cx_1_gt)

                    frame_nums = [f.item() for f in frame_]
                    # frame_nums = [f for f in frame_]  # remove
                    total_loss = torch.tensor(data=0, dtype=torch.float32)
                    img_frames = []
                    linear_pred_centers = []
                    pred_centers = []
                    pred_centers_gt = []
                    gt_based_shifted_points_list = []
                    of_based_shifted_points_list = []
                    actual_points_list = []
                    last_input_velocity = None
                    last_pred_center = None
                    last_true_center = None
                    cap_count = 0

                    first_ts_velocity_linear = None

                    if len(features) == time_steps:
                        cap = cv.VideoCapture(f'{base_path}videos/{vid_label.value}/video{inference_vid_num}/video.mov')

                        # while 1:
                        #     ret, video_frame = cap.read()
                        #     if cap_count in frame_nums:
                        #         img_frames.append(video_frame)
                        #         # if len(img_frames) == len(frame_nums):
                        #         break
                        #     cap_count += 1

                        for i in range(time_steps):
                            feat, bb_center1, bb_center2, bb1, bb2 = features[i].squeeze().float(), \
                                                                     bbox_center1[i].float(), \
                                                                     bbox_center2[i].float(), bbox1[i].float(), bbox2[
                                                                         i].float()
                            try:
                                # center_xy, center_true_uv, center_past_uv, shifted_point = feat[:, 0, :], \
                                #                                                            feat[:, 1, :], \
                                #                                                            feat[:, 2, :], feat[:, 3, :]
                                center_xy, center_true_uv, center_past_uv, shifted_point = feat[0, :], \
                                                                                           feat[1, :], \
                                                                                           feat[2, :], feat[3, :]
                                # gt_based_shifted_points = feat2[:, :2]  # for gt based
                                gt_based_shifted_points = bb_center2  # for bbox_center gt based
                            except IndexError:
                                print('Bad Data')
                                bad_data = True
                                break

                            if i == 0:
                                # last_input_velocity = center_past_uv.unsqueeze(0)  # * 0.4  # bug workaround
                                # last_input_velocity_gt = center_past_uv.unsqueeze(0)
                                last_input_velocity = center_past_uv.unsqueeze(0)
                                last_input_velocity_gt = center_past_uv.unsqueeze(0)
                                last_pred_center = center_xy
                                last_pred_center_gt = center_xy
                                last_true_center = center_xy
                                linear_pred_center = center_xy
                                first_ts_velocity_linear = center_past_uv

                            with torch.no_grad():
                                # logger.info(f'{i} -> Input Velocity: {last_input_velocity}')
                                # logger.info(f'{i} -> Current Center: {last_pred_center}')
                                # block_1 = of_model.block_1(last_input_velocity)
                                # hx, cx = of_model.rnn_cell(block_1, (hx, cx))
                                # block_2 = of_model.block_2(hx)
                                block_2, cx, hx, cx_1, hx_1 = of_model(last_input_velocity, cx, hx, cx_1, hx_1, False)

                                # block_1_gt = gt_model.block_1(last_input_velocity_gt)
                                # hx_gt, cx_gt = gt_model.rnn_cell(block_1_gt, (hx_gt, cx_gt))
                                # block_2_gt = gt_model.block_2(hx_gt)
                                block_2_gt, cx_gt, hx_gt, cx_1_gt, hx_1_gt = gt_model(last_input_velocity_gt, cx_gt,
                                                                                      hx_gt, cx_1_gt, hx_1_gt, False)

                            # gt_based_shifted_points = feat1[:, :2] + feat1[:, 2:4]
                            # moved_points_by_true_of_center = gt_based_shifted_points.mean(0)
                            gt_based_shifted_points_list.append(gt_based_shifted_points)

                            # of_based_shifted_points = feat1[:, :2] + feat1[:, 2:4]
                            of_based_shifted_points = shifted_point
                            # of_based_shifted_points_center = of_based_shifted_points.mean(0)  # remove
                            of_based_shifted_points_list.append(of_based_shifted_points)
                            # pred_center = center_xy + block_2
                            pred_center = last_pred_center + block_2
                            pred_center_gt = last_pred_center_gt + block_2_gt
                            linear_pred_center += first_ts_velocity_linear  # * 0.4  # d = v * t
                            # pred_center = last_pred_center + (block_2 * 0.4)  # velocity * time - bad

                            is_gt_based_shifted_points_inside = is_inside_bbox(point=gt_based_shifted_points.squeeze(),
                                                                               bbox=bb2.squeeze()).item()
                            is_of_based_shifted_points_center_inside = is_inside_bbox(
                                # point=of_based_shifted_points_center.squeeze(),
                                point=of_based_shifted_points.squeeze(),
                                bbox=bb2.squeeze()).item()
                            is_pred_center_inside = is_inside_bbox(point=pred_center.squeeze(),
                                                                   bbox=bb2.squeeze()).item()
                            is_pred_center_gt_inside = is_inside_bbox(point=pred_center_gt.squeeze(),
                                                                      bbox=bb2.squeeze()).item()

                            is_gt_based_shifted_points_inside_list.append(is_gt_based_shifted_points_inside)
                            is_of_based_shifted_points_center_inside_list.append(
                                is_of_based_shifted_points_center_inside)
                            is_pred_center_inside_list.append(is_pred_center_inside)
                            is_pred_center_gt_inside_list.append(is_pred_center_gt_inside)

                            # loss
                            loss_gt_all_points = of_model.cluster_center_loss_meters(gt_based_shifted_points,
                                                                                     pred_center_gt)
                            loss_gt_points_center = of_model. \
                                cluster_center_points_center_loss_meters(gt_based_shifted_points,
                                                                         pred_center_gt)

                            loss_of_all_points = gt_model.cluster_center_loss_meters(of_based_shifted_points,
                                                                                     pred_center)
                            loss_of_points_center = gt_model. \
                                cluster_center_points_center_loss_meters(of_based_shifted_points,
                                                                         pred_center)

                            # logger.info('')
                            # logger.info(f'GT-All points: {loss_gt_all_points}')
                            # logger.info(f'GT-Points center: {loss_gt_points_center}')
                            # logger.info(f'OF-All points: {loss_of_all_points}')
                            # logger.info(f'OF-Points center: {loss_of_points_center}')

                            last_input_velocity = block_2
                            last_input_velocity_gt = block_2_gt
                            # logger.info(f'{i} -> True Center: {moved_points_by_true_of_center}')
                            # logger.info(f'{i} -> True Delta Center: {moved_points_by_true_of_center - last_true_center}')
                            # logger.info(f'{i} -> True Delta Distance: '
                            #             f'{torch.norm(moved_points_by_true_of_center - last_true_center, p=2)}')

                            # logger.info(f'{i} -> Pred Delta Center: {pred_center - last_pred_center}')
                            # logger.info(f'{i} -> Pred Delta Distance: {torch.norm(pred_center - last_pred_center, p=2)}')
                            last_pred_center = pred_center
                            last_pred_center_gt = pred_center_gt
                            # last_true_center = moved_points_by_true_of_center
                            pred_centers.append(pred_center.squeeze(0))
                            pred_centers_gt.append(pred_center_gt.squeeze(0))
                            linear_pred_centers.append(linear_pred_center)

                            # Remove mean for per time step plot
                            actual_points_list.append(bb_center1.squeeze().detach().numpy())
                            # logger.info(f'{i} -> Predicted Velocity: {last_input_velocity}')
                            # logger.info(f'{i} -> True Velocity: {feat1[:, 2:4].mean(dim=0)}')
                            # logger.info(f'*****Next Step*****')
                            # logger.info(f'{i} -> Pred Center: {last_pred_center}')

                        if bad_data:
                            bad_data = False
                            continue

                        predicted_points = [p.detach().numpy() for p in pred_centers]
                        predicted_points_gt = [p.detach().numpy() for p in pred_centers_gt]
                        linear_predicted_points = [p.detach().numpy() for p in linear_pred_centers]

                        # true_points = [p.detach().mean(dim=0).numpy() for p in gt_based_shifted_points_list]
                        # true_points_of = [p.detach().mean(dim=0).numpy() for p in of_based_shifted_points_list]

                        true_points = [p.detach().numpy() for p in gt_based_shifted_points_list]
                        true_points_of = [p.detach().numpy() for p in of_based_shifted_points_list]

                        # logger.info(f'Optical Flow Trajectory Length: '
                        #             f'{trajectory_length(predicted_points)}')
                        # logger.info(f'GT Trajectory Length: '
                        #             f'{trajectory_length(predicted_points_gt)}')
                        # logger.info(f'True GT Trajectory Length: {trajectory_length(true_points)}')
                        # logger.info(f'True OF Trajectory Length: {trajectory_length(true_points_of)}')

                        l2_points = {idx: np.linalg.norm(i - j, 2)
                                     for idx, (i, j) in enumerate(zip(true_points_of, predicted_points))}
                        # logger.info(f'Optical Flow L2 corresponding centers: {l2_points}')

                        l2_points_gt = {idx: np.linalg.norm(i - j, 2)
                                        for idx, (i, j) in enumerate(zip(true_points, predicted_points_gt))}
                        # logger.info(f'GT L2 corresponding centers: {l2_points_gt}')

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
                        # plot_trajectory_rnn_compare(predicted_points=np.stack(predicted_points),
                        #                             predicted_points_gt=np.stack(predicted_points_gt),
                        #                             true_points=np.stack(true_points).squeeze(),
                        #                             true_points_of=np.stack(true_points_of),
                        #                             of_l2=l2_points,
                        #                             gt_l2=l2_points_gt,
                        #                             actual_points=actual_points_list,
                        #                             imgs=img_frames, gt=True,
                        #                             m_ratio=gt_model.val_ratio,
                        #                             show=False,
                        #                             save_path=f'{base_save_path}compare_rgb_{sav_i}')
                        # plot_trajectory_rnn_compare(predicted_points=np.stack(predicted_points),
                        #                             predicted_points_gt=np.stack(predicted_points_gt),
                        #                             true_points=np.stack(true_points).squeeze(),
                        #                             true_points_of=np.stack(true_points_of),
                        #                             of_l2=l2_points,
                        #                             gt_l2=l2_points_gt,
                        #                             actual_points=actual_points_list,
                        #                             imgs=None, gt=True,
                        #                             m_ratio=gt_model.val_ratio,
                        #                             show=False,
                        #                             save_path=f'{base_save_path}compare_traj_{sav_i}')
                        #
                        # plot_trajectory_rnn_compare_side_by_side(predicted_points=np.stack(predicted_points),
                        #                                          predicted_points_gt=np.stack(predicted_points_gt),
                        #                                          true_points=np.stack(true_points).squeeze(),
                        #                                          true_points_of=np.stack(true_points_of),
                        #                                          of_l2=l2_points,
                        #                                          gt_l2=l2_points_gt,
                        #                                          actual_points=actual_points_list,
                        #                                          imgs=img_frames, gt=True,
                        #                                          m_ratio=gt_model.val_ratio,
                        #                                          show=False,
                        #                                          save_path=
                        #                                          f'{base_save_path}compare_side_by_side_rgb_{sav_i}')

                plot_bars_if_inside_bbox([is_gt_based_shifted_points_inside_list,
                                          is_of_based_shifted_points_center_inside_list,
                                          is_pred_center_inside_list,
                                          is_pred_center_gt_inside_list])

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
            else:
                net = SimpleModel(meta=meta, num_frames_to_build_bg_sub_model=12, layers_mode='rnn',
                                  meta_video=SDDVideoDatasets.LITTLE, meta_train_video_number=3,
                                  meta_val_video_number=3, lr=1e-3, train_dataset=train_dataset,
                                  val_dataset=val_dataset, time_steps=t_steps, batch_size=256, num_workers=10)
                logger.info('Setting up network')
                trainer = pl.Trainer(gpus=1, max_epochs=400)
                logger.info('Starting training')
                trainer.fit(net)
        elif graph_debug:
            import hiddenlayer as hl

            file_name = 'time_distributed_velocity_features_with_frame_track_rnn.pt'
            logger.info('Setting up DataLoaders')
            train_feats = torch.load(train_dataset_path + file_name)
            train_x, train_y, train_frames, train_track_ids, train_center_x, train_center_y, train_bbox_x, \
            train_bbox_y = train_feats['x'], train_feats['y'], train_feats['frames'], train_feats['track_ids'], \
                           train_feats['bbox_center_x'], train_feats['bbox_center_y'], train_feats['bbox_x'], \
                           train_feats['bbox_y']

            inference_feats = torch.load(inference_dataset_path + file_name)
            inference_x, inference_y, inference_frames, inference_track_ids, inference_center_x, inference_center_y, \
            inference_bbox_x, inference_bbox_y = \
                inference_feats['x'], inference_feats['y'], inference_feats['frames'], inference_feats['track_ids'], \
                inference_feats['bbox_center_x'], inference_feats['bbox_center_y'], inference_feats['bbox_x'], \
                inference_feats['bbox_y']
            # Split Val and Test
            split_percent = 0.2
            val_length = int(split_percent * len(inference_x))
            inference_length = len(inference_x) - val_length

            inference_dataset = FeaturesDatasetExtra(inference_x, inference_y, frames=inference_frames,
                                                     track_ids=inference_track_ids, mode=FeaturesMode.UV,
                                                     preprocess=False, bbox_center_x=inference_center_x,
                                                     bbox_center_y=inference_center_y,
                                                     bbox_x=inference_bbox_x,
                                                     bbox_y=inference_bbox_y)

            val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset, [val_length, inference_length],
                                                                      generator=torch.Generator().manual_seed(42))

            logger.info('Setting up network')

            train_dataset = FeaturesDatasetExtra(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
                                                 mode=FeaturesMode.UV, preprocess=False, bbox_center_x=train_center_x,
                                                 bbox_center_y=train_center_y,
                                                 bbox_x=train_bbox_x,
                                                 bbox_y=train_bbox_y)
            train_loader = torch.utils.data.DataLoader(train_dataset, 1)

            val_loader = torch.utils.data.DataLoader(val_dataset, 1)

            # val_itr = iter(val_loader)

            # features1, features2, frame_, track_id = next(val_itr)
            net = SimpleModel(meta=meta, num_frames_to_build_bg_sub_model=12, layers_mode='rnn',
                              meta_video=SDDVideoDatasets.QUAD, meta_train_video_number=train_vid_num,
                              meta_val_video_number=inference_vid_num)
            graph_save_path = '../Plots/graph'
            # writer = SummaryWriter()
            for b_idx, b in enumerate(val_loader):
                if len(b[0]) == 5:
                    # o = net(b)
                    # print()
                    # writer.add_graph(net, [b])
                    graph = hl.build_graph(net, b)
                    graph = graph.build_dot()
                    graph.render(f'{graph_save_path}_{b_idx}', view=True, format='png')
                    break
            # writer.close()
        elif overfit:
            # file_name = 'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_t10.pt'
            file_name = 'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox.pt'
            logger.info('Setting up DataLoaders')
            train_feats = torch.load(train_dataset_path + file_name)
            train_x, train_y, train_frames, train_track_ids, train_center_x, train_center_y, train_bbox_x, \
            train_bbox_y = train_feats['x'], train_feats['y'], train_feats['frames'], train_feats['track_ids'], \
                           train_feats['bbox_center_x'], train_feats['bbox_center_y'], train_feats['bbox_x'], \
                           train_feats['bbox_y']

            inference_feats = torch.load(inference_dataset_path + file_name)
            inference_x, inference_y, inference_frames, inference_track_ids, inference_center_x, inference_center_y, \
            inference_bbox_x, inference_bbox_y = \
                inference_feats['x'], inference_feats['y'], inference_feats['frames'], inference_feats['track_ids'], \
                inference_feats['bbox_center_x'], inference_feats['bbox_center_y'], inference_feats['bbox_x'], \
                inference_feats['bbox_y']

            # Split Val and Test
            split_percent = 0.2
            val_length = int(split_percent * len(inference_x))
            inference_length = len(inference_x) - val_length

            inference_dataset = FeaturesDatasetExtra(inference_x, inference_y, frames=inference_frames,
                                                     track_ids=inference_track_ids, mode=FeaturesMode.UV,
                                                     preprocess=False, bbox_center_x=inference_center_x,
                                                     bbox_center_y=inference_center_y,
                                                     bbox_x=inference_bbox_x,
                                                     bbox_y=inference_bbox_y)

            val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset, [val_length, inference_length],
                                                                      generator=torch.Generator().manual_seed(42))

            logger.info('Setting up network')

            train_dataset = FeaturesDatasetExtra(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
                                                 mode=FeaturesMode.UV, preprocess=False, bbox_center_x=train_center_x,
                                                 bbox_center_y=train_center_y,
                                                 bbox_x=train_bbox_x,
                                                 bbox_y=train_bbox_y)

            # train_subset = torch.utils.data.Subset(train_dataset, [i for i in range(16)])
            train_subset = torch.utils.data.Subset(train_dataset, [5, 8, 45, 90, 456])

            train_loader = torch.utils.data.DataLoader(train_subset, 1)

            val_loader = torch.utils.data.DataLoader(val_dataset, 1)

            time_steps = 5

            net = SimpleModel(meta=meta, num_frames_to_build_bg_sub_model=12, layers_mode='rnn',
                              meta_video=SDDVideoDatasets.QUAD, meta_train_video_number=train_vid_num,
                              meta_val_video_number=inference_vid_num, train_dataset=train_dataset,
                              val_dataset=val_dataset, time_steps=time_steps)

            opt = torch.optim.Adam(net.parameters(), lr=1e-2)  # center based 1e-2, points 1e-3
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=1000)
            # scheduler = StepLR(opt, step_size=500, gamma=0.1)
            loss_list = []

            for ep_i in tqdm(range(10002)):
                total_loss = torch.tensor(data=0, dtype=torch.float32)
                # opt.zero_grad()

                for batch_inference in train_loader:
                    # features1, features2, frame_, track_id, bbox_center1, bbox_center2 = batch_inference
                    features1, features2, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch_inference
                    frame_nums = [f.item() for f in frame_]

                    # hx, cx = torch.zeros(size=(1, 32)), torch.zeros(size=(1, 32))
                    # hx, cx = torch.zeros(size=(1, 16)), torch.zeros(size=(1, 16))
                    # hx_1, cx_1 = torch.zeros(size=(1, 32)), torch.zeros(size=(1, 32))
                    hx, cx = torch.zeros(size=(1, 32)), torch.zeros(size=(1, 32))
                    hx_1, cx_1 = torch.zeros(size=(1, 64)), torch.zeros(size=(1, 64))

                    torch.nn.init.xavier_normal_(hx)
                    torch.nn.init.xavier_normal_(cx)
                    torch.nn.init.xavier_normal_(hx_1)
                    torch.nn.init.xavier_normal_(cx_1)

                    total_loss = torch.tensor(data=0, dtype=torch.float32)
                    img_frames = []
                    pred_centers = []
                    gt_based_shifted_points_list = []
                    actual_points_list = []
                    last_input_velocity = None
                    last_pred_center = None
                    last_true_center = None
                    cap_count = 0
                    predictions = torch.zeros((0, 2))
                    true_values = torch.zeros((0, 2))
                    if len(features1) == time_steps:
                        # cap = cv.VideoCapture(f'{base_path}videos/{vid_label.value}/video{inference_vid_num}/video.mov')

                        # while 1:
                        #     ret, video_frame = cap.read()
                        #     if cap_count in frame_nums:
                        #         img_frames.append(video_frame)
                        #         # if len(img_frames) == len(frame_nums):
                        #         break
                        #     cap_count += 1

                        for i in range(time_steps):

                            feat1, feat2, bb_center1, bb_center2 = features1[i].squeeze().float(), features2[
                                i].squeeze().float(), \
                                                                   bbox_center1[i].float(), bbox_center2[i].float()
                            try:
                                center_xy, center_true_uv, center_past_uv = net.find_center_point(feat1)
                                # moved_points_by_true_of = feat2[:, :2]  # for gt based
                                # gt_based_shifted_points = bb_center2  # for bbox_center gt based
                            except IndexError:
                                print('Bad Data')
                                break

                            if i == 0:
                                # l_in_v = torch.zeros((1, 2))
                                last_input_velocity = center_past_uv.unsqueeze(0)  # * 0.4
                                # l_p_center = torch.zeros(2)
                                last_pred_center = center_xy
                                last_true_center = center_xy
                                # l_p_center[0], l_p_center[1] = last_pred_center[1], last_pred_center[0]
                                # last_pred_center = l_p_center
                                # l_in_v[..., 0], l_in_v[..., 1] = last_input_velocity[..., 1], \
                                #                                  last_input_velocity[..., 0]
                                # last_input_velocity = l_in_v

                            # with torch.no_grad():
                            # block_1 = m.block_1(center_past_uv.unsqueeze(0))
                            # hx, cx = m.rnn_cell(block_1, (hx, cx))
                            # block_2 = m.block_2(hx)
                            # logger.info(f'{i} -> Input Velocity: {last_input_velocity}')
                            # logger.info(f'{i} -> Current Center: {last_pred_center}')
                            # with torch.no_grad():
                            # block_1 = net.block_1(last_input_velocity)
                            # hx, cx = net.rnn_cell(block_1, (hx, cx))
                            block_2, cx, hx, cx_1, hx_1 = net(last_input_velocity, cx, hx, cx_1, hx_1)

                            true_velocity = bb_center2  # feat1[:, 2:4].mean(dim=0).unsqueeze(0)

                            gt_based_shifted_points = feat1[:, :2] + feat1[:, 2:4]
                            # gt_based_shifted_points = feat1[:, :2] + (feat1[:, 2:4] * 0.4)
                            moved_points_by_true_of_center = gt_based_shifted_points.mean(0)
                            gt_based_shifted_points_list.append(gt_based_shifted_points)
                            # pred_center = center_xy + block_2
                            pred_center = last_pred_center + block_2
                            # pred_center = last_pred_center + (block_2 * 0.4)  # velocity * time - bad
                            predictions = torch.cat((predictions, block_2), dim=0)
                            true_values = torch.cat((true_values, true_velocity), dim=0)

                            # swapped
                            # pred_center_swapped = torch.zeros_like(pred_center)
                            # pred_center_swapped[:, 0] = pred_center[:, 1]
                            # pred_center_swapped[:, 1] = pred_center[:, 0]

                            last_input_velocity = block_2
                            # logger.info(f'{i} -> True Center: {moved_points_by_true_of_center}')
                            # logger.info(f'{i} -> True Delta Center: {moved_points_by_true_of_center - last_true_center}')
                            # logger.info(f'{i} -> True Delta Distance: '
                            #             f'{torch.norm(moved_points_by_true_of_center - last_true_center, p=2)}')

                            # logger.info(f'{i} -> Pred Delta Center: {pred_center - last_pred_center}')
                            # logger.info(f'{i} -> Pred Delta Distance: {torch.norm(pred_center - last_pred_center, p=2)}')
                            last_pred_center = pred_center
                            last_true_center = moved_points_by_true_of_center
                            pred_centers.append(pred_center.squeeze(0))

                            # swapped
                            # logger.info(f'{i} -> Pred Delta Center: {pred_center_swapped - last_pred_center}')
                            # logger.info(f'{i} -> Pred Delta Distance: '
                            #             f'{torch.norm(pred_center_swapped - last_pred_center, p=2)}')
                            # last_pred_center = pred_center_swapped
                            # last_true_center = moved_points_by_true_of_center
                            # pred_centers.append(pred_center_swapped.squeeze(0))

                            # Remove mean for per time step plot
                            actual_points_list.append(feat1[:, :2].detach().mean(dim=0).numpy())
                            # logger.info(f'{i} -> Predicted Center: {last_pred_center}')
                            # logger.info(f'{i} -> True Center: {last_true_center}')
                            # logger.info(f'{i} -> Predicted Velocity: {last_input_velocity}')
                            # logger.info(f'{i} -> True Velocity: {true_velocity}')
                            # logger.info(f'*****Next Step*****')
                            # logger.info(f'{i} -> Pred Center: {last_pred_center}')

                            total_loss += net.cluster_center_points_center_loss_meters(points=gt_based_shifted_points,
                                                                                       pred_center=pred_center.squeeze(
                                                                                           0))
                        # opt.zero_grad()
                        loss_list.append(total_loss.item())
                        total_loss.backward()
                        predicted_points = [p.detach().numpy() for p in pred_centers]
                        true_points = [p.detach().mean(dim=0).numpy() for p in gt_based_shifted_points_list]
                        # logger.info(f'Pred Trajectory Length: '
                        #             f'{np.linalg.norm(predicted_points[-1] - predicted_points[0], 2)}')
                        # logger.info(f'True Trajectory Length: {np.linalg.norm(true_points[-1] - true_points[0], 2)}')
                        if ep_i % 1000 == 0:
                            plot_trajectory_rnn(predicted_points=np.stack(predicted_points),
                                                true_points=np.stack(true_points),
                                                actual_points=actual_points_list,
                                                imgs=None, gt=True)
                opt.step()
                # logger.info(f'Loss: {total_loss.item()}')
                scheduler.step(total_loss.item())
                opt.zero_grad()

            # torch.save(net, '../Plots/overfit_scheduler_b1_of.pt')
            import matplotlib.pyplot as plt

            plt.plot(loss_list)
            plt.title('Loss')
            plt.show()

            # predictions = predictions.detach().numpy()
            # plt.scatter(predictions[..., 0], predictions[..., 1], label='Predictions')
            # plt.scatter(true_values[..., 0], true_values[..., 1], label='True')
            # plt.title('Predictions')
            # plt.legend(loc="upper right")
            # plt.show()
        elif compare_models:
            base_save_path = '/home/rishabh/Thesis/imgs/'
            # file_name = 'time_distributed_velocity_features_with_frame_track_rnn.pt'
            file_name = 'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox.pt'
            logger.info('Setting up DataLoaders')
            train_feats = torch.load(train_dataset_path + file_name)
            train_x, train_y, train_frames, train_track_ids, train_center_x, train_center_y, train_bbox_x, \
            train_bbox_y = train_feats['x'], train_feats['y'], train_feats['frames'], train_feats['track_ids'], \
                           train_feats['bbox_center_x'], train_feats['bbox_center_y'], train_feats['bbox_x'], \
                           train_feats['bbox_y']

            inference_feats = torch.load(inference_dataset_path + file_name)
            inference_x, inference_y, inference_frames, inference_track_ids, inference_center_x, inference_center_y, \
            inference_bbox_x, inference_bbox_y = \
                inference_feats['x'], inference_feats['y'], inference_feats['frames'], inference_feats['track_ids'], \
                inference_feats['bbox_center_x'], inference_feats['bbox_center_y'], inference_feats['bbox_x'], \
                inference_feats['bbox_y']

            # Split Val and Test
            split_percent = 0.3
            val_length = int(split_percent * len(inference_x))
            inference_length = len(inference_x) - val_length

            inference_dataset = FeaturesDatasetExtra(inference_x, inference_y, frames=inference_frames,
                                                     track_ids=inference_track_ids, mode=FeaturesMode.UV,
                                                     preprocess=False, bbox_center_x=inference_center_x,
                                                     bbox_center_y=inference_center_y,
                                                     bbox_x=inference_bbox_x,
                                                     bbox_y=inference_bbox_y)

            val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset, [val_length, inference_length],
                                                                      generator=torch.Generator().manual_seed(42))

            logger.info('Setting up network')

            train_dataset = FeaturesDatasetExtra(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
                                                 mode=FeaturesMode.UV, preprocess=False, bbox_center_x=train_center_x,
                                                 bbox_center_y=train_center_y,
                                                 bbox_x=train_bbox_x,
                                                 bbox_y=train_bbox_y)

            train_subset = torch.utils.data.Subset(train_dataset, [i for i in range(10)])

            # overfit - active
            train_loader = torch.utils.data.DataLoader(train_dataset, 1)

            val_loader = torch.utils.data.DataLoader(val_dataset, 1)

            test_loader = torch.utils.data.DataLoader(test_dataset, 1)

            kwargs_dict = {
                'meta': meta,
                'layers_mode': 'rnn',
                'meta_video': SDDVideoDatasets.QUAD,
                'meta_train_video_number': train_vid_num,
                'meta_val_video_number': inference_vid_num,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'time_steps': 5
            }
            of_model = SimpleModel(**kwargs_dict)
            # after debug - of based - all points
            # of_model.load_state_dict(torch.load('lightning_logs/version_24/checkpoints/epoch=128.ckpt')['state_dict'])
            # center based
            # of_model.load_state_dict(torch.load('lightning_logs/version_38/checkpoints/epoch=199.ckpt')['state_dict'])
            of_model.load_state_dict(torch.load('lightning_logs/version_40/checkpoints/epoch=9.ckpt')['state_dict'])
            of_model.eval()

            gt_model = SimpleModel(**kwargs_dict)
            # gt_model.load_state_dict(torch.load('lightning_logs/version_27/checkpoints/epoch=30.ckpt')['state_dict'])
            # gt_model.load_state_dict(torch.load('lightning_logs/version_33/checkpoints/epoch=30.ckpt')['state_dict'])
            gt_model.load_state_dict(torch.load('lightning_logs/version_35/checkpoints/epoch=6.ckpt')['state_dict'])
            gt_model.eval()
            time_steps = kwargs_dict['time_steps']

            ade_dataset_of = []
            fde_dataset_of = []

            ade_dataset_gt = []
            fde_dataset_gt = []

            ade_dataset_gt_of_gt = []
            fde_dataset_gt_of_gt = []

            ade_dataset_linear = []
            fde_dataset_linear = []

            is_gt_based_shifted_points_inside_list = []
            is_of_based_shifted_points_center_inside_list = []
            is_pred_center_inside_list = []
            is_pred_center_gt_inside_list = []

            bad_data = False

            for sav_i, batch_inference in enumerate(tqdm(val_loader)):
                features1, features2, frame_, track_id, bbox_center1, bbox_center2, bbox1, bbox2 = batch_inference
                frame_nums = [f.item() for f in frame_]

                hx, cx = torch.zeros(size=(1, 32)), torch.zeros(size=(1, 32))
                hx_gt, cx_gt = torch.zeros(size=(1, 32)), torch.zeros(size=(1, 32))

                total_loss = torch.tensor(data=0, dtype=torch.float32)
                img_frames = []
                linear_pred_centers = []
                pred_centers = []
                pred_centers_gt = []
                gt_based_shifted_points_list = []
                of_based_shifted_points_list = []
                actual_points_list = []
                last_input_velocity = None
                last_pred_center = None
                last_true_center = None
                cap_count = 0

                first_ts_velocity_linear = None

                if len(features1) == time_steps:
                    cap = cv.VideoCapture(f'{base_path}videos/{vid_label.value}/video{inference_vid_num}/video.mov')

                    while 1:
                        ret, video_frame = cap.read()
                        if cap_count in frame_nums:
                            img_frames.append(video_frame)
                            # if len(img_frames) == len(frame_nums):
                            break
                        cap_count += 1

                    for i in range(time_steps):

                        feat1, feat2, bb_center1, bb_center2, bb1, bb2 = features1[i].squeeze().float(), features2[
                            i].squeeze().float(), bbox_center1[i].float(), bbox_center2[i].float(), bbox1[i].float(), \
                                                                         bbox2[i].float()
                        try:
                            center_xy, center_true_uv, center_past_uv = of_model.find_center_point(feat1)
                            # gt_based_shifted_points = feat2[:, :2]  # for gt based
                            gt_based_shifted_points = bb_center2  # for bbox_center gt based
                        except IndexError:
                            print('Bad Data')
                            bad_data = True
                            break

                        if i == 0:
                            last_input_velocity = center_past_uv.unsqueeze(0)  # * 0.4  # bug workaround
                            last_input_velocity_gt = center_past_uv.unsqueeze(0)
                            last_pred_center = center_xy
                            last_pred_center_gt = center_xy
                            last_true_center = center_xy
                            linear_pred_center = center_xy
                            first_ts_velocity_linear = center_past_uv

                        with torch.no_grad():
                            # logger.info(f'{i} -> Input Velocity: {last_input_velocity}')
                            # logger.info(f'{i} -> Current Center: {last_pred_center}')
                            block_1 = of_model.block_1(last_input_velocity)
                            hx, cx = of_model.rnn_cell(block_1, (hx, cx))
                            block_2 = of_model.block_2(hx)

                            block_1_gt = gt_model.block_1(last_input_velocity_gt)
                            hx_gt, cx_gt = gt_model.rnn_cell(block_1_gt, (hx_gt, cx_gt))
                            block_2_gt = gt_model.block_2(hx_gt)

                        # gt_based_shifted_points = feat1[:, :2] + feat1[:, 2:4]
                        # moved_points_by_true_of_center = gt_based_shifted_points.mean(0)
                        gt_based_shifted_points_list.append(gt_based_shifted_points)

                        of_based_shifted_points = feat1[:, :2] + feat1[:, 2:4]
                        of_based_shifted_points_center = of_based_shifted_points.mean(0)
                        of_based_shifted_points_list.append(of_based_shifted_points)
                        # pred_center = center_xy + block_2
                        pred_center = last_pred_center + block_2
                        pred_center_gt = last_pred_center_gt + block_2_gt
                        linear_pred_center += first_ts_velocity_linear * 0.4  # d = v * t
                        # pred_center = last_pred_center + (block_2 * 0.4)  # velocity * time - bad

                        is_gt_based_shifted_points_inside = is_inside_bbox(point=gt_based_shifted_points.squeeze(),
                                                                           bbox=bb2.squeeze()).item()
                        is_of_based_shifted_points_center_inside = is_inside_bbox(
                            point=of_based_shifted_points_center.squeeze(),
                            bbox=bb2.squeeze()).item()
                        is_pred_center_inside = is_inside_bbox(point=pred_center.squeeze(), bbox=bb2.squeeze()).item()
                        is_pred_center_gt_inside = is_inside_bbox(point=pred_center_gt.squeeze(),
                                                                  bbox=bb2.squeeze()).item()

                        is_gt_based_shifted_points_inside_list.append(is_gt_based_shifted_points_inside)
                        is_of_based_shifted_points_center_inside_list.append(is_of_based_shifted_points_center_inside)
                        is_pred_center_inside_list.append(is_pred_center_inside)
                        is_pred_center_gt_inside_list.append(is_pred_center_gt_inside)

                        # loss
                        loss_gt_all_points = of_model.cluster_center_loss_meters(gt_based_shifted_points,
                                                                                 pred_center_gt)
                        loss_gt_points_center = of_model. \
                            cluster_center_points_center_loss_meters(gt_based_shifted_points,
                                                                     pred_center_gt)

                        loss_of_all_points = gt_model.cluster_center_loss_meters(of_based_shifted_points,
                                                                                 pred_center)
                        loss_of_points_center = gt_model. \
                            cluster_center_points_center_loss_meters(of_based_shifted_points,
                                                                     pred_center)

                        # logger.info('')
                        # logger.info(f'GT-All points: {loss_gt_all_points}')
                        # logger.info(f'GT-Points center: {loss_gt_points_center}')
                        # logger.info(f'OF-All points: {loss_of_all_points}')
                        # logger.info(f'OF-Points center: {loss_of_points_center}')

                        last_input_velocity = block_2
                        last_input_velocity_gt = block_2_gt
                        # logger.info(f'{i} -> True Center: {moved_points_by_true_of_center}')
                        # logger.info(f'{i} -> True Delta Center: {moved_points_by_true_of_center - last_true_center}')
                        # logger.info(f'{i} -> True Delta Distance: '
                        #             f'{torch.norm(moved_points_by_true_of_center - last_true_center, p=2)}')

                        # logger.info(f'{i} -> Pred Delta Center: {pred_center - last_pred_center}')
                        # logger.info(f'{i} -> Pred Delta Distance: {torch.norm(pred_center - last_pred_center, p=2)}')
                        last_pred_center = pred_center
                        last_pred_center_gt = pred_center_gt
                        # last_true_center = moved_points_by_true_of_center
                        pred_centers.append(pred_center.squeeze(0))
                        pred_centers_gt.append(pred_center_gt.squeeze(0))
                        linear_pred_centers.append(linear_pred_center)

                        # Remove mean for per time step plot
                        actual_points_list.append(feat1[:, :2].detach().mean(dim=0).numpy())
                        # logger.info(f'{i} -> Predicted Velocity: {last_input_velocity}')
                        # logger.info(f'{i} -> True Velocity: {feat1[:, 2:4].mean(dim=0)}')
                        # logger.info(f'*****Next Step*****')
                        # logger.info(f'{i} -> Pred Center: {last_pred_center}')

                    if bad_data:
                        bad_data = False
                        continue

                    predicted_points = [p.detach().numpy() for p in pred_centers]
                    predicted_points_gt = [p.detach().numpy() for p in pred_centers_gt]
                    linear_predicted_points = [p.detach().numpy() for p in linear_pred_centers]

                    true_points = [p.detach().mean(dim=0).numpy() for p in gt_based_shifted_points_list]
                    true_points_of = [p.detach().mean(dim=0).numpy() for p in of_based_shifted_points_list]

                    # logger.info(f'Optical Flow Trajectory Length: '
                    #             f'{trajectory_length(predicted_points)}')
                    # logger.info(f'GT Trajectory Length: '
                    #             f'{trajectory_length(predicted_points_gt)}')
                    # logger.info(f'True GT Trajectory Length: {trajectory_length(true_points)}')
                    # logger.info(f'True OF Trajectory Length: {trajectory_length(true_points_of)}')

                    l2_points = {idx: np.linalg.norm(i - j, 2)
                                 for idx, (i, j) in enumerate(zip(true_points_of, predicted_points))}
                    # logger.info(f'Optical Flow L2 corresponding centers: {l2_points}')

                    l2_points_gt = {idx: np.linalg.norm(i - j, 2)
                                    for idx, (i, j) in enumerate(zip(true_points, predicted_points_gt))}
                    # logger.info(f'GT L2 corresponding centers: {l2_points_gt}')

                    l2_linear = {idx: np.linalg.norm(i - j, 2)
                                 for idx, (i, j) in enumerate(zip(true_points_of, linear_predicted_points))}

                    # ade_of = compute_ade(np.stack(predicted_points), np.stack(true_points_of))
                    # fde_of = compute_fde(np.stack(predicted_points), np.stack(true_points_of))
                    ade_of = compute_ade(np.stack(predicted_points), np.stack(true_points))
                    fde_of = compute_fde(np.stack(predicted_points), np.stack(true_points))
                    ade_dataset_of.append(ade_of.item())
                    fde_dataset_of.append(fde_of.item())

                    ade_gt = compute_ade(np.stack(predicted_points_gt), np.stack(true_points))
                    fde_gt = compute_fde(np.stack(predicted_points_gt), np.stack(true_points))
                    ade_dataset_gt.append(ade_gt.item())
                    fde_dataset_gt.append(fde_gt.item())

                    ade_gt_of_gt = compute_ade(np.stack(true_points_of), np.stack(true_points))
                    fde_gt_of_gt = compute_fde(np.stack(true_points_of), np.stack(true_points))
                    ade_dataset_gt_of_gt.append(ade_gt_of_gt.item())
                    fde_dataset_gt_of_gt.append(fde_gt_of_gt.item())

                    ade_linear = compute_ade(np.stack(linear_predicted_points), np.stack(true_points))
                    fde_linear = compute_fde(np.stack(linear_predicted_points), np.stack(true_points))
                    ade_dataset_linear.append(ade_linear.item())
                    fde_dataset_linear.append(fde_linear.item())

                    # plot
                    # plot_trajectory_rnn_compare(predicted_points=np.stack(predicted_points),
                    #                             predicted_points_gt=np.stack(predicted_points_gt),
                    #                             true_points=np.stack(true_points),
                    #                             true_points_of=np.stack(true_points_of),
                    #                             of_l2=l2_points,
                    #                             gt_l2=l2_points_gt,
                    #                             actual_points=actual_points_list,
                    #                             imgs=img_frames, gt=True,
                    #                             m_ratio=gt_model.val_ratio,
                    #                             show=False,
                    #                             save_path=f'{base_save_path}compare_rgb_{sav_i}')
                    # plot_trajectory_rnn_compare(predicted_points=np.stack(predicted_points),
                    #                             predicted_points_gt=np.stack(predicted_points_gt),
                    #                             true_points=np.stack(true_points),
                    #                             true_points_of=np.stack(true_points_of),
                    #                             of_l2=l2_points,
                    #                             gt_l2=l2_points_gt,
                    #                             actual_points=actual_points_list,
                    #                             imgs=None, gt=True,
                    #                             m_ratio=gt_model.val_ratio,
                    #                             show=False,
                    #                             save_path=f'{base_save_path}compare_traj_{sav_i}')
                    #
                    # plot_trajectory_rnn_compare_side_by_side(predicted_points=np.stack(predicted_points),
                    #                                          predicted_points_gt=np.stack(predicted_points_gt),
                    #                                          true_points=np.stack(true_points),
                    #                                          true_points_of=np.stack(true_points_of),
                    #                                          of_l2=l2_points,
                    #                                          gt_l2=l2_points_gt,
                    #                                          actual_points=actual_points_list,
                    #                                          imgs=img_frames, gt=True,
                    #                                          m_ratio=gt_model.val_ratio,
                    #                                          show=False,
                    #                                          save_path=
                    #                                          f'{base_save_path}compare_side_by_side_rgb_{sav_i}')

            plot_bars_if_inside_bbox([is_gt_based_shifted_points_inside_list,
                                      is_of_based_shifted_points_center_inside_list,
                                      is_pred_center_inside_list,
                                      is_pred_center_gt_inside_list])

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
        elif compare_of_gt:
            file_name = 'time_distributed_velocity_features_with_frame_track_rnn.pt'
            logger.info('Setting up DataLoaders')
            train_feats = torch.load(train_dataset_path + file_name)
            train_x, train_y, train_frames, train_track_ids, train_center_x, train_center_y, train_bbox_x, \
            train_bbox_y = train_feats['x'], train_feats['y'], train_feats['frames'], train_feats['track_ids'], \
                           train_feats['bbox_center_x'], train_feats['bbox_center_y'], train_feats['bbox_x'], \
                           train_feats['bbox_y']

            inference_feats = torch.load(inference_dataset_path + file_name)
            inference_x, inference_y, inference_frames, inference_track_ids, inference_center_x, inference_center_y, \
            inference_bbox_x, inference_bbox_y = \
                inference_feats['x'], inference_feats['y'], inference_feats['frames'], inference_feats['track_ids'], \
                inference_feats['bbox_center_x'], inference_feats['bbox_center_y'], inference_feats['bbox_x'], \
                inference_feats['bbox_y']

            # Split Val and Test
            split_percent = 0.3
            val_length = int(split_percent * len(inference_x))
            inference_length = len(inference_x) - val_length

            inference_dataset = FeaturesDatasetExtra(inference_x, inference_y, frames=inference_frames,
                                                     track_ids=inference_track_ids, mode=FeaturesMode.UV,
                                                     preprocess=False, bbox_center_x=inference_center_x,
                                                     bbox_center_y=inference_center_y,
                                                     bbox_x=inference_bbox_x,
                                                     bbox_y=inference_bbox_y)

            val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset, [val_length, inference_length])

            logger.info('Setting up network')

            train_dataset = FeaturesDatasetExtra(train_x, train_y, frames=train_frames, track_ids=train_track_ids,
                                                 mode=FeaturesMode.UV, preprocess=False, bbox_center_x=train_center_x,
                                                 bbox_center_y=train_center_y,
                                                 bbox_x=train_bbox_x,
                                                 bbox_y=train_bbox_y)

            train_subset = torch.utils.data.Subset(train_dataset, [i for i in range(10)])

            # overfit - active
            train_loader = torch.utils.data.DataLoader(train_dataset, 1)

            val_loader = torch.utils.data.DataLoader(val_dataset, 1)

            test_loader = torch.utils.data.DataLoader(test_dataset, 1)

            time_steps = 5

            kwargs_dict = {
                'meta': meta,
                'layers_mode': 'rnn',
                'meta_video': SDDVideoDatasets.QUAD,
                'meta_train_video_number': train_vid_num,
                'meta_val_video_number': inference_vid_num
            }
            of_model = SimpleModel(**kwargs_dict)

            dist_list = []
            of_center = 0

            for sav_i, batch_inference in enumerate(tqdm(val_loader)):
                features1, features2, frame_, track_id, bbox_center1, bbox_center2 = batch_inference

                moved_by_of = []
                moved_by_gt = []

                if len(features1) == time_steps:  # Todo: check if the mirror effect is really there - chk screenshot
                    for i in range(time_steps):

                        feat1, feat2, bb_center1, bb_center2 = features1[i].squeeze().float(), features2[
                            i].squeeze().float(), \
                                                               bbox_center1[i].float(), bbox_center2[i].float()
                        try:
                            center_xy, center_true_uv, center_past_uv = of_model.find_center_point(feat1)
                            # center_xy_gt, center_true_uv_gt, center_past_uv_gt = of_model.find_center_point(feat2)
                            # center_xy_gt = feat2[:, :2].mean(0)
                            center_xy_gt = bb_center2  # for bbox_center gt based
                        except IndexError:
                            print('Bad Data')
                            break

                        # moved_points_by_true_of = center_xy + feat1[:, 2:4].mean(0)
                        # moved_points_by_true_of = center_xy + center_true_uv

                        gt_based_shifted_points = feat1[:, :2].mean(0) + feat1[:, 2:4].mean(0)
                        # center_xy_gt = feat2[:, :2].mean(0)

                        moved_by_of.append(gt_based_shifted_points.numpy())
                        moved_by_gt.append(center_xy_gt.numpy())

                        dist = torch.norm(gt_based_shifted_points - center_xy_gt, p=2)
                        dist_list.append(dist.item())

                    plot_trajectory_rnn(predicted_points=np.stack(moved_by_of),
                                        true_points=np.stack(moved_by_gt))

                    # if i == 0: move inside, out due to reformatting
                    #     of_center = center_xy
                    #     gt_center = center_xy
                    #
                    #     dist = torch.norm(of_center - gt_center, p=2)
                    #     dist_list.append(dist.item())
                    # else:
                    #     moved_points_by_true_of = of_center + feat1[:, 2:4].mean(0)
                    #     # moved_points_by_true_of_center = moved_points_by_true_of.mean(0)
                    #
                    #     dist = torch.norm(moved_points_by_true_of - center_xy_gt, p=2)
                    #     dist_list.append(dist.item())
                    #
                    #     of_center = moved_points_by_true_of
            logger.info(f'Error Max : {torch.tensor(dist_list).max()}')
            logger.info(f'Error Min : {torch.tensor(dist_list).min()}')
            logger.info(f'Error Median : {torch.tensor(dist_list).median()}')
            logger.info(f'Error Mean : {torch.tensor(dist_list).mean()}')

            logger.info(f'Error Max [m] : {torch.tensor(dist_list).max() * of_model.val_ratio}')
            logger.info(f'Error Min [m] : {torch.tensor(dist_list).min() * of_model.val_ratio}')
            logger.info(f'Error Median [m] : {torch.tensor(dist_list).median() * of_model.val_ratio}')
            logger.info(f'Error Mean [m] : {torch.tensor(dist_list).mean() * of_model.val_ratio}')
        else:
            logger.info('Setting up DataLoaders')
            train_feats = torch.load(train_dataset_path + file_name)
            train_x, train_y = train_feats['x'], train_feats['y']

            inference_feats = torch.load(inference_dataset_path + file_name)
            inference_x, inference_y = inference_feats['x'], inference_feats['y']

            # Split Val and Test
            split_percent = 0.2
            val_length = int(split_percent * len(inference_x))
            inference_length = len(inference_x) - val_length

            inference_dataset = FeaturesDataset(inference_x, inference_y, mode=FeaturesMode.UV, preprocess=False)
            val_dataset, test_dataset = torch.utils.data.random_split(inference_dataset, [val_length, inference_length])

            logger.info('Setting up network')

            if test:
                pass
            elif inference:
                net_0 = SimpleModel.load_from_checkpoint('lightning_logs/version_8/checkpoints/epoch=4.ckpt'
                                                         , layers_mode='small', meta=meta,
                                                         meta_video=SDDVideoDatasets.QUAD,
                                                         meta_train_video_number=train_vid_num,
                                                         meta_val_video_number=inference_vid_num)
                net_ = SimpleModel.load_from_checkpoint('lightning_logs/version_7/checkpoints/epoch=20.ckpt', meta=meta,
                                                        meta_video=SDDVideoDatasets.QUAD,
                                                        meta_train_video_number=train_vid_num,
                                                        meta_val_video_number=inference_vid_num)
                logger.info(f"Inference Network: {net_.__class__.__name__}")
                logger.info(f"Starting Inference")
                net_.eval()
                loader = torch.utils.data.DataLoader(val_dataset, 1)

                inference_feats_ = torch.load(inference_dataset_path + 'time_distributed_features.pt')
                inference_x_, inference_y_ = inference_feats_['x'], inference_feats_['y']

                # Split Val and Test
                split_percent = 0.2
                val_length_ = int(split_percent * len(inference_x_))
                test_length_ = len(inference_x_) - val_length_

                inference_dataset_ = FeaturesDataset(inference_x_, inference_y_, mode=FeaturesMode.UV, preprocess=False)
                val_dataset_, test_dataset_ = torch.utils.data.random_split(inference_dataset_,
                                                                            [val_length_, test_length_])
                loader_ = torch.utils.data.DataLoader(val_dataset_, 1)

                # train_dataset = FeaturesDataset(train_x, train_y, mode=FeaturesMode.UV, preprocess=False)
                # loader = torch.utils.data.DataLoader(train_dataset, 1)

                # for batch, batch_ in tqdm(zip(loader, loader_)):
                #     features1, features2 = batch
                #     features1, features2 = features1.squeeze().float(), features2.squeeze().float()
                #     # input prior velocity
                #     pred = net_(features1[:, 4:])
                #     # predicted optical flow moved points
                #     moved_points_pred_of = features1[:, :2] + pred
                #     # move points by actual optical flow
                #     moved_points_by_true_of = features1[:, :2] + features1[:, 2:4]
                #
                #     # pred 2
                #     features1_, features2_ = batch_
                #     features1_, features2_ = features1_.squeeze().float(), features2_.squeeze().float()
                #     # input prior velocity
                #     pred_0 = net_0(features1_[:, 4:])
                #     # predicted optical flow moved points
                #     moved_points_pred_of_0 = features1_[:, :2] + pred_0
                #     # move points by actual optical flow
                #     moved_points_by_true_of_0 = features1_[:, :2] + features1_[:, 2:4]
                #
                #     # plot
                #     # plot_points_predicted_and_true(predicted_points=moved_points_pred_of.detach().numpy(),
                #     #                                true_points=moved_points_by_true_of.detach().numpy(),
                #     #                                actual_points=features1[:, :2])  # same frame
                #     plot_and_compare_points_predicted_and_true(predicted_points=moved_points_pred_of.detach().numpy(),
                #                                                true_points=moved_points_by_true_of.detach().numpy(),
                #                                                actual_points=features1[:, :2],
                #                                                predicted_points_1=
                #                                                moved_points_pred_of_0.detach().numpy(),
                #                                                true_points_1=
                #                                                moved_points_by_true_of_0.detach().numpy(),
                #                                                actual_points_1=features1_[:, :2])  # same frame

                of_model = SimpleModel.load_from_checkpoint('lightning_logs/version_10/checkpoints/epoch=15.ckpt',
                                                            layers_mode='small', meta=meta,
                                                            meta_video=SDDVideoDatasets.QUAD,
                                                            meta_train_video_number=train_vid_num,
                                                            meta_val_video_number=inference_vid_num)
                for batch_inference, batch_ in tqdm(zip(loader, loader_)):
                    features1, features2 = batch_inference
                    features1, features2 = features1.squeeze().float(), features2.squeeze().float()
                    # find the center
                    center_xy, center_true_uv, center_past_uv = of_model.find_center_point(features1)
                    # input prior velocity
                    pred = of_model(center_past_uv)
                    pred_center = center_xy + pred
                    # move points by actual optical flow
                    gt_based_shifted_points = features1[:, :2] + features1[:, 2:4]

                    # plot
                    plot_points_predicted_and_true(predicted_points=pred_center.detach().numpy(),
                                                   true_points=gt_based_shifted_points.detach().numpy(),
                                                   actual_points=None)  # same frame

            else:
                if resume:
                    resume_path = 'lightning_logs/version_2/checkpoints/epoch=0.ckpt'
                    trainer = pl.Trainer(gpus=1, max_epochs=20, resume_from_checkpoint=resume_path)
                else:
                    trainer = pl.Trainer(gpus=1, max_epochs=100)

                logger.info('Starting training')
                trainer.fit(net)
