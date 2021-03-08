import os
from pathlib import Path
from typing import Optional

import matplotlib
import torch
import numpy as np
import yaml
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import compute_fde, compute_ade
from baselinev2.config import BASE_PATH, ROOT_PATH, DEBUG_MODE, EVAL_USE_SOCIAL_LSTM_MODEL, EVAL_USE_BATCH_NORM, \
    EVAL_PATH_TO_VIDEO, EVAL_PLOT_PATH, GT_CHECKPOINT_ROOT_PATH, UNSUPERVISED_CHECKPOINT_ROOT_PATH, EVAL_TRAIN_CLASS, \
    EVAL_TRAIN_VIDEO_NUMBER, EVAL_TRAIN_META, EVAL_VAL_CLASS, EVAL_VAL_VIDEO_NUMBER, EVAL_VAL_META, EVAL_TEST_CLASS, \
    EVAL_TEST_VIDEO_NUMBER, EVAL_TEST_META, EVAL_BATCH_SIZE, EVAL_SHUFFLE, EVAL_WORKERS, PLOT_MODE, \
    EVAL_USE_FINAL_POSITIONS_SUPERVISED, EVAL_USE_FINAL_POSITIONS_UNSUPERVISED, EVAL_USE_SIMPLE_MODEL, \
    EVAL_SIMPLE_MODEL_CONFIG_DICT_GT, EVAL_SIMPLE_MODEL_CONFIG_DICT_UNSUPERVISED, SIMPLE_GT_CHECKPOINT_ROOT_PATH, \
    SIMPLE_UNSUPERVISED_CHECKPOINT_ROOT_PATH, EVAL_FOR_WHOLE_CLASS, EVAL_TRAIN_VIDEOS_TO_SKIP, EVAL_VAL_VIDEOS_TO_SKIP, \
    EVAL_TEST_VIDEOS_TO_SKIP, SIMPLE_GT_CHECKPOINT_PATH, SIMPLE_UNSUPERVISED_CHECKPOINT_PATH, DEVICE, BEST_MODEL, \
    EVAL_SINGLE_MODEL, SINGLE_MODEL_CHECKPOINT_PATH, BATCH_PLOT_MODE, EVAL_USE_GENERATED, EVAL_FROM_OVERFIT, \
    EVAL_EXTRACT_STATS
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import get_dataset, ConcatenateDataset
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.models import BaselineRNN, BaselineRNNStacked, BaselineRNNStackedSimple, ConstantLinearBaseline
from baselinev2.nn.train import get_dataset_for_class
from baselinev2.notebooks.utils import get_trajectory_length
from baselinev2.overfit_config import LINEAR_CFG
from baselinev2.utils import social_lstm_parser
from baselinev2.nn.social_lstm.model import BaselineLSTM
from baselinev2.plot_utils import plot_trajectory_alongside_frame, plot_and_compare_trajectory_four_way, \
    plot_and_compare_trajectory_alongside_frame, plot_trajectories
from log import initialize_logging, get_logger

matplotlib.style.use('ggplot')

initialize_logging()
logger = get_logger('baselinev2.nn.evaluate')


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, checkpoint_root_path: str, video_path: str,
                   plot_path: Optional[str] = None):
    checkpoint_path = checkpoint_root_path + 'checkpoints/'
    checkpoint_file = os.listdir(checkpoint_path)[-1]
    model = model.load_from_checkpoint(
        checkpoint_path=checkpoint_path + checkpoint_file,
        hparams_file=f'{checkpoint_root_path}hparams.yaml',
        map_location=None
    )

    model.eval()

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = data

        total_loss = torch.tensor(data=0, dtype=torch.float32)
        predicted_xy, true_xy = [], []

        # Encoder
        b, seq_len = in_uv.size(0), in_uv.size(1)
        h0, c0 = model.init_hidden_states(b_size=b)
        out = model.pre_encoder(in_uv.view(-1, 2))
        out = F.relu(out.view(seq_len, b, -1))
        out, (h_enc, c_enc) = model.encoder(out, (h0, c0))
        # Decoder
        # Last (x,y) and (u,v) position at T=8
        last_xy = in_xy[:, -1, ...]
        last_uv = in_uv[:, -1, ...]

        h_dec, c_dec = h_enc.squeeze(0), c_enc.squeeze(0)
        for gt_pred_xy in gt_xy.permute(1, 0, 2):
            out = model.pre_decoder(last_uv)
            h_dec, c_dec = model.decoder(out, (h_dec, c_dec))
            pred_uv = model.post_decoder(F.relu(h_dec))
            out = last_xy + (pred_uv * 0.4)
            total_loss += model.center_based_loss_meters(gt_center=gt_pred_xy, pred_center=out, ratio=ratio[0].item())

            predicted_xy.append(out.detach().cpu().numpy())
            true_xy.append(gt_pred_xy.detach().cpu().numpy())

            last_xy = out
            last_uv = pred_uv

        ade = compute_ade(np.stack(predicted_xy), np.stack(true_xy)).item()
        fde = compute_fde(np.stack(predicted_xy), np.stack(true_xy)).item()

        plot_frame_number = in_frame_numbers.squeeze()[0].item()
        plot_track_id = in_track_ids.squeeze()[0].item()
        obs_trajectory = in_xy.squeeze().numpy()
        gt_trajectory = np.stack(true_xy).squeeze()
        pred_trajectory = np.stack(predicted_xy).squeeze()
        all_frame_numbers = torch.cat((in_frame_numbers.squeeze(), gt_frame_numbers.squeeze())).tolist()

        plot_trajectory_alongside_frame(
            frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
            obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
            pred_trajectory=pred_trajectory, frame_number=plot_frame_number,
            track_id=plot_track_id, additional_text=f'Frame Numbers: {all_frame_numbers}\nADE: {ade} | FDE: {fde}',
            save_path=f'{plot_path}{checkpoint_file}/'  # None
        )
        # print()


@torch.no_grad()
def evaluate_social_lstm_model(model: nn.Module, data_loader: DataLoader, checkpoint_root_path: str, video_path: str,
                               plot_path: Optional[str] = None):
    checkpoint_path = checkpoint_root_path + 'checkpoints/'
    checkpoint_file = os.listdir(checkpoint_path)[-1]
    model = model.load_from_checkpoint(
        checkpoint_path=checkpoint_path + checkpoint_file,
        hparams_file=f'{checkpoint_root_path}hparams.yaml',
        map_location=None,
        args=social_lstm_parser(pass_final_pos=True)
    )

    model.eval()

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = data

        loss, ade, fde, ratio, pred_trajectory = model.one_step(data)

        plot_frame_number = in_frame_numbers.squeeze()[0].item()
        plot_track_id = in_track_ids.squeeze()[0].item()
        obs_trajectory = in_xy.squeeze().numpy()
        gt_trajectory = gt_xy.squeeze().numpy()
        pred_trajectory = pred_trajectory.squeeze().numpy()
        all_frame_numbers = torch.cat((in_frame_numbers.squeeze(), gt_frame_numbers.squeeze())).tolist()

        plot_trajectory_alongside_frame(
            frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
            obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
            pred_trajectory=pred_trajectory, frame_number=plot_frame_number,
            track_id=plot_track_id, additional_text=f'Frame Numbers: {all_frame_numbers}\nADE: {ade} | FDE: {fde}',
            save_path=f'{plot_path}repo_version/{checkpoint_file}/'
        )


@torch.no_grad()
def evaluate_simple_v2_model(model: Optional[nn.Module], data_loader: DataLoader, checkpoint_path: str,
                             video_path: str, plot_path: Optional[str] = None, generated: bool = False):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    last_model_state_dict = checkpoint['last_model_state_dict']
    # model = BaselineRNNStackedSimple(use_batch_norm=checkpoint['use_batch_norm'],
    #                                  encoder_lstm_num_layers=checkpoint['num_rnn_layers'],
    #                                  decoder_lstm_num_layers=checkpoint['num_rnn_layers'],
    #                                  return_pred=True,
    #                                  generated_dataset=generated,
    #                                  relative_velocities=False,
    #                                  dropout=None, rnn_dropout=0, batch_size=checkpoint['batch_size'],
    #                                  use_gru=checkpoint['use_gru'], learn_hidden_states=True)
    model = BaselineRNNStackedSimple(use_batch_norm=False, arch_config=LINEAR_CFG,
                                     encoder_lstm_num_layers=1,
                                     decoder_lstm_num_layers=1,
                                     return_pred=True,
                                     generated_dataset=generated,
                                     relative_velocities=False,
                                     dropout=None, rnn_dropout=0, batch_size=32,
                                     use_gru=False, learn_hidden_states=False)
    model.load_state_dict(model_state_dict)

    model.eval()

    for idx, data in enumerate(tqdm(data_loader)):
        if generated:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, \
            _, _, ratio = data
        else:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = data

        loss, ade, fde, ratio, pred_trajectory = model.one_step(data)

        for i in range(1):
            im_idx = np.random.choice(checkpoint['batch_size'] - 1, 1).item()

            plot_frame_number = in_frame_numbers.squeeze()[im_idx][0].item()
            plot_track_id = in_track_ids.squeeze()[im_idx][0].item()
            obs_trajectory = in_xy.squeeze().numpy()[im_idx]
            gt_trajectory = gt_xy.squeeze().numpy()[im_idx]
            pred_trajectory_in = pred_trajectory.squeeze()[:, im_idx, ...]
            all_frame_numbers = torch.cat(
                (in_frame_numbers.squeeze()[im_idx], gt_frame_numbers.squeeze()[im_idx])).tolist()

            plot_trajectory_alongside_frame(
                frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
                obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
                pred_trajectory=pred_trajectory_in, frame_number=plot_frame_number,
                track_id=plot_track_id, additional_text=f'Frame Numbers: {all_frame_numbers}\nADE: {ade} | FDE: {fde}',
                save_path=f'{plot_path}'
            )
            print()


def get_models(social_lstm, supervised_checkpoint_root_path, unsupervised_checkpoint_root_path, use_batch_norm,
               supervised_pass_final_pos=True, unsupervised_pass_final_pos=True, use_simple_model_version=False):
    if social_lstm:
        supervised_checkpoint_path = supervised_checkpoint_root_path + 'checkpoints/'
        supervised_checkpoint_file = os.listdir(supervised_checkpoint_path)[-1]
        unsupervised_checkpoint_path = unsupervised_checkpoint_root_path + 'checkpoints/'
        unsupervised_checkpoint_file = os.listdir(unsupervised_checkpoint_path)[-1]

        supervised_net = BaselineLSTM.load_from_checkpoint(
            checkpoint_path=supervised_checkpoint_path + supervised_checkpoint_file,
            hparams_file=f'{supervised_checkpoint_root_path}hparams.yaml',
            map_location=None,
            args=social_lstm_parser(pass_final_pos=supervised_pass_final_pos),
            use_batch_norm=use_batch_norm
        )
        unsupervised_net = BaselineLSTM.load_from_checkpoint(
            checkpoint_path=unsupervised_checkpoint_path + unsupervised_checkpoint_file,
            hparams_file=f'{unsupervised_checkpoint_root_path}hparams.yaml',
            map_location=None,
            args=social_lstm_parser(pass_final_pos=unsupervised_pass_final_pos),
            generated_dataset=False,  # we evaluate on ground-truth trajectories
            use_batch_norm=use_batch_norm
        )
    elif use_simple_model_version:
        # root path is the full path
        supervised_checkpoint_path = supervised_checkpoint_root_path
        unsupervised_checkpoint_path = unsupervised_checkpoint_root_path

        supervised_hyperparameter_path = supervised_checkpoint_root_path[:-15] + 'hparams.yaml'
        unsupervised_hyperparameter_path = unsupervised_checkpoint_root_path[:-15] + 'hparams.yaml'

        try:
            with open(supervised_hyperparameter_path, 'r+') as sup_f:
                supervised_hyperparameter_file = yaml.full_load(sup_f)
        except FileNotFoundError:
            supervised_hyperparameter_file = EVAL_SIMPLE_MODEL_CONFIG_DICT_GT

        # if supervised_hyperparameter_file['generated_data']:  # fixme- remove comment
        #     logger.error('Trying to load model trained on unsupervised data!!')
        #     raise RuntimeError()

        if not supervised_hyperparameter_file['use_simple_model']:
            logger.error('Model to load is not an instance of BaselineRNNStackedSimple!')
            raise RuntimeError()

        supervised_net = BaselineRNNStackedSimple(
            arch_config=supervised_hyperparameter_file['arch_config'],
            batch_size=supervised_hyperparameter_file['batch_size'],
            use_batch_norm=supervised_hyperparameter_file['use_batch_norm'],
            encoder_lstm_num_layers=supervised_hyperparameter_file['num_rnn_layers'],
            decoder_lstm_num_layers=supervised_hyperparameter_file['num_rnn_layers'],
            generated_dataset=EVAL_USE_GENERATED,  # supervised_hyperparameter_file['generated_data'],
            dropout=supervised_hyperparameter_file['dropout'],
            rnn_dropout=supervised_hyperparameter_file['rnn_dropout'],
            use_gru=supervised_hyperparameter_file['use_gru'],
            learn_hidden_states=supervised_hyperparameter_file['learn_hidden_states'],
            feed_model_distances_in_meters=supervised_hyperparameter_file['feed_model_distances_in_meters'],
            relative_velocities=supervised_hyperparameter_file['relative_velocities'])

        try:
            with open(unsupervised_hyperparameter_path, 'r+') as unsup_f:
                unsupervised_hyperparameter_file = yaml.full_load(unsup_f)
        except FileNotFoundError:
            unsupervised_hyperparameter_file = EVAL_SIMPLE_MODEL_CONFIG_DICT_UNSUPERVISED

        if not unsupervised_hyperparameter_file['generated_data']:
            logger.error('Trying to load model trained on supervised data!!')
            raise RuntimeError()

        if not unsupervised_hyperparameter_file['use_simple_model']:
            logger.error('Model to load is not an instance of BaselineRNNStackedSimple!')
            raise RuntimeError()

        unsupervised_net = BaselineRNNStackedSimple(
            arch_config=unsupervised_hyperparameter_file['arch_config'],
            batch_size=unsupervised_hyperparameter_file['batch_size'],
            use_batch_norm=unsupervised_hyperparameter_file['use_batch_norm'],
            encoder_lstm_num_layers=unsupervised_hyperparameter_file['num_rnn_layers'],
            decoder_lstm_num_layers=unsupervised_hyperparameter_file['num_rnn_layers'],
            generated_dataset=EVAL_USE_GENERATED,  # unsupervised_hyperparameter_file['generated_data'],
            dropout=unsupervised_hyperparameter_file['dropout'],
            rnn_dropout=unsupervised_hyperparameter_file['rnn_dropout'],
            use_gru=unsupervised_hyperparameter_file['use_gru'],
            learn_hidden_states=unsupervised_hyperparameter_file['learn_hidden_states'],
            feed_model_distances_in_meters=unsupervised_hyperparameter_file['feed_model_distances_in_meters'],
            relative_velocities=unsupervised_hyperparameter_file['relative_velocities'])

        key = 'model_state_dict' if BEST_MODEL else 'last_model_state_dict'

        supervised_net.load_state_dict(torch.load(supervised_checkpoint_path, map_location=DEVICE)[key])
        unsupervised_net.load_state_dict(torch.load(unsupervised_checkpoint_path, map_location=DEVICE)[key])

    else:
        supervised_checkpoint_path = supervised_checkpoint_root_path + 'checkpoints/'
        supervised_checkpoint_file = os.listdir(supervised_checkpoint_path)[-1]
        unsupervised_checkpoint_path = unsupervised_checkpoint_root_path + 'checkpoints/'
        unsupervised_checkpoint_file = os.listdir(unsupervised_checkpoint_path)[-1]

        NET = BaselineRNNStackedSimple if use_simple_model_version else BaselineRNNStacked
        supervised_net = NET.load_from_checkpoint(
            checkpoint_path=supervised_checkpoint_path + supervised_checkpoint_file,
            hparams_file=f'{supervised_checkpoint_root_path}hparams.yaml',
            map_location=None,
            use_batch_norm=use_batch_norm,
            return_pred=True
        )
        unsupervised_net = NET.load_from_checkpoint(
            checkpoint_path=unsupervised_checkpoint_path + unsupervised_checkpoint_file,
            hparams_file=f'{unsupervised_checkpoint_root_path}hparams.yaml',
            map_location=None,
            generated_dataset=False,  # we evaluate on ground-truth trajectories
            use_batch_norm=use_batch_norm,
            return_pred=True
        )
    supervised_net.eval()
    unsupervised_net.eval()
    return supervised_net, unsupervised_net


def evaluate_per_loader(plot, plot_four_way, plot_path, supervised_caller, loader, unsupervised_caller,
                        video_path, split_name, metrics_in_meters=True, use_simple_model_version=False):
    constant_linear_baseline_caller = ConstantLinearBaseline()

    supervised_ade_list, supervised_fde_list = [], []
    unsupervised_ade_list, unsupervised_fde_list = [], []
    constant_linear_baseline_ade_list, constant_linear_baseline_fde_list = [], []

    for idx, data in enumerate(tqdm(loader)):
        if use_simple_model_version and EVAL_FOR_WHOLE_CLASS:
            data, dataset_idx = data
        if EVAL_USE_GENERATED:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, _, _, ratio = \
                data
        else:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = data

        supervised_loss, supervised_ade, supervised_fde, supervised_ratio, supervised_pred_trajectory = \
            supervised_caller(data)
        unsupervised_loss, unsupervised_ade, unsupervised_fde, unsupervised_ratio, unsupervised_pred_trajectory = \
            unsupervised_caller(data)
        constant_linear_baseline_pred_trajectory, constant_linear_baseline_ade, constant_linear_baseline_fde = \
            constant_linear_baseline_caller.eval(obs_trajectory=in_xy, obs_distances=in_uv,
                                                 gt_trajectory=gt_xy, ratio=ratio)

        # if metrics_in_meters:  # Added in model itself
        #     supervised_ade *= supervised_ratio
        #     supervised_fde *= supervised_ratio
        #     unsupervised_ade *= unsupervised_ratio
        #     unsupervised_fde *= unsupervised_ratio

        if plot:
            plot_frame_number = in_frame_numbers.squeeze()[0].item()
            plot_track_id = in_track_ids.squeeze()[0].item()
            all_frame_numbers = torch.cat((in_frame_numbers.squeeze(), gt_frame_numbers.squeeze())).tolist()

            obs_trajectory = in_xy.squeeze().numpy()
            gt_trajectory = gt_xy.squeeze().numpy()

            supervised_pred_trajectory = supervised_pred_trajectory.squeeze().numpy() \
                if not use_simple_model_version else supervised_pred_trajectory.squeeze()
            unsupervised_pred_trajectory = unsupervised_pred_trajectory.squeeze().numpy() \
                if not use_simple_model_version else unsupervised_pred_trajectory.squeeze()

            if use_simple_model_version and EVAL_FOR_WHOLE_CLASS:
                video_dataset = loader.dataset.datasets[dataset_idx.item()]
                video_path = f'{BASE_PATH}videos/{video_dataset.video_class.value}/' \
                             f'video{video_dataset.video_number}/video.mov'

            if plot_four_way:
                plot_and_compare_trajectory_four_way(
                    frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
                    supervised_obs_trajectory=obs_trajectory,
                    supervised_gt_trajectory=gt_trajectory,
                    supervised_pred_trajectory=supervised_pred_trajectory,
                    unsupervised_obs_trajectory=obs_trajectory,
                    unsupervised_gt_trajectory=gt_trajectory,
                    unsupervised_pred_trajectory=unsupervised_pred_trajectory,
                    frame_number=plot_frame_number,
                    track_id=plot_track_id,
                    additional_text=f'Frame Numbers: {all_frame_numbers}'
                                    f'\nGround Truth -> ADE: {supervised_ade} | FDE: {supervised_fde}'
                                    f'\nUnsupervised -> ADE: {unsupervised_ade} | FDE: {unsupervised_fde}',
                    save_path=f'{plot_path}/{split_name}/model/'
                )
                plot_trajectories(obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
                                  pred_trajectory=constant_linear_baseline_pred_trajectory.squeeze(),
                                  frame_number=plot_frame_number, track_id=plot_track_id,
                                  additional_text=f'Frame Numbers: {all_frame_numbers}'
                                                  f'\nGround Truth -> ADE: {constant_linear_baseline_ade.item()} | '
                                                  f'FDE: {constant_linear_baseline_fde.item()}',
                                  save_path=f'{plot_path}/{split_name}/constant_linear_baseline/'
                                  )
            else:
                plot_and_compare_trajectory_alongside_frame(
                    frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
                    supervised_obs_trajectory=obs_trajectory,
                    supervised_gt_trajectory=gt_trajectory,
                    supervised_pred_trajectory=supervised_pred_trajectory,
                    unsupervised_obs_trajectory=obs_trajectory,
                    unsupervised_gt_trajectory=gt_trajectory,
                    unsupervised_pred_trajectory=unsupervised_pred_trajectory,
                    frame_number=plot_frame_number,
                    track_id=plot_track_id,
                    additional_text=f'Frame Numbers: {all_frame_numbers}'
                                    f'\nGround Truth -> ADE: {supervised_ade} | FDE: {supervised_fde}'
                                    f'\nUnsupervised -> ADE: {unsupervised_ade} | FDE: {unsupervised_fde}',
                    save_path=f'{plot_path}/{split_name}/model/',
                    include_frame=True
                )

        supervised_ade_list.append(supervised_ade)
        supervised_fde_list.append(supervised_fde)
        unsupervised_ade_list.append(unsupervised_ade)
        unsupervised_fde_list.append(unsupervised_fde)
        constant_linear_baseline_ade_list.append(constant_linear_baseline_ade.mean().item())
        constant_linear_baseline_fde_list.append(constant_linear_baseline_fde.mean().item())

    return supervised_ade_list, supervised_fde_list, unsupervised_ade_list, unsupervised_fde_list, \
           constant_linear_baseline_ade_list, constant_linear_baseline_fde_list


def get_metrics(supervised_ade_list, supervised_fde_list, unsupervised_ade_list, unsupervised_fde_list):
    return np.array(supervised_ade_list).mean(), np.array(supervised_fde_list).mean(), \
           np.array(unsupervised_ade_list).mean(), np.array(unsupervised_fde_list).mean()


@torch.no_grad()
def eval_models(supervised_checkpoint_root_path: str, unsupervised_checkpoint_root_path: str, train_loader: DataLoader,
                val_loader: DataLoader, test_loader: DataLoader, social_lstm: bool = True, plot: bool = False,
                use_batch_norm: bool = False, video_path: str = None, plot_path: Optional[str] = None,
                plot_four_way: bool = False, supervised_pass_final_pos: bool = False, use_simple_model_version=False,
                unsupervised_pass_final_pos: bool = False, metrics_in_meters: bool = True):
    supervised_net, unsupervised_net = get_models(social_lstm, supervised_checkpoint_root_path,
                                                  unsupervised_checkpoint_root_path, use_batch_norm,
                                                  supervised_pass_final_pos=supervised_pass_final_pos,
                                                  unsupervised_pass_final_pos=unsupervised_pass_final_pos,
                                                  use_simple_model_version=use_simple_model_version)

    supervised_caller = supervised_net.one_step if social_lstm else supervised_net
    unsupervised_caller = unsupervised_net.one_step if social_lstm else unsupervised_net

    logger.info('Evaluating for Test Set')
    test_supervised_ade_list, test_supervised_fde_list, test_unsupervised_ade_list, test_unsupervised_fde_list, \
    test_constant_linear_baseline_ade_list, test_constant_linear_baseline_fde_list = \
        evaluate_per_loader(
            plot, plot_four_way, plot_path, supervised_caller, test_loader, unsupervised_caller, video_path,
            split_name=NetworkMode.TEST.name, metrics_in_meters=metrics_in_meters,
            use_simple_model_version=use_simple_model_version)

    logger.info('Evaluating for Validation Set')
    val_supervised_ade_list, val_supervised_fde_list, val_unsupervised_ade_list, val_unsupervised_fde_list, \
    val_constant_linear_baseline_ade_list, val_constant_linear_baseline_fde_list = \
        evaluate_per_loader(
            plot, plot_four_way, plot_path, supervised_caller, val_loader, unsupervised_caller, video_path,
            split_name=NetworkMode.VALIDATION.name, metrics_in_meters=metrics_in_meters,
            use_simple_model_version=use_simple_model_version)

    logger.info('Evaluating for Train Set')
    train_supervised_ade_list, train_supervised_fde_list, train_unsupervised_ade_list, train_unsupervised_fde_list, \
    train_constant_linear_baseline_ade_list, train_constant_linear_baseline_fde_list = \
        evaluate_per_loader(
            plot, plot_four_way, plot_path, supervised_caller, train_loader, unsupervised_caller, video_path,
            split_name=NetworkMode.TRAIN.name, metrics_in_meters=metrics_in_meters,
            use_simple_model_version=use_simple_model_version)

    train_supervised_ade, train_supervised_fde, train_unsupervised_ade, train_unsupervised_fde = get_metrics(
        train_supervised_ade_list, train_supervised_fde_list, train_unsupervised_ade_list, train_unsupervised_fde_list
    )

    val_supervised_ade, val_supervised_fde, val_unsupervised_ade, val_unsupervised_fde = get_metrics(
        val_supervised_ade_list, val_supervised_fde_list, val_unsupervised_ade_list, val_unsupervised_fde_list
    )

    test_supervised_ade, test_supervised_fde, test_unsupervised_ade, test_unsupervised_fde = get_metrics(
        test_supervised_ade_list, test_supervised_fde_list, test_unsupervised_ade_list, test_unsupervised_fde_list
    )

    test_constant_linear_baseline_ade, test_constant_linear_baseline_fde = \
        np.array(test_constant_linear_baseline_ade_list).mean(), np.array(test_constant_linear_baseline_fde_list).mean()

    val_constant_linear_baseline_ade, val_constant_linear_baseline_fde = \
        np.array(val_constant_linear_baseline_ade_list).mean(), np.array(val_constant_linear_baseline_fde_list).mean()

    train_constant_linear_baseline_ade, train_constant_linear_baseline_fde = \
        np.array(train_constant_linear_baseline_ade_list).mean(), \
        np.array(train_constant_linear_baseline_fde_list).mean()

    eval_results = {
        'supervised_checkpoint_root_path': supervised_checkpoint_root_path,
        'unsupervised_checkpoint_root_path': unsupervised_checkpoint_root_path,
        'social_lstm': social_lstm,
        'use_batch_norm': use_batch_norm,
        'use_simple_model_version': use_simple_model_version,
        'EVAL_TRAIN_CLASS': EVAL_TRAIN_CLASS,
        'EVAL_VAL_CLASS': EVAL_VAL_CLASS,
        'EVAL_TEST_CLASS': EVAL_TEST_CLASS,
        'EVAL_TRAIN_VIDEO_NUMBER': EVAL_TRAIN_VIDEO_NUMBER,
        'EVAL_VAL_VIDEO_NUMBER': EVAL_VAL_VIDEO_NUMBER,
        'EVAL_TEST_VIDEO_NUMBER': EVAL_TEST_VIDEO_NUMBER,
        'EVAL_TRAIN_VIDEOS_TO_SKIP': EVAL_TRAIN_VIDEOS_TO_SKIP,
        'EVAL_VAL_VIDEOS_TO_SKIP': EVAL_VAL_VIDEOS_TO_SKIP,
        'EVAL_TEST_VIDEOS_TO_SKIP': EVAL_TEST_VIDEOS_TO_SKIP,
        'train':
            {'ade': {'supervised': train_supervised_ade.item(), 'unsupervised': train_unsupervised_ade.item(),
                     'linear': train_constant_linear_baseline_ade.item()},
             'fde': {'supervised': train_supervised_fde.item(), 'unsupervised': train_unsupervised_fde.item(),
                     'linear': train_constant_linear_baseline_fde.item()},
             'num_trajectories': len(train_loader.dataset)},
        'val':
            {'ade': {'supervised': val_supervised_ade.item(), 'unsupervised': val_unsupervised_ade.item(),
                     'linear': val_constant_linear_baseline_ade.item()},
             'fde': {'supervised': val_supervised_fde.item(), 'unsupervised': val_unsupervised_fde.item(),
                     'linear': val_constant_linear_baseline_fde.item()},
             'num_trajectories': len(val_loader.dataset)},
        'test':
            {'ade': {'supervised': test_supervised_ade.item(), 'unsupervised': test_unsupervised_ade.item(),
                     'linear': test_constant_linear_baseline_ade.item()},
             'fde': {'supervised': test_supervised_fde.item(), 'unsupervised': test_unsupervised_fde.item(),
                     'linear': test_constant_linear_baseline_fde.item()},
             'num_trajectories': len(test_loader.dataset)}}

    results_dump_path = f'{plot_path}/eval_results{"_generated" if EVAL_USE_GENERATED else ""}.yaml'
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    with open(results_dump_path, 'w+') as f:
        yaml.dump(eval_results, f)

    logger.info('Train Set')
    logger.info(f'ADE - GT: {train_supervised_ade} | Unsupervised: {train_unsupervised_ade} | '
                f'Linear: {train_constant_linear_baseline_ade}')
    logger.info(f'FDE - GT: {train_supervised_fde} | Unsupervised: {train_unsupervised_fde} | '
                f'Linear: {train_constant_linear_baseline_fde}')

    logger.info('Validation Set')
    logger.info(f'ADE - GT: {val_supervised_ade} | Unsupervised: {val_unsupervised_ade} | '
                f'Linear: {val_constant_linear_baseline_ade}')
    logger.info(f'FDE - GT: {val_supervised_fde} | Unsupervised: {val_unsupervised_fde} | '
                f'Linear: {val_constant_linear_baseline_fde}')

    logger.info('Test Set')
    logger.info(f'ADE - GT: {test_supervised_ade} | Unsupervised: {test_unsupervised_ade} | '
                f'Linear: {test_constant_linear_baseline_ade}')
    logger.info(f'FDE - GT: {test_supervised_fde} | Unsupervised: {test_unsupervised_fde} | '
                f'Linear: {test_constant_linear_baseline_fde}')


def get_model(social_lstm, model_checkpoint_root_path, use_batch_norm,
              model_pass_final_pos=True, use_simple_model_version=False):
    if social_lstm:
        model_checkpoint_path = model_checkpoint_root_path + 'checkpoints/'
        model_checkpoint_file = os.listdir(model_checkpoint_path)[-1]

        model_net = BaselineLSTM.load_from_checkpoint(
            checkpoint_path=model_checkpoint_path + model_checkpoint_file,
            hparams_file=f'{model_checkpoint_root_path}hparams.yaml',
            map_location=None,
            args=social_lstm_parser(pass_final_pos=model_pass_final_pos),
            use_batch_norm=use_batch_norm
        )
    elif use_simple_model_version:
        # root path is the full path
        model_checkpoint_path = model_checkpoint_root_path

        model_hyperparameter_path = model_checkpoint_root_path[:-15] + 'hparams.yaml'

        try:
            with open(model_hyperparameter_path, 'r+') as sup_f:
                model_hyperparameter_file = yaml.full_load(sup_f)
        except FileNotFoundError:
            model_hyperparameter_file = EVAL_SIMPLE_MODEL_CONFIG_DICT_GT

        # if model_hyperparameter_file['generated_data']:  # fixme- remove comment
        #     logger.error('Trying to load model trained on unmodel data!!')
        #     raise RuntimeError()

        if not model_hyperparameter_file['use_simple_model']:
            logger.error('Model to load is not an instance of BaselineRNNStackedSimple!')
            raise RuntimeError()

        model_net = BaselineRNNStackedSimple(
            arch_config=model_hyperparameter_file['arch_config'],
            batch_size=model_hyperparameter_file['batch_size'],
            use_batch_norm=model_hyperparameter_file['use_batch_norm'],
            encoder_lstm_num_layers=model_hyperparameter_file['num_rnn_layers'],
            decoder_lstm_num_layers=model_hyperparameter_file['num_rnn_layers'],
            generated_dataset=EVAL_USE_GENERATED,  # model_hyperparameter_file['generated_data'],
            dropout=model_hyperparameter_file['dropout'],
            rnn_dropout=model_hyperparameter_file['rnn_dropout'],
            use_gru=model_hyperparameter_file['use_gru'],
            learn_hidden_states=model_hyperparameter_file['learn_hidden_states'],
            feed_model_distances_in_meters=model_hyperparameter_file['feed_model_distances_in_meters'],
            relative_velocities=model_hyperparameter_file['relative_velocities'])

        key = 'model_state_dict' if BEST_MODEL else 'last_model_state_dict'

        model_net.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE)[key])
    else:
        model_checkpoint_path = model_checkpoint_root_path + 'checkpoints/'
        model_checkpoint_file = os.listdir(model_checkpoint_path)[-1]

        NET = BaselineRNNStackedSimple if use_simple_model_version else BaselineRNNStacked
        model_net = NET.load_from_checkpoint(
            checkpoint_path=model_checkpoint_path + model_checkpoint_file,
            hparams_file=f'{model_checkpoint_root_path}hparams.yaml',
            map_location=None,
            use_batch_norm=use_batch_norm,
            return_pred=True
        )

    model_net.eval()
    return model_net


def evaluate_per_loader_single_model(plot, plot_path, model_caller, loader, video_path, split_name,
                                     metrics_in_meters=True, use_simple_model_version=False):
    constant_linear_baseline_caller = ConstantLinearBaseline()

    model_ade_list, model_fde_list = [], []
    constant_linear_baseline_ade_list, constant_linear_baseline_fde_list = [], []

    stat_dict = []

    for idx, data in enumerate(tqdm(loader)):
        if use_simple_model_version and EVAL_FOR_WHOLE_CLASS:
            data, dataset_idx = data
        if EVAL_USE_GENERATED:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, _, _, ratio = \
                data
        else:
            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = data

        model_loss, model_ade, model_fde, model_ratio, model_pred_trajectory = \
            model_caller(data)
        constant_linear_baseline_pred_trajectory, constant_linear_baseline_ade, constant_linear_baseline_fde = \
            constant_linear_baseline_caller.eval(obs_trajectory=in_xy, obs_distances=in_uv,
                                                 gt_trajectory=gt_xy, ratio=ratio)

        if EVAL_EXTRACT_STATS:
            obs_trajectory_length, obs_trajectory_length_summed = get_trajectory_length(in_xy)
            gt_trajectory_length, gt_trajectory_length_summed = get_trajectory_length(gt_xy)
            model_pred_trajectory_length, model_pred_trajectory_length_summed = get_trajectory_length(
                model_pred_trajectory.reshape(*gt_xy.shape))
            linear_trajectory_length, linear_trajectory_length_summed = get_trajectory_length(
                constant_linear_baseline_pred_trajectory)

            trajectory_length_stacked = np.stack((np.abs(obs_trajectory_length_summed.sum(-1)),
                                                  np.abs(gt_trajectory_length_summed.sum(-1)),
                                                  np.abs(model_pred_trajectory_length_summed.sum(-1)),
                                                  np.abs(linear_trajectory_length_summed.sum(-1))),
                                                 axis=-1)
            stat_dict.append(trajectory_length_stacked)

        # plot always
        if BATCH_PLOT_MODE:
            im_idx = np.random.choice(EVAL_BATCH_SIZE, 1).item()
            plot_frame_number = in_frame_numbers.squeeze()[im_idx][0].item()
            plot_track_id = in_track_ids.squeeze()[im_idx][0].item()
            all_frame_numbers = torch.cat((in_frame_numbers.squeeze()[im_idx], gt_frame_numbers.squeeze()[im_idx])).tolist()

            obs_trajectory = in_xy.squeeze()[im_idx].numpy()
            gt_trajectory = gt_xy.squeeze()[im_idx].numpy()

            # model_pred_trajectory = model_pred_trajectory.reshape(*gt_xy.shape).squeeze()[im_idx].numpy() \
            #     if not use_simple_model_version else model_pred_trajectory.reshape(*gt_xy.shape).squeeze()[im_idx]
            model_pred_trajectory_instance = model_pred_trajectory.squeeze()[:, im_idx, ...].numpy() \
                if not use_simple_model_version else model_pred_trajectory.squeeze()[:, im_idx, ...]

            if use_simple_model_version and EVAL_FOR_WHOLE_CLASS:
                video_dataset = loader.dataset.datasets[dataset_idx[im_idx].item()]
                video_path = f'{BASE_PATH}videos/{video_dataset.video_class.value}/' \
                             f'video{video_dataset.video_number}/video.mov'

            # plot_trajectory_alongside_frame(
            #     frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
            #     obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
            #     pred_trajectory=constant_linear_baseline_pred_trajectory.squeeze()[im_idx],
            #     frame_number=plot_frame_number, track_id=plot_track_id,
            #     additional_text=f'Frame Numbers: {all_frame_numbers}'
            #                     f'\nGround Truth -> ADE: {constant_linear_baseline_ade[im_idx].item()} | '
            #                     f'FDE: {constant_linear_baseline_fde[im_idx].item()}',
            #     save_path=f'{plot_path}/{split_name}/constant_linear_baseline/'
            # )
            # plot_trajectory_alongside_frame(
            #     frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
            #     obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
            #     pred_trajectory=model_pred_trajectory,
            #     frame_number=plot_frame_number, track_id=plot_track_id,
            #     additional_text=f'Frame Numbers: {all_frame_numbers}'
            #                     f'\nGround Truth -> ADE: {model_ade} | '
            #                     f'FDE: {model_fde}',
            #     save_path=f'{plot_path}/{split_name}/model/'
            # )

            plot_and_compare_trajectory_four_way(
                frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
                supervised_obs_trajectory=obs_trajectory,
                supervised_gt_trajectory=gt_trajectory,
                supervised_pred_trajectory=model_pred_trajectory_instance,
                unsupervised_obs_trajectory=obs_trajectory,
                unsupervised_gt_trajectory=gt_trajectory,
                unsupervised_pred_trajectory=constant_linear_baseline_pred_trajectory.squeeze()[im_idx],
                frame_number=plot_frame_number,
                track_id=plot_track_id,
                additional_text=
                f'Frame Numbers: {all_frame_numbers}'
                f'\nModel -> ADE: {compute_ade(model_pred_trajectory_instance, gt_trajectory) * ratio[0].item()} | '
                f'FDE: {compute_fde(model_pred_trajectory_instance, gt_trajectory) * ratio[0].item()}'
                f'\nLinear -> ADE: '
                f'{compute_ade(constant_linear_baseline_pred_trajectory.squeeze()[im_idx], gt_trajectory) * ratio[0].item()} |'
                f' FDE: '
                f'{compute_fde(constant_linear_baseline_pred_trajectory.squeeze()[im_idx], gt_trajectory) * ratio[0].item()}',
                save_path=f'{plot_path}/{split_name}/model4way/',
                with_linear=True
            )

        if plot:
            plot_frame_number = in_frame_numbers.squeeze()[0].item()
            plot_track_id = in_track_ids.squeeze()[0].item()
            all_frame_numbers = torch.cat((in_frame_numbers.squeeze(), gt_frame_numbers.squeeze())).tolist()

            obs_trajectory = in_xy.squeeze().numpy()
            gt_trajectory = gt_xy.squeeze().numpy()

            model_pred_trajectory = model_pred_trajectory.squeeze().numpy() \
                if not use_simple_model_version else model_pred_trajectory.squeeze()

            if use_simple_model_version and EVAL_FOR_WHOLE_CLASS:
                video_dataset = loader.dataset.datasets[dataset_idx.item()]
                video_path = f'{BASE_PATH}videos/{video_dataset.video_class.value}/' \
                             f'video{video_dataset.video_number}/video.mov'

            plot_trajectory_alongside_frame(
                frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
                obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
                pred_trajectory=constant_linear_baseline_pred_trajectory.squeeze(),
                frame_number=plot_frame_number, track_id=plot_track_id,
                additional_text=f'Frame Numbers: {all_frame_numbers}'
                                f'\nGround Truth -> ADE: {constant_linear_baseline_ade.item()} | '
                                f'FDE: {constant_linear_baseline_fde.item()}',
                save_path=f'{plot_path}/{split_name}/constant_linear_baseline/'
            )
            plot_trajectory_alongside_frame(
                frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
                obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
                pred_trajectory=model_pred_trajectory,
                frame_number=plot_frame_number, track_id=plot_track_id,
                additional_text=f'Frame Numbers: {all_frame_numbers}'
                                f'\nGround Truth -> ADE: {model_ade} | '
                                f'FDE: {model_fde}',
                save_path=f'{plot_path}/{split_name}/model/'
            )

        model_ade_list.append(model_ade)
        model_fde_list.append(model_fde)
        constant_linear_baseline_ade_list.append(constant_linear_baseline_ade.mean().item())
        constant_linear_baseline_fde_list.append(constant_linear_baseline_fde.mean().item())

    stat_dict = np.concatenate(stat_dict, axis=0)
    stat_dict = pd.DataFrame(data=stat_dict, columns=['Observed', 'GT', 'Model', 'Linear'])
    return model_ade_list, model_fde_list, \
           constant_linear_baseline_ade_list, constant_linear_baseline_fde_list


@torch.no_grad()
def eval_model(model_checkpoint_root_path: str, train_loader: DataLoader, val_loader: DataLoader,
               test_loader: DataLoader, social_lstm: bool = True, plot: bool = False,
               use_batch_norm: bool = False, video_path: str = None, plot_path: Optional[str] = None,
               model_pass_final_pos: bool = False, use_simple_model_version=False,
               metrics_in_meters: bool = True):
    model_net = get_model(social_lstm, model_checkpoint_root_path, use_batch_norm,
                          model_pass_final_pos=model_pass_final_pos,
                          use_simple_model_version=use_simple_model_version)

    model_caller = model_net.one_step if social_lstm else model_net

    logger.info('Evaluating for Test Set')
    test_model_ade_list, test_model_fde_list, \
    test_constant_linear_baseline_ade_list, test_constant_linear_baseline_fde_list = \
        evaluate_per_loader_single_model(
            plot, plot_path, model_caller, test_loader, video_path,
            split_name=NetworkMode.TEST.name, metrics_in_meters=metrics_in_meters,
            use_simple_model_version=use_simple_model_version)

    logger.info('Evaluating for Validation Set')
    val_model_ade_list, val_model_fde_list, \
    val_constant_linear_baseline_ade_list, val_constant_linear_baseline_fde_list = \
        evaluate_per_loader_single_model(
            plot, plot_path, model_caller, val_loader, video_path,
            split_name=NetworkMode.VALIDATION.name, metrics_in_meters=metrics_in_meters,
            use_simple_model_version=use_simple_model_version)

    logger.info('Evaluating for Train Set')
    train_model_ade_list, train_model_fde_list, \
    train_constant_linear_baseline_ade_list, train_constant_linear_baseline_fde_list = \
        evaluate_per_loader_single_model(
            plot, plot_path, model_caller, train_loader, video_path,
            split_name=NetworkMode.TRAIN.name, metrics_in_meters=metrics_in_meters,
            use_simple_model_version=use_simple_model_version)

    train_model_ade, train_model_fde = \
        np.array(train_model_ade_list).mean(), np.array(train_model_fde_list).mean()

    val_model_ade, val_model_fde = \
        np.array(val_model_ade_list).mean(), np.array(val_model_fde_list).mean()

    test_model_ade, test_model_fde = \
        np.array(test_model_ade_list).mean(), np.array(test_model_fde_list).mean()

    test_constant_linear_baseline_ade, test_constant_linear_baseline_fde = \
        np.array(test_constant_linear_baseline_ade_list).mean(), np.array(test_constant_linear_baseline_fde_list).mean()

    val_constant_linear_baseline_ade, val_constant_linear_baseline_fde = \
        np.array(val_constant_linear_baseline_ade_list).mean(), np.array(val_constant_linear_baseline_fde_list).mean()

    train_constant_linear_baseline_ade, train_constant_linear_baseline_fde = \
        np.array(train_constant_linear_baseline_ade_list).mean(), \
        np.array(train_constant_linear_baseline_fde_list).mean()

    eval_results = {
        'model_checkpoint_root_path': model_checkpoint_root_path,
        'social_lstm': social_lstm,
        'use_batch_norm': use_batch_norm,
        'use_simple_model_version': use_simple_model_version,
        'EVAL_TRAIN_CLASS': EVAL_TRAIN_CLASS,
        'EVAL_VAL_CLASS': EVAL_VAL_CLASS,
        'EVAL_TEST_CLASS': EVAL_TEST_CLASS,
        'EVAL_TRAIN_VIDEO_NUMBER': EVAL_TRAIN_VIDEO_NUMBER,
        'EVAL_VAL_VIDEO_NUMBER': EVAL_VAL_VIDEO_NUMBER,
        'EVAL_TEST_VIDEO_NUMBER': EVAL_TEST_VIDEO_NUMBER,
        'EVAL_TRAIN_VIDEOS_TO_SKIP': EVAL_TRAIN_VIDEOS_TO_SKIP,
        'EVAL_VAL_VIDEOS_TO_SKIP': EVAL_VAL_VIDEOS_TO_SKIP,
        'EVAL_TEST_VIDEOS_TO_SKIP': EVAL_TEST_VIDEOS_TO_SKIP,
        'train':
            {'ade': {'model': train_model_ade.item(), 'linear': train_constant_linear_baseline_ade.item()},
             'fde': {'model': train_model_fde.item(), 'linear': train_constant_linear_baseline_fde.item()},
             'num_trajectories': len(train_loader.dataset)},
        'val':
            {'ade': {'model': val_model_ade.item(), 'linear': val_constant_linear_baseline_ade.item()},
             'fde': {'model': val_model_fde.item(), 'linear': val_constant_linear_baseline_fde.item()},
             'num_trajectories': len(val_loader.dataset)},
        'test':
            {'ade': {'model': test_model_ade.item(), 'linear': test_constant_linear_baseline_ade.item()},
             'fde': {'model': test_model_fde.item(), 'linear': test_constant_linear_baseline_fde.item()},
             'num_trajectories': len(test_loader.dataset)}}

    results_dump_path = f'{plot_path}/eval_results{"_generated" if EVAL_USE_GENERATED else ""}.yaml'
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    with open(results_dump_path, 'w+') as f:
        yaml.dump(eval_results, f)

    logger.info('Train Set')
    logger.info(f'ADE - GT: {train_model_ade} | '
                f'Linear: {train_constant_linear_baseline_ade}')
    logger.info(f'FDE - GT: {train_model_fde} | '
                f'Linear: {train_constant_linear_baseline_fde}')

    logger.info('Validation Set')
    logger.info(f'ADE - GT: {val_model_ade} | '
                f'Linear: {val_constant_linear_baseline_ade}')
    logger.info(f'FDE - GT: {val_model_fde} | '
                f'Linear: {val_constant_linear_baseline_fde}')

    logger.info('Test Set')
    logger.info(f'ADE - GT: {test_model_ade} | '
                f'Linear: {test_constant_linear_baseline_ade}')
    logger.info(f'FDE - GT: {test_model_fde} | '
                f'Linear: {test_constant_linear_baseline_fde}')


def get_eval_loaders():
    if EVAL_FOR_WHOLE_CLASS:
        train_set = get_dataset_for_class(video_class=EVAL_TRAIN_CLASS, meta_label=EVAL_TRAIN_META,
                                          mode=NetworkMode.TRAIN, get_generated=EVAL_USE_GENERATED,
                                          videos_to_skip=EVAL_TRAIN_VIDEOS_TO_SKIP,
                                          return_dataset_list=True)
        val_set = get_dataset_for_class(video_class=EVAL_VAL_CLASS, meta_label=EVAL_VAL_META,
                                        mode=NetworkMode.VALIDATION, get_generated=EVAL_USE_GENERATED,
                                        videos_to_skip=EVAL_VAL_VIDEOS_TO_SKIP,
                                        return_dataset_list=True)
        test_set = get_dataset_for_class(video_class=EVAL_TEST_CLASS, meta_label=EVAL_TEST_META,
                                         mode=NetworkMode.TEST, get_generated=EVAL_USE_GENERATED,
                                         videos_to_skip=EVAL_TEST_VIDEOS_TO_SKIP,
                                         return_dataset_list=True)
        train_set = ConcatenateDataset(train_set)
        val_set = ConcatenateDataset(val_set)
        test_set = ConcatenateDataset(test_set)
    else:
        train_set = get_dataset(video_clazz=EVAL_TRAIN_CLASS, video_number=EVAL_TRAIN_VIDEO_NUMBER,
                                mode=NetworkMode.TRAIN, meta_label=EVAL_TRAIN_META, get_generated=EVAL_USE_GENERATED)
        val_set = get_dataset(video_clazz=EVAL_VAL_CLASS, video_number=EVAL_VAL_VIDEO_NUMBER,
                              mode=NetworkMode.VALIDATION, meta_label=EVAL_VAL_META, get_generated=EVAL_USE_GENERATED)
        test_set = get_dataset(video_clazz=EVAL_TEST_CLASS, video_number=EVAL_TEST_VIDEO_NUMBER,
                               mode=NetworkMode.TEST, meta_label=EVAL_TEST_META, get_generated=EVAL_USE_GENERATED)

    train_loader = DataLoader(train_set, batch_size=EVAL_BATCH_SIZE, shuffle=EVAL_SHUFFLE, num_workers=EVAL_WORKERS)
    val_loader = DataLoader(val_set, batch_size=EVAL_BATCH_SIZE, shuffle=EVAL_SHUFFLE, num_workers=EVAL_WORKERS)
    test_loader = DataLoader(test_set, batch_size=EVAL_BATCH_SIZE, shuffle=EVAL_SHUFFLE, num_workers=EVAL_WORKERS)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    if DEBUG_MODE:
        num_workers = 0
        shuffle = True
        use_social_lstm_model = False
        use_simple_v2_model = True

        sdd_video_class = SDDVideoClasses.LITTLE
        sdd_meta_class = SDDVideoDatasets.LITTLE
        network_mode = NetworkMode.TRAIN
        sdd_video_number = 3

        path_to_video = f'{BASE_PATH}videos/{sdd_video_class.value}/video{sdd_video_number}/video.mov'

        version = 9

        plot_save_path = f'{ROOT_PATH}Plots/baseline_v2/nn/v{version}/{sdd_video_class.value}{sdd_video_number}/' \
                         f'eval_plots/{network_mode.value}/'

        checkpoint_root_path = f'../baselinev2/lightning_logs/version_{version}/'
        dataset = get_dataset(video_clazz=sdd_video_class, video_number=sdd_video_number, mode=network_mode,
                              meta_label=sdd_meta_class)
        model = BaselineRNN() if not use_social_lstm_model else BaselineLSTM

        if use_social_lstm_model:
            evaluate_social_lstm_model(
                model=model,
                data_loader=DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle),
                checkpoint_root_path=checkpoint_root_path, video_path=path_to_video,
                plot_path=plot_save_path)
        elif use_simple_v2_model:
            generated_dataset = True
            dataset = get_dataset(video_clazz=sdd_video_class, video_number=sdd_video_number, mode=network_mode,
                                  meta_label=sdd_meta_class, get_generated=generated_dataset)
            # checkpoint_path = f'../baselinev2/runs/Feb25_00-30-05_rishabh-Precision-5540baseline/' \
            #                   f'Feb25_00-30-05_rishabh-Precision-5540baseline_checkpoint.ckpt'
            # checkpoint_path = f'../baselinev2/runs/Maar_overfit_experiments/full_train/' \
            #                   f'element_size_None_random_True_lr_0.001_generated_True/' \
            #                   f'element_size_None_random_True_lr_0.001_generated_True_checkpoint.ckpt'
            # checkpoint_path = f'../baselinev2/runs/Maar_overfit_experiments/full_train/' \
            #                   f'element_size_None_random_True_lr_0.002_generated_True/' \
            #                   f'element_size_None_random_True_lr_0.002_generated_True_checkpoint.ckpt'
            checkpoint_path = f'../baselinev2/runs/Maar_overfit_experiments/full_train/' \
                              f'element_size_None_random_True_lr_0.001_generated_True_learn_hidden_False' \
                              f'_rnn_layers_1_2021-03-04 13:37:06.911715/element_size_None_random_True_lr_0.001_' \
                              f'generated_True_learn_hidden_False_rnn_layers_1_2021-03-04 13:37:06.911715' \
                              f'_checkpoint.ckpt'
            experiment_name = os.path.split(checkpoint_path)[-1][:-5]
            plot_save_path = f'{ROOT_PATH}Plots/baseline_v2/nn/EVAL_CUSTOM/{sdd_video_class.value}{sdd_video_number}/' \
                             f'{network_mode.value}/{experiment_name}/'
            evaluate_simple_v2_model(
                model=None,
                data_loader=DataLoader(dataset, batch_size=32, num_workers=num_workers, shuffle=shuffle,
                                       drop_last=True),
                checkpoint_path=checkpoint_path, video_path=path_to_video,
                plot_path=plot_save_path, generated=generated_dataset)
        else:
            evaluate_model(
                model=model,
                data_loader=DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle),
                checkpoint_root_path=checkpoint_root_path, video_path=path_to_video, plot_path=plot_save_path)
    else:
        train_loader, val_loader, test_loader = get_eval_loaders()

        if EVAL_SINGLE_MODEL:
            eval_model(
                model_checkpoint_root_path=SINGLE_MODEL_CHECKPOINT_PATH,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                social_lstm=EVAL_USE_SOCIAL_LSTM_MODEL,
                plot=PLOT_MODE,
                use_batch_norm=EVAL_USE_BATCH_NORM,
                video_path=EVAL_PATH_TO_VIDEO,
                plot_path=EVAL_PLOT_PATH,
                model_pass_final_pos=False,
                use_simple_model_version=EVAL_USE_SIMPLE_MODEL,
                metrics_in_meters=True
            )
        else:
            eval_models(
                supervised_checkpoint_root_path=GT_CHECKPOINT_ROOT_PATH
                if not EVAL_USE_SIMPLE_MODEL else SIMPLE_GT_CHECKPOINT_PATH,
                unsupervised_checkpoint_root_path=UNSUPERVISED_CHECKPOINT_ROOT_PATH
                if not EVAL_USE_SIMPLE_MODEL else SIMPLE_UNSUPERVISED_CHECKPOINT_PATH,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                social_lstm=EVAL_USE_SOCIAL_LSTM_MODEL,
                plot=PLOT_MODE,
                use_batch_norm=EVAL_USE_BATCH_NORM,
                video_path=EVAL_PATH_TO_VIDEO,
                plot_path=EVAL_PLOT_PATH,
                plot_four_way=True,
                supervised_pass_final_pos=EVAL_USE_FINAL_POSITIONS_SUPERVISED,
                unsupervised_pass_final_pos=EVAL_USE_FINAL_POSITIONS_UNSUPERVISED,
                metrics_in_meters=True,
                use_simple_model_version=EVAL_USE_SIMPLE_MODEL
            )
