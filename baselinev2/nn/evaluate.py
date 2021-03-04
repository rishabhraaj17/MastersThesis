import os
from typing import Optional

import matplotlib
import torch
import numpy as np
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
    EVAL_USE_FINAL_POSITIONS_SUPERVISED, EVAL_USE_FINAL_POSITIONS_UNSUPERVISED, EVAL_USE_SIMPLE_MODEL
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import get_dataset
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.models import BaselineRNN, BaselineRNNStacked, BaselineRNNStackedSimple
from baselinev2.overfit_config import LINEAR_CFG
from baselinev2.utils import social_lstm_parser
from baselinev2.nn.social_lstm.model import BaselineLSTM
from baselinev2.plot_utils import plot_trajectory_alongside_frame, plot_and_compare_trajectory_four_way, \
    plot_and_compare_trajectory_alongside_frame
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
            all_frame_numbers = torch.cat((in_frame_numbers.squeeze()[im_idx], gt_frame_numbers.squeeze()[im_idx])).tolist()

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
    supervised_checkpoint_path = supervised_checkpoint_root_path + 'checkpoints/'
    supervised_checkpoint_file = os.listdir(supervised_checkpoint_path)[-1]
    unsupervised_checkpoint_path = unsupervised_checkpoint_root_path + 'checkpoints/'
    unsupervised_checkpoint_file = os.listdir(unsupervised_checkpoint_path)[-1]
    if social_lstm:
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
    else:
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
                        video_path, split_name, metrics_in_meters=True):
    supervised_ade_list, supervised_fde_list = [], []
    unsupervised_ade_list, unsupervised_fde_list = [], []
    for idx, data in enumerate(tqdm(loader)):
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = data

        supervised_loss, supervised_ade, supervised_fde, supervised_ratio, supervised_pred_trajectory = \
            supervised_caller(data)
        unsupervised_loss, unsupervised_ade, unsupervised_fde, unsupervised_ratio, unsupervised_pred_trajectory = \
            unsupervised_caller(data)

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

            supervised_pred_trajectory = supervised_pred_trajectory.squeeze().numpy()
            unsupervised_pred_trajectory = unsupervised_pred_trajectory.squeeze().numpy()

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
                    save_path=f'{plot_path}{split_name}/'
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
                    save_path=f'{plot_path}{split_name}/',
                    include_frame=True
                )

        supervised_ade_list.append(supervised_ade)
        supervised_fde_list.append(supervised_fde)
        unsupervised_ade_list.append(unsupervised_ade)
        unsupervised_fde_list.append(unsupervised_fde)

    return supervised_ade_list, supervised_fde_list, unsupervised_ade_list, unsupervised_fde_list


def get_metrics(supervised_ade_list, supervised_fde_list, unsupervised_ade_list, unsupervised_fde_list):
    return np.array(supervised_ade_list).mean(), np.array(supervised_fde_list).mean(), \
           np.array(unsupervised_ade_list).mean(), np.array(unsupervised_fde_list).mean()


@torch.no_grad()
def eval_models(supervised_checkpoint_root_path: str, unsupervised_checkpoint_root_path: str, train_loader: DataLoader,
                val_loader: DataLoader, test_loader: DataLoader, social_lstm: bool = True, plot: bool = False,
                use_batch_norm: bool = False, video_path: str = None, plot_path: Optional[str] = None,
                plot_four_way: bool = False, supervised_pass_final_pos: bool = True, use_simple_model_version=False,
                unsupervised_pass_final_pos: bool = True, metrics_in_meters: bool = True):
    supervised_net, unsupervised_net = get_models(social_lstm, supervised_checkpoint_root_path,
                                                  unsupervised_checkpoint_root_path, use_batch_norm,
                                                  supervised_pass_final_pos=supervised_pass_final_pos,
                                                  unsupervised_pass_final_pos=unsupervised_pass_final_pos,
                                                  use_simple_model_version=use_simple_model_version)

    supervised_caller = supervised_net.one_step if social_lstm else supervised_net
    unsupervised_caller = unsupervised_net.one_step if social_lstm else unsupervised_net

    logger.info('Evaluating for Test Set')
    test_supervised_ade_list, test_supervised_fde_list, test_unsupervised_ade_list, test_unsupervised_fde_list = \
        evaluate_per_loader(
            plot, plot_four_way, plot_path, supervised_caller, test_loader, unsupervised_caller, video_path,
            split_name=NetworkMode.TEST.name, metrics_in_meters=metrics_in_meters)

    logger.info('Evaluating for Validation Set')
    val_supervised_ade_list, val_supervised_fde_list, val_unsupervised_ade_list, val_unsupervised_fde_list = \
        evaluate_per_loader(
            plot, plot_four_way, plot_path, supervised_caller, val_loader, unsupervised_caller, video_path,
            split_name=NetworkMode.VALIDATION.name, metrics_in_meters=metrics_in_meters)

    logger.info('Evaluating for Train Set')
    train_supervised_ade_list, train_supervised_fde_list, train_unsupervised_ade_list, train_unsupervised_fde_list = \
        evaluate_per_loader(
            plot, plot_four_way, plot_path, supervised_caller, train_loader, unsupervised_caller, video_path,
            split_name=NetworkMode.TRAIN.name, metrics_in_meters=metrics_in_meters)

    train_supervised_ade, train_supervised_fde, train_unsupervised_ade, train_unsupervised_fde = get_metrics(
        train_supervised_ade_list, train_supervised_fde_list, train_unsupervised_ade_list, train_unsupervised_fde_list
    )

    val_supervised_ade, val_supervised_fde, val_unsupervised_ade, val_unsupervised_fde = get_metrics(
        val_supervised_ade_list, val_supervised_fde_list, val_unsupervised_ade_list, val_unsupervised_fde_list
    )

    test_supervised_ade, test_supervised_fde, test_unsupervised_ade, test_unsupervised_fde = get_metrics(
        test_supervised_ade_list, test_supervised_fde_list, test_unsupervised_ade_list, test_unsupervised_fde_list
    )

    logger.info('Train Set')
    logger.info(f'ADE - GT: {train_supervised_ade} | Unsupervised: {train_unsupervised_ade}')
    logger.info(f'FDE - GT: {train_supervised_fde} | Unsupervised: {train_unsupervised_fde}')

    logger.info('Validation Set')
    logger.info(f'ADE - GT: {val_supervised_ade} | Unsupervised: {val_unsupervised_ade}')
    logger.info(f'FDE - GT: {val_supervised_fde} | Unsupervised: {val_unsupervised_fde}')

    logger.info('Test Set')
    logger.info(f'ADE - GT: {test_supervised_ade} | Unsupervised: {test_unsupervised_ade}')
    logger.info(f'FDE - GT: {test_supervised_fde} | Unsupervised: {test_unsupervised_fde}')


def get_eval_loaders():
    train_set = get_dataset(video_clazz=EVAL_TRAIN_CLASS, video_number=EVAL_TRAIN_VIDEO_NUMBER,
                            mode=NetworkMode.TRAIN, meta_label=EVAL_TRAIN_META, get_generated=False)
    val_set = get_dataset(video_clazz=EVAL_VAL_CLASS, video_number=EVAL_VAL_VIDEO_NUMBER,
                          mode=NetworkMode.VALIDATION, meta_label=EVAL_VAL_META, get_generated=False)
    test_set = get_dataset(video_clazz=EVAL_TEST_CLASS, video_number=EVAL_TEST_VIDEO_NUMBER,
                           mode=NetworkMode.TEST, meta_label=EVAL_TEST_META, get_generated=False)

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

        eval_models(
            supervised_checkpoint_root_path=GT_CHECKPOINT_ROOT_PATH,
            unsupervised_checkpoint_root_path=UNSUPERVISED_CHECKPOINT_ROOT_PATH,
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
