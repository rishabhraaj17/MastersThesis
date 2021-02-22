import argparse

import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import BASE_PATH, ROOT_PATH
from baselinev2.constants import NetworkMode
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.models import BaselineRNN, BaselineRNNStacked
from baselinev2.nn.social_lstm.model import BaselineLSTM
from baselinev2.nn.social_lstm.train import bool_flag
from baselinev2.plot_utils import plot_trajectories, plot_trajectory_alongside_frame
from log import initialize_logging, get_logger

matplotlib.style.use('ggplot')

initialize_logging()
logger = get_logger('baselinev2.nn.overfit')


def plot_array(arr, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(arr)
    plt.show()


def reverse_slices(arr):
    arr_cloned = arr.clone()
    arr_cloned[:, 0], arr_cloned[:, 1] = arr[:, 1], arr[:, 0]
    return arr_cloned


def reverse_u_v(batch):
    out_in_xy, out_gt_xy, out_in_uv, out_gt_uv, out_in_track_ids, = [], [], [], [], []
    out_gt_track_ids, out_in_frame_numbers, out_gt_frame_numbers, out_ratio = [], [], [], []
    for data in batch:
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = data
        in_uv = reverse_slices(in_uv)
        gt_uv = reverse_slices(gt_uv)
        
        out_in_xy.append(in_xy)
        out_gt_xy.append(gt_xy)
        out_in_uv.append(in_uv)
        out_gt_uv.append(gt_uv)
        out_in_track_ids.append(in_track_ids)
        out_gt_track_ids.append(gt_track_ids)
        out_in_frame_numbers.append(in_frame_numbers)
        out_gt_frame_numbers.append(gt_frame_numbers)
        out_ratio.append(ratio)
        
    return [torch.stack(out_in_xy), torch.stack(out_gt_xy), torch.stack(out_in_uv), torch.stack(out_gt_uv),
            torch.stack(out_in_track_ids), torch.stack(out_gt_track_ids), torch.stack(out_in_frame_numbers),
            torch.stack(out_gt_frame_numbers), torch.stack(out_ratio)]


def reverse_u_v_generated(batch):
    out_in_xy, out_gt_xy, out_in_uv, out_gt_uv, out_in_track_ids, = [], [], [], [], []
    out_gt_track_ids, out_in_frame_numbers, out_gt_frame_numbers, out_ratio = [], [], [], []
    out_mapped_in_xy, out_mapped_gt_xy = [], []
    for data in batch:
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, \
        mapped_in_xy, mapped_gt_xy, ratio = data
        in_uv = reverse_slices(in_uv)
        gt_uv = reverse_slices(gt_uv)
        
        out_in_xy.append(in_xy)
        out_gt_xy.append(gt_xy)
        out_in_uv.append(in_uv)
        out_gt_uv.append(gt_uv)
        out_in_track_ids.append(in_track_ids)
        out_gt_track_ids.append(gt_track_ids)
        out_in_frame_numbers.append(in_frame_numbers)
        out_gt_frame_numbers.append(gt_frame_numbers)
        out_mapped_in_xy.append(mapped_in_xy)
        out_mapped_gt_xy.append(mapped_gt_xy)
        out_ratio.append(ratio)
        
    return [torch.stack(out_in_xy), torch.stack(out_gt_xy), torch.stack(out_in_uv), torch.stack(out_gt_uv),
            torch.stack(out_in_track_ids), torch.stack(out_gt_track_ids), torch.stack(out_in_frame_numbers),
            torch.stack(out_gt_frame_numbers), torch.stack(out_mapped_in_xy), torch.stack(out_mapped_gt_xy),
            torch.stack(out_ratio)]


def overfit(net, loader, optimizer, num_epochs=5000, batch_mode=False, video_path=None, social_lstm=False):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=500, cooldown=10, verbose=True,
                                                           factor=0.1)
    net.train()
    net.return_pred = True

    network = net.one_step if social_lstm else net

    running_loss, running_ade, running_fde = [], [], []
    with trange(num_epochs) as t:
        for epoch in t:
            for data in loader:
                loss, ade, fde, ratio, pred_trajectory = network(data)
                running_loss.append(loss.item())
                running_ade.append(ade)
                running_fde.append(fde)
                t.set_description('Epoch %i' % epoch)
                t.set_postfix(loss=loss.item(), ade=ade, fde=fde)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step(loss)

                if epoch % 500 == 0:
                    if batch_mode:
                        im_idx = np.random.choice(6, 1).item()
                        plot_trajectory_alongside_frame(
                            frame=extract_frame_from_video(
                                video_path=video_path, frame_number=data[6][im_idx].squeeze()[0].item()),
                            obs_trajectory=data[0][im_idx].squeeze().numpy(),
                            gt_trajectory=data[1][im_idx].squeeze().numpy(),
                            pred_trajectory=pred_trajectory[:, im_idx, ...].squeeze(),
                            frame_number=data[6][im_idx].squeeze()[0].item(),
                            track_id=data[4][im_idx].squeeze()[0].item())
                    else:
                        plot_trajectory_alongside_frame(
                            frame=extract_frame_from_video(
                                video_path=video_path, frame_number=data[6].squeeze()[0].item()),
                            obs_trajectory=data[0].squeeze().numpy(), gt_trajectory=data[1].squeeze().numpy(),
                            pred_trajectory=pred_trajectory.squeeze(), frame_number=data[6].squeeze()[0].item(),
                            track_id=data[4].squeeze()[0].item())

    logger.info(f'Total Loss : {sum(running_loss) / num_epochs}')
    plot_array(running_loss, 'Loss', 'epoch', 'loss')
    plot_array(running_ade, 'ADE', 'epoch', 'ade')
    plot_array(running_fde, 'FDE', 'epoch', 'fde')


def overfit_two_loss(net, loader, optimizer, num_epochs=5000):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=500, cooldown=10, verbose=True)
    net.train()
    net.two_losses = True
    running_dist_loss, running_vel_loss, running_ade, running_fde = [], [], [], []
    with trange(num_epochs) as t:
        for epoch in t:
            for data in loader:
                dist_loss, vel_loss, ade, fde, ratio, pred_trajectory = net(data)

                running_dist_loss.append(dist_loss.item())
                running_vel_loss.append(vel_loss.item())
                running_ade.append(ade)
                running_fde.append(fde)
                t.set_description('Epoch %i' % epoch)
                t.set_postfix(dist_loss=dist_loss.item(), vel_loss=vel_loss.item(), ade=ade, fde=fde)

                loss = dist_loss + vel_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step(loss)

                if epoch % 500 == 0:
                    plot_trajectories(
                        obs_trajectory=data[0].squeeze().numpy(), gt_trajectory=data[1].squeeze().numpy(),
                        pred_trajectory=pred_trajectory.squeeze(), frame_number=data[6].squeeze()[0].item(),
                        track_id=data[4].squeeze()[0].item())

    plot_array(running_dist_loss, 'Distance Loss', 'epoch', 'loss')
    plot_array(running_vel_loss, 'Velocity Loss', 'epoch', 'loss')
    plot_array(running_ade, 'ADE', 'epoch', 'ade')
    plot_array(running_fde, 'FDE', 'epoch', 'fde')


def social_lstm_parser(batch_size=32, learning_rate=0.001, pass_final_pos=False):
    parser = argparse.ArgumentParser("Trajectory Prediction Basics")

    # Configs for Model
    parser.add_argument("--model_name", default="", type=str, help="Define model name for saving")
    parser.add_argument("--model_type", default="lstm", type=str,
                        help="Define type of model. Choose either: linear, lstm or social-lstm")
    parser.add_argument("--save_model", default=False, type=bool_flag, help="Save trained model")
    parser.add_argument("--nl_ADE", default=False, type=bool_flag, help="Use nl_ADE")
    parser.add_argument("--load_model", default=False, type=bool_flag, help="Specify whether to load existing model")
    parser.add_argument("--lstm_pool", default=False, type=bool_flag, help="Specify whether to enable social pooling")
    parser.add_argument("--pooling_type", default="social_pooling", type=str, help="Specify pooling method")
    parser.add_argument("--neighborhood_size", default=10.0, type=float, help="Specify neighborhood size to one side")
    parser.add_argument("--grid_size", default=10, type=int, help="Specify grid size")
    parser.add_argument("--args_set", default="", type=str,
                        help="Specify predefined set of configurations for respective model. "
                             "Choose either: lstm or social-lstm")

    # Configs for data-preparation
    parser.add_argument("--obs_len", default=8, type=int, help="Specify length of observed trajectory")
    parser.add_argument("--pred_len", default=12, type=int, help="Specify length of predicted trajectory")
    parser.add_argument("--data_augmentation", default=False, type=bool_flag,
                        help="Specify whether or not you want to use data augmentation")
    parser.add_argument("--batch_norm", default=False, type=bool_flag, help="Batch Normalization")
    parser.add_argument("--max_num", default=1000000, type=int, help="Specify maximum number of ids")
    parser.add_argument("--skip", default=20, type=int, help="Specify skipping rate")
    parser.add_argument("--PhysAtt", default="", type=str, help="Specify physicalAtt")
    parser.add_argument("--padding", default=False, type=bool_flag, help="Specify if padding should be active")
    parser.add_argument("--final_position", default=pass_final_pos, type=bool_flag,
                        help="Specify whether final positions of pedestrians should be passed to model or not")

    # Configs for training, validation, testing
    parser.add_argument("--batch_size", default=batch_size, type=int, help="Specify batch size")
    parser.add_argument("--wd", default=0.03, type=float, help="Specify weight decay")
    parser.add_argument("--lr", default=learning_rate, type=float, help="Specify learning rate")
    parser.add_argument("--encoder_h_dim", default=64, type=int, help="Specify hidden state dimension h of encoder")
    parser.add_argument("--decoder_h_dim", default=32, type=int, help="Specify hidden state dimension h of decoder")
    parser.add_argument("--emb_dim", default=32, type=int, help="Specify dimension of embedding")
    parser.add_argument("--num_epochs", default=250, type=int, help="Specify number of epochs")
    parser.add_argument("--dropout", default=0.0, type=float, help="Specify dropout rate")
    parser.add_argument("--num_layers", default=1, type=int, help="Specify number of layers of LSTM/Social LSTM Model")
    parser.add_argument("--optim", default="Adam", type=str,
                        help="Specify optimizer. Choose either: adam, rmsprop or sgd")

    # Get arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    single_chunk_fit = False
    generated = True
    num_workers = 12
    shuffle = True
    use_social_lstm_model = True

    do_reverse_slices = False

    sdd_video_class = SDDVideoClasses.LITTLE
    sdd_meta_class = SDDVideoDatasets.LITTLE
    network_mode = NetworkMode.TRAIN
    sdd_video_number = 3

    path_to_video = f'{BASE_PATH}videos/{sdd_video_class.value}/video{sdd_video_number}/video.mov'

    version = 0

    plot_save_path = f'{ROOT_PATH}Plots/baseline_v2/nn/v{version}/{sdd_video_class.value}{sdd_video_number}/' \
                     f'overfit/{network_mode.value}/'

    checkpoint_root_path = f'../baselinev2/lightning_logs/version_{version}/'
    # model = BaselineRNN()
    if use_social_lstm_model:
        model = BaselineLSTM(args=social_lstm_parser(pass_final_pos=False), generated_dataset=generated,
                             use_batch_norm=False)
    else:
        model = BaselineRNNStacked(encoder_lstm_num_layers=1, decoder_lstm_num_layers=1, generated_dataset=generated,
                                   use_batch_norm=True, relative_velocities=False)

    if single_chunk_fit:
        overfit_chunk = 1
        if generated:
            overfit_chunk_path = f"{ROOT_PATH}Datasets/OverfitChunks/generated_overfit{overfit_chunk}.pt"
        else:
            overfit_chunk_path = f"{ROOT_PATH}Datasets/OverfitChunks/overfit{overfit_chunk}.pt"

        overfit_data = torch.load(overfit_chunk_path)

        overfit_dataset = TensorDataset(*overfit_data)
        overfit_dataloader = DataLoader(
            overfit_dataset, 1,
            collate_fn=(reverse_u_v_generated if generated else reverse_u_v) if do_reverse_slices else None)
    else:
        overfit_chunks = [0, 1, 2, 3, 4, 5]
        if generated:
            overfit_data_list = [torch.load(f"{ROOT_PATH}Datasets/OverfitChunks/generated_overfit{o}.pt")
                                 for o in overfit_chunks]
        else:
            overfit_data_list = [torch.load(f"{ROOT_PATH}Datasets/OverfitChunks/overfit{o}.pt") for o in overfit_chunks]
        overfit_data = None
        for o_data in overfit_data_list:
            if overfit_data is None:
                overfit_data = o_data
            else:
                for idx in range(len(overfit_data)):
                    overfit_data[idx] = torch.cat((overfit_data[idx], o_data[idx]))

        overfit_dataset = TensorDataset(*overfit_data)
        overfit_dataloader = DataLoader(
            overfit_dataset, len(overfit_chunks),
            collate_fn=(reverse_u_v_generated if generated else reverse_u_v) if do_reverse_slices else None)

    # lr = 5e-1  # single
    # lr = 1e-2  # batch - good one
    # lr = 7e-3
    # lr = 5e-3  # - best
    lr = 1e-3  # 2e-3
    optim = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    # optim = torch.optim.SGD(model.parameters(), lr=lr)

    overfit(net=model, loader=overfit_dataloader, optimizer=optim, num_epochs=20000, batch_mode=not single_chunk_fit,
            video_path=path_to_video, social_lstm=use_social_lstm_model)
    # overfit_two_loss(net=model, loader=overfit_dataloader, optimizer=optim, num_epochs=5000)

    print()
