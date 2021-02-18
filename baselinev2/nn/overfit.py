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


def overfit(net, loader, optimizer, num_epochs=5000, batch_mode=False, video_path=None):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=500, cooldown=10, verbose=True)
    net.train()
    net.return_pred = True
    running_loss, running_ade, running_fde = [], [], []
    with trange(num_epochs) as t:
        for epoch in t:
            for data in loader:
                loss, ade, fde, ratio, pred_trajectory = net(data)
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


if __name__ == '__main__':
    single_chunk_fit = True
    generated = True
    num_workers = 12
    shuffle = True

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
    model = BaselineRNNStacked(encoder_lstm_num_layers=4, decoder_lstm_num_layers=5, generated_dataset=generated)

    if single_chunk_fit:
        overfit_chunk = 3
        if generated:
            overfit_chunk_path = f"{ROOT_PATH}Datasets/OverfitChunks/generated_overfit{overfit_chunk}.pt"
        else:
            overfit_chunk_path = f"{ROOT_PATH}Datasets/OverfitChunks/overfit{overfit_chunk}.pt"

        overfit_data = torch.load(overfit_chunk_path)

        overfit_dataset = TensorDataset(*overfit_data)
        overfit_dataloader = DataLoader(overfit_dataset, 1)
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
        overfit_dataloader = DataLoader(overfit_dataset, len(overfit_chunks))

    lr = 5e-3
    optim = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    # optim = torch.optim.SGD(model.parameters(), lr=lr)

    overfit(net=model, loader=overfit_dataloader, optimizer=optim, num_epochs=5000, batch_mode=not single_chunk_fit,
            video_path=path_to_video)
    # overfit_two_loss(net=model, loader=overfit_dataloader, optimizer=optim, num_epochs=5000)

    print()
