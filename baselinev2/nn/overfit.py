from pathlib import Path
from typing import List, Union

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm, trange

from baselinev2.nn.dataset import ConcatenateDataset
from baselinev2.overfit_config import *
from baselinev2.constants import NetworkMode, SDDVideoClassAndNumbers
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.models import BaselineRNNStackedSimple
from baselinev2.nn.social_lstm.model import BaselineLSTM
from baselinev2.nn.train import get_train_validation_dataset_for_class, get_social_model, get_simple_model, get_model, \
    get_train_validation_dataset
from baselinev2.plot_utils import plot_trajectories, plot_trajectory_alongside_frame, plot_trajectory_with_relative_data
from baselinev2.utils import social_lstm_parser
from log import initialize_logging, get_logger

matplotlib.style.use('ggplot')

initialize_logging()
logger = get_logger('baselinev2.nn.overfit')


def plot_array(arr, title, xlabel, ylabel, save_path=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(arr)
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + f"{title}.png")
        plt.close()
    else:
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


def overfit(net, loader, optimizer, num_epochs=5000, batch_mode=False, video_path=None, social_lstm=False,
            img_batch_size=6, save_plot_path=None):
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

                if epoch % 1500 == 0:
                    if batch_mode:
                        im_idx = np.random.choice(img_batch_size, 1).item()
                        plot_trajectory_alongside_frame(
                            frame=extract_frame_from_video(
                                video_path=video_path, frame_number=data[6][im_idx].squeeze()[0].item()),
                            obs_trajectory=data[0][im_idx].squeeze().numpy(),
                            gt_trajectory=data[1][im_idx].squeeze().numpy(),
                            pred_trajectory=pred_trajectory[:, im_idx, ...].squeeze(),
                            frame_number=data[6][im_idx].squeeze()[0].item(),
                            track_id=data[4][im_idx].squeeze()[0].item(), save_path=save_plot_path)
                    else:
                        plot_trajectory_alongside_frame(
                            frame=extract_frame_from_video(
                                video_path=video_path, frame_number=data[6].squeeze()[0].item()),
                            obs_trajectory=data[0].squeeze().numpy(), gt_trajectory=data[1].squeeze().numpy(),
                            pred_trajectory=pred_trajectory.squeeze(), frame_number=data[6].squeeze()[0].item(),
                            track_id=data[4].squeeze()[0].item(), save_path=save_plot_path)
                if epoch == list(range(num_epochs))[-1]:
                    for im_idx in range(img_batch_size):
                        plot_trajectory_alongside_frame(
                            frame=extract_frame_from_video(
                                video_path=video_path, frame_number=data[6][im_idx].squeeze()[0].item()),
                            obs_trajectory=data[0][im_idx].squeeze().numpy(),
                            gt_trajectory=data[1][im_idx].squeeze().numpy(),
                            pred_trajectory=pred_trajectory[:, im_idx, ...].squeeze(),
                            frame_number=data[6][im_idx].squeeze()[0].item(),
                            track_id=data[4][im_idx].squeeze()[0].item(), save_path=save_plot_path + 'last_epoch/')

    logger.info(f'Total Loss : {sum(running_loss) / num_epochs}')
    plot_array(running_loss, 'Loss', 'epoch', 'loss', save_path=save_plot_path)
    plot_array(running_ade, 'ADE', 'epoch', 'ade', save_path=save_plot_path)
    plot_array(running_fde, 'FDE', 'epoch', 'fde', save_path=save_plot_path)


def overfit_on_dataset_chunks(train_video_class: Union[SDDVideoClasses, List[SDDVideoClassAndNumbers]],
                              train_video_number: int,
                              train_mode: NetworkMode,
                              train_meta_label: Union[SDDVideoDatasets, List[SDDVideoDatasets]],
                              val_video_class: Union[SDDVideoClasses, List[SDDVideoClassAndNumbers]] = None,
                              val_video_number: int = None, learn_hidden_states=False, get_generated: bool = False,
                              val_mode: NetworkMode = None,
                              val_meta_label: Union[SDDVideoDatasets, List[SDDVideoDatasets]] = None,
                              shuffle: bool = True, lr: float = LR, batch_size: int = BATCH_SIZE,
                              num_workers: int = NUM_WORKERS,
                              pin_memory: bool = True, use_batch_norm: bool = USE_BATCH_NORM,
                              over_fit_mode: bool = OVERFIT,
                              from_checkpoint=None, checkpoint_root_path=None,
                              gpus=1 if torch.cuda.is_available() else None,
                              max_epochs=NUM_EPOCHS, limit_train_batches=1.0, limit_val_batches=1.0,
                              over_fit_batches=0.0,
                              use_social_lstm_model=True, pass_final_pos=True, relative_velocities=False,
                              drop_last=True,
                              use_simple_model: bool = False, use_gru: bool = False, dropout=None, rnn_dropout=0,
                              num_rnn_layers=1,
                              feed_model_distances_in_meters=False, arch_config=LINEAR_CFG, train_for_class=False,
                              train_videos_to_skip=(), val_videos_to_skip=(), resume_custom_path=None,
                              resume_hparam_path=None,
                              resume_additional_epochs=1000, resume_custom_from_last_epoch=True,
                              overfit_element_count=None,
                              keep_overfit_elements_random=False, do_validation=False):
    if train_for_class:
        dataset_train, dataset_val = get_train_validation_dataset_for_class(
            train_video_class=train_video_class, train_meta_label=train_meta_label, val_video_class=val_video_class,
            val_meta_label=val_meta_label, get_generated=get_generated, videos_to_skip_for_train=train_videos_to_skip,
            videos_to_skip_for_val=val_videos_to_skip, return_dataset_list=True)
    else:
        dataset_train, dataset_val = get_train_validation_dataset(
            train_video_class=train_video_class, train_video_number=train_video_number, train_mode=train_mode,
            train_meta_label=train_meta_label, val_video_class=val_video_class, val_video_number=val_video_number,
            val_mode=val_mode, val_meta_label=val_meta_label, get_generated=get_generated)

    dataset_train_superset = ConcatenateDataset(datasets=dataset_train)
    dataset_val_superset = ConcatenateDataset(datasets=dataset_val)
    if overfit_element_count is not None:
        train_indices = np.random.choice(len(dataset_train_superset), size=overfit_element_count, replace=False) \
            if keep_overfit_elements_random else [i for i in range(overfit_element_count)]
        val_indices = np.random.choice(len(dataset_val_superset), size=overfit_element_count, replace=False) \
            if keep_overfit_elements_random else [i for i in range(overfit_element_count)]
        subset_dataset_train = Subset(dataset=dataset_train_superset, indices=train_indices)
        subset_dataset_val = Subset(dataset=dataset_val_superset, indices=val_indices)

        dataset_train = subset_dataset_train
        dataset_val = subset_dataset_val

    if overfit_element_count is None:
        dataset_train = dataset_train_superset
        dataset_val = dataset_val_superset

    loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                              pin_memory=pin_memory, drop_last=drop_last)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                            pin_memory=pin_memory, drop_last=drop_last)
    if use_social_lstm_model:
        model = get_social_model(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=batch_size,
                                 num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm,
                                 over_fit_mode=over_fit_mode, shuffle=shuffle, pin_memory=pin_memory,
                                 generated_dataset=get_generated, from_checkpoint=from_checkpoint,
                                 checkpoint_root_path=checkpoint_root_path, pass_final_pos=pass_final_pos)
    elif use_simple_model:
        model = get_simple_model(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=batch_size,
                                 num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm,
                                 over_fit_mode=over_fit_mode, shuffle=shuffle, pin_memory=pin_memory,
                                 generated_dataset=get_generated, from_checkpoint=from_checkpoint,
                                 checkpoint_root_path=checkpoint_root_path, use_gru=use_gru,
                                 relative_velocities=relative_velocities, dropout=dropout,
                                 rnn_dropout=rnn_dropout, num_rnn_layers=num_rnn_layers,
                                 learn_hidden_states=learn_hidden_states,
                                 feed_model_distances_in_meters=feed_model_distances_in_meters,
                                 arch_config=arch_config)
    else:
        model = get_model(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=batch_size,
                          num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm, over_fit_mode=over_fit_mode,
                          shuffle=shuffle, pin_memory=pin_memory, generated_dataset=get_generated,
                          from_checkpoint=from_checkpoint, checkpoint_root_path=checkpoint_root_path,
                          relative_velocities=relative_velocities, arch_config=arch_config)

    model.to(DEVICE)

    network = model if not use_social_lstm_model else model.one_step
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=AMS_GRAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=SCHEDULER_PATIENCE,
                                                           cooldown=2, verbose=True, factor=SCHEDULER_FACTOR)
    # summary_writer = SummaryWriter(comment='social_lstm' if use_social_lstm_model else 'baseline')

    best_val_loss = 10e7
    resume_dict = {}
    resume_dict_save_root_path = 'runs/Maar_overfit_experiments/' if overfit_element_count is not None else \
        'runs/Maar_overfit_experiments/full_train/'

    resume_dict_save_folder = f'element_size_{overfit_element_count}_random_{keep_overfit_elements_random}_lr_{lr}' \
                              f'_generated_{get_generated}_learn_hidden_{learn_hidden_states}' \
                              f'_rnn_layers_{num_rnn_layers}_{datetime.now()}'
    final_path = f'{resume_dict_save_root_path}{resume_dict_save_folder}/'

    # resume_dict_save_folder = os.listdir(resume_dict_save_root_path)
    # resume_dict_save_folder.sort()
    # resume_dict_save_folder = resume_dict_save_folder[-1]

    start_epoch = 0
    epoch = 0
    per_epoch_to_save_plot = 20 if get_generated else 2

    if resume_custom_path is not None:
        logger.info('Resuming Training!')
        resume_checkpoint = torch.load(resume_custom_path)

        with open(resume_hparam_path, 'r+') as hparam_file:
            resume_hyperparameters = yaml.load(hparam_file)

        if resume_custom_from_last_epoch:
            m_key = 'last_model_state_dict'
            o_key = 'last_optimizer_state_dict'
            s_key = 'last_scheduler_state_dict'
        else:
            m_key = 'model_state_dict'
            o_key = 'optimizer_state_dict'
            s_key = 'scheduler_state_dict'

        model.load_state_dict(resume_checkpoint[m_key])
        optimizer.load_state_dict(resume_checkpoint[o_key])
        scheduler.load_state_dict(resume_checkpoint[s_key])

        start_epoch = resume_hyperparameters['epoch']
        max_epochs += resume_additional_epochs
        best_val_loss = resume_hyperparameters['val_loss']

        # resume_dict_save_folder = os.path.split(os.path.split(resume_custom_path)[0])[-1]

    epoch_t_loss, epoch_t_ade, epoch_t_fde, epoch_v_loss, epoch_v_ade, epoch_v_fde = [], [], [], [], [], []

    try:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            running_t_loss, running_t_ade, running_t_fde = [], [], []
            with tqdm(loader_train, position=0) as t:
                t.set_description('Epoch %i' % epoch)
                for idx, (data, dataset_idx) in enumerate(loader_train):
                    optimizer.zero_grad()

                    data = [d.to(DEVICE) for d in data]
                    loss, ade, fde, ratio, pred_trajectory = network(data)

                    t.set_postfix(loss=loss.item(), ade=ade, fde=fde,
                                  epoch_loss=torch.tensor(epoch_t_loss).mean().item(),
                                  epoch_ade=torch.tensor(epoch_t_ade).mean().item(),
                                  epoch_fde=torch.tensor(epoch_t_fde).mean().item())
                    t.update()

                    loss.backward()
                    optimizer.step()

                    # summary_writer.add_scalar('train/loss', loss.item(), global_step=idx)
                    # summary_writer.add_scalar('train/ade', ade, global_step=idx)
                    # summary_writer.add_scalar('train/fde', fde, global_step=idx)

                    running_t_loss.append(loss.item())
                    running_t_ade.append(ade)
                    running_t_fde.append(fde)

                    # if LOG_HISTOGRAM:
                    #     for name, weight in model.named_parameters():
                    #         summary_writer.add_histogram(name, weight, epoch)
                    #         summary_writer.add_histogram(f'{name}.grad', weight.grad, epoch)

                epoch_t_loss_value = torch.tensor(running_t_loss).mean().item()
                epoch_t_loss.append(epoch_t_loss_value)
                epoch_t_ade.append(torch.tensor(running_t_ade).mean().item())
                epoch_t_fde.append(torch.tensor(running_t_fde).mean().item())

                if epoch % per_epoch_to_save_plot == 0:
                    im_idx = np.random.choice(batch_size, 1).item()
                    video_dataset = dataset_train_superset.datasets[dataset_idx[im_idx].item()]
                    video_path = f'{BASE_PATH}videos/{video_dataset.video_class.value}/' \
                                 f'video{video_dataset.video_number}/video.mov'
                    # plot_trajectory_with_relative_data(np.concatenate((data[0][im_idx].cpu().squeeze().numpy(),
                    #                                                   data[1][im_idx].cpu().squeeze().numpy())),
                    #                                    np.concatenate((data[2][im_idx].cpu().squeeze().numpy(),
                    #                                                   data[3][im_idx].cpu().squeeze().numpy())),
                    #                                    np.concatenate((data[2][im_idx].cpu().squeeze().numpy(),
                    #                                                   data[3][im_idx].cpu().squeeze().numpy())))
                    plot_trajectory_alongside_frame(
                            frame=extract_frame_from_video(
                                video_path=video_path, frame_number=data[6][im_idx].cpu().squeeze()[0].item()),
                            obs_trajectory=data[0][im_idx].cpu().squeeze().numpy(),
                            gt_trajectory=data[1][im_idx].cpu().squeeze().numpy(),
                            pred_trajectory=pred_trajectory[:, im_idx, ...].squeeze(),
                            frame_number=data[6][im_idx].cpu().squeeze()[0].item(), epoch=epoch,
                            track_id=data[4][im_idx].cpu().squeeze()[0].item(), save_path=final_path + 'train/')
                if epoch == list(range(max_epochs))[-1]:
                    for im_idx in range(batch_size):
                        plot_trajectory_alongside_frame(
                            frame=extract_frame_from_video(
                                video_path=video_path, frame_number=data[6][im_idx].cpu().squeeze()[0].item()),
                            obs_trajectory=data[0][im_idx].cpu().squeeze().numpy(),
                            gt_trajectory=data[1][im_idx].cpu().squeeze().numpy(),
                            pred_trajectory=pred_trajectory[:, im_idx, ...].squeeze(),
                            frame_number=data[6][im_idx].cpu().squeeze()[0].item(),
                            track_id=data[4][im_idx].cpu().squeeze()[0].item(), epoch=epoch,
                            save_path=final_path + 'train/' + 'last_epoch/')

                # summary_writer.add_scalar('train/loss_epoch', epoch_t_loss, global_step=epoch)
                # summary_writer.add_scalar('train/ade_epoch', epoch_t_ade, global_step=epoch)
                # summary_writer.add_scalar('train/fde_epoch', epoch_t_fde, global_step=epoch)

            if do_validation:
                model.eval()
                running_v_loss, running_v_ade, running_v_fde = [], [], []
                with tqdm(loader_val, colour='green', position=1) as v:
                    v.set_description('Epoch %i' % epoch)
                    with torch.no_grad():
                        for idx, (data, dataset_idx) in enumerate(tqdm(loader_val)):
                            data = [d.to(DEVICE) for d in data]
                            v_loss, v_ade, v_fde, ratio, pred_trajectory = network(data)

                            v.set_postfix(loss=v_loss.item(), ade=v_ade, fde=v_fde,
                                          epoch_loss=torch.tensor(epoch_v_loss).mean().item(),
                                          epoch_ade=torch.tensor(epoch_v_ade).mean().item(),
                                          epoch_fde=torch.tensor(epoch_v_fde).mean().item())
                            v.update()

                            # summary_writer.add_scalar('val/loss', v_loss.item(), global_step=idx)
                            # summary_writer.add_scalar('val/ade', v_ade, global_step=idx)
                            # summary_writer.add_scalar('val/fde', v_fde, global_step=idx)

                            running_v_loss.append(v_loss.item())
                            running_v_ade.append(v_ade)
                            running_v_fde.append(v_fde)

                    epoch_v_loss_value = torch.tensor(running_v_loss).mean().item()
                    epoch_v_loss.append(epoch_v_loss_value)
                    epoch_v_ade.append(torch.tensor(running_v_ade).mean().item())
                    epoch_v_fde.append(torch.tensor(running_v_fde).mean().item())

                    if epoch % per_epoch_to_save_plot == 0:
                        im_idx = np.random.choice(batch_size, 1).item()
                        video_dataset = dataset_val_superset.datasets[dataset_idx[im_idx].item()]
                        video_path = f'{BASE_PATH}videos/{video_dataset.video_class.value}/' \
                                     f'video{video_dataset.video_number}/video.mov'
                        plot_trajectory_alongside_frame(
                                frame=extract_frame_from_video(
                                    video_path=video_path, frame_number=data[6][im_idx].cpu().squeeze()[0].item()),
                                obs_trajectory=data[0][im_idx].cpu().squeeze().numpy(),
                                gt_trajectory=data[1][im_idx].cpu().squeeze().numpy(),
                                pred_trajectory=pred_trajectory[:, im_idx, ...].squeeze(),
                                frame_number=data[6][im_idx].cpu().squeeze()[0].item(), epoch=epoch,
                                track_id=data[4][im_idx].cpu().squeeze()[0].item(), save_path=final_path + 'val/')
                    if epoch == list(range(max_epochs))[-1]:
                        for im_idx in range(batch_size):
                            plot_trajectory_alongside_frame(
                                frame=extract_frame_from_video(
                                    video_path=video_path, frame_number=data[6][im_idx].cpu().squeeze()[0].item()),
                                obs_trajectory=data[0][im_idx].cpu().squeeze().numpy(),
                                gt_trajectory=data[1][im_idx].cpu().squeeze().numpy(),
                                pred_trajectory=pred_trajectory[:, im_idx, ...].squeeze(),
                                frame_number=data[6][im_idx].cpu().squeeze()[0].item(),
                                track_id=data[4][im_idx].cpu().squeeze()[0].item(), epoch=epoch,
                                save_path=final_path + 'val/' + 'last_epoch/')

                    # summary_writer.add_scalar('val/loss_epoch', epoch_v_loss, global_step=epoch)
                    # summary_writer.add_scalar('val/ade_epoch', epoch_v_ade, global_step=epoch)
                    # summary_writer.add_scalar('val/fde_epoch', epoch_v_fde, global_step=epoch)
                    #
                    # summary_writer.add_scalar('lr',
                    #                           [param_group['lr'] for param_group in optimizer.param_groups][-1],
                    #                           global_step=epoch)
                    # summary_writer.add_scalar('epoch', epoch, global_step=epoch)
                    #
                    # summary_writer.add_scalars(main_tag='loss',
                    #                            tag_scalar_dict={'train_loss': epoch_t_loss, 'val_loss': epoch_v_loss},
                    #                            global_step=epoch)

                    scheduler.step(epoch_v_loss_value)

                if not do_validation:
                    scheduler.step(epoch_t_loss_value)

                if epoch_v_loss_value < best_val_loss:
                    best_val_loss = epoch_v_loss_value
                    resume_dict = {'model_state_dict': model.state_dict(),
                                   'optimizer_state_dict': optimizer.state_dict(),
                                   'scheduler_state_dict': scheduler.state_dict(),
                                   'epoch': epoch,
                                   'val_loss': epoch_v_loss_value,
                                   'lr': lr,
                                   'batch_size': batch_size,
                                   'model_name': 'social_lstm' if use_social_lstm_model else 'baseline',
                                   'architecture': str(model),
                                   'generated_data': get_generated,
                                   'use_batch_norm': use_batch_norm,
                                   'train_class': train_video_class.name
                                   if not train_for_class else [t.name for t in train_video_class],
                                   'train_video_number': train_video_number,
                                   'val_class': val_video_class.name
                                   if not train_for_class else [v.name for v in val_video_class],
                                   'val_video_number': val_video_number,
                                   'use_social_lstm': use_social_lstm_model,
                                   'use_destinations': pass_final_pos,
                                   'use_relative_velocities': relative_velocities,
                                   'num_rnn_layers': num_rnn_layers,
                                   'dropout': dropout,
                                   'rnn_dropout': rnn_dropout,
                                   'use_gru': use_gru,
                                   'use_social_lstm_model': use_social_lstm_model,
                                   'use_simple_model': use_simple_model,
                                   'from_checkpoint': from_checkpoint,
                                   'pass_final_pos': pass_final_pos,
                                   'relative_velocities': relative_velocities,
                                   'learn_hidden_states': learn_hidden_states,
                                   'feed_model_distances_in_meters': feed_model_distances_in_meters,
                                   'arch_config': arch_config
                                   }
                    logger.info(f'Checkpoint Updated at epoch {epoch}, loss {epoch_v_loss_value}')
    except KeyboardInterrupt:
        logger.warning('Keyboard Interrupt: Saving and exiting gracefully.')
    finally:
        # Moved up to avoid replacing later ran experiments
        # resume_dict_save_folder = os.listdir(resume_dict_save_root_path)
        # resume_dict_save_folder.sort()
        # resume_dict_save_folder = resume_dict_save_folder[-1]
        resume_dict.update({
            'last_model_state_dict': model.state_dict(),
            'last_optimizer_state_dict': optimizer.state_dict(),
            'last_scheduler_state_dict': scheduler.state_dict(),
        })
        Path(final_path).mkdir(parents=True, exist_ok=True)
        checkpoint_file_name = f'{final_path}{resume_dict_save_folder}_checkpoint.ckpt' \
            if resume_custom_path is None else f'{final_path}{resume_dict_save_folder}_checkpoint_resumed.ckpt'
        torch.save(resume_dict, checkpoint_file_name)

        hparam_file = f'{final_path}{resume_dict_save_folder}_hparams.yaml' \
            if resume_custom_path is None else f'{final_path}{resume_dict_save_folder}_hparams_resumed.yaml'

        hparam_dict = {'epoch': epoch,
                       'val_loss': epoch_v_loss_value if do_validation else epoch_t_loss_value,
                       'lr': lr,
                       'batch_size': batch_size,
                       'model_name': 'social_lstm' if use_social_lstm_model else 'baseline',
                       'architecture': str(model),
                       'generated_data': get_generated,
                       'use_batch_norm': use_batch_norm,
                       'train_class': train_video_class.name
                       if not train_for_class else [t.name for t in train_video_class],
                       'train_video_number': train_video_number,
                       'val_class': val_video_class.name
                       if not train_for_class else [v.name for v in val_video_class],
                       'val_video_number': val_video_number,
                       'use_social_lstm': use_social_lstm_model,
                       'use_destinations': pass_final_pos,
                       'use_relative_velocities': relative_velocities,
                       'num_rnn_layers': num_rnn_layers,
                       'dropout': dropout,
                       'rnn_dropout': rnn_dropout,
                       'use_gru': use_gru,
                       'use_social_lstm_model': use_social_lstm_model,
                       'use_simple_model': use_simple_model,
                       'from_checkpoint': from_checkpoint,
                       'pass_final_pos': pass_final_pos,
                       'relative_velocities': relative_velocities,
                       'learn_hidden_states': learn_hidden_states,
                       'feed_model_distances_in_meters': feed_model_distances_in_meters,
                       'arch_config': arch_config,
                       'train_for_class': train_for_class,
                       'train_videos_to_skip': train_videos_to_skip,
                       'val_videos_to_skip': val_videos_to_skip,
                       'overfit_element_count': overfit_element_count,
                       'keep_overfit_elements_random': keep_overfit_elements_random,
                       'do_validation': do_validation,
                       'train_indices': train_indices if overfit_element_count is not None else [],
                       'val_indices': val_indices if overfit_element_count is not None else []
                       }
        # Path(hparam_file).mkdir(parents=True, exist_ok=True)
        with open(hparam_file, 'w+') as f:
            if resume_custom_path is not None:
                hparam_dict.update({'resumed_from': [resume_custom_path, resume_hparam_path]})
            yaml.dump(hparam_dict, f)

        logger.info('Saving and exiting gracefully.')
        logger.info(f"Best model at epoch: {hparam_dict['epoch']}")

        plot_array(epoch_t_loss, 'Train Loss', 'epoch', 'loss', save_path=final_path + 'curves/')
        plot_array(epoch_t_ade, 'Train ADE', 'epoch', 'ade', save_path=final_path + 'curves/')
        plot_array(epoch_t_fde, 'Train FDE', 'epoch', 'fde', save_path=final_path + 'curves/')

        plot_array(epoch_v_loss, 'Val Loss', 'epoch', 'loss', save_path=final_path + 'curves/')
        plot_array(epoch_v_ade, 'Val ADE', 'epoch', 'ade', save_path=final_path + 'curves/')
        plot_array(epoch_v_fde, 'Val FDE', 'epoch', 'fde', save_path=final_path + 'curves/')


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
    overfit_chunks = True
    if overfit_chunks:
        video_class_train = TRAIN_CLASS_FOR_WHOLE if TRAIN_FOR_WHOLE_CLASS else TRAIN_CLASS
        video_class_val = VAL_CLASS_FOR_WHOLE if TRAIN_FOR_WHOLE_CLASS else VAL_CLASS
        overfit_on_dataset_chunks(
            train_video_class=video_class_train,
            train_video_number=TRAIN_VIDEO_NUMBER,
            train_mode=NetworkMode.TRAIN,
            train_meta_label=TRAIN_META,
            val_video_class=video_class_val,
            val_video_number=VAL_VIDEO_NUMBER,
            val_mode=NetworkMode.VALIDATION,
            val_meta_label=VAL_META,
            get_generated=USE_GENERATED_DATA,
            shuffle=True,
            lr=LR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            use_batch_norm=USE_BATCH_NORM,
            over_fit_mode=OVERFIT,
            from_checkpoint=RESUME_TRAINING,
            checkpoint_root_path=CHECKPOINT_ROOT,
            gpus=1 if torch.cuda.is_available() else None,
            max_epochs=NUM_EPOCHS,
            limit_train_batches=LIMIT_BATCHES[0],
            limit_val_batches=LIMIT_BATCHES[1],
            over_fit_batches=OVERFIT_BATCHES,
            use_social_lstm_model=USE_SOCIAL_LSTM_MODEL,
            pass_final_pos=USE_FINAL_POSITIONS,
            relative_velocities=USE_RELATIVE_VELOCITIES,
            use_simple_model=USE_SIMPLE_MODEL,
            use_gru=USE_GRU,
            dropout=DROPOUT,
            rnn_dropout=RNN_DROPOUT,
            num_rnn_layers=RNN_LAYERS,
            learn_hidden_states=LEARN_HIDDEN_STATES,
            drop_last=True,
            feed_model_distances_in_meters=FEED_MODEL_DISTANCES_IN_METERS,
            arch_config=LINEAR_CFG,
            train_for_class=TRAIN_FOR_WHOLE_CLASS,
            train_videos_to_skip=TRAIN_VIDEOS_TO_SKIP,
            val_videos_to_skip=VAL_VIDEOS_TO_SKIP,
            resume_custom_path=RESUME_CUSTOM_TRAINING_PATH,
            resume_hparam_path=RESUME_CUSTOM_HPARAM_PATH,
            resume_additional_epochs=RESUME_ADDITIONAL_EPOCH,
            resume_custom_from_last_epoch=RESUME_FROM_LAST_EPOCH,
            overfit_element_count=OVERFIT_ELEMENT_COUNT,
            keep_overfit_elements_random=RANDOM_INDICES_IN_OVERFIT_ELEMENTS,
            do_validation=DO_VALIDATION
        )
    else:
        single_chunk_fit = False
        generated = False
        gt_batch_size = 32
        num_workers = 12
        shuffle = True
        use_social_lstm_model = False

        use_bn = True
        use_gru = False
        learn_hidden_states = True
        rel_velocities = False
        rnn_layers_count = 1
        num_epochs = 20000
        feed_model_distances_in_meters = False

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
            # model = BaselineRNNStacked(encoder_lstm_num_layers=1, decoder_lstm_num_layers=1, generated_dataset=generated,
            #                            use_batch_norm=True, relative_velocities=False)

            model = BaselineRNNStackedSimple(encoder_lstm_num_layers=rnn_layers_count,
                                             decoder_lstm_num_layers=rnn_layers_count, use_gru=use_gru,
                                             generated_dataset=generated,
                                             batch_size=1 if single_chunk_fit else gt_batch_size,
                                             use_batch_norm=use_bn, relative_velocities=rel_velocities,
                                             learn_hidden_states=learn_hidden_states,
                                             feed_model_distances_in_meters=feed_model_distances_in_meters)
            # model = BaselineRNNStackedSimple(encoder_lstm_num_layers=2, decoder_lstm_num_layers=2,
            #                                  generated_dataset=generated, rnn_dropout=0.2,
            #                                  use_batch_norm=True, relative_velocities=False, dropout=0.2)

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
            overfit_chunks = [0, 1, 2, 3, 4, 5] if generated else [i for i in range(gt_batch_size)]
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
        lr = 2e-2  # 1.5e-2  # 2e-3
        optim = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        # optim = torch.optim.SGD(model.parameters(), lr=lr)

        overfit(net=model, loader=overfit_dataloader, optimizer=optim, num_epochs=num_epochs,
                batch_mode=not single_chunk_fit,
                video_path=path_to_video, social_lstm=use_social_lstm_model, img_batch_size=gt_batch_size,
                save_plot_path=f'{ROOT_PATH}Plots/baseline_v2/nn/OVERFIT/{sdd_video_class.value}{sdd_video_number}/' \
                               f'epoch_count{num_epochs}/{"social" if use_social_lstm_model else "baseline"}_'
                               f'generated_{generated}__'
                               f'batch_norm_{use_bn}__generated_{generated}__gru_{use_gru}___lr_{lr}'
                               f'learn_hidden_{learn_hidden_states}__rel_velocities_{rel_velocities}'
                               f'__rnn_layers_{rnn_layers_count}__feed_in_meters_{feed_model_distances_in_meters}/')
        # overfit_two_loss(net=model, loader=overfit_dataloader, optimizer=optim, num_epochs=5000)

        print()
