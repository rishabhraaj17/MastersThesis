import os
from pathlib import Path
from typing import List, Union, Tuple

import yaml

import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import BATCH_SIZE, NUM_WORKERS, LR, USE_BATCH_NORM, OVERFIT, NUM_EPOCHS, LIMIT_BATCHES, \
    OVERFIT_BATCHES, CHECKPOINT_ROOT, RESUME_TRAINING, USE_GENERATED_DATA, TRAIN_CLASS, TRAIN_VIDEO_NUMBER, TRAIN_META, \
    VAL_CLASS, VAL_VIDEO_NUMBER, VAL_META, USE_SOCIAL_LSTM_MODEL, USE_FINAL_POSITIONS, USE_RELATIVE_VELOCITIES, DEVICE, \
    TRAIN_CUSTOM, LOG_HISTOGRAM, USE_SIMPLE_MODEL, USE_GRU, RNN_DROPOUT, RNN_LAYERS, DROPOUT, LEARN_HIDDEN_STATES, \
    FEED_MODEL_DISTANCES_IN_METERS, LINEAR_CFG, TRAIN_FOR_WHOLE_CLASS, TRAIN_CLASS_FOR_WHOLE, VAL_CLASS_FOR_WHOLE, \
    TRAIN_VIDEOS_TO_SKIP, VAL_VIDEOS_TO_SKIP, SCHEDULER_PATIENCE, SCHEDULER_FACTOR, AMS_GRAD, \
    RESUME_CUSTOM_TRAINING_PATH, RESUME_CUSTOM_HPARAM_PATH, RESUME_ADDITIONAL_EPOCH, RESUME_FROM_LAST_EPOCH, \
    OVERFIT_ELEMENT_COUNT, RANDOM_INDICES_IN_OVERFIT_ELEMENTS
from baselinev2.constants import NetworkMode, SDDVideoClassAndNumbers
from baselinev2.nn.dataset import get_dataset
from baselinev2.nn.models import BaselineRNNStacked, BaselineRNNStackedSimple
from baselinev2.utils import social_lstm_parser
from baselinev2.nn.social_lstm.model import BaselineLSTM
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.nn.train')


def get_model(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, lr=LR,
              use_batch_norm=USE_BATCH_NORM, over_fit_mode=OVERFIT, shuffle=True, pin_memory=True,
              generated_dataset=True, from_checkpoint=None, checkpoint_root_path=None, relative_velocities=False,
              arch_config=LINEAR_CFG):
    if from_checkpoint:
        checkpoint_path = checkpoint_root_path + 'checkpoints/'
        checkpoint_file = os.listdir(checkpoint_path)[-1]
        model = BaselineRNNStacked.load_from_checkpoint(
            checkpoint_path=checkpoint_path + checkpoint_file,
            hparams_file=f'{checkpoint_root_path}hparams.yaml',
            map_location=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            generated_dataset=generated_dataset,
            relative_velocities=relative_velocities,
            arch_config=arch_config
        )
    else:
        model = BaselineRNNStacked(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size,
                                   num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm,
                                   overfit_mode=over_fit_mode, shuffle=shuffle, pin_memory=pin_memory,
                                   generated_dataset=generated_dataset, relative_velocities=relative_velocities,
                                   return_pred=True, arch_config=arch_config)
    model.train()
    return model


def get_simple_model(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, lr=LR,
                     use_batch_norm=USE_BATCH_NORM, over_fit_mode=OVERFIT, shuffle=True, pin_memory=True,
                     generated_dataset=True, from_checkpoint=None, checkpoint_root_path=None, use_gru=False,
                     relative_velocities=False, dropout=None, rnn_dropout=0, num_rnn_layers=1,
                     learn_hidden_states=False, feed_model_distances_in_meters=False, arch_config=LINEAR_CFG):
    if from_checkpoint:
        checkpoint_path = checkpoint_root_path + 'checkpoints/'
        checkpoint_file = os.listdir(checkpoint_path)[-1]
        model = BaselineRNNStackedSimple.load_from_checkpoint(
            checkpoint_path=checkpoint_path + checkpoint_file,
            hparams_file=f'{checkpoint_root_path}hparams.yaml',
            map_location=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            generated_dataset=generated_dataset,
            relative_velocities=relative_velocities,
            dropout=dropout,
            rnn_dropout=rnn_dropout,
            encoder_lstm_num_layers=num_rnn_layers,
            decoder_lstm_num_layers=num_rnn_layers,
            use_gru=use_gru,
            learn_hidden_states=learn_hidden_states,
            feed_model_distances_in_meters=feed_model_distances_in_meters,
            arch_config=arch_config
        )
    else:
        model = BaselineRNNStackedSimple(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size,
                                         num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm,
                                         overfit_mode=over_fit_mode, shuffle=shuffle, pin_memory=pin_memory,
                                         generated_dataset=generated_dataset, relative_velocities=relative_velocities,
                                         return_pred=True, dropout=dropout, rnn_dropout=rnn_dropout, use_gru=use_gru,
                                         encoder_lstm_num_layers=num_rnn_layers, decoder_lstm_num_layers=num_rnn_layers,
                                         learn_hidden_states=learn_hidden_states, arch_config=arch_config,
                                         feed_model_distances_in_meters=feed_model_distances_in_meters)
    model.train()
    return model


def get_social_model(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, lr=LR,
                     use_batch_norm=USE_BATCH_NORM, over_fit_mode=OVERFIT, shuffle=True, pin_memory=True,
                     generated_dataset=True, from_checkpoint=None, checkpoint_root_path=None, pass_final_pos=True):
    args = social_lstm_parser(pass_final_pos=pass_final_pos)
    if from_checkpoint:
        checkpoint_path = checkpoint_root_path + 'checkpoints/'
        checkpoint_file = os.listdir(checkpoint_path)[-1]
        model = BaselineLSTM.load_from_checkpoint(
            checkpoint_path=checkpoint_path + checkpoint_file,
            hparams_file=f'{checkpoint_root_path}hparams.yaml',
            map_location=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            generated_dataset=generated_dataset,
            args=args
        )
    else:
        model = BaselineLSTM(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size,
                             num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm,
                             over_fit_mode=over_fit_mode, shuffle=shuffle, pin_memory=pin_memory,
                             generated_dataset=generated_dataset, args=args)
    model.train()
    return model


def get_train_validation_dataset(train_video_class: SDDVideoClasses, train_video_number: int, train_mode: NetworkMode,
                                 train_meta_label: SDDVideoDatasets, val_video_class: SDDVideoClasses = None,
                                 val_video_number: int = None, val_mode: NetworkMode = None,
                                 val_meta_label: SDDVideoDatasets = None, get_generated: bool = False):
    dataset_train = get_dataset(train_video_class, train_video_number, train_mode, meta_label=train_meta_label,
                                get_generated=get_generated)
    dataset_val = get_dataset(val_video_class or train_video_class, val_video_number or train_video_number,
                              val_mode or train_mode, meta_label=val_meta_label or train_meta_label,
                              get_generated=get_generated)
    return dataset_train, dataset_val


def get_dataset_for_one_class(video_class: SDDVideoClassAndNumbers, meta_label: SDDVideoDatasets, mode: NetworkMode,
                              get_generated: bool = False, videos_to_skip: Union[List, Tuple] = ()):
    datasets = []
    all_videos = video_class.value[-1]
    videos_to_consider = np.setdiff1d(all_videos, videos_to_skip)
    for v_num in videos_to_consider:
        datasets.append(get_dataset(video_clazz=video_class.value[0], video_number=v_num, mode=mode,
                                    meta_label=meta_label, get_generated=get_generated))
    return ConcatDataset(datasets=datasets)


def get_dataset_for_class(video_class: List[SDDVideoClassAndNumbers], meta_label: List[SDDVideoDatasets],
                          mode: NetworkMode, get_generated: bool = False,
                          videos_to_skip: Union[List, Tuple] = (), return_dataset_list: bool = False):
    datasets = []
    for v_class, v_meta, skip_videos in zip(video_class, meta_label, videos_to_skip):
        all_videos = v_class.value[-1]
        videos_to_consider = np.setdiff1d(all_videos, skip_videos)
        for v_num in videos_to_consider:
            datasets.append(get_dataset(video_clazz=v_class.value[0], video_number=v_num, mode=mode,
                                        meta_label=v_meta, get_generated=get_generated))
    if return_dataset_list:
        return datasets
    return ConcatDataset(datasets=datasets)


def get_train_validation_dataset_for_class(train_video_class: List[SDDVideoClassAndNumbers],
                                           train_meta_label: List[SDDVideoDatasets],
                                           val_video_class: List[SDDVideoClassAndNumbers] = None,
                                           val_meta_label: List[SDDVideoDatasets] = None, get_generated: bool = False,
                                           videos_to_skip_for_train: Union[List, Tuple] = (),
                                           videos_to_skip_for_val: Union[List, Tuple] = (),
                                           return_dataset_list: bool = False):
    train_dataset = get_dataset_for_class(video_class=train_video_class, meta_label=train_meta_label,
                                          mode=NetworkMode.TRAIN, get_generated=get_generated,
                                          videos_to_skip=videos_to_skip_for_train,
                                          return_dataset_list=return_dataset_list)
    val_dataset = get_dataset_for_class(video_class=val_video_class, meta_label=val_meta_label,
                                        mode=NetworkMode.VALIDATION, get_generated=get_generated,
                                        videos_to_skip=videos_to_skip_for_val,
                                        return_dataset_list=return_dataset_list)
    return train_dataset, val_dataset


def get_trainer(gpus=1 if torch.cuda.is_available() else None, max_epochs=NUM_EPOCHS, limit_train_batches=1.0,
                limit_val_batches=1.0, over_fit_batches=0.0):
    return Trainer(gpus=gpus, max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                   limit_val_batches=limit_val_batches, overfit_batches=over_fit_batches if OVERFIT else 0.0)


def train(train_video_class: Union[SDDVideoClasses, List[SDDVideoClassAndNumbers]], train_video_number: int,
          train_mode: NetworkMode, train_meta_label: Union[SDDVideoDatasets, List[SDDVideoDatasets]],
          val_video_class: Union[SDDVideoClasses, List[SDDVideoClassAndNumbers]] = None, val_video_number: int = None,
          val_mode: NetworkMode = None, val_meta_label: Union[SDDVideoDatasets, List[SDDVideoDatasets]] = None,
          shuffle: bool = True, lr: float = LR, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS,
          pin_memory: bool = True, use_batch_norm: bool = USE_BATCH_NORM, over_fit_mode: bool = OVERFIT,
          from_checkpoint=None, checkpoint_root_path=None, gpus=1 if torch.cuda.is_available() else None,
          max_epochs=NUM_EPOCHS, limit_train_batches=1.0, limit_val_batches=1.0, over_fit_batches=0.0,
          use_social_lstm_model=True, pass_final_pos=True, relative_velocities=False, get_generated: bool = False,
          use_simple_model: bool = False, use_gru: bool = False, dropout=None, rnn_dropout=0, num_rnn_layers=1,
          learn_hidden_states=False, drop_last=True, feed_model_distances_in_meters=True, arch_config=LINEAR_CFG,
          train_for_class=False, train_videos_to_skip=(), val_videos_to_skip=(), resume_custom_from_last_epoch=True,
          resume_custom_path=None, resume_hparam_path=None, resume_additional_epochs=1000,
          overfit_element_count=None, keep_overfit_elements_random=False):
    if train_for_class:
        dataset_train, dataset_val = get_train_validation_dataset_for_class(
            train_video_class=train_video_class, train_meta_label=train_meta_label, val_video_class=val_video_class,
            val_meta_label=val_meta_label, get_generated=get_generated, videos_to_skip_for_train=train_videos_to_skip,
            videos_to_skip_for_val=val_videos_to_skip)
    else:
        dataset_train, dataset_val = get_train_validation_dataset(
            train_video_class=train_video_class, train_video_number=train_video_number, train_mode=train_mode,
            train_meta_label=train_meta_label, val_video_class=val_video_class, val_video_number=val_video_number,
            val_mode=val_mode, val_meta_label=val_meta_label, get_generated=get_generated)

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
    trainer = get_trainer(gpus=gpus, max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                          limit_val_batches=limit_val_batches,
                          over_fit_batches=over_fit_batches if over_fit_mode else 0.0)

    trainer.fit(model=model)


def train_custom(train_video_class: Union[SDDVideoClasses, List[SDDVideoClassAndNumbers]], train_video_number: int,
                 train_mode: NetworkMode, train_meta_label: Union[SDDVideoDatasets, List[SDDVideoDatasets]],
                 val_video_class: Union[SDDVideoClasses, List[SDDVideoClassAndNumbers]] = None,
                 val_video_number: int = None, learn_hidden_states=False, get_generated: bool = False,
                 val_mode: NetworkMode = None, val_meta_label: Union[SDDVideoDatasets, List[SDDVideoDatasets]] = None,
                 shuffle: bool = True, lr: float = LR, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS,
                 pin_memory: bool = True, use_batch_norm: bool = USE_BATCH_NORM, over_fit_mode: bool = OVERFIT,
                 from_checkpoint=None, checkpoint_root_path=None, gpus=1 if torch.cuda.is_available() else None,
                 max_epochs=NUM_EPOCHS, limit_train_batches=1.0, limit_val_batches=1.0, over_fit_batches=0.0,
                 use_social_lstm_model=True, pass_final_pos=True, relative_velocities=False, drop_last=True,
                 use_simple_model: bool = False, use_gru: bool = False, dropout=None, rnn_dropout=0, num_rnn_layers=1,
                 feed_model_distances_in_meters=False, arch_config=LINEAR_CFG, train_for_class=False,
                 train_videos_to_skip=(), val_videos_to_skip=(), resume_custom_path=None, resume_hparam_path=None,
                 resume_additional_epochs=1000, resume_custom_from_last_epoch=True, overfit_element_count=None, 
                 keep_overfit_elements_random=False):
    if train_for_class:
        dataset_train, dataset_val = get_train_validation_dataset_for_class(
            train_video_class=train_video_class, train_meta_label=train_meta_label, val_video_class=val_video_class,
            val_meta_label=val_meta_label, get_generated=get_generated, videos_to_skip_for_train=train_videos_to_skip,
            videos_to_skip_for_val=val_videos_to_skip)
    else:
        dataset_train, dataset_val = get_train_validation_dataset(
            train_video_class=train_video_class, train_video_number=train_video_number, train_mode=train_mode,
            train_meta_label=train_meta_label, val_video_class=val_video_class, val_video_number=val_video_number,
            val_mode=val_mode, val_meta_label=val_meta_label, get_generated=get_generated)
        
    if overfit_element_count is not None:
        train_indices = np.random.choice(len(dataset_train), size=overfit_element_count, replace=False) \
            if keep_overfit_elements_random else [i for i in range(overfit_element_count)]
        val_indices = np.random.choice(len(dataset_val), size=overfit_element_count, replace=False) \
            if keep_overfit_elements_random else [i for i in range(overfit_element_count)]
        subset_dataset_train = Subset(dataset=dataset_train, indices=train_indices)
        subset_dataset_val = Subset(dataset=dataset_val, indices=val_indices)

        dataset_train = subset_dataset_train
        dataset_val = subset_dataset_val

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
    summary_writer = SummaryWriter(comment='social_lstm' if use_social_lstm_model else 'baseline')

    best_val_loss = 10e7
    resume_dict = {}
    resume_dict_save_root_path = 'runs/'

    resume_dict_save_folder = os.listdir(resume_dict_save_root_path)
    resume_dict_save_folder.sort()
    resume_dict_save_folder = resume_dict_save_folder[-1]

    start_epoch = 0

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

    total_train_iter_in_one_epoch = len(dataset_train) // batch_size
    total_val_iter_in_one_epoch = len(dataset_val) // batch_size

    epoch_t_loss, epoch_t_ade, epoch_t_fde, epoch_v_loss, epoch_v_ade, epoch_v_fde = None, None, None, None, None, None

    try:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            running_t_loss, running_t_ade, running_t_fde = [], [], []
            with tqdm(loader_train, position=0) as t:
                t.set_description('Epoch %i' % epoch)
                for idx, data in enumerate(loader_train):
                    optimizer.zero_grad()

                    data = [d.to(DEVICE) for d in data]
                    loss, ade, fde, ratio, pred_trajectory = network(data)

                    t.set_postfix(loss=loss.item(), ade=ade, fde=fde,
                                  epoch_loss=epoch_t_loss, epoch_ade=epoch_t_ade, epoch_fde=epoch_t_fde)
                    t.update()

                    loss.backward()
                    optimizer.step()

                    summary_writer.add_scalar('train/loss', loss.item(),
                                              global_step=((total_train_iter_in_one_epoch * epoch) + idx))
                    summary_writer.add_scalar('train/ade', ade,
                                              global_step=((total_train_iter_in_one_epoch * epoch) + idx))
                    summary_writer.add_scalar('train/fde', fde,
                                              global_step=((total_train_iter_in_one_epoch * epoch) + idx))

                    running_t_loss.append(loss.item())
                    running_t_ade.append(ade)
                    running_t_fde.append(fde)

                    if LOG_HISTOGRAM:
                        for name, weight in model.named_parameters():
                            summary_writer.add_histogram(name, weight, epoch)
                            summary_writer.add_histogram(f'{name}.grad', weight.grad, epoch)

                epoch_t_loss = torch.tensor(running_t_loss).mean().item()
                epoch_t_ade = torch.tensor(running_t_ade).mean().item()
                epoch_t_fde = torch.tensor(running_t_fde).mean().item()

                summary_writer.add_scalar('train/loss_epoch', epoch_t_loss, global_step=epoch)
                summary_writer.add_scalar('train/ade_epoch', epoch_t_ade, global_step=epoch)
                summary_writer.add_scalar('train/fde_epoch', epoch_t_fde, global_step=epoch)

            model.eval()
            running_v_loss, running_v_ade, running_v_fde = [], [], []
            with tqdm(loader_val, colour='green', position=1) as v:
                v.set_description('Epoch %i' % epoch)
                with torch.no_grad():
                    for idx, data in enumerate(tqdm(loader_val)):
                        data = [d.to(DEVICE) for d in data]
                        v_loss, v_ade, v_fde, ratio, pred_trajectory = network(data)

                        v.set_postfix(loss=v_loss.item(), ade=v_ade, fde=v_fde,
                                      epoch_loss=epoch_v_loss, epoch_ade=epoch_v_ade, epoch_fde=epoch_v_fde)
                        v.update()

                        summary_writer.add_scalar('val/loss', v_loss.item(),
                                                  global_step=((total_val_iter_in_one_epoch * epoch) + idx))
                        summary_writer.add_scalar('val/ade', v_ade,
                                                  global_step=((total_val_iter_in_one_epoch * epoch) + idx))
                        summary_writer.add_scalar('val/fde', v_fde,
                                                  global_step=((total_val_iter_in_one_epoch * epoch) + idx))

                        running_v_loss.append(v_loss.item())
                        running_v_ade.append(v_ade)
                        running_v_fde.append(v_fde)

                epoch_v_loss = torch.tensor(running_v_loss).mean().item()
                epoch_v_ade = torch.tensor(running_v_ade).mean().item()
                epoch_v_fde = torch.tensor(running_v_fde).mean().item()

                summary_writer.add_scalar('val/loss_epoch', epoch_v_loss, global_step=epoch)
                summary_writer.add_scalar('val/ade_epoch', epoch_v_ade, global_step=epoch)
                summary_writer.add_scalar('val/fde_epoch', epoch_v_fde, global_step=epoch)

                summary_writer.add_scalar('lr',
                                          [param_group['lr'] for param_group in optimizer.param_groups][-1],
                                          global_step=epoch)
                summary_writer.add_scalar('epoch', epoch, global_step=epoch)

                summary_writer.add_scalars(main_tag='loss',
                                           tag_scalar_dict={'train_loss': epoch_t_loss, 'val_loss': epoch_v_loss},
                                           global_step=epoch)

                scheduler.step(epoch_v_loss)

                if epoch_v_loss < best_val_loss:
                    best_val_loss = epoch_v_loss
                    resume_dict = {'model_state_dict': model.state_dict(),
                                   'optimizer_state_dict': optimizer.state_dict(),
                                   'scheduler_state_dict': scheduler.state_dict(),
                                   'epoch': epoch,
                                   'val_loss': epoch_v_loss,
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
                    logger.info(f'Checkpoint Updated at epoch {epoch}, loss {epoch_v_loss}')
                    final_path = f'{resume_dict_save_root_path}{resume_dict_save_folder}/'
                    checkpoint_file_name = f'{final_path}{resume_dict_save_folder}_checkpoint_epoch_{epoch}.ckpt' \
                        if resume_custom_path is None else f'{final_path}{resume_dict_save_folder}' \
                                                           f'_checkpoint_resumed_epoch_{epoch}.ckpt'
                    torch.save(resume_dict, checkpoint_file_name)
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
        final_path = f'{resume_dict_save_root_path}{resume_dict_save_folder}/'
        checkpoint_file_name = f'{final_path}{resume_dict_save_folder}_checkpoint.ckpt' \
            if resume_custom_path is None else f'{final_path}{resume_dict_save_folder}_checkpoint_resumed.ckpt'
        torch.save(resume_dict, checkpoint_file_name)

        hparam_file = f'{final_path}{resume_dict_save_folder}_hparams.yaml' \
            if resume_custom_path is None else f'{final_path}{resume_dict_save_folder}_hparams_resumed.yaml'

        hparam_dict = {'epoch': epoch,
                       'val_loss': epoch_v_loss,
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
                       'keep_overfit_elements_random': keep_overfit_elements_random
                       }
        # Path(hparam_file).mkdir(parents=True, exist_ok=True)
        with open(hparam_file, 'w+') as f:
            if resume_custom_path is not None:
                hparam_dict.update({'resumed_from': [resume_custom_path, resume_hparam_path]})
            yaml.dump(hparam_dict, f)

        logger.info('Saving and exiting gracefully.')
        logger.info(f"Best model at epoch: {resume_dict['epoch']}")


if __name__ == '__main__':
    trainer_method = train_custom if TRAIN_CUSTOM else train
    video_class_train = TRAIN_CLASS_FOR_WHOLE if TRAIN_FOR_WHOLE_CLASS else TRAIN_CLASS
    video_class_val = VAL_CLASS_FOR_WHOLE if TRAIN_FOR_WHOLE_CLASS else VAL_CLASS
    trainer_method(
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
        keep_overfit_elements_random=RANDOM_INDICES_IN_OVERFIT_ELEMENTS
    )
