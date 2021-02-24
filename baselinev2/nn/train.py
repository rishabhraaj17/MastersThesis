import os

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import BATCH_SIZE, NUM_WORKERS, LR, USE_BATCH_NORM, OVERFIT, NUM_EPOCHS, LIMIT_BATCHES, \
    OVERFIT_BATCHES, CHECKPOINT_ROOT, RESUME_TRAINING, USE_GENERATED_DATA, TRAIN_CLASS, TRAIN_VIDEO_NUMBER, TRAIN_META, \
    VAL_CLASS, VAL_VIDEO_NUMBER, VAL_META, USE_SOCIAL_LSTM_MODEL, USE_FINAL_POSITIONS, USE_RELATIVE_VELOCITIES, DEVICE, \
    TRAIN_CUSTOM, LOG_HISTOGRAM
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import get_dataset
from baselinev2.nn.models import BaselineRNNStacked
from baselinev2.nn.overfit import social_lstm_parser
from baselinev2.nn.social_lstm.model import BaselineLSTM
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.nn.train')


def get_model(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, lr=LR,
              use_batch_norm=USE_BATCH_NORM, over_fit_mode=OVERFIT, shuffle=True, pin_memory=True,
              generated_dataset=True, from_checkpoint=None, checkpoint_root_path=None, relative_velocities=False):
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
            relative_velocities=relative_velocities
        )
    else:
        model = BaselineRNNStacked(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size,
                                   num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm,
                                   overfit_mode=over_fit_mode, shuffle=shuffle, pin_memory=pin_memory,
                                   generated_dataset=generated_dataset, relative_velocities=relative_velocities,
                                   return_pred=True)
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


def get_trainer(gpus=1 if torch.cuda.is_available() else None, max_epochs=NUM_EPOCHS, limit_train_batches=1.0,
                limit_val_batches=1.0, over_fit_batches=0.0):
    return Trainer(gpus=gpus, max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                   limit_val_batches=limit_val_batches, overfit_batches=over_fit_batches if OVERFIT else 0.0)


def train(train_video_class: SDDVideoClasses, train_video_number: int, train_mode: NetworkMode,
          train_meta_label: SDDVideoDatasets, val_video_class: SDDVideoClasses = None, val_video_number: int = None,
          val_mode: NetworkMode = None, val_meta_label: SDDVideoDatasets = None, get_generated: bool = False,
          shuffle: bool = True, lr: float = LR, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS,
          pin_memory: bool = True, use_batch_norm: bool = USE_BATCH_NORM, over_fit_mode: bool = OVERFIT,
          from_checkpoint=None, checkpoint_root_path=None, gpus=1 if torch.cuda.is_available() else None,
          max_epochs=NUM_EPOCHS, limit_train_batches=1.0, limit_val_batches=1.0, over_fit_batches=0.0,
          use_social_lstm_model=True, pass_final_pos=True, relative_velocities=False):
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
    else:
        model = get_model(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=batch_size,
                          num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm, over_fit_mode=over_fit_mode,
                          shuffle=shuffle, pin_memory=pin_memory, generated_dataset=get_generated,
                          from_checkpoint=from_checkpoint, checkpoint_root_path=checkpoint_root_path,
                          relative_velocities=relative_velocities)
    trainer = get_trainer(gpus=gpus, max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                          limit_val_batches=limit_val_batches,
                          over_fit_batches=over_fit_batches if over_fit_mode else 0.0)

    trainer.fit(model=model)


def train_custom(train_video_class: SDDVideoClasses, train_video_number: int, train_mode: NetworkMode,
                 train_meta_label: SDDVideoDatasets, val_video_class: SDDVideoClasses = None,
                 val_video_number: int = None,
                 val_mode: NetworkMode = None, val_meta_label: SDDVideoDatasets = None, get_generated: bool = False,
                 shuffle: bool = True, lr: float = LR, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS,
                 pin_memory: bool = True, use_batch_norm: bool = USE_BATCH_NORM, over_fit_mode: bool = OVERFIT,
                 from_checkpoint=None, checkpoint_root_path=None, gpus=1 if torch.cuda.is_available() else None,
                 max_epochs=NUM_EPOCHS, limit_train_batches=1.0, limit_val_batches=1.0, over_fit_batches=0.0,
                 use_social_lstm_model=True, pass_final_pos=True, relative_velocities=False):
    dataset_train, dataset_val = get_train_validation_dataset(
        train_video_class=train_video_class, train_video_number=train_video_number, train_mode=train_mode,
        train_meta_label=train_meta_label, val_video_class=val_video_class, val_video_number=val_video_number,
        val_mode=val_mode, val_meta_label=val_meta_label, get_generated=get_generated)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                              pin_memory=pin_memory)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                            pin_memory=pin_memory)
    if use_social_lstm_model:
        model = get_social_model(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=batch_size,
                                 num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm,
                                 over_fit_mode=over_fit_mode, shuffle=shuffle, pin_memory=pin_memory,
                                 generated_dataset=get_generated, from_checkpoint=from_checkpoint,
                                 checkpoint_root_path=checkpoint_root_path, pass_final_pos=pass_final_pos)
    else:
        model = get_model(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=batch_size,
                          num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm, over_fit_mode=over_fit_mode,
                          shuffle=shuffle, pin_memory=pin_memory, generated_dataset=get_generated,
                          from_checkpoint=from_checkpoint, checkpoint_root_path=checkpoint_root_path,
                          relative_velocities=relative_velocities)

    model.to(DEVICE)

    network = model.one_step if use_social_lstm_model else model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=20, cooldown=2, verbose=True,
                                                           factor=0.1)
    summary_writer = SummaryWriter(comment='social_lstm' if use_social_lstm_model else 'baseline')

    best_val_loss = 10e7
    resume_dict = {}
    resume_dict_save_root_path = 'runs/'

    try:
        for epoch in range(max_epochs):
            model.train()
            with tqdm(loader_train, position=0) as t:
                t.set_description('Epoch %i' % epoch)
                for idx, data in enumerate(loader_train):
                    optimizer.zero_grad()

                    data = [d.to(DEVICE) for d in data]
                    loss, ade, fde, ratio, pred_trajectory = network(data)

                    t.set_postfix(loss=loss.item(), ade=ade, fde=fde)
                    t.update()

                    loss.backward()
                    optimizer.step()

                    summary_writer.add_scalar('train/loss', loss.item(), global_step=idx)
                    summary_writer.add_scalar('train/ade', ade, global_step=idx)
                    summary_writer.add_scalar('train/fde', fde, global_step=idx)

                    summary_writer.add_scalar('train/loss_epoch', loss.item(), global_step=epoch)
                    summary_writer.add_scalar('train/ade_epoch', ade, global_step=epoch)
                    summary_writer.add_scalar('train/fde_epoch', fde, global_step=epoch)

                    if LOG_HISTOGRAM:
                        for name, weight in model.named_parameters():
                            summary_writer.add_histogram(name, weight, epoch)
                            summary_writer.add_histogram(f'{name}.grad', weight.grad, epoch)

            model.eval()
            running_v_loss = []
            with tqdm(loader_val, colour='green', position=1) as v:
                v.set_description('Epoch %i' % epoch)
                with torch.no_grad():
                    for idx, data in enumerate(tqdm(loader_val)):
                        data = [d.to(DEVICE) for d in data]
                        v_loss, v_ade, v_fde, ratio, pred_trajectory = network(data)

                        v.set_postfix(loss=v_loss.item(), ade=v_ade, fde=v_fde)
                        v.update()

                        summary_writer.add_scalar('val/loss', v_loss.item(), global_step=idx)
                        summary_writer.add_scalar('val/ade', v_ade, global_step=idx)
                        summary_writer.add_scalar('val/fde', v_fde, global_step=idx)

                        summary_writer.add_scalar('val/loss_epoch', v_loss.item(), global_step=epoch)
                        summary_writer.add_scalar('val/ade_epoch', v_ade, global_step=epoch)
                        summary_writer.add_scalar('val/fde_epoch', v_fde, global_step=epoch)

                        summary_writer.add_scalar('lr',
                                                  [param_group['lr'] for param_group in optimizer.param_groups][-1],
                                                  global_step=idx)
                        summary_writer.add_scalar('epoch', epoch, global_step=epoch)

                        running_v_loss.append(v_loss.item())

                epoch_v_loss = torch.tensor(running_v_loss).mean().item()
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
                                   'train_class': train_video_class.name,
                                   'train_video_number': train_video_number,
                                   'val_class': val_video_class.name,
                                   'val_video_number': val_video_number,
                                   'use_social_lstm': use_social_lstm_model,
                                   'use_destinations': pass_final_pos,
                                   'use_relative_velocities': relative_velocities
                                   }
                    logger.info(f'Checkpoint Updated at epoch {epoch}, loss {epoch_v_loss}')
    except KeyboardInterrupt:
        logger.warning('Keyboard Interrupt: Saving and exiting gracefully.')
    finally:
        resume_dict_save_folder = os.listdir(resume_dict_save_root_path)
        resume_dict_save_folder.sort()
        resume_dict_save_folder = resume_dict_save_folder[-1]
        resume_dict.update({
            'last_model_state_dict': model.state_dict(),
            'last_optimizer_state_dict': optimizer.state_dict(),
            'last_scheduler_state_dict': scheduler.state_dict(),
        })
        torch.save(resume_dict, f'{resume_dict_save_root_path}{resume_dict_save_folder}/'
                                f'{resume_dict_save_folder}_checkpoint.ckpt')
        logger.info('Saving and exiting gracefully.')
        logger.info(f"Best model at epoch: {resume_dict['epoch']}")


if __name__ == '__main__':
    trainer_method = train_custom if TRAIN_CUSTOM else train
    trainer_method(
        train_video_class=TRAIN_CLASS,
        train_video_number=TRAIN_VIDEO_NUMBER,
        train_mode=NetworkMode.TRAIN,
        train_meta_label=TRAIN_META,
        val_video_class=VAL_CLASS,
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
        relative_velocities=USE_RELATIVE_VELOCITIES
    )
