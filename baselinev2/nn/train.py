import os

import torch
from pytorch_lightning import Trainer

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import BATCH_SIZE, NUM_WORKERS, LR, USE_BATCH_NORM, OVERFIT, NUM_EPOCHS, LIMIT_BATCHES, \
    OVERFIT_BATCHES, CHECKPOINT_ROOT, RESUME_TRAINING, USE_GENERATED_DATA, TRAIN_CLASS, TRAIN_VIDEO_NUMBER, TRAIN_META, \
    VAL_CLASS, VAL_VIDEO_NUMBER, VAL_META
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import get_dataset
from baselinev2.nn.models import BaselineRNNStacked


def get_model(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, lr=LR,
              use_batch_norm=USE_BATCH_NORM, over_fit_mode=OVERFIT, shuffle=True, pin_memory=True,
              generated_dataset=True, from_checkpoint=None, checkpoint_root_path=None):
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
            generated_dataset=generated_dataset
        )
    else:
        model = BaselineRNNStacked(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size,
                                   num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm,
                                   overfit_mode=over_fit_mode, shuffle=shuffle, pin_memory=pin_memory,
                                   generated_dataset=generated_dataset)
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
          max_epochs=NUM_EPOCHS, limit_train_batches=1.0, limit_val_batches=1.0, over_fit_batches=0.0):
    dataset_train, dataset_val = get_train_validation_dataset(
        train_video_class=train_video_class, train_video_number=train_video_number, train_mode=train_mode,
        train_meta_label=train_meta_label, val_video_class=val_video_class, val_video_number=val_video_number,
        val_mode=val_mode, val_meta_label=val_meta_label, get_generated=get_generated)
    model = get_model(train_dataset=dataset_train, val_dataset=dataset_val, batch_size=batch_size,
                      num_workers=num_workers, lr=lr, use_batch_norm=use_batch_norm, over_fit_mode=over_fit_mode,
                      shuffle=shuffle, pin_memory=pin_memory, generated_dataset=get_generated,
                      from_checkpoint=from_checkpoint, checkpoint_root_path=checkpoint_root_path)
    trainer = get_trainer(gpus=gpus, max_epochs=max_epochs, limit_train_batches=limit_train_batches,
                          limit_val_batches=limit_val_batches,
                          over_fit_batches=over_fit_batches if over_fit_mode else 0.0)

    trainer.fit(model=model)


if __name__ == '__main__':
    train(
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
        over_fit_batches=OVERFIT_BATCHES
    )