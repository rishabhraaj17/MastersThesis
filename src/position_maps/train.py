import os

import albumentations as A
import hydra
import torch
from pytorch_lightning import seed_everything, Trainer
from torch.nn import CrossEntropyLoss

from average_image.constants import SDDVideoClasses
from log import get_logger
from position_maps.dataset import SDDFrameAndAnnotationDataset
from position_maps.models import PositionMapUNet
from position_maps.utils import heat_map_collate_fn

seed_everything(42)
logger = get_logger(__name__)


@hydra.main(config_path="config", config_name="config")
def train(cfg):
    logger.info(f'Setting up DataLoader and Model...')

    height, width = cfg.desired_size
    transform = A.Compose(
        [A.Resize(height=height, width=width),
         A.RandomBrightnessContrast(p=0.3),
         # A.RandomRotate90(p=0.3),  # possible on square images
         A.VerticalFlip(p=0.3),
         A.HorizontalFlip(p=0.3)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )

    train_dataset = SDDFrameAndAnnotationDataset(
        root=cfg.root, video_label=getattr(SDDVideoClasses, cfg.video_class),
        num_videos=cfg.train.num_videos, transform=transform if cfg.data_augmentation else None,
        num_workers=cfg.dataset_workers, scale=cfg.scale_factor,
        video_number_to_use=cfg.train.video_number_to_use,
        multiple_videos=cfg.train.multiple_videos,
        use_generated=cfg.use_generated_dataset,
        sigma=cfg.sigma,
        plot=cfg.plot_samples,
        desired_size=cfg.desired_size
    )

    val_dataset = SDDFrameAndAnnotationDataset(
        root=cfg.root, video_label=getattr(SDDVideoClasses, cfg.video_class),
        num_videos=cfg.val.num_videos, transform=transform if cfg.data_augmentation else None,
        num_workers=cfg.dataset_workers, scale=cfg.scale_factor,
        video_number_to_use=cfg.val.video_number_to_use,
        multiple_videos=cfg.val.multiple_videos,
        use_generated=cfg.use_generated_dataset,
        sigma=cfg.sigma,
        plot=cfg.plot_samples,
        desired_size=cfg.desired_size
    )

    loss_fn = CrossEntropyLoss()
    model = PositionMapUNet(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                            loss_function=loss_fn, collate_fn=heat_map_collate_fn)

    logger.info(f'Setting up Trainer...')

    if cfg.warm_restart.enable:
        checkpoint_root_path = f'{cfg.warm_restart.checkpoint.root}{cfg.warm_restart.checkpoint.path}' \
                               f'{cfg.warm_restart.checkpoint.version}/checkpoints/'
        hparams_path = f'{cfg.warm_restart.checkpoint.root}{cfg.warm_restart.checkpoint.path}' \
                       f'{cfg.warm_restart.checkpoint.version}/hparams.yaml'
        model_path = checkpoint_root_path + os.listdir(checkpoint_root_path)[0]
        logger.info(f'Resuming from : {model_path}')
        if cfg.warm_restart.custom_load:
            logger.info(f'Loading weights manually as custom load is {cfg.warm_restart.custom_load}')
            load_dict = torch.load(model_path)

            model.load_state_dict(load_dict['state_dict'])
            model.to(cfg.device)
            model.train()
        else:
            model = PositionMapUNet.load_from_checkpoint(
                checkpoint_path=model_path,
                hparams_file=hparams_path,
                map_location='cuda:0',
                train_dataset=train_dataset, val_dataset=val_dataset,
                loss_function=loss_fn, collate_fn=heat_map_collate_fn)

        trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                          fast_dev_run=cfg.trainer.fast_dev_run)
    else:
        if cfg.resume_mode:
            checkpoint_path = f'{cfg.resume.checkpoint.path}{cfg.resume.checkpoint.version}/checkpoints/'
            checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]

            trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                              fast_dev_run=cfg.trainer.fast_dev_run, resume_from_checkpoint=checkpoint_file)
        else:
            trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                              fast_dev_run=cfg.trainer.fast_dev_run)
    logger.info(f'Starting training...')

    trainer.fit(model)


if __name__ == '__main__':
    train()
