import os
import warnings

import albumentations as A
import hydra
import numpy as np
import torch
from kornia.losses import FocalLoss, BinaryFocalLossWithLogits
from pytorch_lightning import seed_everything, Trainer
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Subset

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from log import get_logger
from dataset import SDDFrameAndAnnotationDataset
import models as model_zoo
from utils import heat_map_collate_fn, plot_predictions

seed_everything(42)
logger = get_logger(__name__)


def get_resize_dims(cfg):
    meta = SDDMeta(cfg.root + 'H_SDD.txt')

    train_reference_img_path = f'{cfg.root}annotations/{getattr(SDDVideoClasses, cfg.video_class).value}/' \
                               f'video{cfg.train.video_number_to_use}/reference.jpg'
    train_w, train_h = meta.get_new_scale(img_path=train_reference_img_path,
                                          dataset=getattr(SDDVideoDatasets, cfg.video_meta_class),
                                          sequence=cfg.train.video_number_to_use,
                                          desired_ratio=cfg.desired_pixel_to_meter_ratio)

    val_reference_img_path = f'{cfg.root}annotations/{getattr(SDDVideoClasses, cfg.video_class).value}/' \
                             f'video{cfg.val.video_number_to_use}/reference.jpg'
    val_w, val_h = meta.get_new_scale(img_path=val_reference_img_path,
                                      dataset=getattr(SDDVideoDatasets, cfg.video_meta_class),
                                      sequence=cfg.val.video_number_to_use,
                                      desired_ratio=cfg.desired_pixel_to_meter_ratio)
    return [int(train_w), int(train_h)], [int(val_w), int(val_h)]


def setup_dataset(cfg, train_transform, val_transform):
    train_dataset = SDDFrameAndAnnotationDataset(
        root=cfg.root, video_label=getattr(SDDVideoClasses, cfg.video_class),
        num_videos=cfg.train.num_videos, transform=train_transform if cfg.data_augmentation else None,
        num_workers=cfg.dataset_workers, scale=cfg.scale_factor,
        video_number_to_use=cfg.train.video_number_to_use,
        multiple_videos=cfg.train.multiple_videos,
        use_generated=cfg.use_generated_dataset,
        sigma=cfg.sigma,
        plot=cfg.plot_samples,
        desired_size=cfg.desired_size,
        heatmap_shape=cfg.heatmap_shape,
        return_combined_heatmaps=cfg.return_combined_heatmaps,
        seg_map_objectness_threshold=cfg.seg_map_objectness_threshold,
        meta_label=getattr(SDDVideoDatasets, cfg.video_meta_class)
    )
    val_dataset = SDDFrameAndAnnotationDataset(
        root=cfg.root, video_label=getattr(SDDVideoClasses, cfg.video_class),
        num_videos=cfg.val.num_videos, transform=val_transform if cfg.data_augmentation else None,
        num_workers=cfg.dataset_workers, scale=cfg.scale_factor,
        video_number_to_use=cfg.val.video_number_to_use,
        multiple_videos=cfg.val.multiple_videos,
        use_generated=cfg.use_generated_dataset,
        sigma=cfg.sigma,
        plot=cfg.plot_samples,
        desired_size=cfg.desired_size,
        heatmap_shape=cfg.heatmap_shape,
        return_combined_heatmaps=cfg.return_combined_heatmaps,
        seg_map_objectness_threshold=cfg.seg_map_objectness_threshold,
        meta_label=getattr(SDDVideoDatasets, cfg.video_meta_class)
    )
    return train_dataset, val_dataset


def setup_trainer(cfg, loss_fn, model, train_dataset, val_dataset):
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
            network_type = getattr(model_zoo, cfg.postion_map_network_type)
            model = network_type.load_from_checkpoint(
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
    return model, trainer


@hydra.main(config_path="config", config_name="config")
def train(cfg):
    logger.info(f'Setting up DataLoader and Model...')

    (train_w, train_h), (val_w, val_h) = get_resize_dims(cfg)
    train_transform = A.Compose(
        [A.Resize(height=train_h, width=train_w),
         A.RandomBrightnessContrast(p=0.3),
         # A.RandomRotate90(p=0.3),  # possible on square images
         A.VerticalFlip(p=0.3),
         A.HorizontalFlip(p=0.3)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )
    val_transform = A.Compose(
        [A.Resize(height=val_h, width=val_w),
         A.RandomBrightnessContrast(p=0.3),
         # A.RandomRotate90(p=0.3),  # possible on square images
         A.VerticalFlip(p=0.3),
         A.HorizontalFlip(p=0.3)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )

    train_dataset, val_dataset = setup_dataset(cfg, train_transform=train_transform, val_transform=val_transform)

    network_type = getattr(model_zoo, cfg.postion_map_network_type)

    if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation', 'PositionMapUNetClassMapSegmentation']:
        loss_fn = BinaryFocalLossWithLogits(alpha=0.8, reduction='mean')  # CrossEntropyLoss()
    else:
        loss_fn = MSELoss()

    model = network_type(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                         loss_function=loss_fn, collate_fn=heat_map_collate_fn)

    logger.info(f'Setting up Trainer...')

    model, trainer = setup_trainer(cfg, loss_fn, model, train_dataset, val_dataset)
    logger.info(f'Starting training...')

    trainer.fit(model)


@hydra.main(config_path="config", config_name="config")
def overfit(cfg):
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

    train_dataset, val_dataset = setup_dataset(cfg, transform)

    network_type = getattr(model_zoo, cfg.postion_map_network_type)

    if cfg.class_map_segmentation or cfg.position_map_segmentation:
        # loss_fn = CrossEntropyLoss()
        # loss_fn = FocalLoss(alpha=0.9, reduction='mean')
        loss_fn = BinaryFocalLossWithLogits(alpha=0.8, reduction='mean')
    else:
        loss_fn = MSELoss()

    model = network_type(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                         loss_function=loss_fn, collate_fn=heat_map_collate_fn)
    model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)

    train_subset = Subset(dataset=train_dataset, indices=list(cfg.overfit.subset_indices))
    train_loader = DataLoader(train_subset, batch_size=cfg.overfit.batch_size, shuffle=False,
                              num_workers=cfg.overfit.num_workers, collate_fn=heat_map_collate_fn,
                              pin_memory=cfg.overfit.pin_memory, drop_last=cfg.overfit.drop_last)
    for epoch in range(cfg.overfit.num_epochs):
        model.train()

        train_loss = []
        for data in train_loader:
            opt.zero_grad()

            frames, heat_masks, position_map, distribution_map, class_maps, meta = data

            if cfg.class_map_segmentation:
                frames, class_maps = frames.to(cfg.device), class_maps.to(cfg.device)
            elif cfg.position_map_segmentation:
                frames, position_map = frames.to(cfg.device), position_map.to(cfg.device)
            else:
                frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

            out = model(frames)

            if cfg.class_map_segmentation:
                loss = loss_fn(out, class_maps.long().squeeze(dim=1))
            elif cfg.position_map_segmentation:
                loss = loss_fn(out, position_map.long().squeeze(dim=1))
            else:
                loss = loss_fn(out, heat_masks)

            train_loss.append(loss.item())

            loss.backward()
            opt.step()

        logger.info(f"Epoch: {epoch} | Train Loss: {np.array(train_loss).mean()}")

        if epoch % cfg.overfit.plot_checkpoint == 0:
            model.eval()
            val_loss = []

            for data in train_loader:
                frames, heat_masks, position_map, distribution_map, class_maps, meta = data

                if cfg.class_map_segmentation:
                    frames, class_maps = frames.to(cfg.device), class_maps.to(cfg.device)
                elif cfg.position_map_segmentation:
                    frames, position_map = frames.to(cfg.device), position_map.to(cfg.device)
                else:
                    frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

                with torch.no_grad():
                    out = model(frames)

                if cfg.class_map_segmentation:
                    loss = loss_fn(out, class_maps.long().squeeze(dim=1))
                elif cfg.position_map_segmentation:
                    loss = loss_fn(out, position_map.long().squeeze(dim=1))
                else:
                    loss = loss_fn(out, heat_masks)

                val_loss.append(loss.item())

                random_idx = np.random.choice(cfg.overfit.batch_size, 1, replace=False).item()

                if cfg.class_map_segmentation or cfg.position_map_segmentation:
                    # pred_mask = torch.cat((torch.softmax(out, dim=1),
                    #                        torch.zeros(size=(out.shape[0], 1, out.shape[2], out.shape[3]),
                    #                                    device=cfg.device)), dim=1).squeeze().cpu().permute(1, 2, 0)
                    pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     class_maps[random_idx].squeeze().cpu()
                                     if cfg.class_map_segmentation else position_map[random_idx].squeeze().cpu(),
                                     pred_mask[random_idx].int() * 255,
                                     additional_text=f"{network_type.__name__} | "
                                                     f"{loss_fn._get_name()} | Epoch: {epoch}")
                else:
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     out[random_idx].squeeze().cpu(),
                                     additional_text=f"{network_type.__name__} | "
                                                     f"{loss_fn._get_name()} | Epoch: {epoch}")

            logger.info(f"Epoch: {epoch} | Validation Loss: {np.array(val_loss).mean()}")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        overfit()
        # train()
