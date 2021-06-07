import os
import warnings

import albumentations as A
import hydra
import numpy as np
import torch
from kornia.losses import FocalLoss, BinaryFocalLossWithLogits
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Subset, ConcatDataset

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from log import get_logger
from dataset import SDDFrameAndAnnotationDataset
import models as model_zoo
from utils import heat_map_collate_fn, plot_predictions, get_scaled_shapes_with_pad_values, rgb_transform

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


def get_resize_shape(cfg, sdd_meta, video_class, video_number, desired_ratio):
    reference_img_path = f'{cfg.root}annotations/{getattr(SDDVideoClasses, video_class).value}/' \
                         f'video{video_number}/reference.jpg'
    w, h = sdd_meta.get_new_scale(img_path=reference_img_path,
                                  dataset=getattr(SDDVideoDatasets, video_class),
                                  sequence=video_number,
                                  desired_ratio=desired_ratio)
    return w, h


def setup_dataset(cfg):
    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=[cfg.video_class, cfg.video_class],
        video_numbers=[[cfg.train.video_number_to_use], [cfg.val.video_number_to_use]],
        desired_ratio=cfg.desired_pixel_to_meter_ratio_rgb)

    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=[cfg.video_class, cfg.video_class],
        video_numbers=[[cfg.train.video_number_to_use], [cfg.val.video_number_to_use]],
        desired_ratio=cfg.desired_pixel_to_meter_ratio)

    train_condition = (df.CLASS == cfg.video_class) & (df.NUMBER == cfg.train.video_number_to_use)
    train_h, train_w = df[train_condition].RESCALED_SHAPE.values.item()
    train_pad_values = df[train_condition].PAD_VALUES.values.item()

    target_train_h, target_train_w = df_target[train_condition].RESCALED_SHAPE.values.item()
    target_train_pad_values = df_target[train_condition].PAD_VALUES.values.item()

    val_condition = (df.CLASS == cfg.video_class) & (df.NUMBER == cfg.val.video_number_to_use)
    val_h, val_w = df[val_condition].RESCALED_SHAPE.values.item()
    val_pad_values = df[val_condition].PAD_VALUES.values.item()

    target_val_h, target_val_w = df_target[val_condition].RESCALED_SHAPE.values.item()
    target_val_pad_values = df_target[val_condition].PAD_VALUES.values.item()

    # (train_w, train_h), (val_w, val_h) = get_resize_dims(cfg)

    train_transform, val_transform = setup_transforms_for_single_instance(train_h=target_train_h,
                                                                          train_w=target_train_w,
                                                                          val_h=target_val_h,
                                                                          val_w=target_val_w)
    rgb_train_transform, rgb_val_transform = setup_transforms_for_single_instance(train_h=train_h,
                                                                                  train_w=train_w,
                                                                                  val_h=val_h,
                                                                                  val_w=val_w)
    rgb_plot_train_transform, _ = setup_transforms_for_single_instance(train_h=rgb_max_shape[0],
                                                                       train_w=rgb_max_shape[1],
                                                                       val_h=val_h,
                                                                       val_w=val_w)
    common_transform = setup_single_common_transform(use_replay_compose=cfg.using_replay_compose)

    train_dataset = setup_single_dataset_instance(cfg, train_transform, video_class=cfg.video_class,
                                                  num_videos=cfg.train.num_videos,
                                                  video_number_to_use=cfg.train.video_number_to_use,
                                                  multiple_videos=cfg.train.multiple_videos,
                                                  rgb_transform_fn=rgb_train_transform,
                                                  rgb_new_shape=(train_h, train_w),
                                                  rgb_pad_value=train_pad_values,
                                                  target_pad_value=target_train_pad_values,
                                                  rgb_plot_transform=rgb_plot_train_transform,
                                                  common_transform=common_transform,
                                                  using_replay_compose=cfg.using_replay_compose)
    val_dataset = setup_single_dataset_instance(cfg, val_transform, video_class=cfg.video_class,
                                                num_videos=cfg.val.num_videos,
                                                video_number_to_use=cfg.val.video_number_to_use,
                                                multiple_videos=cfg.val.multiple_videos,
                                                rgb_transform_fn=rgb_val_transform,
                                                rgb_new_shape=(val_h, val_w),
                                                rgb_pad_value=val_pad_values,
                                                target_pad_value=target_val_pad_values,
                                                rgb_plot_transform=rgb_plot_train_transform,
                                                common_transform=common_transform,
                                                using_replay_compose=cfg.using_replay_compose)
    return train_dataset, val_dataset, target_max_shape


def setup_single_dataset_instance(cfg, transform, video_class, num_videos, video_number_to_use, multiple_videos,
                                  rgb_transform_fn, rgb_new_shape, rgb_pad_value, target_pad_value,
                                  rgb_plot_transform, common_transform, using_replay_compose):
    dataset = SDDFrameAndAnnotationDataset(
        root=cfg.root, video_label=getattr(SDDVideoClasses, video_class),
        num_videos=num_videos, transform=transform if cfg.data_augmentation else None,
        num_workers=cfg.dataset_workers, scale=cfg.scale_factor,
        video_number_to_use=video_number_to_use,
        multiple_videos=multiple_videos,
        use_generated=cfg.use_generated_dataset,
        sigma=cfg.sigma,
        plot=cfg.plot_samples,
        desired_size=cfg.desired_size,
        heatmap_shape=cfg.heatmap_shape,
        return_combined_heatmaps=cfg.return_combined_heatmaps,
        seg_map_objectness_threshold=cfg.seg_map_objectness_threshold,
        meta_label=getattr(SDDVideoDatasets, video_class),
        heatmap_region_limit_threshold=cfg.heatmap_region_limit_threshold,
        downscale_only_target_maps=cfg.downscale_only_target_maps,
        rgb_transform=rgb_transform_fn,
        rgb_new_shape=rgb_new_shape,
        rgb_pad_value=rgb_pad_value,
        target_pad_value=target_pad_value,
        rgb_plot_transform=rgb_plot_transform,
        common_transform=common_transform,
        using_replay_compose=using_replay_compose
    )
    return dataset


def setup_transforms_for_single_instance(train_h, train_w, val_h, val_w):
    train_transform = setup_single_transform(train_h, train_w)
    val_transform = setup_single_transform(val_h, val_w)
    return train_transform, val_transform


def setup_single_transform(height, width):
    transform = A.Compose(
        [A.Resize(height=height, width=width)],
        # A.RandomBrightnessContrast(p=0.3),
        # A.RandomRotate90(p=0.3),  # possible on square images
        # A.VerticalFlip(p=0.3),
        # A.HorizontalFlip(p=0.3)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )
    return transform


def setup_single_common_transform(use_replay_compose=False):
    if use_replay_compose:
        transform = A.ReplayCompose(
            [A.RandomBrightnessContrast(p=0.3),
             A.VerticalFlip(p=0.3),
             A.HorizontalFlip(p=0.3)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            keypoint_params=A.KeypointParams(format='xy')
        )
    else:
        transform = A.Compose(
            [A.RandomBrightnessContrast(p=0.3),
             A.VerticalFlip(p=0.3),
             A.HorizontalFlip(p=0.3)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            keypoint_params=A.KeypointParams(format='xy'),
            additional_targets={'image0': 'image',
                                'keypoints0': 'keypoints',
                                'bboxes0': 'bboxes'}
        )
    return transform


def setup_multiple_datasets(cfg):
    meta = SDDMeta(cfg.root + 'H_SDD.txt')
    video_classes = cfg.train.video_classes_to_use
    video_numbers = cfg.train.video_numbers_to_use
    for val_clz, val_v_num in zip(cfg.val.video_classes_to_use, cfg.val.video_numbers_to_use):
        if val_clz in video_classes:
            idx = video_classes.index(val_clz)
            video_numbers[idx] = (list(set(video_numbers[idx] + val_v_num)))
        else:
            video_classes.append(val_clz)
            video_numbers.append(val_v_num)

    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=video_classes,
        video_numbers=video_numbers,
        desired_ratio=cfg.desired_pixel_to_meter_ratio_rgb)

    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=video_classes,
        video_numbers=video_numbers,
        desired_ratio=cfg.desired_pixel_to_meter_ratio)

    train_datasets = setup_multiple_datasets_core(cfg, meta, video_classes_to_use=cfg.train.video_classes_to_use,
                                                  video_numbers_to_use=cfg.train.video_numbers_to_use,
                                                  num_videos=cfg.train.num_videos,
                                                  multiple_videos=cfg.train.multiple_videos,
                                                  df=df, df_target=df_target, rgb_max_shape=rgb_max_shape)
    val_datasets = setup_multiple_datasets_core(cfg, meta, video_classes_to_use=cfg.val.video_classes_to_use,
                                                video_numbers_to_use=cfg.val.video_numbers_to_use,
                                                num_videos=cfg.val.num_videos,
                                                multiple_videos=cfg.val.multiple_videos,
                                                df=df, df_target=df_target, rgb_max_shape=rgb_max_shape)
    return train_datasets, val_datasets, target_max_shape


def setup_multiple_datasets_core(cfg, meta, video_classes_to_use, video_numbers_to_use, num_videos, multiple_videos,
                                 df, df_target, rgb_max_shape, use_common_transforms=True):
    datasets = []
    for idx, v_clz in enumerate(video_classes_to_use):
        for v_num in video_numbers_to_use[idx]:
            condition = (df.CLASS == v_clz) & (df.NUMBER == v_num)
            h, w = df[condition].RESCALED_SHAPE.values.item()
            pad_values = df[condition].PAD_VALUES.values.item()

            target_h, target_w = df_target[condition].RESCALED_SHAPE.values.item()
            target_pad_values = df_target[condition].PAD_VALUES.values.item()

            transform = setup_single_transform(height=target_h, width=target_w)
            rgb_transform_fn = setup_single_transform(height=h, width=w)
            rgb_plot_transform = setup_single_transform(height=rgb_max_shape[0], width=rgb_max_shape[1])
            if use_common_transforms:
                common_transform = setup_single_common_transform(use_replay_compose=cfg.using_replay_compose)
            else:
                common_transform = None

            datasets.append(setup_single_dataset_instance(cfg=cfg, transform=transform,
                                                          video_class=v_clz,
                                                          num_videos=num_videos,
                                                          video_number_to_use=v_num,
                                                          multiple_videos=multiple_videos,
                                                          rgb_transform_fn=rgb_transform_fn,
                                                          rgb_new_shape=(h, w),
                                                          rgb_pad_value=pad_values,
                                                          target_pad_value=target_pad_values,
                                                          rgb_plot_transform=rgb_plot_transform,
                                                          common_transform=common_transform,
                                                          using_replay_compose=cfg.using_replay_compose))
    return ConcatDataset(datasets)


def setup_trainer(cfg, loss_fn, model, train_dataset, val_dataset):
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.monitor,
        save_top_k=cfg.trainer.num_checkpoints_to_save,
        mode=cfg.mode,
        verbose=cfg.verbose
    )
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
                map_location=cfg.device,
                train_dataset=train_dataset, val_dataset=val_dataset,
                loss_function=loss_fn, collate_fn=heat_map_collate_fn)

        trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                          fast_dev_run=cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback])
    else:
        if cfg.resume_mode:
            checkpoint_path = f'{cfg.resume.checkpoint.path}{cfg.resume.checkpoint.version}/checkpoints/'
            checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]

            trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                              fast_dev_run=cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback],
                              resume_from_checkpoint=checkpoint_file)
        else:
            trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                              fast_dev_run=cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback])
    return model, trainer


@hydra.main(config_path="config", config_name="config")
def train(cfg):
    logger.info(f'Setting up DataLoader and Model...')

    train_dataset, val_dataset, target_max_shape = setup_dataset(cfg)

    network_type = getattr(model_zoo, cfg.postion_map_network_type)

    if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation', 'PositionMapUNetClassMapSegmentation',
                                 'PositionMapUNetHeatmapSegmentation']:
        loss_fn = BinaryFocalLossWithLogits(alpha=cfg.focal_loss_alpha, reduction='mean')  # CrossEntropyLoss()
    else:
        loss_fn = MSELoss()

    model = network_type(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                         loss_function=loss_fn, collate_fn=heat_map_collate_fn, desired_output_shape=target_max_shape)

    logger.info(f'Setting up Trainer...')

    model, trainer = setup_trainer(cfg, loss_fn, model, train_dataset, val_dataset)
    logger.info(f'Starting training...')

    trainer.fit(model)


@hydra.main(config_path="config", config_name="config")
def overfit(cfg):
    logger.info(f'Overfit - Setting up DataLoader and Model...')

    train_dataset, val_dataset, target_max_shape = setup_dataset(cfg)

    network_type = getattr(model_zoo, cfg.overfit.postion_map_network_type)
    if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation',
                                 'PositionMapUNetClassMapSegmentation',
                                 'PositionMapUNetHeatmapSegmentation',
                                 'PositionMapStackedHourGlass']:
        loss_fn = BinaryFocalLossWithLogits(alpha=cfg.overfit.focal_loss_alpha, reduction='mean')  # CrossEntropyLoss()
    else:
        loss_fn = MSELoss()

    model = network_type(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                         loss_function=loss_fn, collate_fn=heat_map_collate_fn, desired_output_shape=target_max_shape)
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

            if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                frames, class_maps = frames.to(cfg.device), class_maps.to(cfg.device)
            elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                frames, position_map = frames.to(cfg.device), position_map.to(cfg.device)
            elif network_type.__name__ in ['PositionMapUNetHeatmapSegmentation',
                                           'PositionMapStackedHourGlass']:
                frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)
            else:
                frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

            out = model(frames)

            if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                loss = loss_fn(out, class_maps.long().squeeze(dim=1))
            elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                loss = loss_fn(out, position_map.long().squeeze(dim=1))
            elif network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                loss = loss_fn(out, heat_masks)
            elif network_type.__name__ == 'PositionMapStackedHourGlass':
                loss = model.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
                loss = loss.mean()
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

                if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                    frames, class_maps = frames.to(cfg.device), class_maps.to(cfg.device)
                elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                    frames, position_map = frames.to(cfg.device), position_map.to(cfg.device)
                elif network_type.__name__ in ['PositionMapUNetHeatmapSegmentation',
                                               'PositionMapStackedHourGlass']:
                    frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)
                else:
                    frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

                with torch.no_grad():
                    out = model(frames)

                if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                    loss = loss_fn(out, class_maps.long().squeeze(dim=1))
                elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                    loss = loss_fn(out, position_map.long().squeeze(dim=1))
                elif network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                    loss = loss_fn(out, heat_masks)
                elif network_type.__name__ == 'PositionMapStackedHourGlass':
                    loss = model.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
                    loss = loss.mean()
                else:
                    loss = loss_fn(out, heat_masks)

                val_loss.append(loss.item())

                random_idx = np.random.choice(cfg.overfit.batch_size, 1, replace=False).item()

                if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation',
                                             'PositionMapUNetClassMapSegmentation']:
                    pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     class_maps[random_idx].squeeze().cpu()
                                     if cfg.class_map_segmentation else position_map[random_idx].squeeze().cpu(),
                                     pred_mask[random_idx].int() * 255,
                                     additional_text=f"{network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")
                elif network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                    pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     pred_mask[random_idx].int() * 255,
                                     additional_text=f"{network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")
                elif network_type.__name__ == 'PositionMapStackedHourGlass':
                    pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     class_maps[random_idx].squeeze().cpu()
                                     if cfg.class_map_segmentation else position_map[random_idx].squeeze().cpu(),
                                     pred_mask[-1][random_idx].int().squeeze(dim=0) * 255,
                                     additional_text=f"{network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")
                    plot_predictions(pred_mask[-3][random_idx].int().squeeze(dim=0) * 255,
                                     pred_mask[-2][random_idx].int().squeeze(dim=0) * 255,
                                     pred_mask[-1][random_idx].int().squeeze(dim=0) * 255,
                                     additional_text=f"{network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}\nLast 3 HeatMaps", all_heatmaps=True)
                else:
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     out[random_idx].squeeze().cpu(),
                                     additional_text=f"{network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")

            logger.info(f"Epoch: {epoch} | Validation Loss: {np.array(val_loss).mean()}")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        overfit()
        # train()
