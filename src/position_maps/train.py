import copy
import os
import warnings

import albumentations as A
import hydra
import numpy as np
import torch
import torchvision
from kornia.losses import FocalLoss, BinaryFocalLossWithLogits
from mmdet.models import GaussianFocalLoss
from omegaconf import ListConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, ConcatDataset

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from log import get_logger
from dataset import SDDFrameAndAnnotationDataset
import models as model_zoo
from losses import CenterNetFocalLoss
import src_lib.models_hub as hub
from src_lib.models_hub.ccnet import CCNet
from src_lib.models_hub.danet import DANet
from src_lib.models_hub.deeplab import DeepLabV3, DeepLabV3Plus, UNetDeepLabV3Plus
from src_lib.models_hub.dmnet import DMNet
from src_lib.models_hub.hrnet import HRNetwork, HRPoseNetwork
from src_lib.models_hub.msanet import MSANet
from src_lib.models_hub.ocrnet import OCRNet, OCRResNet
from src_lib.models_hub.pspnet import PSPUNet, PSPNet
from src_lib.models_hub.trans_unet import TransUNet
from src_lib.models_hub.unets import R2AttentionUNet, AttentionUNet
from src_lib.models_hub.vis_trans import VisionTransformerSegmentation
from src.position_maps.patch_utils import extract_patches_2d, reconstruct_from_patches_2d, quick_viz
from utils import heat_map_collate_fn, plot_predictions, get_scaled_shapes_with_pad_values, ImagePadder, \
    TensorDatasetForTwo, plot_predictions_v2

warnings.filterwarnings("ignore")

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
        using_replay_compose=using_replay_compose,
        manual_annotation_processing=cfg.manual_annotation_processing
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
    video_classes = copy.deepcopy(cfg.train.video_classes_to_use)
    video_numbers = copy.deepcopy(cfg.train.video_numbers_to_use)
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


def setup_single_video_dataset(cfg):
    meta = SDDMeta(cfg.root + 'H_SDD.txt')

    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.single_video_mode.video_classes_to_use,
        video_numbers=cfg.single_video_mode.video_numbers_to_use,
        desired_ratio=cfg.desired_pixel_to_meter_ratio_rgb)

    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.single_video_mode.video_classes_to_use,
        video_numbers=cfg.single_video_mode.video_numbers_to_use,
        desired_ratio=cfg.desired_pixel_to_meter_ratio)

    dataset = setup_multiple_datasets_core(
        cfg, meta, video_classes_to_use=cfg.single_video_mode.video_classes_to_use,
        video_numbers_to_use=cfg.single_video_mode.video_numbers_to_use,
        num_videos=cfg.single_video_mode.num_videos,
        multiple_videos=cfg.single_video_mode.multiple_videos,
        df=df, df_target=df_target, rgb_max_shape=rgb_max_shape)

    val_dataset_len = round(len(dataset) * cfg.single_video_mode.val_percent)
    train_indices = torch.arange(start=0, end=len(dataset) - val_dataset_len)
    val_indices = torch.arange(start=len(dataset) - val_dataset_len, end=len(dataset))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset, target_max_shape


def setup_multiple_datasets_core(cfg, meta, video_classes_to_use, video_numbers_to_use, num_videos, multiple_videos,
                                 df, df_target, rgb_max_shape, use_common_transforms=True):
    datasets = []
    for idx, v_clz in enumerate(video_classes_to_use):
        for v_num in video_numbers_to_use[idx]:
            logger.info(f"Setting up {v_clz} - {v_num}")
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

    plugins = None
    if cfg.trainer.accelerator in ['ddp', 'ddp_cpu']:
        plugins = DDPPlugin(find_unused_parameters=cfg.trainer.find_unused_parameters)

    if cfg.warm_restart.enable:
        checkpoint_root_path = f'{cfg.warm_restart.checkpoint.root}{cfg.warm_restart.checkpoint.path}' \
                               f'{cfg.warm_restart.checkpoint.version}/checkpoints/'
        hparams_path = f'{cfg.warm_restart.checkpoint.root}{cfg.warm_restart.checkpoint.path}' \
                       f'{cfg.warm_restart.checkpoint.version}/hparams.yaml'

        checkpoint_files = os.listdir(checkpoint_root_path)

        epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
        epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
        checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

        model_path = checkpoint_root_path + checkpoint_files[-cfg.warm_restart.checkpoint.top_k]
        # model_path = checkpoint_root_path + os.listdir(checkpoint_root_path)[0]
        logger.info(f'Resuming from : {model_path}')
        if cfg.warm_restart.custom_load:
            logger.info(f'Loading weights manually as custom load is {cfg.warm_restart.custom_load}')
            load_dict = torch.load(model_path, map_location=cfg.device)

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
                          fast_dev_run=cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback],
                          accelerator=cfg.trainer.accelerator, deterministic=cfg.trainer.deterministic,
                          replace_sampler_ddp=cfg.trainer.replace_sampler_ddp,
                          num_nodes=cfg.trainer.num_nodes, plugins=plugins,
                          gradient_clip_val=cfg.trainer.gradient_clip_val)
    else:
        if cfg.resume_mode:
            checkpoint_path = f'{cfg.resume.checkpoint.path}{cfg.resume.checkpoint.version}/checkpoints/'
            checkpoint_files = os.listdir(checkpoint_path)

            epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
            epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
            checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

            checkpoint_file = checkpoint_path + checkpoint_files[-cfg.resume.checkpoint.top_k]

            trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                              fast_dev_run=cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback],
                              resume_from_checkpoint=checkpoint_file, accelerator=cfg.trainer.accelerator,
                              deterministic=cfg.trainer.deterministic,
                              replace_sampler_ddp=cfg.trainer.replace_sampler_ddp,
                              num_nodes=cfg.trainer.num_nodes, plugins=plugins,
                              gradient_clip_val=cfg.trainer.gradient_clip_val)
        else:
            trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                              fast_dev_run=cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback],
                              accelerator=cfg.trainer.accelerator, deterministic=cfg.trainer.deterministic,
                              replace_sampler_ddp=cfg.trainer.replace_sampler_ddp,
                              num_nodes=cfg.trainer.num_nodes, plugins=plugins,
                              gradient_clip_val=cfg.trainer.gradient_clip_val)
    return model, trainer


@hydra.main(config_path="config", config_name="config")
def train(cfg):
    logger.info(f'Setting up DataLoader and Model...')

    # runtime config adjustments
    if cfg.trainer.accelerator in ['ddp', 'ddp_cpu']:
        cfg.num_workers = 0
        cfg.dataset_workers = 0

    # train_dataset, val_dataset, target_max_shape = setup_dataset(cfg)
    train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    network_type = getattr(model_zoo, cfg.postion_map_network_type)

    if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation', 'PositionMapUNetClassMapSegmentation',
                                 'PositionMapUNetHeatmapSegmentation']:
        loss_fn = BinaryFocalLossWithLogits(alpha=cfg.focal_loss_alpha, reduction='mean')  # CrossEntropyLoss()
    elif network_type.__name__ in ['HourGlassPositionMapNetwork', 'HourGlassPositionMapNetworkDDP']:
        loss_fn = CenterNetFocalLoss() if cfg.use_center_net_gaussian_focal_loss \
            else GaussianFocalLoss(alpha=cfg.gaussuan_focal_loss_alpha, reduction='mean')
    else:
        loss_fn = MSELoss()

    if network_type.__name__ in ['HourGlassPositionMapNetwork', 'HourGlassPositionMapNetworkDDP']:
        model = network_type.from_config(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                                         loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                                         desired_output_shape=target_max_shape)
    else:
        model = network_type(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                             loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                             desired_output_shape=target_max_shape)

    logger.info(f'Setting up Trainer...')

    model, trainer = setup_trainer(cfg, loss_fn, model, train_dataset, val_dataset)
    logger.info(f'Starting training...')

    trainer.fit(model)


def build_loss(cfg):
    loss_fn = BinaryFocalLossWithLogits(
        alpha=cfg.loss.bfl.alpha, gamma=cfg.loss.bfl.gamma, reduction=cfg.loss.reduction)
    gauss_loss_fn = CenterNetFocalLoss()
    return loss_fn, [gauss_loss_fn]


def build_model(cfg, train_dataset, val_dataset, loss_function, additional_loss_functions=(),
                collate_fn=None, desired_output_shape=None):
    return getattr(hub, cfg.model)(
        config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
        loss_function=loss_function, collate_fn=collate_fn, additional_loss_functions=additional_loss_functions,
        desired_output_shape=desired_output_shape)


@hydra.main(config_path="config", config_name="config")
def train_v1(cfg):
    logger.info(f'Setting up DataLoader and Model...')

    # runtime config adjustments
    if cfg.trainer.accelerator in ['ddp', 'ddp_cpu']:
        cfg.num_workers = 0
        cfg.dataset_workers = 0

    if cfg.single_video_mode.enabled:
        train_dataset, val_dataset, target_max_shape = setup_single_video_dataset(cfg)
    else:
        train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    loss_fn, gaussian_loss_fn = build_loss(cfg)

    model = build_model(cfg, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_fn,
                        additional_loss_functions=gaussian_loss_fn, collate_fn=heat_map_collate_fn,
                        desired_output_shape=target_max_shape)

    logger.info(f'Setting up Trainer...')

    model, trainer = setup_trainer(cfg, loss_fn, model, train_dataset, val_dataset)
    logger.info(f'Starting training...')

    trainer.fit(model)


@hydra.main(config_path="config", config_name="config")
def overfit(cfg):
    logger.info(f'Overfit - Setting up DataLoader and Model...')

    if cfg.single_video_mode.enabled:
        train_dataset, val_dataset, target_max_shape = setup_single_video_dataset(cfg)
    else:
        train_dataset, val_dataset, target_max_shape = setup_dataset(cfg)
    # train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    reduction = 'mean'
    network_type = getattr(model_zoo, cfg.overfit.postion_map_network_type)
    if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation',
                                 'PositionMapUNetClassMapSegmentation',
                                 'PositionMapUNetHeatmapSegmentation',
                                 'PositionMapStackedHourGlass',
                                 'HourGlassPositionMapNetwork']:
        # loss_fn = BinaryFocalLossWithLogits(alpha=cfg.overfit.focal_loss_alpha, reduction='mean')
        # loss_fn = GaussianFocalLoss(alpha=cfg.overfit.gaussuan_focal_loss_alpha, reduction='mean')
        loss_fn = BinaryFocalLossWithLogits(alpha=0.85, gamma=4, reduction=reduction)
    else:
        loss_fn = MSELoss()

    # bfl_fn = BinaryFocalLossWithLogits(alpha=0.9, gamma=4)
    gauss_loss_fn = CenterNetFocalLoss()
    g_weight = 0.5
    image_pad_factor = 16 if cfg.model_hub.model in ['PSPUNet', 'UNetDeepLabV3Plus'] else 8

    model_hub_models = ['DeepLabV3', 'DeepLabV3Plus', 'PSPUNet', 'DANet', 'DMNet', 'PSPNet', 'CCNet', 'HRNet', 'OCRNet',
                        'OCRResNet', 'UNetDeepLabV3Plus', 'HRPoseNetwork']
    if cfg.from_model_hub:
        if cfg.model_hub.model == 'MSANet':
            model = MSANet(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                           loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                           desired_output_shape=target_max_shape)
        elif cfg.model_hub.model == 'attn_u_net':
            model = AttentionUNet(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                                  loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                                  desired_output_shape=target_max_shape)
        elif cfg.model_hub.model == 'DeepLabV3Plus':
            model = DeepLabV3Plus(config=cfg, train_dataset=None, val_dataset=None,
                                  loss_function=loss_fn, collate_fn=None,
                                  desired_output_shape=None)
        elif cfg.model_hub.model == 'DeepLabV3':
            keys_to_remove = ['classifier.4.weight', 'classifier.4.bias',
                              'aux_classifier.4.weight', 'aux_classifier.4.bias']
            pretrained_state_dict = torch.load(
                '/home/rishabh/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth',
                map_location=cfg.device)
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k not in keys_to_remove}

            model = DeepLabV3(config=cfg, train_dataset=None, val_dataset=None,
                              loss_function=loss_fn, collate_fn=None,
                              desired_output_shape=None)
            model.net.load_state_dict(pretrained_state_dict, strict=False)
        elif cfg.model_hub.model == 'PSPUNet':
            model = PSPUNet(config=cfg, train_dataset=None, val_dataset=None,
                            loss_function=loss_fn, collate_fn=None,
                            desired_output_shape=None)
        elif cfg.model_hub.model == 'PSPNet':
            model = PSPNet(config=cfg, train_dataset=None, val_dataset=None,
                           loss_function=loss_fn, collate_fn=None,
                           desired_output_shape=None)
        elif cfg.model_hub.model == 'DANet':
            model = DANet(config=cfg, train_dataset=None, val_dataset=None,
                          loss_function=loss_fn, collate_fn=None,
                          desired_output_shape=None)
        elif cfg.model_hub.model == 'DMNet':
            model = DMNet(config=cfg, train_dataset=None, val_dataset=None,
                          loss_function=loss_fn, collate_fn=None,
                          desired_output_shape=None)
        elif cfg.model_hub.model == 'CCNet':
            model = CCNet(config=cfg, train_dataset=None, val_dataset=None,
                          loss_function=loss_fn, collate_fn=None,
                          desired_output_shape=None)
        elif cfg.model_hub.model == 'HRNet':
            model = HRNetwork(config=cfg, train_dataset=None, val_dataset=None,
                              loss_function=loss_fn, collate_fn=None,
                              desired_output_shape=None)
        elif cfg.model_hub.model == 'OCRNet':
            model = OCRNet(config=cfg, train_dataset=None, val_dataset=None,
                           loss_function=loss_fn, collate_fn=None,
                           desired_output_shape=None)
        elif cfg.model_hub.model == 'OCRResNet':
            model = OCRResNet(config=cfg, train_dataset=None, val_dataset=None,
                              loss_function=loss_fn, collate_fn=None,
                              desired_output_shape=None)
        elif cfg.model_hub.model == 'UNetDeepLabV3Plus':
            model = UNetDeepLabV3Plus(config=cfg, train_dataset=None, val_dataset=None,
                                      loss_function=loss_fn, collate_fn=None,
                                      desired_output_shape=None)
        elif cfg.model_hub.model == 'HRPoseNetwork':
            model = HRPoseNetwork(config=cfg, train_dataset=None, val_dataset=None,
                                  loss_function=loss_fn, collate_fn=None,
                                  desired_output_shape=None)
    else:
        if network_type.__name__ == 'HourGlassPositionMapNetwork':
            model = network_type.from_config(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                                             loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                                             desired_output_shape=target_max_shape)
        else:
            model = network_type(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                                 loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                                 desired_output_shape=target_max_shape)

    if cfg.overfit.use_pretrained.enabled:
        checkpoint_path = f'{cfg.overfit.use_pretrained.checkpoint.root}' \
                          f'{cfg.overfit.use_pretrained.checkpoint.path}' \
                          f'{cfg.overfit.use_pretrained.checkpoint.version}/checkpoints/'
        checkpoint_files = os.listdir(checkpoint_path)

        epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
        epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
        checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

        checkpoint_file = checkpoint_path + checkpoint_files[-cfg.overfit.use_pretrained.checkpoint.top_k]

        logger.info(f'Loading weights from: {checkpoint_file}')
        load_dict = torch.load(checkpoint_file, map_location=cfg.device)

        model.load_state_dict(load_dict['state_dict'])

    model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)
    sch = ReduceLROnPlateau(opt,
                            patience=cfg.patch_mode.patience,
                            verbose=cfg.patch_mode.verbose,
                            factor=cfg.patch_mode.factor,
                            min_lr=cfg.patch_mode.min_lr)

    if isinstance(cfg.overfit.subset_indices, (list, ListConfig)):
        indices = list(cfg.overfit.subset_indices)
    else:
        indices = np.random.choice(len(train_dataset), cfg.overfit.subset_indices, replace=False)
    train_subset = Subset(dataset=train_dataset, indices=indices)
    train_loader = DataLoader(train_subset, batch_size=cfg.overfit.batch_size, shuffle=False,
                              num_workers=cfg.overfit.num_workers, collate_fn=heat_map_collate_fn,
                              pin_memory=cfg.overfit.pin_memory, drop_last=cfg.overfit.drop_last)
    for epoch in range(cfg.overfit.num_epochs):
        model.train()

        train_loss = []
        for t_idx, data in enumerate(train_loader):
            # opt.zero_grad()

            frames, heat_masks, position_map, distribution_map, class_maps, meta = data

            if cfg.from_model_hub:
                padder = ImagePadder(frames.shape[-2:], factor=image_pad_factor)
                frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]
                frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)
            else:
                if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                    frames, class_maps = frames.to(cfg.device), class_maps.to(cfg.device)
                elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                    frames, position_map = frames.to(cfg.device), position_map.to(cfg.device)
                elif network_type.__name__ in ['PositionMapUNetHeatmapSegmentation',
                                               'PositionMapStackedHourGlass',
                                               'HourGlassPositionMapNetwork']:
                    frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)
                else:
                    frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

            out = model(frames)

            if cfg.from_model_hub:
                if cfg.model_hub.model in model_hub_models:
                    loss = getattr(torch.Tensor, reduction)(model.calculate_loss(out, heat_masks)) + \
                           getattr(torch.Tensor, reduction)(model.calculate_additional_loss(
                               gauss_loss_fn, out, heat_masks, apply_sigmoid=True, weight_factor=g_weight))
                else:
                    loss = model.calculate_loss(out, heat_masks)
            else:
                if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                    loss = loss_fn(out, class_maps.long().squeeze(dim=1))
                elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                    loss = loss_fn(out, position_map.long().squeeze(dim=1))
                elif network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                    loss = loss_fn(out, heat_masks)
                elif network_type.__name__ == 'PositionMapStackedHourGlass':
                    loss = model.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
                    loss = loss.mean()
                elif network_type.__name__ == 'HourGlassPositionMapNetwork':
                    out = model_zoo.post_process_multi_apply(out)
                    loss = model.calc_loss(out, heat_masks).mean()
                else:
                    loss = loss_fn(out, heat_masks)

            train_loss.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            # opt.step()

            if t_idx % 6 == 0 or t_idx == (len(train_loader)) - 1:
                opt.step()
                opt.zero_grad()
        sch.step(np.array(train_loss).mean())

        # if epoch % 6 == 0 or epoch == cfg.overfit.num_epochs - 1:
        #     opt.step()
        #     opt.zero_grad()
        # sch.step(np.array(train_loss).mean())

        logger.info(f"Epoch: {epoch} | Train Loss: {np.array(train_loss).mean()}")

        if epoch % cfg.overfit.plot_checkpoint == 0:
            model.eval()
            val_loss = []

            for data in train_loader:
                frames, heat_masks, position_map, distribution_map, class_maps, meta = data

                if cfg.from_model_hub:
                    padder = ImagePadder(frames.shape[-2:], factor=image_pad_factor)
                    frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]
                    frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)
                else:
                    if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                        frames, class_maps = frames.to(cfg.device), class_maps.to(cfg.device)
                    elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                        frames, position_map = frames.to(cfg.device), position_map.to(cfg.device)
                    elif network_type.__name__ in ['PositionMapUNetHeatmapSegmentation',
                                                   'PositionMapStackedHourGlass',
                                                   'HourGlassPositionMapNetwork']:
                        frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)
                    else:
                        frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

                with torch.no_grad():
                    out = model(frames)

                if cfg.from_model_hub:
                    if cfg.model_hub.model in model_hub_models:
                        loss = getattr(torch.Tensor, reduction)(model.calculate_loss(out, heat_masks)) + \
                               getattr(torch.Tensor, reduction)(model.calculate_additional_loss(
                                   gauss_loss_fn, out, heat_masks, apply_sigmoid=True, weight_factor=g_weight))
                    else:
                        loss = model.calculate_loss(out, heat_masks)
                        # loss = torch.tensor([0])  # model.calculate_loss(out, heat_masks)
                else:
                    if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                        loss = loss_fn(out, class_maps.long().squeeze(dim=1))
                    elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                        loss = loss_fn(out, position_map.long().squeeze(dim=1))
                    elif network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                        loss = loss_fn(out, heat_masks)
                    elif network_type.__name__ == 'PositionMapStackedHourGlass':
                        loss = model.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
                        loss = loss.mean()
                    elif network_type.__name__ == 'HourGlassPositionMapNetwork':
                        out = model_zoo.post_process_multi_apply(out)
                        loss = model.calc_loss(out, heat_masks).sum()
                    else:
                        loss = loss_fn(out, heat_masks)

                val_loss.append(loss.item())

                random_idx = np.random.choice(cfg.overfit.batch_size, 1, replace=False).item()

                if cfg.from_model_hub:
                    show = np.random.choice(2, 1, replace=False, p=[0.85, 0.15]).item()

                    if cfg.model_hub.model in model_hub_models:
                        if isinstance(out, (list, tuple)):
                            if cfg.model_hub.model == 'DANet':
                                out0, out1 = out
                                out = [out0[0], out1]

                            out = [o.cpu().squeeze(1) for o in out]
                            if show:
                                plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                                    heat_masks[random_idx].squeeze().cpu(),
                                                    torch.nn.functional.threshold(out[0][random_idx].sigmoid(),
                                                                                  threshold=cfg.prediction.threshold,
                                                                                  value=cfg.prediction.fill_value,
                                                                                  inplace=True),
                                                    logits_mask=out[0][random_idx].sigmoid(),
                                                    additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                                    f"| Epoch: {epoch} "
                                                                    f"| Threshold: {cfg.prediction.threshold}")
                                plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                                    heat_masks[random_idx].squeeze().cpu(),
                                                    torch.nn.functional.threshold(out[-1][random_idx].sigmoid(),
                                                                                  threshold=cfg.prediction.threshold,
                                                                                  value=cfg.prediction.fill_value,
                                                                                  inplace=True),
                                                    logits_mask=out[-1][random_idx].sigmoid(),
                                                    additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                                    f"| Epoch: {epoch} "
                                                                    f"| Threshold: {cfg.prediction.threshold}")

                                if cfg.model_hub.model == 'DeepLabV3Plus':
                                    plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                                        heat_masks[random_idx].squeeze().cpu(),
                                                        torch.nn.functional.threshold(
                                                            out[-2][random_idx].sigmoid(),
                                                            threshold=cfg.prediction.threshold,
                                                            value=cfg.prediction.fill_value,
                                                            inplace=True),
                                                        logits_mask=out[-2][random_idx].sigmoid(),
                                                        additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                                        f"| Epoch: {epoch} "
                                                                        f"| Threshold: {cfg.prediction.threshold}")
                        else:
                            out = out.cpu().squeeze(1)
                            if show:
                                plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                                    heat_masks[random_idx].squeeze().cpu(),
                                                    torch.nn.functional.threshold(out[random_idx].sigmoid(),
                                                                                  threshold=cfg.prediction.threshold,
                                                                                  value=cfg.prediction.fill_value,
                                                                                  inplace=True),
                                                    logits_mask=out[random_idx].sigmoid(),
                                                    additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                                    f"| Epoch: {epoch} "
                                                                    f"| Threshold: {cfg.prediction.threshold}")

                    else:
                        pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                        plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                         heat_masks[random_idx].squeeze().cpu(),
                                         pred_mask[random_idx].int() * 255,
                                         additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                         f"| Epoch: {epoch}")
                else:
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
                    elif network_type.__name__ == 'HourGlassPositionMapNetwork':
                        pred_mask = [torch.round(torch.sigmoid(o)).squeeze(dim=1).cpu() for o in out]
                        # pred_mask = [o.squeeze(dim=1).cpu() for o in out]
                        plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                         heat_masks[random_idx].squeeze().cpu(),
                                         pred_mask[-1][random_idx].int().squeeze(dim=0) * 255,
                                         additional_text=f"{network_type.__name__} | {loss_fn._get_name()} "
                                                         f"| Epoch: {epoch}")
                        plot_predictions(heat_masks[random_idx].squeeze().cpu(),
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


@hydra.main(config_path="config", config_name="config")
def patch_based_overfit(cfg):
    logger.info(f'Patch Based Overfit - Setting up DataLoader and Model...')

    train_dataset, val_dataset, target_max_shape = setup_dataset(cfg)
    # train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    loss_fn = CenterNetFocalLoss()
    bfl_fn = BinaryFocalLossWithLogits(alpha=0.9, gamma=4)

    if cfg.patch_mode.model == 'MSANet':
        model = MSANet(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                       loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                       desired_output_shape=target_max_shape)
    elif cfg.patch_mode.model == 'TransUNet':
        model = TransUNet(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                          loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                          desired_output_shape=target_max_shape)
    elif cfg.patch_mode.model == 'ViT':
        model = VisionTransformerSegmentation(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                                              loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                                              desired_output_shape=target_max_shape)
    elif cfg.patch_mode.model == 'r2_attn_u_net':
        model = R2AttentionUNet(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                                loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                                desired_output_shape=target_max_shape)
    elif cfg.patch_mode.model == 'attn_u_net':
        model = AttentionUNet(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                              loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                              desired_output_shape=target_max_shape)
    elif cfg.patch_mode.model == 'HourGlass':
        model = model_zoo.HourGlassPositionMapNetwork.from_config(
            config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
            loss_function=loss_fn, collate_fn=heat_map_collate_fn,
            desired_output_shape=None)
    else:
        raise NotImplementedError

    if cfg.patch_mode.use_pretrained.enabled:
        checkpoint_path = f'{cfg.patch_mode.use_pretrained.checkpoint.root}' \
                          f'{cfg.patch_mode.use_pretrained.checkpoint.path}' \
                          f'{cfg.patch_mode.use_pretrained.checkpoint.version}/checkpoints/'
        checkpoint_files = os.listdir(checkpoint_path)

        epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
        epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
        checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

        checkpoint_file = checkpoint_path + checkpoint_files[-cfg.patch_mode.use_pretrained.checkpoint.top_k]

        logger.info(f'Loading weights from: {checkpoint_file}')
        load_dict = torch.load(checkpoint_file, map_location=cfg.device)

        model.load_state_dict(load_dict['state_dict'])

    model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)

    sch = ReduceLROnPlateau(opt,
                            patience=cfg.patch_mode.patience,
                            verbose=cfg.patch_mode.verbose,
                            factor=cfg.patch_mode.factor,
                            min_lr=cfg.patch_mode.min_lr)

    train_subset = Subset(dataset=train_dataset, indices=list(cfg.patch_mode.subset_indices))
    train_loader = DataLoader(train_subset, batch_size=cfg.patch_mode.batch_size, shuffle=False,
                              num_workers=cfg.patch_mode.num_workers, collate_fn=heat_map_collate_fn,
                              pin_memory=cfg.patch_mode.pin_memory, drop_last=cfg.patch_mode.drop_last)
    for epoch in range(cfg.patch_mode.num_epochs):
        model.train()

        train_loss = []
        for data in train_loader:
            opt.zero_grad()

            frames, heat_masks, position_map, distribution_map, class_maps, meta = data

            padder = ImagePadder(frames.shape[-2:])
            frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]

            # frames_whole, heat_masks_whole = frames.clone(), heat_masks.clone()

            frames = extract_patches_2d(frames, cfg.patch_mode.patch_size, batch_first=True)
            heat_masks = extract_patches_2d(heat_masks, cfg.patch_mode.patch_size, batch_first=True)
            # viz
            # random_idx = np.random.choice(cfg.patch_mode.batch_size, 1, replace=False).item()

            # stitched_frames = reconstruct_from_patches_2d(
            #     frames, frames_whole.shape[-2:], batch_first=True)
            # stitched_masks = reconstruct_from_patches_2d(
            #     heat_masks, heat_masks_whole.shape[-2:], batch_first=True)
            # 
            # p_frames = torchvision.utils.make_grid(
            #     frames[random_idx].squeeze(0), nrow=frames_whole.shape[2] // cfg.patch_mode.patch_size[0])
            # p_masks = torchvision.utils.make_grid(
            #     heat_masks[random_idx].squeeze(0), nrow=heat_masks_whole.shape[2] // cfg.patch_mode.patch_size[0])

            # quick_viz(p_frames)
            # quick_viz(frames_whole, stitched_frames)
            #
            # quick_viz(p_masks)
            # quick_viz(heat_masks_whole, stitched_masks)

            # reshape for input and loss
            frames = frames.contiguous().view(-1, *frames.shape[2:]).float()
            heat_masks = heat_masks.contiguous().view(-1, *heat_masks.shape[2:]).float()

            # filter out null tiles
            # valid_idx = [idx for idx, h in enumerate(heat_masks) if h.max() > 0]
            # frames = frames[valid_idx]
            # heat_masks = heat_masks[valid_idx]

            frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

            # out = model(frames)
            #
            # loss = model.calculate_loss(out, heat_masks)
            #
            # train_loss.append(loss.item())

            # till = frames.shape[0] - cfg.patch_mode.mini_batch_size + 1 \
            #     if frames.shape[0] % 2 == 0 or cfg.patch_mode.mini_batch_size == 1 \
            #     else frames.shape[0] - cfg.patch_mode.mini_batch_size + 2
            till = (frames.shape[0] + cfg.patch_mode.mini_batch_size) - \
                   (frames.shape[0] % cfg.patch_mode.mini_batch_size) - cfg.patch_mode.mini_batch_size \
                if frames.shape[0] % cfg.patch_mode.mini_batch_size == 0 \
                else (frames.shape[0] + cfg.patch_mode.mini_batch_size) - \
                     (frames.shape[0] % cfg.patch_mode.mini_batch_size)
            for f_idx in range(0, till, cfg.patch_mode.mini_batch_size):
                out = model(frames[f_idx: f_idx + cfg.patch_mode.mini_batch_size])

                if cfg.patch_mode.model == 'HourGlass':
                    out = model_zoo.post_process_multi_apply(out)

                if isinstance(out, (tuple, list)):
                    out = out
                else:
                    out = out.sigmoid()
                loss = model.calculate_loss(out, heat_masks[f_idx: f_idx + cfg.patch_mode.mini_batch_size]) + \
                       bfl_fn(out, heat_masks[f_idx: f_idx + cfg.patch_mode.mini_batch_size]).abs().sum()

                if cfg.patch_mode.model == 'HourGlass':
                    loss = loss.mean()

                train_loss.append(loss.item())

                loss.backward()
            opt.step()
            sch.step(np.array(train_loss).mean())

            # loss.backward()
            # opt.step()

        logger.info(f"Epoch: {epoch} | Train Loss: {np.array(train_loss).mean()}")

        if epoch % cfg.patch_mode.plot_checkpoint == 0:
            model.eval()
            val_loss = []

            for data in train_loader:
                frames, heat_masks, position_map, distribution_map, class_maps, meta = data

                padder = ImagePadder(frames.shape[-2:])
                frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]

                frames_whole, heat_masks_whole = frames.clone(), heat_masks.clone()

                frames = extract_patches_2d(frames, cfg.patch_mode.patch_size, batch_first=True)
                heat_masks = extract_patches_2d(heat_masks, cfg.patch_mode.patch_size, batch_first=True)
                # viz
                random_idx = np.random.choice(cfg.patch_mode.batch_size, 1, replace=False).item()

                # stitched_frames = reconstruct_from_patches_2d(
                #     frames, frames_whole.shape[-2:], batch_first=True)
                # stitched_masks = reconstruct_from_patches_2d(
                #     heat_masks, heat_masks_whole.shape[-2:], batch_first=True)
                # 
                # p_frames = torchvision.utils.make_grid(
                #     frames[random_idx].squeeze(0), nrow=frames_whole.shape[2] // cfg.patch_mode.patch_size[0])
                # p_masks = torchvision.utils.make_grid(
                #     heat_masks[random_idx].squeeze(0), nrow=heat_masks_whole.shape[2] // cfg.patch_mode.patch_size[0])

                # quick_viz(p_frames)
                # quick_viz(frames_whole, stitched_frames)
                #
                # quick_viz(p_masks)
                # quick_viz(heat_masks_whole, stitched_masks)

                # reshape for input and loss
                frames = frames.contiguous().view(-1, *frames.shape[2:]).float()
                heat_masks = heat_masks.contiguous().view(-1, *heat_masks.shape[2:]).float()

                # filter out null tiles
                # valid_idx = [idx for idx, h in enumerate(heat_masks) if h.max() > 0]
                # frames = frames[valid_idx]
                # heat_masks = heat_masks[valid_idx]

                frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

                # with torch.no_grad():
                #     out = model(frames)

                outs = []
                till = (frames.shape[0] + cfg.patch_mode.mini_batch_size) - \
                       (frames.shape[0] % cfg.patch_mode.mini_batch_size) - cfg.patch_mode.mini_batch_size \
                    if frames.shape[0] % cfg.patch_mode.mini_batch_size == 0 \
                    else (frames.shape[0] + cfg.patch_mode.mini_batch_size) - \
                         (frames.shape[0] % cfg.patch_mode.mini_batch_size)
                with torch.no_grad():
                    for f_idx in range(
                            0, till, cfg.patch_mode.mini_batch_size):
                        out = model(frames[f_idx: f_idx + cfg.patch_mode.mini_batch_size])

                        # view_dim0 = frames.shape[0] % cfg.patch_mode.mini_batch_size \
                        #     if frames.shape[0] == till and frames.shape[0] % 2 != 0 \
                        #     else cfg.patch_mode.mini_batch_size
                        view_dim0 = out.shape[0]
                        if cfg.patch_mode.model == 'HourGlass':
                            out = model_zoo.post_process_multi_apply(out)
                            outs.append(
                                [o.view(view_dim0, -1, 1, *frames.shape[2:]).cpu() for o in out])
                        else:
                            outs.append(out.view(view_dim0, -1, 1, *frames.shape[2:]).cpu())

                        if isinstance(out, (tuple, list)):
                            out = out
                        else:
                            out = out.sigmoid()

                        if cfg.patch_mode.model == 'MSANet':
                            loss = torch.tensor([0])
                        else:
                            loss = model.calculate_loss(
                                out, heat_masks[f_idx: f_idx + cfg.patch_mode.mini_batch_size]) + \
                                   bfl_fn(out, heat_masks[f_idx: f_idx + cfg.patch_mode.mini_batch_size]).abs().sum()

                        if cfg.patch_mode.model == 'HourGlass':
                            loss = loss.mean()

                        val_loss.append(loss.item())

                # val_loss.append(loss.item())

                if cfg.patch_mode.model == 'HourGlass':
                    out1, out2 = [], []
                    for o in outs:
                        out1.append(o[0])
                        out2.append(o[1])

                    pred_mask1 = torch.sigmoid(torch.cat(out1, dim=1)).cpu()
                    stitched_pred_masks1 = reconstruct_from_patches_2d(
                        pred_mask1, heat_masks_whole.shape[-2:], batch_first=True).squeeze(dim=1)

                    pred_mask2 = torch.sigmoid(torch.cat(out2, dim=1)).cpu()
                    stitched_pred_masks2 = reconstruct_from_patches_2d(
                        pred_mask2, heat_masks_whole.shape[-2:], batch_first=True).squeeze(dim=1)

                    quick_viz(stitched_pred_masks1, stitched_pred_masks2)

                    pred_mask1 = torch.round(torch.sigmoid(torch.cat(out1, dim=1))).cpu()
                    stitched_pred_masks1 = reconstruct_from_patches_2d(
                        pred_mask1, heat_masks_whole.shape[-2:], batch_first=True).squeeze(dim=1)

                    pred_mask2 = torch.round(torch.sigmoid(torch.cat(out2, dim=1))).cpu()
                    stitched_pred_masks2 = reconstruct_from_patches_2d(
                        pred_mask2, heat_masks_whole.shape[-2:], batch_first=True).squeeze(dim=1)

                    quick_viz(stitched_pred_masks1, stitched_pred_masks2)

                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     stitched_pred_masks2[random_idx].int() * 255,
                                     additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")
                else:
                    # pred_mask0 = torch.sigmoid(torch.cat(outs, dim=1)).cpu()
                    pred_mask0 = torch.sigmoid(torch.cat(outs, dim=0)).cpu().transpose(0, 1)

                    # pred_mask = torch.round(torch.sigmoid(torch.cat(outs, dim=1))).cpu()
                    pred_mask = torch.round(torch.sigmoid(torch.cat(outs, dim=0))).cpu().transpose(0, 1)

                    stitched_pred_masks0 = reconstruct_from_patches_2d(
                        pred_mask0, heat_masks_whole.shape[-2:], batch_first=True).squeeze(dim=1)

                    stitched_pred_masks = reconstruct_from_patches_2d(
                        pred_mask, heat_masks_whole.shape[-2:], batch_first=True).squeeze(dim=1)
                    quick_viz(stitched_pred_masks0, stitched_pred_masks)

                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     stitched_pred_masks[random_idx].int() * 255,
                                     additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")

            logger.info(f"Epoch: {epoch} | Validation Loss: {np.array(val_loss).mean()}")


@hydra.main(config_path="config", config_name="config")
def selected_patch_overfit(cfg):
    patches = torch.load('patch_data_256_8_imgs.pt')

    # patches = {'frames': [], 'masks': []}
    # for stuff in os.listdir(os.getcwd()):
    #     if stuff.startswith('patch'):
    #         item = torch.load(stuff)
    #         patches['frames'].append(item['frames'])
    #         patches['masks'].append(item['masks'])
    #
    # patches['frames'] = torch.cat(patches['frames'], dim=0)
    # patches['masks'] = torch.cat(patches['masks'], dim=0)

    f_patches, m_patches = patches['frames'], patches['masks']

    # selected_patches = [17]  # [24]

    # filter out null tiles
    selected_patches = [idx for idx, h in enumerate(m_patches) if h.max() > 0.5]

    f_patches, m_patches = f_patches[selected_patches].to('cpu'), m_patches[selected_patches].to('cpu')
    dataset = TensorDatasetForTwo(f_patches, m_patches)

    logger.info(f'Mini Patch Based Overfit - Setting up DataLoader and Model...')

    # loss_fn = CenterNetFocalLoss()
    gauss_loss_fn = CenterNetFocalLoss()
    loss_fn = BinaryFocalLossWithLogits(alpha=0.8, gamma=4)

    if cfg.patch_mode.model == 'MSANet':
        model = MSANet(config=cfg, train_dataset=None, val_dataset=None,
                       loss_function=loss_fn, collate_fn=None,
                       desired_output_shape=None)
    elif cfg.patch_mode.model == 'TransUNet':
        model = TransUNet(config=cfg, train_dataset=None, val_dataset=None,
                          loss_function=loss_fn, collate_fn=None,
                          desired_output_shape=None)
    elif cfg.patch_mode.model == 'ViT':
        model = VisionTransformerSegmentation(config=cfg, train_dataset=None, val_dataset=None,
                                              loss_function=loss_fn, collate_fn=None,
                                              desired_output_shape=None)
    elif cfg.patch_mode.model == 'r2_attn_u_net':
        model = R2AttentionUNet(config=cfg, train_dataset=None, val_dataset=None,
                                loss_function=loss_fn, collate_fn=None,
                                desired_output_shape=None)
    elif cfg.patch_mode.model == 'attn_u_net':
        model = AttentionUNet(config=cfg, train_dataset=None, val_dataset=None,
                              loss_function=loss_fn, collate_fn=None,
                              desired_output_shape=None)
    elif cfg.patch_mode.model == 'DeepLabV3Plus':
        model = DeepLabV3Plus(config=cfg, train_dataset=None, val_dataset=None,
                              loss_function=loss_fn, collate_fn=None,
                              desired_output_shape=None)
    elif cfg.patch_mode.model == 'HourGlass':
        model = model_zoo.HourGlassPositionMapNetwork.from_config(
            config=cfg, train_dataset=None, val_dataset=None,
            loss_function=loss_fn, collate_fn=None,
            desired_output_shape=None)
    else:
        raise NotImplementedError

    model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)

    sch = ReduceLROnPlateau(opt,
                            patience=cfg.patch_mode.patience,
                            verbose=cfg.patch_mode.verbose,
                            factor=cfg.patch_mode.factor,
                            min_lr=cfg.patch_mode.min_lr)

    g_weight = 0.5

    train_loader = DataLoader(dataset, batch_size=cfg.patch_mode.batch_size, shuffle=False,
                              num_workers=cfg.patch_mode.num_workers, collate_fn=None,
                              pin_memory=cfg.patch_mode.pin_memory, drop_last=cfg.patch_mode.drop_last)
    interpolate_size = (128, 128)  # for cam and pam
    for epoch in range(cfg.patch_mode.num_epochs):
        model.train()
        train_loss = []

        for t_idx, data in enumerate(train_loader):
            # opt.zero_grad()

            frames, mask = data

            frames = interpolate(frames, size=interpolate_size)
            mask = interpolate(mask, size=interpolate_size)

            frames, mask = frames.to(cfg.device), mask.to(cfg.device)

            out = model(frames)

            if cfg.patch_mode.model == 'HourGlass':
                out = model_zoo.post_process_multi_apply(out)
                loss = model.calculate_loss(out, mask).abs().sum()  # .mean()
            elif cfg.patch_mode.model == 'MSANet':
                loss = model.calculate_loss(out, mask)
            elif cfg.patch_mode.model in ['DeepLabV3', 'DeepLabV3Plus']:
                loss = model.calculate_loss(out, mask).abs().sum() + \
                       torch.stack([g_weight * gauss_loss_fn(o.sigmoid(), mask) for o in out]).sum()
            else:
                # loss = model.calculate_loss(out.sigmoid(), mask)
                loss = model.calculate_loss(out, mask).abs().sum() + (g_weight * gauss_loss_fn(out.sigmoid(), mask))

            train_loss.append(loss.item())

            loss.backward()
            if t_idx % 6 == 0 or t_idx == len(train_loader) - 1:
                opt.step()
                opt.zero_grad()

        sch.step(np.array(train_loss).mean())

        logger.info(f"Epoch: {epoch} | Train Loss: {np.array(train_loss).mean()}")

        if epoch % cfg.patch_mode.plot_checkpoint == 0:
            model.eval()
            val_loss = []

            for data in train_loader:
                frames, mask = data

                frames = interpolate(frames, size=interpolate_size)
                mask = interpolate(mask, size=interpolate_size)

                frames, mask = frames.to(cfg.device), mask.to(cfg.device)

                with torch.no_grad():
                    out = model(frames)

                if cfg.patch_mode.model == 'HourGlass':
                    out = model_zoo.post_process_multi_apply(out)
                    loss = model.calculate_loss(out, mask).abs().sum()  # .mean()
                elif cfg.patch_mode.model == 'MSANet':
                    loss = torch.tensor([0])
                elif cfg.patch_mode.model in ['DeepLabV3', 'DeepLabV3Plus']:
                    loss = model.calculate_loss(out, mask).abs().sum() + \
                           torch.stack([g_weight * gauss_loss_fn(o.sigmoid(), mask) for o in out]).sum()
                else:
                    # loss = model.calculate_loss(out.sigmoid(), mask)
                    loss = model.calculate_loss(out, mask).abs().sum() + (g_weight * gauss_loss_fn(out.sigmoid(), mask))

                val_loss.append(loss.item())

                random_idx = np.random.choice(frames.shape[0], 1, replace=False).item()

                if cfg.patch_mode.model in ['HourGlass', 'DeepLabV3', 'DeepLabV3Plus']:
                    out = [o.cpu().squeeze(1) for o in out]
                    # plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                    #                     mask[random_idx].squeeze().cpu(),
                    #                     out[0][random_idx].sigmoid().round(),
                    #                     logits_mask=out[0][random_idx].sigmoid(),
                    #                     additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                    #                                     f"| Epoch: {epoch}")
                    # plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                    #                     mask[random_idx].squeeze().cpu(),
                    #                     out[-1][random_idx].sigmoid().round(),
                    #                     logits_mask=out[-1][random_idx].sigmoid(),
                    #                     additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                    #                                     f"| Epoch: {epoch}")
                else:
                    out = out.cpu().squeeze(1)
                    # plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                    #                     mask[random_idx].squeeze().cpu(),
                    #                     out[random_idx].sigmoid().round(),
                    #                     logits_mask=out[random_idx].sigmoid(),
                    #                     additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                    #                                     f"| Epoch: {epoch}")

            logger.info(f"Epoch: {epoch} | Validation Loss: {np.array(val_loss).mean()}")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # selected_patch_overfit()
        # patch_based_overfit()
        # overfit()
        # train()
        train_v1()
