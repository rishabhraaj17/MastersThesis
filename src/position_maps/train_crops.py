import os
import warnings
from typing import Union, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast, ShiftScaleRotate
from omegaconf import ListConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader, Dataset, random_split
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from baselinev2.exceptions import TimeoutException
from baselinev2.improve_metrics.crop_utils import sample_random_crops, replace_overlapping_boxes, REPLACEMENT_TIMEOUT
from baselinev2.improve_metrics.model import plot_predictions, make_datasets_simple, people_collate_fn, \
    people_collate_fn_tuple
from baselinev2.plot_utils import add_box_to_axes
from baselinev2.utils import get_bbox_center
from log import get_logger
from src.position_maps.classic_nn_mixin import DATASET_TO_CROP_MODEL
from src.position_maps.location_utils import locations_from_heatmaps, \
    get_adjusted_object_locations_rgb, get_processed_patches_to_train_rgb_only
from src_lib.models_hub.crop_classifiers import CropClassifier, CropClassifierDDP
from train import setup_single_video_dataset, setup_multiple_datasets, build_model, build_loss
from utils import heat_map_collate_fn, ImagePadder

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


class CropsTensorDataset(Dataset):
    def __init__(self, path: Union[List[str], str]):
        self.path = path
        if isinstance(self.path, (list, tuple)):
            self.gt_crops, self.gt_labels = [], []
            self.tn_crops, self.tn_labels = [], []
            for p in path:
                gt_data, tn_data = torch.load(p)

                self.gt_crops.append(gt_data['images'])
                self.gt_labels.append(gt_data['labels'])

                self.tn_crops.append(tn_data['images'])
                self.tn_labels.append(torch.zeros_like(tn_data['labels']))

            self.gt_crops = torch.cat(self.gt_crops)
            self.gt_labels = torch.cat(self.gt_labels)
            self.tn_crops = torch.cat(self.tn_crops)
            self.tn_labels = torch.cat(self.tn_labels)
        else:
            gt_data, tn_data = torch.load(self.path)

            self.gt_crops = gt_data['images']
            self.gt_labels = gt_data['labels']

            self.tn_crops = tn_data['images']
            self.tn_labels = torch.zeros_like(tn_data['labels'])

    def __len__(self):
        return len(self.gt_crops)

    def __getitem__(self, item):
        crops = torch.stack((self.gt_crops[item], self.tn_crops[item]))
        labels = torch.stack((self.gt_labels[item], self.tn_labels[item]))
        return crops, labels


@hydra.main(config_path="config", config_name="config")
def train_crop_classifier(cfg):
    # todo: train with augmentations
    # path = 'death_circle_crops0_1.pt'
    # path = os.path.join(os.getcwd(), path)

    path = ['death_circle_crops0_1.pt', 'death_circle_crops_2_3_4.pt']
    path = [os.path.join(os.getcwd(), p) for p in path]

    val_ratio = 0.2

    dataset = CropsTensorDataset(path=path)
    val_len = int(len(dataset) * val_ratio)
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_len, val_len], generator=torch.Generator().manual_seed(42))

    model = CropClassifier(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset, desired_output_shape=None,
                           loss_function=nn.BCEWithLogitsLoss())

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.crop_classifier.monitor,
        save_top_k=cfg.crop_classifier.num_checkpoints_to_save,
        mode=cfg.crop_classifier.mode,
        verbose=cfg.crop_classifier.verbose
    )

    trainer = Trainer(max_epochs=cfg.crop_classifier.num_epochs, gpus=1,
                      fast_dev_run=False, callbacks=[checkpoint_callback],
                      deterministic=True, gradient_clip_val=2.0,
                      accumulate_grad_batches=1)
    trainer.fit(model)


@hydra.main(config_path="config", config_name="config")
def train_crop_classifier_v1(cfg):
    crop_cfg = cfg.crop_classifier
    img_t = Compose([
        HorizontalFlip(p=0.4),
        VerticalFlip(p=0.4),
        Rotate(p=0.4),
        RandomBrightnessContrast(p=0.2),
        ShiftScaleRotate(p=0.2),
    ], p=0.7)

    logger.info(f'Setting up datasets...')
    video_class = getattr(SDDVideoClasses, crop_cfg.dataset.video_class)
    logger.info(f'Video Class: {video_class.name}')
    train_dataset, val_dataset = make_datasets_simple(
        crop_cfg, video_class=video_class, return_test_split=False, plot=False,
        server_mode=crop_cfg.server_mode,
        transforms=img_t if crop_cfg.data_augmentation else None)
    logger.info(f'Setting up model...')

    if crop_cfg.trainer.accelerator in ['ddp', 'ddp_cpu']:
        model = CropClassifierDDP(
            config=cfg, train_dataset=train_dataset, val_dataset=val_dataset, desired_output_shape=None,
            loss_function=nn.BCEWithLogitsLoss(), collate_fn=people_collate_fn_tuple)
    else:
        model = CropClassifier(
            config=cfg, train_dataset=train_dataset, val_dataset=val_dataset, desired_output_shape=None,
            loss_function=nn.BCEWithLogitsLoss(), collate_fn=people_collate_fn_tuple)

    checkpoint_callback = ModelCheckpoint(
        monitor=crop_cfg.monitor,
        save_top_k=crop_cfg.num_checkpoints_to_save,
        mode=crop_cfg.mode,
        verbose=crop_cfg.verbose
    )

    train_logger = WandbLogger(project=crop_cfg.project_name, name=f'{video_class.name}_Classifier')
    train_logger.log_hyperparams(cfg)
    # train_logger = False

    sync_bn = False
    plugins = None
    if crop_cfg.trainer.accelerator in ['ddp', 'ddp_cpu']:
        plugins = DDPPlugin(find_unused_parameters=crop_cfg.trainer.find_unused_parameters)
        sync_bn = True

    if crop_cfg.warm_restart.enable:
        version_name = f"{crop_cfg.warm_restart.checkpoint.run_name}".split('-')[-1]
        checkpoint_root_path = f'{crop_cfg.warm_restart.checkpoint.root}' \
                               f'{crop_cfg.warm_restart.checkpoint.run_name}' \
                               f'{crop_cfg.warm_restart.checkpoint.tail_path}' \
                               f'{crop_cfg.warm_restart.checkpoint.project_name}/' \
                               f'{version_name}/checkpoints/'

        checkpoint_files = os.listdir(checkpoint_root_path)

        epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
        epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
        checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

        model_path = checkpoint_root_path + checkpoint_files[-cfg.warm_restart.checkpoint.top_k]

        if crop_cfg.warm_restart.custom_load:
            logger.info(f'Loading weights manually as custom load is {cfg.warm_restart.custom_load}')
            load_dict = torch.load(model_path, map_location=cfg.device)

            model.load_state_dict(load_dict['state_dict'])
            model.to(cfg.device)
            model.train()

            trainer = Trainer(max_epochs=crop_cfg.trainer.num_epochs, gpus=crop_cfg.trainer.gpus,
                              fast_dev_run=crop_cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback],
                              accelerator=crop_cfg.trainer.accelerator, deterministic=crop_cfg.trainer.deterministic,
                              replace_sampler_ddp=crop_cfg.trainer.replace_sampler_ddp,
                              num_nodes=crop_cfg.trainer.num_nodes, plugins=plugins,
                              gradient_clip_val=crop_cfg.trainer.gradient_clip_val,
                              sync_batchnorm=sync_bn,
                              accumulate_grad_batches=crop_cfg.trainer.accumulate_grad_batches,
                              logger=train_logger)
        else:
            trainer = Trainer(max_epochs=crop_cfg.trainer.num_epochs, gpus=crop_cfg.trainer.gpus,
                              fast_dev_run=crop_cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback],
                              accelerator=crop_cfg.trainer.accelerator, deterministic=crop_cfg.trainer.deterministic,
                              replace_sampler_ddp=crop_cfg.trainer.replace_sampler_ddp,
                              num_nodes=crop_cfg.trainer.num_nodes, plugins=plugins,
                              gradient_clip_val=crop_cfg.trainer.gradient_clip_val,
                              sync_batchnorm=sync_bn,
                              accumulate_grad_batches=crop_cfg.trainer.accumulate_grad_batches,
                              logger=train_logger, resume_from_checkpoint=model_path)
    else:
        trainer = Trainer(max_epochs=crop_cfg.trainer.num_epochs, gpus=crop_cfg.trainer.gpus,
                          fast_dev_run=crop_cfg.trainer.fast_dev_run, callbacks=[checkpoint_callback],
                          accelerator=crop_cfg.trainer.accelerator, deterministic=crop_cfg.trainer.deterministic,
                          replace_sampler_ddp=crop_cfg.trainer.replace_sampler_ddp,
                          num_nodes=crop_cfg.trainer.num_nodes, plugins=plugins,
                          gradient_clip_val=crop_cfg.trainer.gradient_clip_val,
                          sync_batchnorm=sync_bn,
                          accumulate_grad_batches=crop_cfg.trainer.accumulate_grad_batches,
                          logger=train_logger)
    trainer.fit(model)


@hydra.main(config_path="config", config_name="config")
def eval_crop_classifier(cfg):
    path = ['death_circle_crops0_1.pt', 'death_circle_crops_2_3_4.pt']
    path = [os.path.join(os.getcwd(), p) for p in path]

    val_ratio = 0.2

    dataset = CropsTensorDataset(path=path)
    val_len = int(len(dataset) * val_ratio)
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_len, val_len], generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.crop_classifier.batch_size * 2, shuffle=cfg.crop_classifier.shuffle,
        num_workers=cfg.crop_classifier.num_workers * 0, collate_fn=None,
        pin_memory=cfg.crop_classifier.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    model = CropClassifier(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset, desired_output_shape=None,
                           loss_function=nn.BCEWithLogitsLoss())
    model.load_state_dict(
        torch.load('lightning_logs/version_513924/checkpoints/epoch=27-step=21531.ckpt')['state_dict'])
    model.cuda()

    val_loss, accuracies = [], []
    for b_idx, data in enumerate(tqdm(val_loader)):
        crops, labels = data
        crops, labels = crops.view(-1, *crops.shape[2:]).cuda(), labels.view(-1, 1).cuda()
        out = model(crops)

        loss = model.calculate_loss(out, labels)
        val_loss.append(loss.item())

        acc = out.sigmoid().round().eq(labels).float().mean()
        accuracies.append(acc.item())

        if acc.item() < 0.96:
            plot_predictions(labels, crops, out.sigmoid().round(), b_idx)

    logger.info(f"Test Loss: {np.array(val_loss).mean()} | Accuracy: {np.array(accuracies).mean()}")


@hydra.main(config_path="config", config_name="config")
def eval_crop_classifier_v1(cfg):
    crop_cfg = cfg.crop_classifier
    train_dataset, val_dataset = make_datasets_simple(
        crop_cfg, video_class=SDDVideoClasses.BOOKSTORE, return_test_split=False, plot=False,
        server_mode=False,
        transforms=None)

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.crop_classifier.batch_size, shuffle=cfg.crop_classifier.shuffle,
        num_workers=cfg.crop_classifier.num_workers * 0, collate_fn=people_collate_fn_tuple,
        pin_memory=cfg.crop_classifier.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    model = CropClassifier(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset, desired_output_shape=None,
                           loss_function=nn.BCEWithLogitsLoss())
    checkpoint_root_path = f'{DATASET_TO_CROP_MODEL[SDDVideoClasses.BOOKSTORE]}'
    checkpoint_files = os.listdir(checkpoint_root_path)

    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

    model_path = checkpoint_root_path + checkpoint_files[-1]

    load_dict = torch.load(model_path, map_location='cuda:0')

    model.load_state_dict(load_dict['state_dict'])
    model.to('cuda:0')
    model.eval()

    val_loss, accuracies = [], []
    for b_idx, data in enumerate(tqdm(val_loader)):
        crops, labels = data
        crops, labels = crops.cuda(), labels.view(-1, 1).cuda()
        out = model(crops)

        pred_labels = out.sigmoid().round()

        loss = model.calculate_loss(out, labels)
        val_loss.append(loss.item())

        acc = out.sigmoid().round().eq(labels).float().mean()
        accuracies.append(acc.item())

        if acc.item() < 0.96:
            plot_predictions(labels, crops, out.sigmoid().round(), b_idx)

    logger.info(f"Test Loss: {np.array(val_loss).mean()} | Accuracy: {np.array(accuracies).mean()}")


@hydra.main(config_path="config", config_name="config")
def generate_crops_v0(cfg):
    logger.info(f'Patch Classifier')
    logger.info(f'Setting up DataLoader and Model...')

    # adjust config here
    cfg.device = 'cpu'  # 'cuda:0'
    cfg.single_video_mode.enabled = False
    cfg.preproccesing.pad_factor = 8
    cfg.frame_rate = 20
    cfg.video_based.enabled = False

    if cfg.single_video_mode.enabled:
        # config adapt
        cfg.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
        cfg.single_video_mode.video_numbers_to_use = [[4]]
        cfg.desired_pixel_to_meter_ratio_rgb = 0.07
        cfg.desired_pixel_to_meter_ratio = 0.07

        train_dataset, val_dataset, target_max_shape = setup_single_video_dataset(cfg, use_common_transforms=False)
    else:
        cfg.train.video_classes_to_use = ['DEATH_CIRCLE']
        cfg.train.video_numbers_to_use = [[2, 3, 4]]
        cfg.desired_pixel_to_meter_ratio_rgb = 0.07
        cfg.desired_pixel_to_meter_ratio = 0.07
        cfg.frame_rate = 1.

        train_dataset, val_dataset, target_max_shape = setup_multiple_datasets(cfg)

    # loss config params are ok!
    loss_fn, gaussian_loss_fn = build_loss(cfg)

    # position map model config
    cfg.model = 'DeepLabV3Plus'
    position_model = build_model(cfg, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=loss_fn,
                                 additional_loss_functions=gaussian_loss_fn, collate_fn=heat_map_collate_fn,
                                 desired_output_shape=target_max_shape)

    if cfg.crop_classifier.use_pretrained.enabled:
        checkpoint_path = f'{cfg.crop_classifier.use_pretrained.checkpoint.root}' \
                          f'{cfg.crop_classifier.use_pretrained.checkpoint.path}' \
                          f'{cfg.crop_classifier.use_pretrained.checkpoint.version}/checkpoints/'
        checkpoint_files = os.listdir(checkpoint_path)

        epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
        epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
        checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

        checkpoint_file = checkpoint_path + checkpoint_files[-cfg.crop_classifier.use_pretrained.checkpoint.top_k]

        logger.info(f'Loading weights from: {checkpoint_file}')
        load_dict = torch.load(checkpoint_file, map_location=cfg.device)

        position_model.load_state_dict(load_dict['state_dict'], strict=False)

    position_model.to(cfg.device)
    position_model.eval()

    # classifier model
    model = CropClassifier.from_config(config=cfg)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.crop_classifier.lr,
                           weight_decay=cfg.crop_classifier.weight_decay, amsgrad=cfg.crop_classifier.amsgrad)
    sch = ReduceLROnPlateau(opt,
                            patience=cfg.crop_classifier.patience,
                            verbose=cfg.crop_classifier.verbose,
                            factor=cfg.crop_classifier.factor,
                            min_lr=cfg.crop_classifier.min_lr)

    if isinstance(cfg.crop_classifier.subset_indices, (list, ListConfig)):
        indices = list(cfg.crop_classifier.subset_indices)
    else:
        indices = np.random.choice(len(train_dataset), cfg.crop_classifier.subset_indices, replace=False)
    # train_subset = Subset(dataset=train_dataset, indices=indices)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.crop_classifier.batch_size, shuffle=cfg.crop_classifier.shuffle,
        num_workers=cfg.crop_classifier.num_workers, collate_fn=heat_map_collate_fn,
        pin_memory=cfg.crop_classifier.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.crop_classifier.batch_size * 2, shuffle=cfg.crop_classifier.shuffle,
        num_workers=cfg.crop_classifier.num_workers, collate_fn=heat_map_collate_fn,
        pin_memory=cfg.crop_classifier.pin_memory, drop_last=cfg.interplay_v0.drop_last)

    # training + logic

    for epoch in range(cfg.crop_classifier.num_epochs):
        opt.zero_grad()
        model.train()

        train_loss = []
        gt_crops, gt_boxes = [], []  # gt boxes in xyxy
        crops, boxes = [], []
        for t_idx, data in enumerate(tqdm(train_loader)):
            frames, heat_masks, _, _, _, meta = data

            padder = ImagePadder(frames.shape[-2:], factor=cfg.preproccesing.pad_factor)
            frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]
            frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

            with torch.no_grad():
                pred_position_maps = position_model(frames)

            pred_object_locations = locations_from_heatmaps(
                frames, cfg.crop_classifier.objectness.kernel,
                cfg.crop_classifier.objectness.loc_cutoff,
                cfg.crop_classifier.objectness.marker_size, pred_position_maps,
                vis_on=False, threshold=cfg.crop_classifier.objectness.threshold)

            selected_head = pred_object_locations[cfg.crop_classifier.objectness.index_select]

            selected_head, frames_scaled = get_adjusted_object_locations_rgb(
                selected_head, frames, meta)
            selected_head = [torch.from_numpy(np.stack(s)) for s in selected_head if len(s) != 0]
            # frames_scaled = torch.from_numpy(frames_scaled).permute(0, 3, 1, 2)
            frames_scaled = [torch.from_numpy(fs).permute(2, 0, 1) for fs in frames_scaled]

            # get tp crops
            for l_idx, locs in enumerate(selected_head):
                if locs.numel() == 0:
                    continue

                crops_filtered, valid_boxes = get_gt_crops_and_boxes(cfg, frames_scaled, l_idx, locs)

                gt_crops.append(crops_filtered)
                gt_boxes.append(valid_boxes)

                # viz
                viz_boxes_on_img(frames_scaled[l_idx], valid_boxes, show=False)
                viz_crops(crops_filtered, show=False)
                if len(crops_filtered) == 0 or len(valid_boxes) == 0:
                    continue

            # get tn crops

            if gt_crops is None:
                continue

            for rgb_frame, gt_c, gt_b in zip(frames_scaled, gt_crops, gt_boxes):
                tn_boxes, tn_boxes_xyxy, tn_crops = get_initial_tn_crops_and_boxes(cfg, gt_c, rgb_frame)

                gt_bbox_centers, tn_boxes, tn_boxes_xyxy, tn_crops = get_radius_filtered_tn_crops_and_boxes(
                    cfg, gt_b, tn_boxes, tn_boxes_xyxy, tn_crops)

                boxes_to_replace = gt_c.shape[0] - tn_boxes_xyxy.shape[0]
                replaceable_boxes, replaceable_tn_boxes, replaceable_crops = [], [], []
                replacement_required = False

                try:
                    replacement_required = replace_overlapping_boxes(
                        (cfg.crop_classifier.crop_size[0],
                         cfg.crop_classifier.crop_size[1]), boxes_to_replace,
                        gt_bbox_centers, gt_b, rgb_frame,
                        cfg.crop_classifier.radius_elimination, replaceable_boxes, replaceable_crops,
                        replaceable_tn_boxes, replacement_required)
                except TimeoutException:
                    logger.warning(f'Replacement timed-out : {REPLACEMENT_TIMEOUT}!! Skipping frame')
                    return {}, {}

                tn_boxes_xyxy, tn_crops = replace_tn_data_if_required(replaceable_boxes, replaceable_crops,
                                                                      replaceable_tn_boxes, replacement_required,
                                                                      tn_boxes, tn_boxes_xyxy, tn_crops)

                viz_crops(tn_crops, show=False)
                boxes.append(tn_boxes_xyxy)
                crops.append(tn_crops)
            # train loop ends
        # epoch loop ends
        gt_crops = torch.cat(gt_crops)
        crops = torch.cat(crops)
        gt_data = {'images': gt_crops, 'labels': torch.ones((gt_crops.shape[0]))}
        tn_data = {'images': crops, 'labels': torch.zeros((crops.shape[0]))}
        torch.save((gt_data, tn_data), 'death_circle_crops_2_3_4.pt')


def replace_tn_data_if_required(replaceable_boxes, replaceable_crops, replaceable_tn_boxes, replacement_required,
                                tn_boxes, tn_boxes_xyxy, tn_crops):
    if replacement_required:
        replaceable_boxes = torch.cat(replaceable_boxes)
        replaceable_tn_boxes = torch.cat(replaceable_tn_boxes)
        replaceable_crops = torch.cat(replaceable_crops)

        tn_boxes = torch.cat((tn_boxes, replaceable_boxes))
        tn_boxes_xyxy = torch.cat((tn_boxes_xyxy, replaceable_tn_boxes))
        tn_crops = torch.cat((tn_crops, replaceable_crops))
    return tn_boxes_xyxy, tn_crops


def get_radius_filtered_tn_crops_and_boxes(cfg, gt_b, tn_boxes, tn_boxes_xyxy, tn_crops):
    boxes_iou = torchvision.ops.box_iou(gt_b, tn_boxes_xyxy)
    gt_box_match, tn_boxes_xyxy_match = torch.where(boxes_iou)
    tn_boxes_xyxy_match_numpy = tn_boxes_xyxy_match.numpy()
    l2_distances_matrix = np.zeros(shape=(gt_b.shape[0], tn_boxes_xyxy.shape[0]))
    if cfg.crop_classifier.radius_elimination is not None:
        tn_boxes_xyxy_centers = np.stack(
            [get_bbox_center(tn_box) for tn_box in tn_boxes_xyxy.numpy()]).squeeze()
        gt_bbox_centers = np.stack(
            [get_bbox_center(g_b) for g_b in gt_b.numpy()]).squeeze()

        # for g_idx, gt_center in enumerate(gt_bbox_centers):
        for g_idx, gt_center in enumerate(gt_bbox_centers):
            if tn_boxes_xyxy_centers.ndim == 1:
                tn_boxes_xyxy_centers = np.expand_dims(tn_boxes_xyxy_centers, axis=0)
            for f_idx, fp_center in enumerate(tn_boxes_xyxy_centers):
                l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)

        # l2_distances_matrix = l2_distances_matrix[:gt_bbox_centers.shape[0], ...]
        l2_distances_matrix = l2_distances_matrix[:gt_bbox_centers.shape[0], ...]
        gt_r_invalid, fp_r_invalid = np.where(l2_distances_matrix < cfg.crop_classifier.radius_elimination)

        tn_boxes_xyxy_match_numpy = np.union1d(tn_boxes_xyxy_match.numpy(), fp_r_invalid)
    valid_tn_boxes_xyxy_idx = np.setdiff1d(np.arange(tn_boxes_xyxy.shape[0]), tn_boxes_xyxy_match_numpy)
    tn_boxes_xyxy = tn_boxes_xyxy[valid_tn_boxes_xyxy_idx]
    tn_boxes = tn_boxes[valid_tn_boxes_xyxy_idx]
    tn_crops = tn_crops[valid_tn_boxes_xyxy_idx]
    return gt_bbox_centers, tn_boxes, tn_boxes_xyxy, tn_crops


def get_initial_tn_crops_and_boxes(cfg, gt_c, rgb_frame):
    tn_crops, tn_boxes = sample_random_crops(rgb_frame,
                                             (cfg.crop_classifier.crop_size[0],
                                              cfg.crop_classifier.crop_size[1]),
                                             gt_c.shape[0])
    tn_boxes = [torch.tensor((b[1], b[0], b[2], b[3])) for b in tn_boxes]
    tn_boxes = torch.stack(tn_boxes)
    tn_boxes_xyxy = torchvision.ops.box_convert(tn_boxes, 'xywh', 'xyxy')
    return tn_boxes, tn_boxes_xyxy, tn_crops


def get_gt_crops_and_boxes(cfg, frames_scaled, l_idx, locs):
    crops_filtered, valid_boxes = get_processed_patches_to_train_rgb_only(
        cfg.crop_classifier.crop_size[0], cfg.crop_classifier.crop_size[1], frames_scaled,
        l_idx, locs)
    valid_boxes = [torch.tensor((b[1], b[0], b[2], b[3])) for b in valid_boxes]
    valid_boxes = torch.stack(valid_boxes)
    valid_boxes = torchvision.ops.box_convert(valid_boxes, 'xywh', 'xyxy')
    return crops_filtered, valid_boxes


def viz_boxes_on_img(frames_scaled, valid_boxes, show=True):
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    ax.imshow(frames_scaled.permute(1, 2, 0))
    add_box_to_axes(ax, valid_boxes)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def viz_crops(crops_filtered, show=True):
    crops_viz_image = torchvision.utils.make_grid(crops_filtered)
    plt.imshow(crops_viz_image.permute(1, 2, 0))
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # train_crop_classifier()
    # train_crop_classifier_v1()
    eval_crop_classifier_v1()
    # eval_crop_classifier()
    # generate_crops_v0()
