import os
import warnings

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import torch
from kornia.losses import BinaryFocalLossWithLogits
from pytorch_lightning import seed_everything
from torch.nn import MSELoss
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from log import get_logger
import models as model_zoo
from dataset import SDDFrameAndAnnotationDataset
from train import setup_multiple_datasets_core, setup_single_transform, setup_single_common_transform
from utils import heat_map_collate_fn, plot_predictions, get_blob_count, overlay_images, plot_predictions_with_overlay, \
    get_scaled_shapes_with_pad_values

seed_everything(42)
logger = get_logger(__name__)


def get_supervised_boxes(cfg, current_random_frame, meta, random_idx, test_loader, test_transform, frame):
    # fixme: return two boxes for rgb shape and target shape
    gt_annotation_path = f'{cfg.root}annotations/{test_loader.dataset.video_label.value}/' \
                         f'video{test_loader.dataset.video_number_to_use}/annotation_augmented.csv'
    gt_annotation_df = pd.read_csv(gt_annotation_path)
    gt_annotation_df = gt_annotation_df.drop(gt_annotation_df.columns[[0]], axis=1)
    frame_annotation = get_frame_annotations_and_skip_lost(gt_annotation_df, current_random_frame)
    gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                        original_scale=meta[random_idx]['original_shape'],
                                                        new_scale=meta[random_idx]['original_shape'],
                                                        return_track_id=False,
                                                        tracks_with_annotations=True)
    supervised_boxes = gt_annotations[:, :-1]
    inside_boxes_idx = [b for b, box in enumerate(supervised_boxes)
                        if (box[0] > 0 and box[2] < meta[random_idx]['original_shape'][1])
                        and (box[1] > 0 and box[3] < meta[random_idx]['original_shape'][0])]
    supervised_boxes = supervised_boxes[inside_boxes_idx]
    gt_bbox_centers = gt_bbox_centers[inside_boxes_idx]
    class_labels = ['object'] * supervised_boxes.shape[0]

    frame = interpolate(frame.unsqueeze(0), size=meta[random_idx]['original_shape'])

    out = test_transform(image=frame.squeeze(dim=0).permute(1, 2, 0).numpy(), bboxes=supervised_boxes,
                         keypoints=gt_bbox_centers,
                         class_labels=class_labels)
    return out['bboxes']


def get_resize_dims(cfg):
    meta = SDDMeta(cfg.eval.root + 'H_SDD.txt')

    test_reference_img_path = f'{cfg.eval.root}annotations/{getattr(SDDVideoClasses, cfg.eval.video_class).value}/' \
                              f'video{cfg.eval.test.video_number_to_use}/reference.jpg'
    test_w, test_h = meta.get_new_scale(img_path=test_reference_img_path,
                                        dataset=getattr(SDDVideoDatasets, cfg.eval.video_meta_class),
                                        sequence=cfg.eval.test.video_number_to_use,
                                        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio)
    return [int(test_w), int(test_h)]


def setup_dataset(cfg):
    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=[cfg.eval.video_class],
        video_numbers=[[cfg.eval.test.video_number_to_use]],
        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio_rgb)

    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=[cfg.eval.video_class],
        video_numbers=[[cfg.eval.test.video_number_to_use]],
        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio)

    condition = (df.CLASS == cfg.eval.video_class) & (df.NUMBER == cfg.eval.test.video_number_to_use)
    h, w = df[condition].RESCALED_SHAPE.values.item()
    pad_values = df[condition].PAD_VALUES.values.item()

    target_h, target_w = df_target[condition].RESCALED_SHAPE.values.item()
    target_pad_values = df_target[condition].PAD_VALUES.values.item()

    transform = setup_single_transform(height=target_h, width=target_w)
    rgb_transform_fn = setup_single_transform(height=h, width=w)
    rgb_plot_transform = setup_single_transform(height=rgb_max_shape[0], width=rgb_max_shape[1])
    if cfg.eval.resize_transform_only:
        common_transform = None
    else:
        common_transform = setup_single_common_transform(use_replay_compose=cfg.using_replay_compose)

    test_dataset = SDDFrameAndAnnotationDataset(
        root=cfg.eval.root, video_label=getattr(SDDVideoClasses, cfg.eval.video_class),
        num_videos=cfg.eval.test.num_videos, transform=transform if cfg.eval.data_augmentation else None,
        num_workers=cfg.eval.dataset_workers, scale=cfg.eval.scale_factor,
        video_number_to_use=cfg.eval.test.video_number_to_use,
        multiple_videos=cfg.eval.test.multiple_videos,
        use_generated=cfg.eval.use_generated_dataset,
        sigma=cfg.eval.sigma,
        plot=cfg.eval.plot_samples,
        desired_size=cfg.eval.desired_size,
        heatmap_shape=cfg.eval.heatmap_shape,
        return_combined_heatmaps=cfg.eval.return_combined_heatmaps,
        seg_map_objectness_threshold=cfg.eval.seg_map_objectness_threshold,
        meta_label=getattr(SDDVideoDatasets, cfg.eval.video_meta_class),
        heatmap_region_limit_threshold=cfg.eval.heatmap_region_limit_threshold,
        downscale_only_target_maps=cfg.eval.downscale_only_target_maps,
        rgb_transform=rgb_transform_fn,
        rgb_new_shape=(h, w),
        rgb_pad_value=pad_values,
        target_pad_value=target_pad_values,
        rgb_plot_transform=rgb_plot_transform,
        common_transform=common_transform,
        using_replay_compose=cfg.eval.using_replay_compose
    )
    return test_dataset, transform, target_max_shape


def setup_test_transform(cfg, test_h, test_w):
    if cfg.eval.resize_transform_only:
        transform = A.Compose(
            [A.Resize(height=test_h, width=test_w)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            keypoint_params=A.KeypointParams(format='xy')
        )
    else:
        transform = A.Compose(
            [A.Resize(height=test_h, width=test_w),
             A.RandomBrightnessContrast(p=0.3),
             A.VerticalFlip(p=0.3),
             A.HorizontalFlip(p=0.3)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            keypoint_params=A.KeypointParams(format='xy')
        )
    return transform


def setup_multiple_test_datasets(cfg):
    meta = SDDMeta(cfg.root + 'H_SDD.txt')
    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.eval.test.video_classes_to_use,
        video_numbers=cfg.eval.test.video_numbers_to_use,
        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio_rgb)

    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.eval.test.video_classes_to_use,
        video_numbers=cfg.eval.test.video_numbers_to_use,
        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio)
    # note: downscale_only_target_maps=cfg.downscale_only_target_maps may not point to eval cfg
    datasets = setup_multiple_datasets_core(cfg, meta, video_classes_to_use=cfg.eval.test.video_classes_to_use,
                                            video_numbers_to_use=cfg.eval.test.video_numbers_to_use,
                                            num_videos=cfg.eval.test.num_videos,
                                            multiple_videos=cfg.eval.test.multiple_videos,
                                            df=df, df_target=df_target, rgb_max_shape=rgb_max_shape)
    return datasets, target_max_shape


def setup_eval(cfg):
    logger.info(f'Setting up DataLoader')
    test_dataset, test_transform, target_max_shape = setup_dataset(cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval.batch_size, shuffle=False,
                             num_workers=cfg.eval.num_workers, collate_fn=heat_map_collate_fn,
                             pin_memory=cfg.eval.pin_memory, drop_last=cfg.eval.drop_last)

    network_type = getattr(model_zoo, cfg.eval.postion_map_network_type)
    if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation',
                                 'PositionMapUNetClassMapSegmentation',
                                 'PositionMapUNetHeatmapSegmentation',
                                 'PositionMapStackedHourGlass']:
        loss_fn = BinaryFocalLossWithLogits(alpha=cfg.eval.focal_loss_alpha, reduction='mean')  # CrossEntropyLoss()
    else:
        loss_fn = MSELoss()

    model = network_type(config=cfg, train_dataset=None, val_dataset=None,
                         loss_function=loss_fn, collate_fn=heat_map_collate_fn, desired_output_shape=target_max_shape)

    logger.info(f'Setting up Model')

    checkpoint_path = f'{cfg.eval.checkpoint.root}{cfg.eval.checkpoint.path}{cfg.eval.checkpoint.version}/checkpoints/'
    checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]

    if cfg.eval.use_lightning_loader:
        hparams_file = f'{cfg.eval.checkpoint.root}{cfg.eval.checkpoint.path}{cfg.eval.checkpoint.version}/hparams.yaml'
        model = network_type.load_from_checkpoint(
            checkpoint_file,
            hparams_file=hparams_file,
            map_location=cfg.eval.device,
            train_dataset=None, val_dataset=None,
            loss_function=loss_fn, collate_fn=heat_map_collate_fn, desired_output_shape=target_max_shape)
    else:
        logger.info(f'Loading weights from: {checkpoint_file}')
        load_dict = torch.load(checkpoint_file, map_location=cfg.eval.device)

        model.load_state_dict(load_dict['state_dict'])
        model.to(cfg.eval.device)
    model.eval()

    return loss_fn, model, network_type, test_loader, checkpoint_file, test_transform


@hydra.main(config_path="config", config_name="config")
def evaluate(cfg):
    to_pil = ToPILImage()

    loss_fn, model, network_type, test_loader, checkpoint_file, test_transform = setup_eval(cfg)

    logger.info(f'Starting evaluation...')

    total_loss = []
    for idx, data in enumerate(tqdm(test_loader)):
        frames, heat_masks, position_map, distribution_map, class_maps, meta = data

        if network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
            frames, class_maps = frames.to(cfg.eval.device), class_maps.to(cfg.eval.device)
        elif network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
            frames, position_map = frames.to(cfg.eval.device), position_map.to(cfg.eval.device)
        elif network_type.__name__ in ['PositionMapUNetHeatmapSegmentation',
                                       'PositionMapStackedHourGlass']:
            frames, heat_masks = frames.to(cfg.eval.device), heat_masks.to(cfg.eval.device)
        else:
            frames, heat_masks = frames.to(cfg.eval.device), heat_masks.to(cfg.eval.device)

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

        total_loss.append(loss.item())

        if cfg.eval.show_plots and idx % cfg.eval.plot_checkpoint == 0:
            random_idx = np.random.choice(cfg.eval.batch_size, 1, replace=False).item()
            current_random_frame = meta[random_idx]['item']

            save_dir = f'{cfg.eval.plot_save_dir}{network_type.__name__}_{loss_fn._get_name()}/' \
                       f'version_{cfg.eval.checkpoint.version}/{os.path.split(checkpoint_file)[-1][:-5]}/'
            save_image_name = f'frame_{current_random_frame}'

            if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation',
                                         'PositionMapUNetClassMapSegmentation']:
                pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                 class_maps[random_idx].squeeze().cpu()
                                 if cfg.class_map_segmentation else position_map[random_idx].squeeze().cpu(),
                                 pred_mask[random_idx].int() * 255,
                                 additional_text=f"{network_type.__name__} | {loss_fn._get_name()} | "
                                                 f"Frame: {current_random_frame}")
            elif network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                num_objects_gt = meta[random_idx]['bbox_centers'].shape[0]

                pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()

                num_objects_pred = get_blob_count(
                    pred_mask[random_idx].numpy().astype(np.uint8) * 255,
                    kernel_size=(cfg.eval.blob_counter.kernel[0], cfg.eval.blob_counter.kernel[1]),
                    plot=cfg.eval.blob_counter.plot)

                additional_text = f"{network_type.__name__} | {loss_fn._get_name()} | Frame: {current_random_frame}\n" \
                                  f"Agent Count : [GT: {num_objects_gt} | Prediction: {num_objects_pred}]"
                if cfg.eval.plot_with_overlay:
                    unsupervised_boxes = meta[random_idx]['boxes']
                    unsupervised_rgb_boxes = meta[random_idx]['rgb_boxes']
                    supervised_boxes = get_supervised_boxes(cfg, current_random_frame, meta, random_idx, test_loader,
                                                            test_transform, frames[random_idx].cpu()) \
                        if cfg.eval.resize_transform_only else None
                    superimposed_image = overlay_images(transformer=to_pil, background=frames[random_idx],
                                                        overlay=pred_mask[random_idx])
                    plot_predictions_with_overlay(
                        img=frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                        mask=heat_masks[random_idx].squeeze().cpu(),
                        pred_mask=pred_mask[random_idx].int() * 255,
                        overlay_image=superimposed_image,
                        additional_text=additional_text,
                        save_dir=save_dir + 'overlay/',
                        img_name=save_image_name,
                        supervised_boxes=supervised_boxes,
                        unsupervised_rgb_boxes=unsupervised_rgb_boxes,
                        unsupervised_boxes=unsupervised_boxes)
                else:
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     pred_mask[random_idx].int() * 255,
                                     additional_text=additional_text,
                                     save_dir=save_dir + 'simple/',
                                     img_name=save_image_name)
            elif network_type.__name__ == 'PositionMapStackedHourGlass':
                pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                 class_maps[random_idx].squeeze().cpu()
                                 if cfg.class_map_segmentation else position_map[random_idx].squeeze().cpu(),
                                 pred_mask[-1][random_idx].int().squeeze(dim=0) * 255,
                                 additional_text=f"{network_type.__name__} | {loss_fn._get_name()} "
                                                 f"| Frame: {current_random_frame}")
                plot_predictions(pred_mask[-3][random_idx].int().squeeze(dim=0) * 255,
                                 pred_mask[-2][random_idx].int().squeeze(dim=0) * 255,
                                 pred_mask[-1][random_idx].int().squeeze(dim=0) * 255,
                                 additional_text=f"{network_type.__name__} | {loss_fn._get_name()} "
                                                 f"| Frame: {current_random_frame}\nLast 3 HeatMaps", all_heatmaps=True)
            else:
                plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                 heat_masks[random_idx].squeeze().cpu(),
                                 out[random_idx].squeeze().cpu(),
                                 additional_text=f"{network_type.__name__} | {loss_fn._get_name()} |"
                                                 f" Frame: {current_random_frame}")

    logger.info(f"Test Loss: {np.array(total_loss).mean()}")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        evaluate()
