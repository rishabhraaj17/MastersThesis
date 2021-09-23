import copy
import os
import warnings
from pathlib import Path

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import scipy
import skimage
import torch
import torchvision
from kornia.losses import BinaryFocalLossWithLogits
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mmdet.models import GaussianFocalLoss
from mmdet.models.utils.gaussian_target import get_local_maximum
from pytorch_lightning import seed_everything
from torch.nn import MSELoss
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import models as model_zoo
from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from dataset import SDDFrameAndAnnotationDataset
from interact import extract_agents_locations, correct_locations, get_position_correction_transform
from log import get_logger
from losses import CenterNetFocalLoss
from patch_utils import quick_viz
from train import setup_multiple_datasets_core, setup_single_transform, setup_single_common_transform, \
    setup_single_dataset_instance
from utils import heat_map_collate_fn, plot_predictions, get_blob_count, overlay_images, plot_predictions_with_overlay, \
    get_scaled_shapes_with_pad_values, plot_image_with_features, ImagePadder, get_ensemble, plot_predictions_v2, \
    heat_map_temporal_4d_collate_fn
import src_lib.models_hub as hub

seed_everything(42)
logger = get_logger(__name__)


def get_supervised_boxes(cfg, current_random_frame, meta, random_idx, test_loader, test_transform, frame):
    # fixme: return two boxes for rgb shape and target shape
    gt_annotations, gt_bbox_centers = get_supervised_annotation_per_frame(cfg, current_random_frame, meta, random_idx,
                                                                          test_loader)
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


def get_supervised_annotation_per_frame(cfg, current_random_frame, meta, random_idx, test_loader):
    gt_annotation_df = get_supervised_df(cfg, test_loader)
    frame_annotation = get_frame_annotations_and_skip_lost(gt_annotation_df, current_random_frame)
    gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                        original_scale=meta[random_idx]['original_shape'],
                                                        new_scale=meta[random_idx]['original_shape'],
                                                        return_track_id=False,
                                                        tracks_with_annotations=True)
    return gt_annotations, gt_bbox_centers


def get_supervised_df(cfg, test_loader):
    gt_annotation_path = f'{cfg.root}annotations/{test_loader.dataset.video_label.value}/' \
                         f'video{test_loader.dataset.video_number_to_use}/annotation_augmented.csv'
    gt_annotation_df = pd.read_csv(gt_annotation_path)
    gt_annotation_df = gt_annotation_df.drop(gt_annotation_df.columns[[0]], axis=1)
    return gt_annotation_df


def get_blobs_per_image_for_metrics(cfg, meta, out):
    blobs_per_image, masks = extract_agents_locations(blob_threshold=cfg.eval.blob_threshold,
                                                      mask=out.clone().detach().cpu(),
                                                      objectness_threshold=cfg.eval.objectness_threshold)
    blobs_per_image = [correct_locations(b) for b in blobs_per_image]

    adjusted_locations, scaled_images = [], []
    for blobs, m, mask in zip(blobs_per_image, meta, masks):
        original_shape = m['original_shape']
        transform = get_position_correction_transform(original_shape)
        out = transform(image=mask.squeeze(0).numpy(), keypoints=blobs)
        adjusted_locations.append(out['keypoints'])
        scaled_images.append(out['image'])

    blobs_per_image = copy.deepcopy(adjusted_locations)
    masks = np.stack(scaled_images)

    return blobs_per_image, out


def get_adjusted_object_locations(locations, heat_masks, meta):
    adjusted_locations, scaled_images = [], []
    for blobs, m, mask in zip(locations, meta, heat_masks):
        original_shape = m['original_shape']
        transform = get_position_correction_transform(original_shape)
        out = transform(image=mask.squeeze(0).numpy(), keypoints=blobs.numpy())
        adjusted_locations.append(out['keypoints'])
        scaled_images.append(out['image'])

    masks = np.stack(scaled_images)

    return adjusted_locations, masks


def get_gt_annotations_for_metrics(blobs_per_image, cfg, f, frame_number, meta, rgb_frame, test_loader):
    gt_annotation_df = get_supervised_df(cfg, test_loader)
    frame_annotation = get_frame_annotations_and_skip_lost(gt_annotation_df, frame_number)
    gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                        original_scale=meta[f]['original_shape'],
                                                        new_scale=meta[f]['original_shape'],
                                                        return_track_id=False,
                                                        tracks_with_annotations=True)
    supervised_boxes = gt_annotations[:, :-1]

    inside_boxes_idx = [b for b, box in enumerate(supervised_boxes)
                        if (box[0] > 0 and box[2] < meta[f]['original_shape'][1])
                        and (box[1] > 0 and box[3] < meta[f]['original_shape'][0])]
    supervised_boxes = supervised_boxes[inside_boxes_idx]
    gt_bbox_centers = gt_bbox_centers[inside_boxes_idx]
    class_labels = ['object'] * supervised_boxes.shape[0]

    rgb_frame = interpolate(rgb_frame.unsqueeze(0), size=meta[f]['original_shape'])
    # transform = get_position_correction_transform(meta[f]['original_shape'])
    # out = transform(image=rgb_frame.squeeze(dim=0).permute(1, 2, 0).numpy(),
    #                 keypoints=gt_bbox_centers)
    # gt_bbox_centers = out['keypoints']
    # gt_bbox_centers = np.stack(gt_bbox_centers)
    pred_centers = blobs_per_image[f]

    return gt_bbox_centers, pred_centers, rgb_frame, supervised_boxes


def get_precision_recall_for_metrics(cfg, gt_bbox_centers, pred_centers, ratio):
    distance_matrix = np.zeros(shape=(len(gt_bbox_centers), len(pred_centers)))
    for gt_i, gt_loc in enumerate(gt_bbox_centers):
        for pred_i, pred_loc in enumerate(pred_centers):
            dist = np.linalg.norm((gt_loc - pred_loc), 2) * ratio
            distance_matrix[gt_i, pred_i] = dist

    distance_matrix = cfg.eval.gt_pred_loc_distance_threshold - distance_matrix
    distance_matrix[distance_matrix < 0] = 1000

    # Hungarian
    match_rows, match_cols = scipy.optimize.linear_sum_assignment(distance_matrix)
    actually_matched_mask = distance_matrix[match_rows, match_cols] < 1000

    match_rows = match_rows[actually_matched_mask]
    match_cols = match_cols[actually_matched_mask]

    tp = len(match_rows)
    fp = len(pred_centers) - len(match_rows)
    fn = len(gt_bbox_centers) - len(match_rows)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return fn, fp, precision, recall, tp


def get_image_array_from_figure(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    video_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
    return video_frame


def process_numpy_video_frame_to_tensor(video_frame):
    video_frame = torch.from_numpy(video_frame).permute(2, 0, 1)
    image_padder = ImagePadder(video_frame.shape)
    video_frame = image_padder.pad(video_frame.unsqueeze(0).float())[0].to(dtype=torch.uint8)
    return video_frame


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
        using_replay_compose=cfg.eval.using_replay_compose,
        manual_annotation_processing=cfg.eval.manual_annotation_processing,
        frame_rate=cfg.eval.frame_rate,
        config=cfg,
        frames_per_clip=cfg.eval.video_based.frames_per_clip if cfg.eval.video_based.enabled else 1
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


def setup_multiple_test_datasets_core(cfg, video_classes_to_use, video_numbers_to_use, num_videos, multiple_videos,
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
                                                          using_replay_compose=cfg.eval.using_replay_compose,
                                                          frame_rate=cfg.eval.frame_rate))
    return ConcatDataset(datasets)


def setup_multiple_test_datasets(cfg, return_dummy_transform=True):
    meta = SDDMeta(cfg.root + 'H_SDD.txt')
    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.eval.test.video_classes_to_use,
        video_numbers=cfg.eval.test.video_numbers_to_use,
        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio_rgb)

    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.eval.test.video_classes_to_use,
        video_numbers=cfg.eval.test.video_numbers_to_use,
        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio)
    # downscale_only_target_maps=cfg.downscale_only_target_maps may not point to eval cfg but not being used
    datasets = setup_multiple_test_datasets_core(cfg, video_classes_to_use=cfg.eval.test.video_classes_to_use,
                                                 video_numbers_to_use=cfg.eval.test.video_numbers_to_use,
                                                 num_videos=cfg.eval.test.num_videos,
                                                 multiple_videos=cfg.eval.test.multiple_videos,
                                                 df=df, df_target=df_target, rgb_max_shape=rgb_max_shape,
                                                 use_common_transforms=False)
    if return_dummy_transform:
        return datasets, None, target_max_shape
    return datasets, target_max_shape


def setup_single_video_dataset(cfg):
    meta = SDDMeta(cfg.eval.root + 'H_SDD.txt')

    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.eval.root, video_classes=cfg.eval.test.single_video_mode.video_classes_to_use,
        video_numbers=cfg.eval.test.single_video_mode.video_numbers_to_use,
        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio_rgb)

    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.eval.root, video_classes=cfg.eval.test.single_video_mode.video_classes_to_use,
        video_numbers=cfg.eval.test.single_video_mode.video_numbers_to_use,
        desired_ratio=cfg.eval.desired_pixel_to_meter_ratio)

    dataset = setup_multiple_datasets_core(
        cfg.eval, meta, video_classes_to_use=cfg.eval.test.single_video_mode.video_classes_to_use,
        video_numbers_to_use=cfg.eval.test.single_video_mode.video_numbers_to_use,
        num_videos=cfg.eval.test.single_video_mode.num_videos,
        multiple_videos=cfg.eval.test.single_video_mode.multiple_videos,
        df=df, df_target=df_target, rgb_max_shape=rgb_max_shape)

    val_dataset_len = round(len(dataset) * cfg.eval.test.single_video_mode.val_percent)
    train_indices = torch.arange(start=0, end=len(dataset) - val_dataset_len)
    val_indices = torch.arange(start=len(dataset) - val_dataset_len, end=len(dataset))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, val_indices)

    return train_dataset, test_dataset, target_max_shape


def setup_eval(cfg):
    logger.info(f'Setting up DataLoader')

    # test_dataset, test_transform, target_max_shape = setup_dataset(cfg)
    test_dataset, test_transform, target_max_shape = setup_multiple_test_datasets(cfg)

    test_loader = DataLoader(test_dataset, batch_size=cfg.eval.batch_size, shuffle=False,
                             num_workers=cfg.eval.num_workers, collate_fn=heat_map_collate_fn,
                             pin_memory=cfg.eval.pin_memory, drop_last=cfg.eval.drop_last)

    network_type = getattr(model_zoo, cfg.eval.postion_map_network_type)
    if network_type.__name__ in ['PositionMapUNetPositionMapSegmentation',
                                 'PositionMapUNetClassMapSegmentation',
                                 'PositionMapUNetHeatmapSegmentation',
                                 'PositionMapStackedHourGlass']:
        loss_fn = BinaryFocalLossWithLogits(alpha=cfg.eval.focal_loss_alpha, reduction='mean')  # CrossEntropyLoss()
    elif network_type.__name__ == 'HourGlassPositionMapNetwork':
        # loss_fn = GaussianFocalLoss(alpha=cfg.eval.gaussuan_focal_loss_alpha, reduction='mean')
        loss_fn = CenterNetFocalLoss()
    else:
        loss_fn = MSELoss()

    if network_type.__name__ == 'HourGlassPositionMapNetwork':
        model = network_type.from_config(config=cfg, train_dataset=None, val_dataset=None,
                                         loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                                         desired_output_shape=target_max_shape)
    else:
        model = network_type(config=cfg, train_dataset=None, val_dataset=None,
                             loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                             desired_output_shape=target_max_shape)

    logger.info(f'Setting up Model')

    checkpoint_path = f'{cfg.eval.checkpoint.root}{cfg.eval.checkpoint.path}{cfg.eval.checkpoint.version}/checkpoints/'
    checkpoint_files = os.listdir(checkpoint_path)

    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

    checkpoint_file = checkpoint_path + checkpoint_files[-cfg.eval.checkpoint.top_k]

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
        if cfg.eval.use_ensemble:
            ensemble_files = [checkpoint_path + c for c in checkpoint_files]
            state_dict = [torch.load(e, map_location=cfg.eval.device)['state_dict'] for e in ensemble_files]
            state_dict = get_ensemble(state_dict)
            model.load_state_dict(state_dict)
        else:
            load_dict = torch.load(checkpoint_file, map_location=cfg.eval.device)
            model.load_state_dict(load_dict['state_dict'])

        model.to(cfg.eval.device)
    model.eval()

    return loss_fn, model, network_type, test_loader, checkpoint_file, test_transform


def locations_from_heatmaps(frames, kernel, loc_cutoff, marker_size, out, vis_on=False):
    out = [o.sigmoid() for o in out]
    pruned_locations = []
    loc_maxima_per_output = [get_local_maximum(o, kernel) for o in out]
    for li, loc_max_out in enumerate(loc_maxima_per_output):
        temp_locations = []
        for out_img_idx in range(loc_max_out.shape[0]):
            h_loc, w_loc = torch.where(loc_max_out[out_img_idx].squeeze(0) > loc_cutoff)
            loc = torch.stack((w_loc, h_loc)).t()

            temp_locations.append(loc)

            # viz
            if vis_on:
                plt.imshow(frames[out_img_idx].cpu().permute(1, 2, 0))
                plt.plot(w_loc, h_loc, 'o', markerfacecolor='r', markeredgecolor='k', markersize=marker_size)

                plt.title(f'Out - {li} - {out_img_idx}')
                plt.tight_layout()
                plt.show()

        pruned_locations.append(temp_locations)
    return pruned_locations


@hydra.main(config_path="config", config_name="config")
def evaluate(cfg):
    sdd_meta = SDDMeta(cfg.eval.root + 'H_SDD.txt')
    to_pil = ToPILImage()

    loss_fn, model, network_type, test_loader, checkpoint_file, test_transform = setup_eval(cfg)

    ratio = float(sdd_meta.get_meta(getattr(SDDVideoDatasets, cfg.eval.video_meta_class)
                                    , cfg.eval.test.video_number_to_use)[0]['Ratio'].to_numpy()[0])

    logger.info(f'Starting evaluation...')

    total_loss = []
    tp_list, fp_list, fn_list = [], [], []

    video_frames = []
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
        elif network_type.__name__ == 'HourGlassPositionMapNetwork':
            out = model_zoo.post_process_multi_apply(out)
            loss = model.calculate_loss(out, heat_masks).mean()
        else:
            loss = loss_fn(out, heat_masks)

        total_loss.append(loss.item())

        if cfg.eval.evaluate_precision_recall:
            blobs_per_image, _ = get_blobs_per_image_for_metrics(cfg, meta, out)

            for f in range(len(meta)):
                frame_number = meta[f]['item']
                rgb_frame = frames[f].cpu()
                gt_heatmap = heat_masks[f].cpu()
                pred_heatmap = out.sigmoid().round()[f].cpu()

                gt_bbox_centers, pred_centers, rgb_frame, supervised_boxes = get_gt_annotations_for_metrics(
                    blobs_per_image, cfg, f, frame_number, meta, rgb_frame, test_loader)

                fn, fp, precision, recall, tp = get_precision_recall_for_metrics(cfg, gt_bbox_centers, pred_centers,
                                                                                 ratio)

                if cfg.eval.show_plots or cfg.eval.make_video:
                    fig = plot_image_with_features(
                        rgb_frame.squeeze(dim=0).permute(1, 2, 0).numpy(), gt_bbox_centers,
                        np.stack(pred_centers), boxes=supervised_boxes,
                        txt=f'Frame Number: {frame_number}\n'
                            f'Agent Count: GT-{len(gt_bbox_centers)} | Pred-{len(pred_centers)}'
                            f'\nPrecision: {precision} | Recall: {recall}',
                        footnote_txt=f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                                     f'Video Number: {cfg.eval.test.video_number_to_use}'
                                     f'\n\nL2 Matching Threshold: '
                                     f'{cfg.eval.gt_pred_loc_distance_threshold}m',
                        video_mode=cfg.eval.make_video,
                        plot_heatmaps=True,
                        gt_heatmap=gt_heatmap.squeeze(dim=0).numpy(),
                        pred_heatmap=pred_heatmap.squeeze(dim=0).numpy())

                    if cfg.eval.make_video:
                        video_frame = get_image_array_from_figure(fig)

                        if video_frame.shape[0] != meta[f]['original_shape'][1] \
                                or video_frame.shape[1] != meta[f]['original_shape'][0]:
                            video_frame = skimage.transform.resize(
                                video_frame, (meta[f]['original_shape'][1], meta[f]['original_shape'][0]))
                            video_frame = (video_frame * 255).astype(np.uint8)

                            video_frame = process_numpy_video_frame_to_tensor(video_frame)
                            video_frames.append(video_frame)

                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)

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
            elif network_type.__name__ in ['PositionMapUNetHeatmapSegmentation', 'HourGlassPositionMapNetwork']:
                num_objects_gt = meta[random_idx]['bbox_centers'].shape[0]

                if network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                    pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                elif network_type.__name__ == 'HourGlassPositionMapNetwork':
                    temp_mask = [torch.sigmoid(o).squeeze(dim=1).cpu() for o in out]
                    pred_mask = [torch.round(torch.sigmoid(o)).squeeze(dim=1).cpu() for o in out]
                    pred_mask = pred_mask[cfg.eval.pick_heatmap_from_stack_number]
                else:
                    return NotImplementedError

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
                        unsupervised_boxes=unsupervised_boxes,
                        do_nothing=cfg.eval.plots_do_nothing)
                else:
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     pred_mask[random_idx].int() * 255,
                                     additional_text=additional_text,
                                     save_dir=save_dir + 'simple/',
                                     img_name=save_image_name,
                                     do_nothing=cfg.eval.plots_do_nothing)
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

    final_precision = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fp_list).sum())
    final_recall = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fn_list).sum())

    logger.info(f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                f'Video Number: {cfg.eval.test.video_number_to_use}')
    logger.info(f"Threshold: {cfg.eval.gt_pred_loc_distance_threshold}m")
    logger.info(f"Test Loss: {np.array(total_loss).mean()}")
    logger.info(f"Precision: {final_precision} | Recall: {final_recall}")

    if cfg.eval.make_video:
        logger.info(f"Writing Video")
        Path(os.path.join(os.getcwd(), 'videos')).mkdir(parents=True, exist_ok=True)
        torchvision.io.write_video(
            f'videos/{getattr(SDDVideoClasses, cfg.eval.video_meta_class).name}_'
            f'{cfg.eval.test.video_number_to_use}.avi',
            torch.cat(video_frames).permute(0, 2, 3, 1),
            cfg.eval.video_fps)


@hydra.main(config_path="config", config_name="config")
def evaluate_v1(cfg):
    sdd_meta = SDDMeta(cfg.eval.root + 'H_SDD.txt')
    # to_pil = ToPILImage()

    logger.info(f'Setting up DataLoader')

    if cfg.eval.test.single_video_mode.enabled:
        train_dataset, test_dataset, target_max_shape = setup_single_video_dataset(cfg)
    elif cfg.eval.mutiple_dataset_mode.enabled:
        whole_test_dataset, target_max_shape = setup_multiple_test_datasets(cfg, return_dummy_transform=False)
        test_dataset = []
        for dset in whole_test_dataset.datasets:
            test_dataset.append(
                Subset(dset, indices=np.random.choice(
                    len(dset), cfg.eval.mutiple_dataset_mode.samples_per_dataset, replace=False))
            )
        test_dataset = ConcatDataset(test_dataset)
    else:
        whole_test_dataset, _, target_max_shape = setup_dataset(cfg)
        test_dataset = Subset(whole_test_dataset, indices=np.random.choice(
            len(whole_test_dataset), cfg.eval.test.samples_per_dataset, replace=False))

    collate_fn = heat_map_temporal_4d_collate_fn if cfg.eval.video_based.enabled else heat_map_collate_fn
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval.batch_size, shuffle=cfg.eval.shuffle,
                             num_workers=cfg.eval.num_workers, collate_fn=collate_fn,
                             pin_memory=cfg.eval.pin_memory, drop_last=cfg.eval.drop_last)

    loss_fn = BinaryFocalLossWithLogits(
        alpha=cfg.eval.loss.bfl.alpha, gamma=cfg.eval.loss.bfl.gamma, reduction=cfg.eval.loss.reduction)
    gauss_loss_fn = [CenterNetFocalLoss()]

    model = getattr(hub, cfg.eval.model)(
        config=cfg, train_dataset=None, val_dataset=None,
        loss_function=loss_fn, collate_fn=heat_map_collate_fn, additional_loss_functions=gauss_loss_fn,
        desired_output_shape=None)

    if cfg.eval.checkpoint.wandb.enabled:
        version_name = f"{cfg.eval.checkpoint.wandb.checkpoint.run_name}".split('-')[-1]
        checkpoint_path = f'{cfg.eval.checkpoint.wandb.checkpoint.root}' \
                          f'{cfg.eval.checkpoint.wandb.checkpoint.run_name}' \
                          f'{cfg.eval.checkpoint.wandb.checkpoint.tail_path}' \
                          f'{cfg.eval.checkpoint.wandb.checkpoint.project_name}/' \
                          f'{version_name}/checkpoints/'
    else:
        checkpoint_path = f'{cfg.eval.checkpoint.root}{cfg.eval.checkpoint.path}' \
                          f'{cfg.eval.checkpoint.version}/checkpoints/'
    checkpoint_files = os.listdir(checkpoint_path)

    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

    checkpoint_file = checkpoint_path + checkpoint_files[-cfg.eval.checkpoint.top_k]
    logger.info(f'Loading weights : {checkpoint_file}')

    load_dict = torch.load(checkpoint_file, map_location=cfg.eval.device)
    model.load_state_dict(load_dict['state_dict'])

    model.to(cfg.eval.device)
    model.eval()

    ratio = float(sdd_meta.get_meta(getattr(SDDVideoDatasets, cfg.eval.video_meta_class)
                                    , cfg.eval.test.video_number_to_use)[0]['Ratio'].to_numpy()[0])

    logger.info(f'Starting evaluation...')

    total_loss = []
    tp_list, fp_list, fn_list = [], [], []

    video_frames = []
    for idx, data in enumerate(tqdm(test_loader)):
        frames, heat_masks, position_map, distribution_map, class_maps, meta = data

        padder = ImagePadder(frames.shape[-2:], factor=cfg.eval.preproccesing.pad_factor)
        frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]
        frames, heat_masks = frames.to(cfg.eval.device), heat_masks.to(cfg.eval.device)

        with torch.no_grad():
            out = model(frames)

        if cfg.eval.video_based.enabled:
            frames = frames[:, -3:, ...]
            heat_masks = heat_masks[:, cfg.eval.video_based.gt_idx, None, ...]

        loss1 = getattr(torch.Tensor, cfg.eval.loss.reduction)(model.calculate_loss(out, heat_masks))
        loss2 = getattr(torch.Tensor, cfg.eval.loss.reduction)(model.calculate_additional_losses(
            out, heat_masks, cfg.eval.loss.gaussian_weight, cfg.eval.loss.apply_sigmoid))
        loss = loss1 + loss2

        total_loss.append(loss.item())

        if cfg.eval.evaluate_precision_recall:
            locations = locations_from_heatmaps(frames, cfg.eval.objectness.kernel,
                                                cfg.eval.objectness.loc_cutoff,
                                                cfg.eval.objectness.marker_size, out, vis_on=False)
            metrics_out = out[cfg.eval.objectness.index_select]
            blobs_per_image, _ = get_adjusted_object_locations(
                locations[cfg.eval.objectness.index_select], metrics_out, meta)

            for f in range(len(meta)):
                frame_number = meta[f]['item']
                rgb_frame = frames[f].cpu()
                gt_heatmap = heat_masks[f].cpu()
                pred_heatmap = metrics_out.sigmoid()[f].cpu()

                gt_bbox_centers, pred_centers, rgb_frame, supervised_boxes = get_gt_annotations_for_metrics(
                    blobs_per_image, cfg, f, frame_number, meta, rgb_frame, test_loader)

                fn, fp, precision, recall, tp = get_precision_recall_for_metrics(cfg, gt_bbox_centers, pred_centers,
                                                                                 ratio)

                if cfg.eval.show_plots or cfg.eval.make_video:
                    fig = plot_image_with_features(
                        rgb_frame.squeeze(dim=0).permute(1, 2, 0).numpy(), gt_bbox_centers,
                        np.stack(pred_centers), boxes=supervised_boxes,
                        txt=f'Frame Number: {frame_number}\n'
                            f'Agent Count: GT-{len(gt_bbox_centers)} | Pred-{len(pred_centers)}'
                            f'\nPrecision: {precision} | Recall: {recall}',
                        footnote_txt=f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                                     f'Video Number: {cfg.eval.test.video_number_to_use}'
                                     f'\n\nL2 Matching Threshold: '
                                     f'{cfg.eval.gt_pred_loc_distance_threshold}m',
                        video_mode=cfg.eval.make_video,
                        plot_heatmaps=True,
                        gt_heatmap=gt_heatmap.squeeze(dim=0).numpy(),
                        pred_heatmap=pred_heatmap.squeeze(dim=0).numpy())

                    if cfg.eval.make_video:
                        video_frame = get_image_array_from_figure(fig)

                        if video_frame.shape[0] != meta[f]['original_shape'][1] \
                                or video_frame.shape[1] != meta[f]['original_shape'][0]:
                            video_frame = skimage.transform.resize(
                                video_frame, (meta[f]['original_shape'][1], meta[f]['original_shape'][0]))
                            video_frame = (video_frame * 255).astype(np.uint8)

                            video_frame = process_numpy_video_frame_to_tensor(video_frame)
                            video_frames.append(video_frame)

                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)

        if cfg.eval.show_plots and idx % cfg.eval.plot_checkpoint == 0:
            random_idx = np.random.choice(cfg.eval.batch_size, 1, replace=False).item()
            current_random_frame = meta[random_idx]['item']

            if cfg.eval.save_plots:
                save_dir = f'{cfg.eval.plot_save_dir}{model._get_name()}_{loss_fn._get_name()}/' \
                           f'version_{cfg.eval.checkpoint.version}/{os.path.split(checkpoint_file)[-1][:-5]}/'
                if cfg.eval.checkpoint.wandb.enabled:
                    save_dir = f'{cfg.eval.plot_save_dir}{model._get_name()}_{loss_fn._get_name()}/' \
                               f'version_{cfg.eval.checkpoint.wandb.checkpoint.run_name}/' \
                               f'{os.path.split(checkpoint_file)[-1][:-5]}/'
                if cfg.eval.mutiple_dataset_mode.enabled:
                    save_dir = save_dir + 'multiple_dataset_mode/'
                elif cfg.eval.test.single_video_mode.enabled:
                    save_dir = save_dir + 'single_dataset_mode/'
                else:
                    save_dir = save_dir + 'original_image_rescaled/'
            else:
                save_dir = None
            save_image_name = f'frame_{current_random_frame}'

            additional_text = f"{model._get_name()} | {loss_fn._get_name()} | Frame: {current_random_frame}\n" \
                              f""
            show = np.random.choice(2, 1, replace=False, p=[0.01, 0.99]).item()

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
                                                    f"| Epoch: {idx} "
                                                    f"| Frame Number: {current_random_frame} "
                                                    f"| Threshold: {cfg.prediction.threshold} | "
                                                    f"Out Idx: 0",
                                    save_dir=save_dir,
                                    img_name=save_image_name + '_head0')
                plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                    heat_masks[random_idx].squeeze().cpu(),
                                    torch.nn.functional.threshold(out[-1][random_idx].sigmoid(),
                                                                  threshold=cfg.prediction.threshold,
                                                                  value=cfg.prediction.fill_value,
                                                                  inplace=True),
                                    logits_mask=out[-1][random_idx].sigmoid(),
                                    additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                    f"| Epoch: {idx} "
                                                    f"| Frame Number: {current_random_frame} "
                                                    f"| Threshold: {cfg.prediction.threshold} | "
                                                    f"Out Idx: -1",
                                    save_dir=save_dir, img_name=save_image_name + '_head2')

                if cfg.model_hub.model == 'DeepLabV3Plus' and len(out) > 2:
                    plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                        heat_masks[random_idx].squeeze().cpu(),
                                        torch.nn.functional.threshold(
                                            out[-2][random_idx].sigmoid(),
                                            threshold=cfg.prediction.threshold,
                                            value=cfg.prediction.fill_value,
                                            inplace=True),
                                        logits_mask=out[-2][random_idx].sigmoid(),
                                        additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                                        f"| Epoch: {idx} "
                                                        f"| Frame Number: {current_random_frame} "
                                                        f"| Threshold: {cfg.prediction.threshold} | "
                                                        f"Out Idx: -2",
                                        save_dir=save_dir, img_name=save_image_name + '_head1')

    final_precision = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fp_list).sum())
    final_recall = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fn_list).sum())

    logger.info(f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                f'Video Number: {cfg.eval.test.video_number_to_use}')
    logger.info(f"Threshold: {cfg.eval.gt_pred_loc_distance_threshold}m | "
                f"Max-Pool kernel size: {cfg.eval.objectness.kernel} | "
                f"Head Used: {cfg.eval.objectness.index_select}")
    logger.info(f"Test Loss: {np.array(total_loss).mean()}")
    logger.info(f"Precision: {final_precision} | Recall: {final_recall}")

    if cfg.eval.make_video:
        logger.info(f"Writing Video")
        Path(os.path.join(os.getcwd(), 'videos')).mkdir(parents=True, exist_ok=True)
        torchvision.io.write_video(
            f'videos/{getattr(SDDVideoClasses, cfg.eval.video_meta_class).name}_'
            f'{cfg.eval.test.video_number_to_use}.avi',
            torch.cat(video_frames).permute(0, 2, 3, 1),
            cfg.eval.video_fps)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # evaluate()
        evaluate_v1()
