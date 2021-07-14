import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
from kornia.losses import BinaryFocalLossWithLogits
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

import src_lib.models_hub as hub
from average_image.constants import SDDVideoDatasets, SDDVideoClasses
from average_image.utils import SDDMeta
from log import get_logger
from src.position_maps.evaluate import setup_single_video_dataset, setup_dataset, get_gt_annotations_for_metrics
from src.position_maps.location_utils import locations_from_heatmaps, get_adjusted_object_locations
from src.position_maps.losses import CenterNetFocalLoss
from src.position_maps.utils import heat_map_temporal_4d_collate_fn, heat_map_collate_fn, ImagePadder, \
    plot_image_with_features

seed_everything(42)
logger = get_logger(__name__)


def adjust_config(cfg):
    cfg.eval.device = 'cpu'

    # Pixel to Meter 
    cfg.eval.desired_pixel_to_meter_ratio_rgb = 0.07
    cfg.eval.desired_pixel_to_meter_ratio = 0.07

    # if single video mode
    cfg.eval.test.single_video_mode.enabled = False
    cfg.eval.test.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
    cfg.eval.test.single_video_mode.video_numbers_to_use = [[4]]
    cfg.eval.test.single_video_mode.num_videos = -1
    cfg.eval.test.single_video_mode.multiple_videos = False

    # one video at a time
    cfg.eval.video_class = 'DEATH_CIRCLE'
    cfg.eval.video_meta_class = 'DEATH_CIRCLE'
    cfg.eval.test.video_number_to_use = 4
    cfg.eval.test.num_videos = -1
    cfg.eval.dataset_workers = 12
    cfg.eval.test.multiple_videos = False
    cfg.eval.plot_samples = False
    cfg.eval.frame_rate = 30.
    cfg.eval.resize_transform_only = True

    # Temporal 2D
    cfg.eval.video_based.enabled = False
    cfg.eval.video_based.frames_per_clip = 4
    cfg.eval.video_based.gt_idx = -1

    # Loader
    cfg.eval.batch_size = 4
    cfg.eval.shuffle = False
    cfg.eval.num_workers = 0
    cfg.eval.pin_memory = False
    cfg.eval.drop_last = False

    # model
    cfg.eval.model = 'DeepLabV3Plus'  # DeepLabV3PlusTemporal2D or DeepLabV3Plus

    # checkpoint
    cfg.eval.checkpoint.version = 509506
    cfg.eval.checkpoint.top_k = 1

    # object
    cfg.eval.objectness.kernel = 3
    cfg.eval.objectness.loc_cutoff = 0.05
    cfg.eval.objectness.marker_size = 3
    cfg.eval.objectness.index_select = -1

    cfg.eval.gt_pred_loc_distance_threshold = 2  # in meters


@hydra.main(config_path="config", config_name="config")
def evaluate_and_store_predicted_maps(cfg):
    adjust_config(cfg)
    # sdd_meta = SDDMeta(cfg.eval.root + 'H_SDD.txt')

    logger.info(f'Setting up DataLoader')

    if cfg.eval.test.single_video_mode.enabled:
        train_dataset, test_dataset, target_max_shape = setup_single_video_dataset(cfg)
    else:
        # test_dataset, target_max_shape = setup_multiple_test_datasets(cfg, return_dummy_transform=False)
        test_dataset, _, target_max_shape = setup_dataset(cfg)

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

    checkpoint_path = f'{cfg.eval.checkpoint.root}{cfg.eval.checkpoint.path}{cfg.eval.checkpoint.version}/checkpoints/'
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

    # ratio = float(sdd_meta.get_meta(getattr(SDDVideoDatasets, cfg.eval.video_meta_class)
    #                                 , cfg.eval.test.video_number_to_use)[0]['Ratio'].to_numpy()[0])

    logger.info(f'Starting evaluation for storing heatmaps...')

    total_loss = []
    # tp_list, fp_list, fn_list = [], [], []

    pred_head_0, pred_head_1, pred_head_2, frames_sequence = [], [], [], []
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

        pred_head_0.append(out[0])
        pred_head_1.append(out[1])
        pred_head_2.append(out[2])
        frames_sequence.extend([m['item'] for m in meta])

        # locations = locations_from_heatmaps(frames, cfg.eval.objectness.kernel,
        #                                     cfg.eval.objectness.loc_cutoff,
        #                                     cfg.eval.objectness.marker_size, out, vis_on=False)
        # metrics_out = out[cfg.eval.objectness.index_select]
        # blobs_per_image, _ = get_adjusted_object_locations(
        #     locations[cfg.eval.objectness.index_select], metrics_out, meta)
        #
        # for f in range(len(meta)):
        #     frame_number = meta[f]['item']
        #     rgb_frame = frames[f].cpu()
        #     gt_heatmap = heat_masks[f].cpu()
        #     pred_heatmap = metrics_out.sigmoid()[f].cpu()
        #
        #     gt_bbox_centers, pred_centers, rgb_frame, supervised_boxes = get_gt_annotations_for_metrics(
        #         blobs_per_image, cfg, f, frame_number, meta, rgb_frame, test_loader)
        #
        #     # fn, fp, precision, recall, tp = get_precision_recall_for_metrics(cfg, gt_bbox_centers, pred_centers,
        #     #                                                                  ratio)
        #     # tp_list.append(tp)
        #     # fp_list.append(fp)
        #     # fn_list.append(fn)
        #
        #     fig = plot_image_with_features(
        #         rgb_frame.squeeze(dim=0).permute(1, 2, 0).numpy(), gt_bbox_centers,
        #         np.stack(pred_centers), boxes=supervised_boxes,
        #         txt=f'Frame Number: {frame_number}\n'
        #             f'Agent Count: GT-{len(gt_bbox_centers)} | Pred-{len(pred_centers)}',
        #         footnote_txt=f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
        #                      f'Video Number: {cfg.eval.test.video_number_to_use}'
        #                      f'\n\nL2 Matching Threshold: '
        #                      f'{cfg.eval.gt_pred_loc_distance_threshold}m',
        #         video_mode=False,
        #         plot_heatmaps=True,
        #         gt_heatmap=gt_heatmap.squeeze(dim=0).numpy(),
        #         pred_heatmap=pred_heatmap.squeeze(dim=0).numpy())

    # final_precision = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fp_list).sum())
    # final_recall = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fn_list).sum())

    logger.info(f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                f'Video Number: {cfg.eval.test.video_number_to_use}')
    logger.info(f"Threshold: {cfg.eval.gt_pred_loc_distance_threshold}m | "
                f"Max-Pool kernel size: {cfg.eval.objectness.kernel} | "
                f"Head Used: {cfg.eval.objectness.index_select}")
    logger.info(f"Test Loss: {np.array(total_loss).mean()}")

    save_path = os.path.join(os.getcwd(),
                             f'HeatMapPredictions'
                             f'/{getattr(SDDVideoClasses, cfg.eval.video_class).name}'
                             f'/{cfg.eval.test.video_number_to_use}/')  # only for single dataset setup
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_dict = {
        'out_head_0': torch.cat(pred_head_0),
        'out_head_1': torch.cat(pred_head_1),
        'out_head_2': torch.cat(pred_head_2),
        'frames_sequence': frames_sequence,
    }
    torch.save(save_dict, save_path)
    logger.info(f"Saved Predictions at {save_path}")
    # logger.info(f"Precision: {final_precision} | Recall: {final_recall}")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        evaluate_and_store_predicted_maps()
