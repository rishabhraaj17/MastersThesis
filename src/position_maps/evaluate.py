import os
import warnings

import albumentations as A
import hydra
import numpy as np
import torch
from kornia.losses import BinaryFocalLossWithLogits
from pytorch_lightning import seed_everything
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from log import get_logger
import models as model_zoo
from dataset import SDDFrameAndAnnotationDataset
from train import setup_multiple_datasets_core
from utils import heat_map_collate_fn, plot_predictions, get_blob_count

seed_everything(42)
logger = get_logger(__name__)


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
    test_w, test_h = get_resize_dims(cfg)

    transform = setup_test_transform(cfg, test_h, test_w)

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
        heatmap_region_limit_threshold=cfg.eval.heatmap_region_limit_threshold
    )
    return test_dataset


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
    datasets = setup_multiple_datasets_core(cfg, meta, video_classes_to_use=cfg.eval.test.video_classes_to_use,
                                            video_numbers_to_use=cfg.eval.test.video_numbers_to_use,
                                            num_videos=cfg.eval.test.num_videos,
                                            multiple_videos=cfg.eval.test.multiple_videos)
    return datasets


def setup_eval(cfg):
    logger.info(f'Setting up DataLoader')
    test_dataset = setup_dataset(cfg)
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
                         loss_function=loss_fn, collate_fn=heat_map_collate_fn)

    logger.info(f'Setting up Model')
    checkpoint_path = f'{cfg.eval.checkpoint.root}{cfg.eval.checkpoint.path}{cfg.eval.checkpoint.version}/checkpoints/'
    checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]

    logger.info(f'Loading weights from: {checkpoint_file}')
    load_dict = torch.load(checkpoint_file)

    model.load_state_dict(load_dict['state_dict'])
    model.to(cfg.eval.device)
    model.eval()

    return loss_fn, model, network_type, test_loader, checkpoint_file


@hydra.main(config_path="config", config_name="config")
def evaluate(cfg):
    loss_fn, model, network_type, test_loader, checkpoint_file = setup_eval(cfg)

    logger.info(f'Starting evaluation...')

    total_loss = []
    for idx, data in enumerate(tqdm(test_loader)):
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

        total_loss.append(loss.item())

        random_idx = np.random.choice(cfg.eval.batch_size, 1, replace=False).item()

        if idx % cfg.eval.plot_checkpoint == 0:
            current_random_frame = meta[random_idx]['item']
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

                plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                 heat_masks[random_idx].squeeze().cpu(),
                                 pred_mask[random_idx].int() * 255,
                                 additional_text=f"{network_type.__name__} | {loss_fn._get_name()} | "
                                                 f"Frame: {current_random_frame}\n"
                                                 f"Agent Count : [GT: {num_objects_gt}| "
                                                 f"Prediction: {num_objects_pred}]",
                                 save_dir=f'{cfg.eval.plot_save_dir}{network_type.__name__}_{loss_fn._get_name()}/'
                                          f'version_{cfg.eval.checkpoint.version}/'
                                          f'{os.path.split(checkpoint_file)[-1][:-5]}/',
                                 img_name=f'frame_{current_random_frame}')
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
