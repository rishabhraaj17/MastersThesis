import warnings

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import torch
from kornia.losses import BinaryFocalLossWithLogits
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from skimage.feature import blob_log
import torch.distributions as D
from torch.nn import MSELoss
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import ToPILImage

from average_image.constants import SDDVideoClasses
from average_image.utils import SDDMeta
from baselinev2.utils import get_generated_frame_annotations
from gmm import GaussianMixture
from log import get_logger
import models as model_zoo
from train import setup_multiple_datasets_core
from utils import generate_position_map, overlay_images, get_scaled_shapes_with_pad_values, heat_map_collate_fn, \
    plot_predictions

seed_everything(42)
logger = get_logger(__name__)


def plot_to_debug(im):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(8, 10))
    axs.imshow(im)

    plt.tight_layout()
    plt.show()


def get_position_correction_transform(new_shape):
    h, w = new_shape
    transform = A.Compose(
        [A.Resize(height=h, width=w)],
        keypoint_params=A.KeypointParams(format='xy')
    )
    return transform


def correct_locations(blob, v0: bool = False):
    if v0:
        locations = blob[:, :2]
        rolled = np.rollaxis(locations, -1).tolist()
        locations_x, locations_y = rolled[1], rolled[0]
    else:
        locations_x, locations_y = blob[:, 1], blob[:, 0]

    locations = np.stack([locations_x, locations_y]).T
    return locations


def extract_agents_locations(blob_threshold, mask, objectness_threshold):
    mask = mask.sigmoid()
    mask[mask < objectness_threshold] = 0.
    blobs = []
    for m in mask:
        blobs.append(blob_log(m.squeeze(0).round(), threshold=blob_threshold))
    return blobs, mask


def create_mixture_of_gaussians(masks, meta_list, rgb_img,
                                video_class: SDDVideoClasses, video_number: int,
                                objectness_threshold: float = 0.5, blob_threshold: float = 0.45,
                                blob_overlap: float = 0.5, get_generated: bool = True):
    blobs_per_image, masks = extract_agents_locations(blob_threshold, masks, objectness_threshold)

    adjusted_locations, scaled_images = [], []
    for blobs, meta, mask in zip(blobs_per_image, meta_list, masks):
        original_shape = meta['original_shape']
        blobs = correct_locations(blobs)
        transform = get_position_correction_transform(original_shape)
        out = transform(image=mask.squeeze(0).numpy(), keypoints=blobs)
        adjusted_locations.append(out['keypoints'])
        scaled_images.append(out['image'])

    adjusted_locations = np.stack(adjusted_locations)

    # verify
    detected_maps, scaled_detected_maps = [], []
    for blob, scaled_blobs, scaled_image in zip(blobs_per_image, adjusted_locations, scaled_images):
        locations = correct_locations(blob)
        detected_maps.append(generate_position_map([masks.shape[-2], masks.shape[-1]], locations, sigma=1.5,
                                                   heatmap_shape=None,
                                                   return_combined=True, hw_mode=True))

        scaled_detected_maps.append(generate_position_map([scaled_image.shape[-2], scaled_image.shape[-1]],
                                                          np.stack(scaled_blobs),
                                                          sigma=1.5 * 4,
                                                          heatmap_shape=None,
                                                          return_combined=True, hw_mode=True))
    # verify ends

    original_shape = meta_list[0]['original_shape']
    h, w = original_shape

    annotation_root = '../../Datasets/SDD/filtered_generated_annotations/'
    annotation_path = f'{annotation_root}{video_class.value}/video{video_number}/generated_annotations.csv'
    dataset = pd.read_csv(annotation_path)

    frame_annotation = get_generated_frame_annotations(df=dataset, frame_number=0)

    boxes = frame_annotation[:, 1:5].astype(np.int)
    track_idx = frame_annotation[:, 0].astype(np.int)
    bbox_centers = frame_annotation[:, 7:9].astype(np.int)

    inside_boxes_idx = [b for b, box in enumerate(boxes)
                        if (box[0] > 0 and box[2] < w) and (box[1] > 0 and box[3] < h)]

    boxes = boxes[inside_boxes_idx]
    track_idx = track_idx[inside_boxes_idx]
    bbox_centers = bbox_centers[inside_boxes_idx]

    trajectory_map = generate_position_map([original_shape[-2], original_shape[-1]],
                                           bbox_centers,
                                           sigma=1.5 * 4,
                                           heatmap_shape=None,
                                           return_combined=True, hw_mode=True)

    superimposed_image = overlay_images(transformer=ToPILImage(), background=rgb_img[0],
                                        overlay=torch.from_numpy(trajectory_map).unsqueeze(0))
    superimposed_image_flip = overlay_images(transformer=ToPILImage(), background=rgb_img[0],
                                             overlay=torch.from_numpy(scaled_detected_maps[0]).unsqueeze(0))

    # we dont need to learn the mixture u and sigma?? We just need gradients from each position
    gmm = GaussianMixture(n_components=adjusted_locations[0].shape[0],
                          n_features=2,
                          mu_init=torch.from_numpy(adjusted_locations[0]).unsqueeze(0),
                          var_init=(torch.ones(adjusted_locations[0].shape) * 1.5).unsqueeze(0))

    loss = - gmm.score_samples(torch.from_numpy(bbox_centers))

    print()


def setup_interact_dataset(cfg):
    meta = SDDMeta(cfg.root + 'H_SDD.txt')
    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.interact.video_classes_to_use,
        video_numbers=cfg.interact.video_numbers_to_use,
        desired_ratio=cfg.interact.desired_pixel_to_meter_ratio_rgb)

    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=cfg.interact.video_classes_to_use,
        video_numbers=cfg.interact.video_numbers_to_use,
        desired_ratio=cfg.interact.desired_pixel_to_meter_ratio)

    interact_datasets = setup_multiple_datasets_core(cfg, meta, video_classes_to_use=cfg.interact.video_classes_to_use,
                                                     video_numbers_to_use=cfg.interact.video_numbers_to_use,
                                                     num_videos=cfg.interact.num_videos,
                                                     multiple_videos=cfg.interact.multiple_videos,
                                                     df=df, df_target=df_target, rgb_max_shape=rgb_max_shape)
    # val_datasets = setup_multiple_datasets_core(cfg, meta, video_classes_to_use=cfg.val.video_classes_to_use,
    #                                             video_numbers_to_use=cfg.val.video_numbers_to_use,
    #                                             num_videos=cfg.val.num_videos,
    #                                             multiple_videos=cfg.val.multiple_videos,
    #                                             df=df, df_target=df_target, rgb_max_shape=rgb_max_shape)
    return interact_datasets, target_max_shape


@hydra.main(config_path="config", config_name="config")
def interact_demo(cfg):
    logger.info(f'Interact - Setting up DataLoader and Model...')

    train_dataset, target_max_shape = setup_interact_dataset(cfg)

    position_map_network_type = getattr(model_zoo, cfg.interact.postion_map_network_type)
    if position_map_network_type.__name__ in ['PositionMapUNetPositionMapSegmentation',
                                              'PositionMapUNetClassMapSegmentation',
                                              'PositionMapUNetHeatmapSegmentation',
                                              'PositionMapStackedHourGlass']:
        loss_fn = BinaryFocalLossWithLogits(alpha=cfg.interact.focal_loss_alpha, reduction='mean')  # CrossEntropyLoss()
    else:
        loss_fn = MSELoss()

    position_map_model = position_map_network_type(config=cfg, train_dataset=None, val_dataset=None,
                                                   loss_function=loss_fn, collate_fn=heat_map_collate_fn,
                                                   desired_output_shape=target_max_shape)

    trajectory_model = model_zoo.TrajectoryModel(cfg)

    model = model_zoo.PositionMapWithTrajectories(config=cfg, position_map_model=position_map_model,
                                                  trajectory_model=trajectory_model, train_dataset=train_dataset,
                                                  val_dataset=None, desired_output_shape=target_max_shape,
                                                  loss_function=loss_fn, collate_fn=heat_map_collate_fn)

    model.to(cfg.interact.device)

    opt = torch.optim.Adam(position_map_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                           amsgrad=cfg.amsgrad)

    train_subset = Subset(dataset=train_dataset, indices=list(cfg.interact.subset_indices))
    train_loader = DataLoader(train_subset, batch_size=cfg.interact.batch_size, shuffle=False,
                              num_workers=cfg.interact.num_workers, collate_fn=heat_map_collate_fn,
                              pin_memory=cfg.interact.pin_memory, drop_last=cfg.interact.drop_last)

    last_iter_output = None
    for epoch in range(cfg.interact.num_epochs):
        model.train()

        train_loss = []
        for data in train_loader:
            opt.zero_grad()

            frames, heat_masks, position_map, distribution_map, class_maps, meta = data

            if position_map_network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                frames, class_maps = frames.to(cfg.device), class_maps.to(cfg.device)
            elif position_map_network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                frames, position_map = frames.to(cfg.device), position_map.to(cfg.device)
            elif position_map_network_type.__name__ in ['PositionMapUNetHeatmapSegmentation',
                                                        'PositionMapStackedHourGlass']:
                frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)
            else:
                frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

            out = model.position_map_model(frames)

            if position_map_network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                loss = loss_fn(out, class_maps.long().squeeze(dim=1))
            elif position_map_network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                loss = loss_fn(out, position_map.long().squeeze(dim=1))
            elif position_map_network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                loss = loss_fn(out, heat_masks)
            elif position_map_network_type.__name__ == 'PositionMapStackedHourGlass':
                loss = position_map_model.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
                loss = loss.mean()
            else:
                loss = loss_fn(out, heat_masks)

            train_loss.append(loss.item())

            loss.backward()
            opt.step()

        logger.info(f"Epoch: {epoch} | Train Loss: {np.array(train_loss).mean()}")

        if epoch % cfg.interact.plot_checkpoint == 0:
            position_map_model.eval()
            val_loss = []

            for data in train_loader:
                frames, heat_masks, position_map, distribution_map, class_maps, meta = data

                if position_map_network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                    frames, class_maps = frames.to(cfg.device), class_maps.to(cfg.device)
                elif position_map_network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                    frames, position_map = frames.to(cfg.device), position_map.to(cfg.device)
                elif position_map_network_type.__name__ in ['PositionMapUNetHeatmapSegmentation',
                                                            'PositionMapStackedHourGlass']:
                    frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)
                else:
                    frames, heat_masks = frames.to(cfg.device), heat_masks.to(cfg.device)

                with torch.no_grad():
                    out = position_map_model(frames)

                if position_map_network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                    loss = loss_fn(out, class_maps.long().squeeze(dim=1))
                elif position_map_network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                    loss = loss_fn(out, position_map.long().squeeze(dim=1))
                elif position_map_network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                    loss = loss_fn(out, heat_masks)
                elif position_map_network_type.__name__ == 'PositionMapStackedHourGlass':
                    loss = position_map_model.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
                    loss = loss.mean()
                else:
                    loss = loss_fn(out, heat_masks)

                val_loss.append(loss.item())

                random_idx = np.random.choice(cfg.interact.batch_size, 1, replace=False).item()

                if position_map_network_type.__name__ in ['PositionMapUNetPositionMapSegmentation',
                                                          'PositionMapUNetClassMapSegmentation']:
                    pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     class_maps[random_idx].squeeze().cpu()
                                     if cfg.class_map_segmentation else position_map[random_idx].squeeze().cpu(),
                                     pred_mask[random_idx].int() * 255,
                                     additional_text=f"{position_map_network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")
                elif position_map_network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                    pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     pred_mask[random_idx].int() * 255,
                                     additional_text=f"{position_map_network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")
                elif position_map_network_type.__name__ == 'PositionMapStackedHourGlass':
                    pred_mask = torch.round(torch.sigmoid(out)).squeeze(dim=1).cpu()
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     class_maps[random_idx].squeeze().cpu()
                                     if cfg.class_map_segmentation else position_map[random_idx].squeeze().cpu(),
                                     pred_mask[-1][random_idx].int().squeeze(dim=0) * 255,
                                     additional_text=f"{position_map_network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")
                    plot_predictions(pred_mask[-3][random_idx].int().squeeze(dim=0) * 255,
                                     pred_mask[-2][random_idx].int().squeeze(dim=0) * 255,
                                     pred_mask[-1][random_idx].int().squeeze(dim=0) * 255,
                                     additional_text=f"{position_map_network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}\nLast 3 HeatMaps", all_heatmaps=True)
                else:
                    plot_predictions(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                                     heat_masks[random_idx].squeeze().cpu(),
                                     out[random_idx].squeeze().cpu(),
                                     additional_text=f"{position_map_network_type.__name__} | {loss_fn._get_name()} "
                                                     f"| Epoch: {epoch}")

            logger.info(f"Epoch: {epoch} | Validation Loss: {np.array(val_loss).mean()}")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        interact_demo()
    # sample_path = '../../Plots/proposed_method/v0/position_maps/' \
    #               'PositionMapUNetHeatmapSegmentation_BinaryFocalLossWithLogits/' \
    #               'version_424798/epoch=61-step=72787/sample.pt'
    # sample = torch.load(sample_path)
    # rgb_im, heat_mask, pred_mask, m = sample['rgb'], sample['mask'], sample['out'], sample['meta']
    # create_mixture_of_gaussians(masks=pred_mask, blob_overlap=0.2, blob_threshold=0.2,
    #                             get_generated=False, meta_list=m, rgb_img=rgb_im,
    #                             video_class=SDDVideoClasses.DEATH_CIRCLE, video_number=2)  # these params look good
