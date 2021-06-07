import os
import warnings

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import scipy
import torch
from kornia.losses import BinaryFocalLossWithLogits
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from skimage.feature import blob_log
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


class PositionBasedTrack(object):
    def __init__(self, idx: int, frames: np.ndarray, locations: np.ndarray):
        super(PositionBasedTrack, self).__init__()
        self.idx = idx
        self.frames = frames
        self.locations = locations

    def __eq__(self, other):
        return self.idx == other.idx

    def __repr__(self):
        return f"Track ID: {self.idx}\nFrames: {self.frames}\nTrack Positions: {self.locations}"


def plot_to_debug(im, txt=''):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(8, 10))
    axs.imshow(im)

    plt.tight_layout(pad=1.58)
    plt.title(txt)
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
                                                     df=df, df_target=df_target, rgb_max_shape=rgb_max_shape,
                                                     use_common_transforms=False)
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
    checkpoint_path = f'{cfg.interact.checkpoint.root}{cfg.interact.checkpoint.path}{cfg.interact.checkpoint.version}/' \
                      f'checkpoints/'
    checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]
    logger.info(f'Loading Position Map Model weights from: {checkpoint_file}')
    load_dict = torch.load(checkpoint_file, map_location=cfg.interact.device)

    position_map_model.load_state_dict(load_dict['state_dict'])

    trajectory_model = model_zoo.TrajectoryModel(cfg)

    model = model_zoo.PositionMapWithTrajectories(config=cfg, position_map_model=position_map_model,
                                                  trajectory_model=trajectory_model, train_dataset=train_dataset,
                                                  val_dataset=None, desired_output_shape=target_max_shape,
                                                  loss_function=loss_fn, collate_fn=heat_map_collate_fn)

    model.freeze_position_map_model()
    model.to(cfg.interact.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.interact.lr,
                           weight_decay=cfg.interact.weight_decay,
                           amsgrad=cfg.interact.amsgrad)

    train_subset = Subset(dataset=train_dataset, indices=list(cfg.interact.subset_indices))
    train_loader = DataLoader(train_subset, batch_size=cfg.interact.batch_size, shuffle=False,
                              num_workers=cfg.interact.num_workers, collate_fn=heat_map_collate_fn,
                              pin_memory=cfg.interact.pin_memory, drop_last=cfg.interact.drop_last)

    last_iter_output, last_iter_blobs, last_frame_number = None, None, None
    running_tracks, track_ids_used = [], []
    in_xy, in_dxdy, out_xy = [], [], []
    for epoch in range(cfg.interact.num_epochs):
        model.train()

        train_loss = []
        for t_idx, data in enumerate(train_loader):
            # opt.zero_grad()

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

            # plot_to_debug(frames.cpu().squeeze().permute(1, 2, 0), txt=f'RGB @ T = {t_idx}')
            # plot_to_debug(out.cpu().detach().squeeze().sigmoid().round(), txt=f'Pred Mask @ T = {t_idx}')

            agents_count_gt = meta[0]['bbox_centers'].shape[0]
            blobs_per_image, masks = extract_agents_locations(blob_threshold=cfg.interact.blob_threshold,
                                                              mask=out.clone().detach().cpu(),
                                                              objectness_threshold=cfg.interact.objectness_threshold)
            detected_maps = []
            for blob in blobs_per_image:
                locations = correct_locations(blob)
                detected_maps.append(generate_position_map([masks.shape[-2], masks.shape[-1]], locations, sigma=1.5,
                                                           heatmap_shape=None,
                                                           return_combined=True, hw_mode=True))

            # plot_to_debug(detected_maps[0], txt=f'Blobs->Mask @ T = {t_idx}')
            blobs = correct_locations(blobs_per_image[0])  # for batch_size=1

            if t_idx > 0:
                current_frame_number = meta[0]['item']
                blob_distance_matrix = np.zeros((last_iter_blobs.shape[0], blobs.shape[0]))
                for p_idx, prev_blob in enumerate(last_iter_blobs):
                    for b_idx, blob in enumerate(blobs):
                        blob_distance_matrix[p_idx, b_idx] = np.linalg.norm((prev_blob - blob), ord=2)

                # Hungarian
                match_rows, match_cols = scipy.optimize.linear_sum_assignment(blob_distance_matrix)

                last_track_id_used = track_ids_used[-1] if len(track_ids_used) != 0 else 0
                for r, c in zip(match_rows, match_cols):
                    in_xy.append(last_iter_blobs[r])
                    out_xy.append(blobs[c])
                    in_dxdy.append(blobs[c] - last_iter_blobs[r])
                    is_part_of_live_track = [(i.locations[-1] == last_iter_blobs[r]).all() for i in running_tracks]

                    if np.array(is_part_of_live_track).any():
                        track_idx = np.where(is_part_of_live_track)[0]
                        if track_idx.shape[0] > 1:
                            for t in track_idx:
                                is_part_of_live_track[t] = False if running_tracks[t].frames.shape[0] > t_idx \
                                    else is_part_of_live_track[t]

                        track_idx = np.where(is_part_of_live_track)[0].item()

                        running_tracks[track_idx].frames = np.append(running_tracks[track_idx].frames,
                                                                     [current_frame_number])
                        running_tracks[track_idx].locations = np.append(running_tracks[track_idx].locations,
                                                                        [blobs[c]], axis=0)
                    else:
                        running_tracks.append(PositionBasedTrack(
                            idx=last_track_id_used,
                            frames=np.array([last_frame_number, current_frame_number]),
                            locations=np.array([last_iter_blobs[r], blobs[c]])))

                        track_ids_used.append(last_track_id_used)
                        last_track_id_used += 1

                # plot_to_debug(last_iter_output.cpu().squeeze().sigmoid().round(), txt=f'Pred Mask @ T = {t_idx - 1}')

                in_xy, in_dxdy, out_xy = np.stack(in_xy), np.stack(in_dxdy), np.stack(out_xy)
                in_xy, in_dxdy, out_xy = torch.from_numpy(in_xy), torch.from_numpy(in_dxdy), torch.from_numpy(out_xy)
                in_xy, in_dxdy, out_xy = \
                    in_xy.float().unsqueeze(0).to(cfg.interact.device), \
                    in_dxdy.float().unsqueeze(0).to(cfg.interact.device), \
                    out_xy.float().unsqueeze(0).to(cfg.interact.device)
                batch = {'in_xy': in_xy, 'in_dxdy': in_dxdy, 'out_xy': out_xy}

                in_map = generate_position_map([heat_masks.shape[-2], heat_masks.shape[-1]],
                                               in_xy.squeeze().detach().clone().cpu().numpy(),
                                               sigma=1.5,
                                               heatmap_shape=None,
                                               return_combined=True, hw_mode=True)
                plot_to_debug(in_map, txt=f'Input locations @ T = {t_idx}')
                true_out_map = generate_position_map([heat_masks.shape[-2], heat_masks.shape[-1]],
                                                     out_xy.squeeze().detach().clone().cpu().numpy(),
                                                     sigma=1.5,
                                                     heatmap_shape=None,
                                                     return_combined=True, hw_mode=True)
                plot_to_debug(true_out_map, txt=f'True locations @ T = {t_idx}')

                gmm = GaussianMixture(n_components=out_xy.shape[1],
                                      n_features=out_xy.shape[-1],
                                      mu_init=out_xy,
                                      var_init=torch.ones_like(out_xy) * 1.5)
                for t_i in range(9000):
                    opt.zero_grad()
                    traj_out = model.trajectory_model(batch)
                    traj_out_xy = traj_out['out_xy']

                    # loss = - gmm.score_samples(traj_out_xy.squeeze(0)).sum()
                    loss = torch.linalg.norm((out_xy - traj_out_xy))
                    logger.info(f'Loss @ {t_i}: {loss.item()}')
                    loss.backward()
                    opt.step()

                trajectory_map = generate_position_map([heat_masks.shape[-2], heat_masks.shape[-1]],
                                                       traj_out_xy.squeeze().detach().clone().cpu().numpy(),
                                                       sigma=1.5,
                                                       heatmap_shape=None,
                                                       return_combined=True, hw_mode=True)
                plot_to_debug(trajectory_map, txt=f'Predicted locations @ T = {t_idx}')

            last_iter_output = out.clone().detach()
            last_iter_blobs = np.copy(blobs)
            last_frame_number = meta[0]['item']
            in_xy, in_dxdy, out_xy = [], [], []

            if position_map_network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                loss = loss_fn(out, class_maps.long().squeeze(dim=1))
            elif position_map_network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                loss = loss_fn(out, position_map.long().squeeze(dim=1))
            elif position_map_network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                loss = torch.tensor([0.])  # loss_fn(out, heat_masks)
            elif position_map_network_type.__name__ == 'PositionMapStackedHourGlass':
                loss = model.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
                loss = loss.mean()
            else:
                loss = loss_fn(out, heat_masks)

            train_loss.append(loss.item())

            # loss.backward()
            # opt.step()

        logger.info(f"Epoch: {epoch} | Train Loss: {np.array(train_loss).mean()}")

        if epoch % cfg.interact.plot_checkpoint == 0:
            model.eval()
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
                    out = model(frames)

                if position_map_network_type.__name__ == 'PositionMapUNetClassMapSegmentation':
                    loss = loss_fn(out, class_maps.long().squeeze(dim=1))
                elif position_map_network_type.__name__ == 'PositionMapUNetPositionMapSegmentation':
                    loss = loss_fn(out, position_map.long().squeeze(dim=1))
                elif position_map_network_type.__name__ == 'PositionMapUNetHeatmapSegmentation':
                    loss = loss_fn(out, heat_masks)
                elif position_map_network_type.__name__ == 'PositionMapStackedHourGlass':
                    loss = model.network.calc_loss(combined_hm_preds=out, heatmaps=heat_masks)
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
