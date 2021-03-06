from typing import List, Tuple, Optional

import albumentations as A
import numpy as np
import timeout_decorator
import torch
import torchvision
from matplotlib import pyplot as plt
from mmdet.models.utils.gaussian_target import get_local_maximum
from torch.nn.functional import pad

from average_image.constants import SDDVideoClasses
from baseline.extracted_of_optimization import find_points_inside_circle, is_point_inside_circle
from baselinev2.exceptions import TimeoutException

TIMEOUT = 10  # seconds


def locations_from_heatmaps(frames, kernel, loc_cutoff, marker_size, out, vis_on=False, threshold=None):
    if threshold is not None:
        out = [torch.threshold(o.sigmoid(), threshold=threshold, value=0) for o in out]
    else:
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


def get_position_correction_transform(new_shape):
    h, w = new_shape
    transform = A.Compose(
        [A.Resize(height=h, width=w)],
        keypoint_params=A.KeypointParams(format='xy')
    )
    return transform


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


def get_adjusted_object_locations_rgb(locations, rgb_frames, meta):
    adjusted_locations, scaled_images = [], []
    for blobs, m, frame in zip(locations, meta, rgb_frames):
        original_shape = m['original_shape']
        transform = get_position_correction_transform(original_shape)
        out = transform(image=frame.permute(1, 2, 0).numpy(), keypoints=blobs.numpy())
        adjusted_locations.append(out['keypoints'])
        scaled_images.append(out['image'])

    # masks = np.stack(scaled_images)  # problems in stacking diff images size
    masks = scaled_images

    return adjusted_locations, masks


def get_processed_patches_to_train(crop_h, crop_w, frames, heat_masks, l_idx, locations):
    crops_filtered, target_crops_filtered, filtered_idx = [], [], []
    crop_box_ijwh, crops, target_crops = get_patches(crop_h, crop_w, frames, heat_masks, l_idx, locations)
    for f_idx, (c, tc) in enumerate(zip(crops, target_crops)):
        if c.numel() != 0:
            if c.shape[-1] != crop_w or c.shape[-2] != crop_h:
                diff_h = crop_h - c.shape[2]
                diff_w = crop_w - c.shape[3]

                c = pad(c, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
                        mode='replicate')
                tc = pad(tc, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
                         mode='constant')
            crops_filtered.append(c)
            target_crops_filtered.append(tc)
            filtered_idx.append(f_idx)
    crops_filtered = torch.cat(crops_filtered) if len(crops_filtered) != 0 else []
    target_crops = torch.cat(target_crops_filtered) if len(target_crops) != 0 else []
    valid_boxes = crop_box_ijwh[filtered_idx].to(dtype=torch.int32) if len(filtered_idx) != 0 else []
    return crops_filtered, target_crops, valid_boxes


def get_processed_patches_to_train_rgb_only(crop_h, crop_w, frames, l_idx, locations):
    crops_filtered, filtered_idx = [], []
    crop_box_ijwh, crops = get_patches_rgb_only(crop_h, crop_w, frames, l_idx, locations)
    for f_idx, c in enumerate(crops):
        if c.numel() != 0:
            if c.shape[-1] != crop_w or c.shape[-2] != crop_h:
                diff_h = crop_h - c.shape[2]
                diff_w = crop_w - c.shape[3]

                c = pad(c, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
                        mode='replicate')
            crops_filtered.append(c)
            filtered_idx.append(f_idx)
    crops_filtered = torch.cat(crops_filtered) if len(crops_filtered) != 0 else []
    valid_boxes = crop_box_ijwh[filtered_idx].to(dtype=torch.int32) if len(filtered_idx) != 0 else []
    return crops_filtered, valid_boxes


def get_patches(crop_h, crop_w, frames, heat_masks, l_idx, locations):
    crop_box_ijwh = get_boxes_for_patches(crop_h, crop_w, locations)
    crops = [torchvision.transforms.F.crop(
        frames[l_idx].unsqueeze(0).cpu(), box[0].item(), box[1].item(), box[2].item(), box[3].item())
        for box in crop_box_ijwh.to(dtype=torch.int32)]
    target_crops = [torchvision.transforms.F.crop(
        heat_masks[l_idx].unsqueeze(0).cpu(), box[0].item(), box[1].item(), box[2].item(), box[3].item())
        for box in crop_box_ijwh.to(dtype=torch.int32)]
    return crop_box_ijwh, crops, target_crops


def get_patches_rgb_only(crop_h, crop_w, frames, l_idx, locations):
    crop_box_ijwh = get_boxes_for_patches(crop_h, crop_w, locations)
    crops = [torchvision.transforms.F.crop(
        frames[l_idx].unsqueeze(0).cpu(), box[0].item(), box[1].item(), box[2].item(), box[3].item())
        for box in crop_box_ijwh.to(dtype=torch.int32)]
    return crop_box_ijwh, crops


def get_boxes_for_patches(crop_h, crop_w, locations):
    crop_box_cxcywh = torch.stack([torch.tensor([kp[0], kp[1], crop_w, crop_h]) for kp in locations])
    crop_box_ijwh = torchvision.ops.box_convert(crop_box_cxcywh, 'cxcywh', 'xywh')
    crop_box_ijwh = torch.stack([torch.tensor([b[1], b[0], b[2], b[3]]) for b in crop_box_ijwh])
    return crop_box_ijwh


@timeout_decorator.timeout(seconds=TIMEOUT, timeout_exception=TimeoutException)
def prune_locations_proximity_based(cluster_centers, radius):
    rejected_cluster_centers = []
    rejected_cluster_centers_idx = []
    pruned_cluster_centers = []
    pruned_cluster_centers_idx = []
    for cluster_center in cluster_centers:
        if not np.isin(cluster_center, pruned_cluster_centers).all() \
                or not np.isin(cluster_center, rejected_cluster_centers).all():

            centers_inside_idx = find_points_inside_circle(cluster_centers,
                                                           circle_center=cluster_center,
                                                           circle_radius=radius)
            for center_inside_idx in centers_inside_idx:
                if not is_cluster_center_in_the_radius_of_one_of_pruned_centers(
                        cluster_centers[center_inside_idx], pruned_cluster_centers, radius, rejected_cluster_centers):

                    if not np.isin(cluster_centers[center_inside_idx], pruned_cluster_centers).all() and \
                            not np.isin(cluster_centers[center_inside_idx], rejected_cluster_centers).all():
                        pruned_cluster_centers.append(cluster_centers[center_inside_idx])
                        pruned_cluster_centers_idx.append(center_inside_idx)
                else:
                    if not np.isin(cluster_centers[center_inside_idx], rejected_cluster_centers).all() and \
                            not np.isin(cluster_centers[center_inside_idx], pruned_cluster_centers).all():
                        rejected_cluster_centers.append(cluster_centers[center_inside_idx])
                        rejected_cluster_centers_idx.append(center_inside_idx)

    pruned_cluster_centers = np.stack(pruned_cluster_centers) if len(pruned_cluster_centers) != 0 else np.zeros((0, 2))
    return pruned_cluster_centers, pruned_cluster_centers_idx


def is_cluster_center_in_the_radius_of_one_of_pruned_centers(cluster_center, pruned_cluster_centers, radius,
                                                             rejected_cluster_centers):
    for pruned_cluster_center in pruned_cluster_centers:
        if is_point_inside_circle(circle_x=pruned_cluster_center[0], circle_y=pruned_cluster_center[1],
                                  rad=radius, x=cluster_center[0], y=cluster_center[1]):
            # rejected_cluster_centers.append(cluster_center)
            return True

    return False


class Location(object):
    def __init__(self, frame_number: int, locations: np.ndarray, pruned_locations: np.ndarray,
                 scaled_locations: np.ndarray):
        # pruned locations are scaled
        self.frame_number = frame_number
        self.locations = locations
        self.scaled_locations = scaled_locations
        self.pruned_locations = pruned_locations

    def __repr__(self):
        return f"Frame: {self.frame_number}"


class Locations(object):
    def __init__(self, locations: List[Location]):
        self.locations = locations

    def __repr__(self):
        frames_yet = [loc.frame_number for loc in self.locations]
        return f"Covered Frames{frames_yet}"


class ExtractedLocations(object):
    def __init__(self, video_class: SDDVideoClasses, video_numbers: int,
                 shape: Tuple[int, int], scaled_shape: Tuple[int, int],
                 padded_shape: Tuple[int, int],
                 head0: Locations, head1: Locations, head2: Optional[Locations] = None):
        self.video_class = video_class
        self.video_numbers = video_numbers
        self.head0 = head0
        self.head1 = head1
        self.head2 = head2
        self.shape = shape
        self.scaled_shape = scaled_shape
        self.padded_shape = padded_shape
