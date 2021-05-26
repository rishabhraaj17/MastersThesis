import copy
import os
import shutil
from typing import Tuple, List, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from torch.nn.functional import interpolate
from tqdm import tqdm

from average_image.bbox_utils import _process_scale, cal_centers
from average_image.constants import SDDVideoClasses
from baselinev2.plot_utils import add_box_to_axes, add_features_to_axis


def gaussian_v0(x: int, y: int, height: int, width: int, sigma: int = 5) -> 'np.ndarray':
    channel = [np.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(height) for c in range(width)]
    channel = np.array(channel, dtype=np.float32).reshape((height, width))
    return channel


def generate_position_map_v0(shape: Tuple[int, int], bounding_boxes_centers: List[List[int]],
                             sigma: int = 2, return_as_one: bool = True,
                             normalized: bool = True,
                             gaussian_to_use: str = 'v2') -> Union[List['np.ndarray'], 'np.ndarray']:
    if gaussian_to_use == 'v0':
        core = gaussian_v0
    elif gaussian_to_use == 'v1':
        core = gaussian_v1
    elif gaussian_to_use == 'v2':
        core = gaussian_v2
    else:
        return NotImplemented
    h, w = shape
    masks = []
    for center in bounding_boxes_centers:
        x, y = center
        masks.append(core(x=x, y=y, height=h, width=w, sigma=sigma))
    if return_as_one:
        return np.stack(masks).sum(axis=0)
    return masks


def plot_samples(img, mask, boxes, box_centers, plot_boxes=False, add_feats_to_mask=False, add_feats_to_img=False,
                 additional_text=''):
    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
    img_axis, mask_axis = axs
    if img is not None:
        img_axis.imshow(img)

    if mask is not None:
        mask_axis.imshow(mask, cmap='hot')

    if boxes is not None and plot_boxes:
        add_box_to_axes(img_axis, boxes)
        add_box_to_axes(mask_axis, boxes)

    if add_feats_to_img:
        add_features_to_axis(img_axis, box_centers)
    if add_feats_to_mask:
        add_features_to_axis(mask_axis, box_centers)

    img_axis.set_title('RGB')
    mask_axis.set_title('Mask')

    fig.suptitle(additional_text)

    plt.show()


def plot_predictions(img, mask, pred_mask, additional_text='', all_heatmaps=False):
    fig, axs = plt.subplots(1, 3, sharex='none', sharey='none', figsize=(16, 8))
    img_axis, mask_axis, pred_mask_axis = axs
    if img is not None:
        if all_heatmaps:
            img_axis.imshow(img, cmap='hot')
        else:
            img_axis.imshow(img)

    if mask is not None:
        mask_axis.imshow(mask, cmap='hot')

    if pred_mask is not None:
        pred_mask_axis.imshow(pred_mask, cmap='hot')

    if all_heatmaps:
        img_axis.set_title('Predicted Mask -3')
        mask_axis.set_title('Predicted Mask -2')
        pred_mask_axis.set_title('Predicted Mask -1')
    else:
        img_axis.set_title('RGB')
        mask_axis.set_title('Mask')
        pred_mask_axis.set_title('Predicted Mask')

    fig.suptitle(additional_text)

    plt.show()


def gaussian_one_dimensional(grid, side_length, loc, sigma, normalized=True):
    if normalized:
        return 1 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(grid - loc) ** 2 / (2 * sigma ** 2))
    return np.sqrt(side_length) * np.exp(- (grid - loc) ** 2 / (2 * sigma ** 2))


def gaussian_v1(x: int, y: int, height: int, width: int, sigma: int = 5, normalized: bool = True) -> 'np.ndarray':
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    gaussian_x = gaussian_one_dimensional(xx, width, x, sigma, normalized=normalized)
    gaussian_y = gaussian_one_dimensional(yy, height, y, sigma, normalized=normalized)
    return gaussian_x * gaussian_y


def gaussian_v2(x: int, y: int, height: int, width: int, sigma: int = 5, seed: int = 42) -> 'np.ndarray':
    img = np.dstack(np.mgrid[0:height:1, 0:width:1])
    rv = multivariate_normal(mean=[y, x], cov=sigma, seed=seed)
    return rv.pdf(img)


# not usable
def gaussian_v3(empty_map, x, y, stride, sigma):
    n_sigma = 4
    tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
    map_h, map_w = empty_map.shape
    br[0] = min(br[0], map_w * stride)
    br[1] = min(br[1], map_h * stride)

    shift = stride / 2 - 0.5
    for map_y in range(tl[1] // stride, br[1] // stride):
        for map_x in range(tl[0] // stride, br[0] // stride):
            d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                 (map_y * stride + shift - y) * (map_y * stride + shift - y)
            exponent = d2 / 2 / sigma / sigma
            if exponent > 4.6052:  # threshold, ln(100), ~0.01
                continue
            empty_map[map_y, map_x] += np.exp(-exponent)
            if empty_map[map_y, map_x] > 1:
                empty_map[map_y, map_x] = 1
    return empty_map


def generate_position_map(image_shape, object_locations, sigma: float, heatmap_shape=None, return_combined=False,
                          hw_mode=True):
    heatmap_shape = copy.copy(image_shape) if heatmap_shape is None else heatmap_shape
    if hw_mode:
        image_shape[0], image_shape[1] = image_shape[1], image_shape[0]
        heatmap_shape[0], heatmap_shape[1] = heatmap_shape[1], heatmap_shape[0]
    heatmap_shape, image_shape = np.array(heatmap_shape), np.array(image_shape)
    num_objects = object_locations.shape[0]
    heatmaps = np.zeros((num_objects,
                         heatmap_shape[1],
                         heatmap_shape[0]),
                        dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_objects):
        feat_stride = image_shape / heatmap_shape
        mu_x = int(object_locations[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(object_locations[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_shape[0] or ul[1] >= heatmap_shape[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_shape[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_shape[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_shape[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_shape[1])

        heatmaps[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    if return_combined:
        return heatmaps.sum(axis=0)
    return heatmaps


def copy_filtered_annotations(root_path, other_root_path):
    filtered_generated_path = root_path + '/filtered_generated_annotations'
    v_clazzes = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
                 SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS,
                 SDDVideoClasses.QUAD]
    v_numbers = [[i for i in range(7)], [i for i in range(4)], [i for i in range(5)], [i for i in range(9)],
                 [i for i in range(15)], [i for i in range(4)], [i for i in range(12)], [i for i in range(4)]]

    for idx, v_clz in enumerate(tqdm(v_clazzes)):
        for v_num in v_numbers[idx]:
            print(f"************** Processing Features: {v_clz.name} - {v_num} *******************************")
            to_move_path = f"{filtered_generated_path}/{v_clz.value}/video{v_num}/generated_annotations.csv"
            os.remove(to_move_path)
            annotation_save_path = f'{other_root_path}Plots/baseline_v2/v0/{v_clz.value}{v_num}/csv_annotation/' \
                                   f'filtered_generated_annotations.csv'
            shutil.copyfile(annotation_save_path, to_move_path)
    print("Done!")


def scale_annotations(bbox, original_shape, new_shape):
    x_min, y_min, x_max, y_max = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x, y, w, h = x_min, y_min, (x_max - x_min), (y_max - y_min)
    h, w, x, y = _process_scale(h, new_shape, original_shape, w, x, y)
    x_min_, y_min_, x_max_, y_max_ = x, y, x + w, y + h

    scaled_annotations = np.vstack((x_min_, y_min_, x_max_, y_max_)).T
    scaled_centers = cal_centers(scaled_annotations)

    return scaled_annotations, scaled_centers


def resize_transform(img, boxes, scale, desired_size, mode='bilinear'):
    original_shape = (img.shape[-2], img.shape[-1])
    new_shape = ()

    if scale is not None:
        resized_image = interpolate(img, scale_factor=scale, mode=mode, recompute_scale_factor=True)
        new_shape = (resized_image.shape[-2], resized_image.shape[-1])
    if desired_size is not None:
        desired_size = list(desired_size)
        resized_image = interpolate(img, size=desired_size, mode=mode)
        new_shape = desired_size

    new_boxes, new_centers = scale_annotations(
        boxes,
        original_shape=original_shape,
        new_shape=new_shape)

    new_boxes = torch.from_numpy(new_boxes)
    new_centers = torch.from_numpy(new_centers)

    return resized_image, new_boxes, new_centers, original_shape, new_shape


def heat_map_collate_fn(batch):
    rgb_img_list, mask_list, position_map_list, distribution_map_list, cm_list, meta_list = [], [], [], [], [], []
    for batch_item in batch:
        rgb, mask, position_map, distribution_map, class_maps, meta = batch_item
        rgb_img_list.append(rgb)
        mask_list.append(mask)
        meta_list.append(meta)
        position_map_list.append(position_map)
        distribution_map_list.append(distribution_map)
        cm_list.append(class_maps)

    rgb_img_list = torch.cat(rgb_img_list)
    mask_list = torch.stack(mask_list).unsqueeze(1)
    position_map_list = torch.stack(position_map_list).unsqueeze(1)
    distribution_map_list = torch.stack(distribution_map_list)
    cm_list = torch.stack(cm_list).unsqueeze(1)

    return rgb_img_list, mask_list, position_map_list, distribution_map_list, cm_list, meta_list


if __name__ == '__main__':
    s = [480, 320]
    b_centers = [[50, 50], [76, 82], [12, 67], [198, 122]]
    out1 = generate_position_map(s, b_centers, sigma=5)
    plt.imshow(out1)
    plt.show()

    out2 = generate_position_map(s, b_centers, sigma=5)
    plt.imshow(out2)
    plt.show()

    # root_pth = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD'
    # other_root_pth = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/'

    # root_pth = '/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD'
    # other_root_pth = '/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/'
    # copy_filtered_annotations(root_pth, other_root_pth)
    print()
