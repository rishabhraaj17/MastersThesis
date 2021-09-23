import copy
import os
import shutil
from pathlib import Path
from typing import Tuple, List, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt, patches, lines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
from torch.nn.functional import interpolate, pad
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from average_image.bbox_utils import _process_scale, cal_centers
from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import SDDMeta
from baselinev2.plot_utils import add_box_to_axes, add_features_to_axis


class ImagePadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', factor=8):
        self.factor = factor
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // factor) + 1) * factor - self.ht) % factor
        pad_wd = (((self.wd // factor) + 1) * factor - self.wd) % factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs, mode='replicate'):
        return [pad(x, self._pad, mode=mode) for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


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


def plot_samples(img, mask, rgb_boxes, rgb_box_centers, boxes, box_centers, plot_boxes=False,
                 add_feats_to_mask=False, add_feats_to_img=False, additional_text=''):
    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
    img_axis, mask_axis = axs
    if img is not None:
        img_axis.imshow(img)

    if mask is not None:
        mask_axis.imshow(mask, cmap='hot')

    if boxes is not None and plot_boxes:
        add_box_to_axes(img_axis, rgb_boxes)
        add_box_to_axes(mask_axis, boxes)

    if add_feats_to_img:
        add_features_to_axis(img_axis, rgb_box_centers)
    if add_feats_to_mask:
        add_features_to_axis(mask_axis, box_centers)

    img_axis.set_title('RGB')
    mask_axis.set_title('Mask')

    fig.suptitle(additional_text)

    plt.tight_layout()
    plt.show()


def plot_image_with_features(im, feat1=None, feat2=None, boxes=None, txt='', marker_size=5, add_on_both_axes=True,
                             footnote_txt='', video_mode=False, plot_heatmaps=True, gt_heatmap=None, pred_heatmap=None,
                             h_factor=0, w_factor=0):
    if plot_heatmaps:
        fig, axs = plt.subplots(2, 2, sharex='none', sharey='none', figsize=(22 + h_factor, 16 + w_factor))
        rgb_ax, feat_ax, gt_heatmap_ax, pred_heatmap_ax = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
        if gt_heatmap is not None:
            gt_heatmap_ax.imshow(gt_heatmap, cmap='hot')
        if pred_heatmap is not None:
            pred_heatmap_ax.imshow(pred_heatmap, cmap='hot')
        gt_heatmap_ax.set_title('GT Heatmap')
        pred_heatmap_ax.set_title('Pred Heatmap')
    else:
        fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
        rgb_ax, feat_ax = axs

    rgb_ax.imshow(im)
    feat_ax.imshow(im)

    legends_dict = {}
    if feat1 is not None:
        add_features_to_axis(feat_ax, feat1, marker_size=marker_size, marker_color='b')
        legends_dict.update({'b': 'Locations GT'})

    if feat2 is not None:
        add_features_to_axis(feat_ax, feat2, marker_size=marker_size, marker_color='g')
        legends_dict.update({'g': 'Locations Pred'})

    if boxes is not None:
        add_box_to_axes(feat_ax, boxes)
        if add_on_both_axes:
            add_box_to_axes(rgb_ax, boxes)
        legends_dict.update({'r': 'GT Boxes'})

    rgb_ax.set_title('RGB')
    feat_ax.set_title('RGB + Locations')

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout(pad=1.58)

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.suptitle(txt)
    plt.figtext(0.99, 0.01, footnote_txt, horizontalalignment='right')

    if video_mode:
        plt.close()
    else:
        plt.show()

    return fig


def plot_for_location_visualizations(im, feat1=None, feat2=None, boxes=None, txt='', marker_size=5,
                                     add_on_both_axes=True, radius=None,
                                     footnote_txt='', video_mode=False):
    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
    rgb_ax, feat_ax = axs

    rgb_ax.imshow(im)
    feat_ax.imshow(im)

    legends_dict = {}
    if feat1 is not None:
        add_features_to_axis(feat_ax, feat1, marker_size=marker_size, marker_color='b')
        legends_dict.update({'b': 'Locations GT'})

    if feat2 is not None:
        add_features_to_axis(feat_ax, feat2, marker_size=marker_size, marker_color='g')
        legends_dict.update({'g': 'Locations Pred'})

    if boxes is not None:
        add_box_to_axes(feat_ax, boxes)
        if add_on_both_axes:
            add_box_to_axes(rgb_ax, boxes)
        legends_dict.update({'r': 'GT Boxes'})

    if radius is not None:
        for c_center in feat2:
            feat_ax.add_artist(plt.Circle((c_center[0], c_center[1]), radius, color='yellow', fill=False))
        legends_dict.update({'yellow': 'Prune Radius'})

    rgb_ax.set_title('RGB')
    feat_ax.set_title('RGB + Locations')

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout(pad=1.58)

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.suptitle(txt)
    plt.figtext(0.99, 0.01, footnote_txt, horizontalalignment='right')

    if video_mode:
        plt.close()
    else:
        plt.show()

    return fig


def plot_predictions(img, mask, pred_mask, additional_text='', all_heatmaps=False, save_dir=None, img_name='',
                     tight_layout=True, do_nothing=False):
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
    if tight_layout:
        plt.tight_layout()

    if do_nothing:
        plt.close()
        return fig

    if save_dir is None:
        plt.show()
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + f'{img_name}.png')
        plt.close()


def plot_predictions_v2(img, mask, pred_mask, logits_mask, additional_text='',
                        all_heatmaps=False, save_dir=None, img_name='',
                        tight_layout=True, do_nothing=False, show_colorbar=False):
    fig_size = (20, 8) if img.shape[0] > img.shape[1] else (18, 4)
    fig, axs = plt.subplots(1, 4, sharex='none', sharey='none', figsize=fig_size)
    img_axis, mask_axis, pred_mask_axis, logits_axis = axs
    if img is not None:
        if all_heatmaps:
            img_axis.imshow(img, cmap='hot')
        else:
            img_axis.imshow(img)

    if mask is not None:
        mask_axis.imshow(mask, cmap='hot')

    if pred_mask is not None:
        pred_mask_axis.imshow(pred_mask, cmap='hot')

    if logits_mask is not None:
        lm = logits_axis.imshow(logits_mask, cmap='hot')
        if show_colorbar:
            divider = make_axes_locatable(logits_axis)
            cax = divider.append_axes('right', size='5%', pad=0.10)
            fig.colorbar(lm, cax=cax, orientation='vertical')

    if all_heatmaps:
        logits_axis.set_title('Predicted Mask -4')
        img_axis.set_title('Predicted Mask -3')
        mask_axis.set_title('Predicted Mask -2')
        pred_mask_axis.set_title('Predicted Mask -1')
    else:
        img_axis.set_title('RGB')
        mask_axis.set_title('Mask')
        pred_mask_axis.set_title('Predicted Mask')
        logits_axis.set_title('Logits Mask')

    img_axis.grid(False)
    mask_axis.grid(False)
    pred_mask_axis.grid(False)
    logits_axis.grid(False)

    fig.suptitle(additional_text)
    if tight_layout:
        plt.tight_layout()

    if do_nothing:
        plt.close()
        return fig

    if save_dir is None:
        plt.show()
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + f'{img_name}.png')
        plt.close()


def plot_predictions_with_overlay(img, mask, pred_mask, overlay_image, supervised_boxes=None,
                                  unsupervised_rgb_boxes=None, unsupervised_boxes=None,
                                  additional_text='', all_heatmaps=False, save_dir=None, img_name='',
                                  do_nothing=False):
    fig, axs = plt.subplots(2, 2, sharex='none', sharey='none', figsize=(16, 16))
    img_axis, mask_axis, overlay_axis, pred_mask_axis = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    if img is not None:
        if all_heatmaps:
            img_axis.imshow(img, cmap='hot')
        else:
            img_axis.imshow(img)

    if overlay_image is not None:
        if all_heatmaps:
            overlay_axis.imshow(overlay_image, cmap='hot')
        else:
            overlay_axis.imshow(overlay_image)

    if mask is not None:
        mask_axis.imshow(mask, cmap='hot')

    if pred_mask is not None:
        pred_mask_axis.imshow(pred_mask, cmap='hot')

    legends_dict = {}
    if supervised_boxes is not None:
        add_box_to_axes(img_axis, supervised_boxes)
        add_box_to_axes(overlay_axis, supervised_boxes)

        legends_dict.update({'r': 'Supervised Bounding Box'})

    if unsupervised_rgb_boxes is not None:
        add_box_to_axes(img_axis, unsupervised_rgb_boxes, edge_color='g')

    if unsupervised_boxes is not None:
        add_box_to_axes(overlay_axis, unsupervised_boxes, edge_color='g')

        legends_dict.update({'g': 'Unsupervised Bounding Box'})

    if all_heatmaps:
        img_axis.set_title('Predicted Mask -4')
        mask_axis.set_title('Predicted Mask -3')
        overlay_axis.set_title('Predicted Mask -2')
        pred_mask_axis.set_title('Predicted Mask -1')
    else:
        img_axis.set_title('RGB')
        mask_axis.set_title('Mask')
        pred_mask_axis.set_title('Predicted Mask')
        overlay_axis.set_title('RGB/Prediction Overlay')

    fig.suptitle(additional_text)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if do_nothing:
        plt.close()
        return fig

    if save_dir is None:
        plt.show()
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + f'{img_name}.png')
        plt.close()


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
    if hw_mode:  # height-width format
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


def get_each_dims(root_path, desired_ratio=0.25, save_csv=False):
    filtered_generated_path = root_path + '/annotations'
    meta = SDDMeta(root_path + '/H_SDD.txt')
    v_clazzes = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
                 SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS,
                 SDDVideoClasses.QUAD]
    v_numbers = [[i for i in range(7)], [i for i in range(4)], [i for i in range(5)], [i for i in range(9)],
                 [i for i in range(15)], [i for i in range(4)], [i for i in range(12)], [i for i in range(4)]]

    vid_list, vid_num_list, vid_shape_list, ratio_list = [], [], [], []
    desired_ratio_list, new_shape_list, orientation = [], [], []
    for idx, v_clz in enumerate(v_clazzes):
        for v_num in v_numbers[idx]:
            # print(f"************** Processing Features: {v_clz.name} - {v_num} *******************************")
            im_path = f"{filtered_generated_path}/{v_clz.value}/video{v_num}/reference.jpg"

            (new_w, new_h), ratio = meta.get_new_scale(img_path=im_path, dataset=getattr(SDDVideoDatasets, v_clz.name),
                                                       sequence=v_num, desired_ratio=desired_ratio,
                                                       return_original_ratio=True, use_v2=True)

            img = Image.open(im_path)

            vid_list.append(v_clz.name)
            vid_num_list.append(v_num)
            vid_shape_list.append((img.height, img.width))
            ratio_list.append(ratio)

            desired_ratio_list.append(desired_ratio)
            new_shape_list.append((round(new_h), round(new_w)))
            orientation.append("Portrait" if img.height > img.width else "Landscape")

    df = pd.DataFrame.from_dict({'CLASS': vid_list, "NUMBER": vid_num_list,
                                 'ORIENTATION': orientation, 'SHAPE': vid_shape_list,
                                 'ORIGINAL_RATIO': ratio_list, 'DESIRED_RATIO': desired_ratio_list,
                                 'RESCALED_SHAPE': new_shape_list})
    portrait_df = df[df.ORIENTATION == "Portrait"]
    landscape_df = df[df.ORIENTATION == "Landscape"]
    if save_csv:
        df.to_csv(f"{root_path}/shapes_summary.csv", index=False)
    return df, (portrait_df, landscape_df)


def get_metadata_for_videos(root_path, video_classes, video_numbers, desired_ratio=0.25):
    df, _ = get_each_dims(root_path=root_path, desired_ratio=desired_ratio, save_csv=False)

    filtered_dfs = []
    for idx, v_clz in enumerate(video_classes):
        filtered_dfs.append(df[(df.CLASS == v_clz) & df.NUMBER.isin(video_numbers[idx])])

    return pd.concat(filtered_dfs)


def get_scaled_shapes_with_pad_values(root_path, video_classes, video_numbers, desired_ratio=0.25):
    df = get_metadata_for_videos(root_path=root_path, video_classes=video_classes,
                                 video_numbers=video_numbers, desired_ratio=desired_ratio)

    max_shape = df.RESCALED_SHAPE.max()
    pad_params_list = []
    for idx, row in df.iterrows():
        pad_params = get_pad_parameters_with_shapes(original_shape=row.RESCALED_SHAPE, desired_shape=max_shape)
        pad_params_list.append(pad_params)

    # df.insert(loc=-1, column='PAD_VALUES', value=pad_params_list, allow_duplicates=True)
    df['PAD_VALUES'] = pad_params_list
    return df, max_shape


def resize_and_pad(root_path, video_class, video_number, desired_shape, plot=False):
    annotation_root_path = root_path + '/annotations'
    im_path = f"{annotation_root_path}/{video_class.value}/video{video_number}/reference.jpg"
    img = Image.open(im_path)
    img_tensor = to_tensor(img)

    out = pad(img_tensor, get_pad_parameters(img_tensor=img_tensor, desired_shape=desired_shape))

    if plot:
        fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
        img_axis, padded_img_axis = axs

        img_axis.imshow(img_tensor.permute(1, 2, 0))
        padded_img_axis.imshow(out.permute(1, 2, 0))

        img_axis.set_title('Image')
        padded_img_axis.set_title('Padded Image')

        plt.show()

    return out


def get_pad_parameters(img_tensor, desired_shape):
    # Pad img_tensor shape to the desired_shape
    desired_height, desired_width = desired_shape
    diff_h = desired_height - img_tensor.shape[-2]
    diff_w = desired_width - img_tensor.shape[-1]

    return [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]


def get_pad_parameters_with_shapes(original_shape, desired_shape):
    # Pad img_tensor shape to the desired_shape
    desired_height, desired_width = desired_shape
    original_height, original_width = original_shape
    diff_h = desired_height - original_height
    diff_w = desired_width - original_width

    return [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]


def rgb_transform(img, new_shape, pad_values):
    img = interpolate(img, size=new_shape, mode='bilinear')
    img = pad(img, pad_values)
    return img


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


def heat_map_temporal_collate_fn(batch):
    rgb_img_list, mask_list, position_map_list, distribution_map_list, cm_list, meta_list = [], [], [], [], [], []
    for batch_item in batch:
        rgb, mask, position_map, distribution_map, class_maps, meta = batch_item
        rgb_img_list.append(rgb)
        mask_list.append(mask)
        meta_list.append(meta)
        position_map_list.append(position_map)
        distribution_map_list.append(distribution_map)
        cm_list.append(class_maps)

    rgb_img_list = torch.stack(rgb_img_list).transpose(1, 2)
    mask_list = torch.stack(mask_list).transpose(1, 2)
    position_map_list = torch.stack(position_map_list).transpose(1, 2)
    distribution_map_list = torch.stack(distribution_map_list).transpose(1, 2)
    cm_list = torch.stack(cm_list).transpose(1, 2)

    return rgb_img_list, mask_list, position_map_list, distribution_map_list, cm_list, meta_list


def heat_map_temporal_4d_collate_fn(batch):
    rgb_img_list, mask_list, position_map_list, distribution_map_list, cm_list, meta_list = [], [], [], [], [], []
    for batch_item in batch:
        rgb, mask, position_map, distribution_map, class_maps, meta = batch_item
        rgb_img_list.append(rgb.view(-1, *rgb.shape[-2:]))
        mask_list.append(mask.view(-1, *mask.shape[-2:]))
        meta_list.append(meta)
        position_map_list.append(position_map.view(-1, *position_map.shape[-2:]))
        distribution_map_list.append(distribution_map.view(-1, *distribution_map.shape[-2:]))
        cm_list.append(class_maps.view(-1, *class_maps.shape[-2:]))

    rgb_img_list = torch.stack(rgb_img_list)
    mask_list = torch.stack(mask_list)
    position_map_list = torch.stack(position_map_list)
    distribution_map_list = torch.stack(distribution_map_list)
    cm_list = torch.stack(cm_list)

    return rgb_img_list, mask_list, position_map_list, distribution_map_list, cm_list, meta_list


# https://stackoverflow.com/questions/58125495/how-to-count-how-many-white-balls-there-are-in-an-image-with-opencv-python
# https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
def get_blob_count(image, kernel_size=(1, 1), plot=False):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    blobs = 0
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(image, [c], -1, (36, 255, 12), -1)
        if area > 13000:
            blobs += 2
        else:
            blobs += 1

    if plot:
        plt.imshow(image)
        plt.show()

        plt.imshow(thresh)
        plt.show()

        plt.imshow(opening)
        plt.show()

    return blobs


def overlay_images(transformer, background, overlay):
    background = transformer(background)
    overlay = transformer(overlay)

    if background.height != overlay.height or background.width != overlay.width:
        background = background.resize((overlay.width, overlay.height), Image.ANTIALIAS)

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.5)
    return np.array(new_img)


def get_ensemble(state_dictionaries):
    base_dict = state_dictionaries[0]
    out_dict = copy.deepcopy(state_dictionaries[0])

    num_state_dicts = float(len(state_dictionaries))

    # Average all parameters
    for key in base_dict:
        temp = 0
        for state_dict in state_dictionaries:
            temp += state_dict[key]
        out_dict[key] = temp / num_state_dicts
    return out_dict


class TensorDatasetForTwo(Dataset):
    def __init__(self, a, b):
        super(TensorDatasetForTwo, self).__init__()
        assert a.shape[0] == b.shape[0]
        self.a = a
        self.b = b

    def __getitem__(self, index):
        return self.a[index], self.b[index]

    def __len__(self):
        return self.a.shape[0]


if __name__ == '__main__':
    # get_each_dims('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD', 0.05)
    get_scaled_shapes_with_pad_values(
        '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD',
        video_classes=[SDDVideoClasses.LITTLE.name, SDDVideoClasses.GATES.name],
        video_numbers=[[0, 1], [0, 2, 4]],
        desired_ratio=0.05)

    # resize_and_pad('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD',
    #                video_class=SDDVideoClasses.LITTLE, video_number=2,
    #                desired_shape=(2002, 1445), plot=True)
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
