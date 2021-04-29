import numpy as np
import pandas as pd
import timeout_decorator
import torch
import torchvision.io as tio
import torchvision.ops
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
from matplotlib import pyplot as plt, patches
import albumentations as aug_transforms

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, scale_annotations
from average_image.constants import SDDVideoClasses
from baseline.extracted_of_optimization import is_point_inside_circle
from baselinev2.exceptions import TimeoutException
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.plot_utils import add_box_to_axes
from baselinev2.utils import get_generated_frame_annotations, get_bbox_center, get_generated_track_annotations_for_frame
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.improve_metrics.crop_utils')

REPLACEMENT_TIMEOUT = 10


class CustomRandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant",
                 return_correct_box=False):
        super(CustomRandomCrop, self).__init__(size=size, padding=padding, pad_if_needed=pad_if_needed, fill=fill,
                                               padding_mode=padding_mode)
        self.return_correct_box = return_correct_box

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = tvf.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = tvf._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = tvf.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = tvf.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        if self.return_correct_box:
            return tvf.crop(img, i, j, h, w), (j, i, h, w)
        return tvf.crop(img, i, j, h, w), (i, j, h, w)


class CustomRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=tvf.InterpolationMode.BILINEAR):
        super(CustomRandomResizedCrop, self).__init__(size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return tvf.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)


def read_annotation_file(path):
    dff = pd.read_csv(path)
    dff = dff.drop(dff.columns[[0]], axis=1)
    return dff


def add_box_to_axes_xyhw(ax, boxes, edge_color='r'):
    for box in boxes:
        rect = patches.Rectangle(xy=(box[1], box[0]), width=box[2], height=box[3],
                                 edgecolor=edge_color, fill=False,
                                 linewidth=None)
        ax.add_patch(rect)


def add_box_to_axes_torch_converted_xyhw(ax, boxes, edge_color='r'):
    # we hv an issue, conversion swaps x and y
    for box in boxes:
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2], height=box[3],
                                 edgecolor=edge_color, fill=False,
                                 linewidth=None)
        ax.add_patch(rect)


def show_crop(im, crop_im, box):
    fig, ax = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(14, 12))
    image_axis, cropped_image_axis = ax

    image_axis.imshow(im)
    cropped_image_axis.imshow(crop_im)
    add_box_to_axes_xyhw(image_axis, box)

    image_axis.scatter(box[0][1], box[0][0])

    plt.show()


def show_image_with_crop_boxes(im, cropped_boxes, true_boxes, xyxy_mode=True, xywh_mode_v2=True,
                               overlapping_boxes=None, title=''):
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(14, 12))
    image_axis = ax

    image_axis.imshow(im)

    if xyxy_mode:
        add_box_to_axes(image_axis, cropped_boxes)
        add_box_to_axes(image_axis, true_boxes, edge_color='g')
        if overlapping_boxes is not None:
            add_box_to_axes(image_axis, overlapping_boxes[0], edge_color='aqua')
            add_box_to_axes(image_axis, overlapping_boxes[1], edge_color='yellow')
    elif xywh_mode_v2:
        add_box_to_axes_torch_converted_xyhw(image_axis, cropped_boxes)
        add_box_to_axes_torch_converted_xyhw(image_axis, true_boxes, edge_color='g')
    else:
        add_box_to_axes_xyhw(image_axis, cropped_boxes)
        add_box_to_axes_xyhw(image_axis, true_boxes, edge_color='g')

    plt.suptitle(title)
    plt.show()


def sample_random_crops(img, size, n, return_correct_box=False):
    crops, boxes = [], []
    random_cropper = CustomRandomCrop(size=size, return_correct_box=return_correct_box)
    for idx in range(n):
        crop, b_box = random_cropper(img)
        crops.append(crop)
        boxes.append(torch.tensor(b_box))

    try:
        crops = torch.stack(crops)
        boxes = torch.stack(boxes)
    except RuntimeError:
        logger.warning('Skipping as frame not found!')
    return crops, boxes
    # return torch.stack(crops), torch.stack(boxes)


def patches_and_labels_debug(image, bounding_box_size, annotations, frame_number, num_patches=None, new_shape=None,
                             use_generated=True, debug_mode=False):
    original_shape = (image.shape[2], image.shape[3]) if image.ndim == 4 else (image.shape[1], image.shape[2])
    new_shape = original_shape if new_shape is None else original_shape

    if use_generated:
        frame_annotation = get_generated_frame_annotations(annotations, frame_number)
        gt_boxes = torch.from_numpy(frame_annotation[:, 1:5].astype(np.int))
        generated_track_idx = frame_annotation[:, 0]
    else:
        frame_annotation = get_frame_annotations_and_skip_lost(annotations, frame_number)
        gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                            original_scale=original_shape,
                                                            new_scale=new_shape, return_track_id=False,
                                                            tracks_with_annotations=True)
        gt_boxes = torch.from_numpy(gt_annotations[:, :-1])

    num_patches = frame_annotation.shape[0] if num_patches is None else num_patches
    crops, boxes = sample_random_crops(image, bounding_box_size, num_patches)

    gt_boxes_xywh = torchvision.ops.box_convert(gt_boxes, 'xyxy', 'xywh')
    gt_boxes_xywh = [torch.tensor((b[1], b[0], b[2], b[3])) for b in gt_boxes_xywh]
    gt_boxes_xywh = torch.stack(gt_boxes_xywh)

    fp_boxes = torchvision.ops.box_convert(boxes, 'xywh', 'xyxy')

    gt_crops = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in gt_boxes_xywh]
    gt_crops_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                        for c in gt_crops if c.shape[1] != 0 and c.shape[2] != 0]

    boxes_iou = torchvision.ops.box_iou(gt_boxes, fp_boxes)
    gt_box_match, fp_boxes_match = torch.where(boxes_iou)

    valid_fp_boxes_idx = np.setdiff1d(np.arange(fp_boxes.shape[0]), fp_boxes_match.numpy())

    overlapping_boxes = [fp_boxes[fp_boxes_match], gt_boxes[gt_box_match]]

    # show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh)
    show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, overlapping_boxes=overlapping_boxes)
    gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
    plt.imshow(gt_crops_grid.permute(1, 2, 0))
    plt.show()

    fp_boxes = fp_boxes[valid_fp_boxes_idx]
    boxes = boxes[valid_fp_boxes_idx]
    crops = crops[valid_fp_boxes_idx]

    boxes_to_replace = num_patches - fp_boxes.shape[0]
    replaceable_boxes, replaceable_fp_boxes, replaceable_crops = [], [], []
    replacement_required = False

    while boxes_to_replace != 0:
        replacement_required = True
        temp_crops, temp_boxes = sample_random_crops(image, bounding_box_size, boxes_to_replace)
        temp_fp_boxes = torchvision.ops.box_convert(temp_boxes, 'xywh', 'xyxy')

        temp_boxes_iou = torchvision.ops.box_iou(gt_boxes, temp_fp_boxes)
        temp_gt_box_match, temp_fp_boxes_match = torch.where(temp_boxes_iou)

        temp_valid_fp_boxes_idx = np.setdiff1d(np.arange(temp_fp_boxes.shape[0]), temp_fp_boxes_match.numpy())
        replaceable_boxes.append(temp_boxes[temp_valid_fp_boxes_idx])
        replaceable_fp_boxes.append(temp_fp_boxes[temp_valid_fp_boxes_idx])
        replaceable_crops.append(temp_crops[temp_valid_fp_boxes_idx])

        boxes_to_replace -= temp_valid_fp_boxes_idx.size

    if replacement_required:
        replaceable_boxes = torch.cat(replaceable_boxes)
        replaceable_fp_boxes = torch.cat(replaceable_fp_boxes)
        replaceable_crops = torch.cat(replaceable_crops)

        boxes = torch.cat((boxes, replaceable_boxes))
        fp_boxes = torch.cat((fp_boxes, replaceable_fp_boxes))
        crops = torch.cat((crops, replaceable_crops))

    show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes)
    fp_crops_grid = torchvision.utils.make_grid(crops)
    plt.imshow(fp_crops_grid.permute(1, 2, 0))
    plt.show()

    gt_crops_resized = torch.stack(gt_crops_resized)
    gt_patches_and_labels = {'patches': gt_crops_resized, 'labels': torch.ones(size=(gt_crops_resized.shape[0],))}
    fp_patches_and_labels = {'patches': crops, 'labels': torch.zeros(size=(crops.shape[0],))}

    if debug_mode:
        gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

        # show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes)
        show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh)

        for crop_idx in range(len(gt_crops)):
            show_crop(image.permute(1, 2, 0), gt_crops[crop_idx].permute(1, 2, 0), [gt_boxes_xywh[crop_idx]])
            # show_crop(image[0].permute(1, 2, 0), gt_crops[crop_idx][0].permute(1, 2, 0), [gt_boxes_xywh[crop_idx]])
            # show_crop(image[1].permute(1, 2, 0), gt_crops[crop_idx][1].permute(1, 2, 0), [gt_boxes_xywh[crop_idx]])
            print()

    return gt_patches_and_labels, fp_patches_and_labels


def patches_and_labels(image, bounding_box_size, annotations, frame_number, num_patches=None, new_shape=None,
                       use_generated=True, radius_elimination=None, plot=False, only_long_trajectories=False,
                       track_length_threshold=60, img_transforms=None, additional_w=None, additional_h=None):
    original_shape = (image.shape[2], image.shape[3]) if image.ndim == 4 else (image.shape[1], image.shape[2])
    new_shape = original_shape if new_shape is None else original_shape

    if only_long_trajectories:
        if use_generated:
            frame_annotation = get_generated_frame_annotations(annotations, frame_number)
            track_annotations = get_generated_track_annotations_for_frame(annotations, frame_number)
            track_lengths = np.array([t.shape[0] for t in track_annotations])
            feasible_track_length = track_lengths > track_length_threshold
            feasible_frame_annotations = frame_annotation[feasible_track_length]
            gt_boxes = torch.from_numpy(feasible_frame_annotations[:, 1:5].astype(np.int))
            generated_track_idx = feasible_frame_annotations[:, 0]
            gt_bbox_centers = feasible_frame_annotations[:, 7:9]
        else:
            return NotImplemented
    else:
        if use_generated:
            frame_annotation = get_generated_frame_annotations(annotations, frame_number)
            gt_boxes = torch.from_numpy(frame_annotation[:, 1:5].astype(np.int))
            generated_track_idx = frame_annotation[:, 0]
            gt_bbox_centers = frame_annotation[:, 7:9]
        else:
            frame_annotation = get_frame_annotations_and_skip_lost(annotations, frame_number)
            gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                                original_scale=original_shape,
                                                                new_scale=new_shape, return_track_id=False,
                                                                tracks_with_annotations=True)
            gt_boxes = torch.from_numpy(gt_annotations[:, :-1])

    if frame_annotation.size == 0 or feasible_frame_annotations.size == 0:
        return {}, {}

    num_patches = frame_annotation.shape[0] if num_patches is None else num_patches
    crops, boxes = sample_random_crops(image, bounding_box_size, num_patches)

    gt_boxes_xywh = torchvision.ops.box_convert(gt_boxes, 'xyxy', 'xywh')
    gt_boxes_xywh = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w)) for b in gt_boxes_xywh]
    gt_boxes_xywh = torch.stack(gt_boxes_xywh)

    fp_boxes = torchvision.ops.box_convert(boxes, 'xywh', 'xyxy')

    gt_crops = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in gt_boxes_xywh]
    gt_crops_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                        for c in gt_crops if c.shape[1] != 0 and c.shape[2] != 0]

    boxes_iou = torchvision.ops.box_iou(gt_boxes, fp_boxes)
    gt_box_match, fp_boxes_match = torch.where(boxes_iou)

    fp_boxes_match_numpy = fp_boxes_match.numpy()

    l2_distances_matrix = np.zeros(shape=(fp_boxes.shape[0], fp_boxes.shape[0]))
    if radius_elimination is not None:
        fp_boxes_centers = np.stack([get_bbox_center(fp_box) for fp_box in fp_boxes.numpy()]).squeeze()

        for g_idx, gt_center in enumerate(gt_bbox_centers):
            if fp_boxes_centers.ndim == 1:
                fp_boxes_centers = np.expand_dims(fp_boxes_centers, axis=0)
            for f_idx, fp_center in enumerate(fp_boxes_centers):
                l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)

        gt_r_invalid, fp_r_invalid = np.where(l2_distances_matrix < radius_elimination)
        fp_boxes_match_numpy = np.union1d(fp_boxes_match.numpy(), fp_r_invalid)

    valid_fp_boxes_idx = np.setdiff1d(np.arange(fp_boxes.shape[0]), fp_boxes_match_numpy)

    overlapping_boxes = [fp_boxes[fp_boxes_match], gt_boxes[gt_box_match]]

    if plot:
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, overlapping_boxes=overlapping_boxes)
        gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

    fp_boxes = fp_boxes[valid_fp_boxes_idx]
    boxes = boxes[valid_fp_boxes_idx]
    crops = crops[valid_fp_boxes_idx]

    boxes_to_replace = num_patches - fp_boxes.shape[0]
    replaceable_boxes, replaceable_fp_boxes, replaceable_crops = [], [], []
    replacement_required = False

    while boxes_to_replace != 0:
        replacement_required = True
        temp_crops, temp_boxes = sample_random_crops(image, bounding_box_size, boxes_to_replace)
        temp_fp_boxes = torchvision.ops.box_convert(temp_boxes, 'xywh', 'xyxy')

        temp_boxes_iou = torchvision.ops.box_iou(gt_boxes, temp_fp_boxes)
        temp_gt_box_match, temp_fp_boxes_match = torch.where(temp_boxes_iou)

        temp_fp_boxes_match_numpy = temp_fp_boxes_match.numpy()
        if radius_elimination:
            temp_l2_distances_matrix = np.zeros(shape=(gt_boxes.shape[0], temp_fp_boxes.shape[0]))
            temp_fp_boxes_centers = np.stack([get_bbox_center(fp_box) for fp_box in temp_fp_boxes.numpy()]).squeeze()

            for g_idx, gt_center in enumerate(gt_bbox_centers):
                if temp_fp_boxes_centers.ndim == 1:
                    temp_fp_boxes_centers = np.expand_dims(temp_fp_boxes_centers, axis=0)
                for f_idx, fp_center in enumerate(temp_fp_boxes_centers):
                    temp_l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)

            temp_gt_r_invalid, temp_fp_r_invalid = np.where(temp_l2_distances_matrix < radius_elimination)
            temp_fp_boxes_match_numpy = np.union1d(temp_fp_boxes_match.numpy(), temp_fp_r_invalid)

        temp_valid_fp_boxes_idx = np.setdiff1d(np.arange(temp_fp_boxes.shape[0]), temp_fp_boxes_match_numpy)
        replaceable_boxes.append(temp_boxes[temp_valid_fp_boxes_idx])
        replaceable_fp_boxes.append(temp_fp_boxes[temp_valid_fp_boxes_idx])
        replaceable_crops.append(temp_crops[temp_valid_fp_boxes_idx])

        boxes_to_replace -= temp_valid_fp_boxes_idx.size

    if replacement_required:
        replaceable_boxes = torch.cat(replaceable_boxes)
        replaceable_fp_boxes = torch.cat(replaceable_fp_boxes)
        replaceable_crops = torch.cat(replaceable_crops)

        boxes = torch.cat((boxes, replaceable_boxes))
        fp_boxes = torch.cat((fp_boxes, replaceable_fp_boxes))
        crops = torch.cat((crops, replaceable_crops))

    if plot:
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes)
        fp_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(fp_crops_grid.permute(1, 2, 0))
        plt.show()

    gt_crops_resized = torch.stack(gt_crops_resized)

    # data-augmentation
    if img_transforms:
        out = [img_transforms(image=np.transpose(im, [1, 2, 0])) for im in gt_crops_resized.numpy()]
        out = [torch.from_numpy(o['image']).permute(2, 0, 1) for o in out]
        gt_crops_resized = torch.stack(out)

        out = [img_transforms(image=np.transpose(im, [1, 2, 0])) for im in crops.numpy()]
        out = [torch.from_numpy(o['image']).permute(2, 0, 1) for o in out]
        crops = torch.stack(out)

    if plot:
        gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

        gt_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

    gt_patches_and_labels = {'patches': gt_crops_resized, 'labels': torch.ones(size=(gt_crops_resized.shape[0],))}
    fp_patches_and_labels = {'patches': crops, 'labels': torch.zeros(size=(crops.shape[0],))}

    return gt_patches_and_labels, fp_patches_and_labels


def patches_and_labels_with_anchors(image, bounding_box_size, annotations, frame_number, num_patches=None,
                                    new_shape=None,
                                    use_generated=True, radius_elimination=None, plot=False,
                                    only_long_trajectories=False,
                                    track_length_threshold=60, img_transforms=None, additional_w=None,
                                    additional_h=None,
                                    aspect_ratios=(), scales=()):
    original_shape = (image.shape[2], image.shape[3]) if image.ndim == 4 else (image.shape[1], image.shape[2])
    new_shape = original_shape if new_shape is None else original_shape

    if only_long_trajectories:
        if use_generated:
            frame_annotation = get_generated_frame_annotations(annotations, frame_number)
            track_annotations = get_generated_track_annotations_for_frame(annotations, frame_number)
            track_lengths = np.array([t.shape[0] for t in track_annotations])
            feasible_track_length = track_lengths > track_length_threshold
            feasible_frame_annotations = frame_annotation[feasible_track_length]
            gt_boxes = torch.from_numpy(feasible_frame_annotations[:, 1:5].astype(np.int))
            generated_track_idx = feasible_frame_annotations[:, 0]
            gt_bbox_centers = feasible_frame_annotations[:, 7:9]
        else:
            return NotImplemented
    else:
        if use_generated:
            frame_annotation = get_generated_frame_annotations(annotations, frame_number)
            gt_boxes = torch.from_numpy(frame_annotation[:, 1:5].astype(np.int))
            generated_track_idx = frame_annotation[:, 0]
            gt_bbox_centers = frame_annotation[:, 7:9]
        else:
            frame_annotation = get_frame_annotations_and_skip_lost(annotations, frame_number)
            gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                                original_scale=original_shape,
                                                                new_scale=new_shape, return_track_id=False,
                                                                tracks_with_annotations=True)
            gt_boxes = torch.from_numpy(gt_annotations[:, :-1])

    if frame_annotation.size == 0 or feasible_frame_annotations.size == 0:
        return {}, {}

    gt_boxes_xywh = torchvision.ops.box_convert(gt_boxes, 'xyxy', 'xywh')
    gt_boxes_xywh = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w)) for b in gt_boxes_xywh]

    new_gt_boxes_xywh_w = []
    new_gt_boxes_xywh_h = []

    gt_boxes_cxcywh = torchvision.ops.box_convert(gt_boxes, 'xyxy', 'cxcywh')
    if len(aspect_ratios) != 0 and len(scales) != 0:
        for scale in scales:
            for ratio in aspect_ratios:
                for box in gt_boxes_cxcywh:
                    _x, _y, _w, _h = box
                    adjusted_box = torch.tensor((_x, _y, int(_w * ratio * scale), int(_h * (1 / ratio) * scale)))
                    new_gt_boxes_xywh_w.append(adjusted_box)
                    adjusted_box = torch.tensor((_x, _y, int(_w * 1 / ratio * scale), int(_h * ratio) * scale))
                    new_gt_boxes_xywh_h.append(adjusted_box)

    new_gt_boxes_xywh_w = torchvision.ops.box_convert(torch.stack(new_gt_boxes_xywh_w), 'cxcywh', 'xywh').int()
    new_gt_boxes_xywh_w = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w))
                           for b in new_gt_boxes_xywh_w]

    new_gt_boxes_xywh_h = torchvision.ops.box_convert(torch.stack(new_gt_boxes_xywh_h), 'cxcywh', 'xywh').int()
    new_gt_boxes_xywh_h = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w))
                           for b in new_gt_boxes_xywh_h]
    _gt_crops_w = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in new_gt_boxes_xywh_w]
    _gt_crops_h = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in new_gt_boxes_xywh_h]

    _gt_crops_w_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                           for c in _gt_crops_w if c.shape[1] != 0 and c.shape[2] != 0]
    _gt_crops_h_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                           for c in _gt_crops_h if c.shape[1] != 0 and c.shape[2] != 0]

    # k = 0
    # for i in range(40):
    #     for j in range(4):
    #         try:
    #             # ax[i, j].axis('off')
    #             fig, ax = plt.subplots(1, 2)
    #
    #             ax[0].set_title(f'{k}')
    #             # ax[0].imshow(_gt_crops_w[k].permute(1, 2, 0))
    #             ax[0].imshow(_gt_crops_w_resized[k].permute(1, 2, 0))
    #
    #             ax[1].set_title(f'{k}')
    #             # ax[1].imshow(_gt_crops_h[k].permute(1, 2, 0))
    #             ax[1].imshow(_gt_crops_h_resized[k].permute(1, 2, 0))
    #
    #             k += 1
    #             plt.show()
    #
    #         except IndexError:
    #             continue

    gt_boxes_xywh = torch.stack(gt_boxes_xywh)

    gt_crops = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in gt_boxes_xywh]
    gt_crops_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                        for c in gt_crops if c.shape[1] != 0 and c.shape[2] != 0]

    gt_crops_resized.extend(_gt_crops_w_resized)
    gt_crops_resized.extend(_gt_crops_h_resized)

    num_patches = (len(gt_crops) + len(_gt_crops_w) + len(_gt_crops_h)) if num_patches is None else num_patches
    crops, boxes = sample_random_crops(image, bounding_box_size, num_patches)

    # correct mapping now
    boxes_t = [torch.tensor((b[1], b[0], b[2], b[3])) for b in boxes]
    boxes_t = torch.stack(boxes_t)
    fp_boxes = torchvision.ops.box_convert(boxes_t, 'xywh', 'xyxy')

    # fp_boxes = torchvision.ops.box_convert(boxes, 'xywh', 'xyxy') - incorrect or let crop random switch it

    boxes_iou = torchvision.ops.box_iou(gt_boxes, fp_boxes)
    gt_box_match, fp_boxes_match = torch.where(boxes_iou)

    fp_boxes_match_numpy = fp_boxes_match.numpy()

    l2_distances_matrix = np.zeros(shape=(fp_boxes.shape[0], fp_boxes.shape[0]))
    if radius_elimination is not None:
        fp_boxes_centers = np.stack([get_bbox_center(fp_box) for fp_box in fp_boxes.numpy()]).squeeze()

        for g_idx, gt_center in enumerate(gt_bbox_centers):
            if fp_boxes_centers.ndim == 1:
                fp_boxes_centers = np.expand_dims(fp_boxes_centers, axis=0)
            for f_idx, fp_center in enumerate(fp_boxes_centers):
                l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)

        l2_distances_matrix = l2_distances_matrix[:gt_bbox_centers.shape[0], ...]
        gt_r_invalid, fp_r_invalid = np.where(l2_distances_matrix < radius_elimination)

        fp_boxes_match_numpy = np.union1d(fp_boxes_match.numpy(), fp_r_invalid)

    valid_fp_boxes_idx = np.setdiff1d(np.arange(fp_boxes.shape[0]), fp_boxes_match_numpy)

    overlapping_boxes = [fp_boxes[fp_boxes_match], gt_boxes[gt_box_match]]

    if plot:
        # show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, overlapping_boxes=overlapping_boxes)
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, title='xyxy',
                                   overlapping_boxes=overlapping_boxes)
        # show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
        #                            title='xywh')
        gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

        fp_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(fp_crops_grid.permute(1, 2, 0))
        plt.show()

    fp_boxes = fp_boxes[valid_fp_boxes_idx]
    boxes = boxes[valid_fp_boxes_idx]
    crops = crops[valid_fp_boxes_idx]
    # crops = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in boxes]
    # crops = torch.stack(crops)

    if plot:
        show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
                                   title='xywh')
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, title='xyxy')
        gt_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

    boxes_to_replace = num_patches - fp_boxes.shape[0]
    replaceable_boxes, replaceable_fp_boxes, replaceable_crops = [], [], []
    replacement_required = False

    while boxes_to_replace != 0:
        replacement_required = True
        temp_crops, temp_boxes = sample_random_crops(image, bounding_box_size, boxes_to_replace)
        boxes_temp = [torch.tensor((b[1], b[0], b[2], b[3])) for b in temp_boxes]
        boxes_temp = torch.stack(boxes_temp)
        temp_fp_boxes = torchvision.ops.box_convert(boxes_temp, 'xywh', 'xyxy')

        temp_boxes_iou = torchvision.ops.box_iou(gt_boxes, temp_fp_boxes)
        temp_gt_box_match, temp_fp_boxes_match = torch.where(temp_boxes_iou)

        temp_fp_boxes_match_numpy = temp_fp_boxes_match.numpy()
        if radius_elimination:
            temp_l2_distances_matrix = np.zeros(shape=(gt_boxes.shape[0], temp_fp_boxes.shape[0]))
            temp_fp_boxes_centers = np.stack([get_bbox_center(fp_box) for fp_box in temp_fp_boxes.numpy()]).squeeze()

            for g_idx, gt_center in enumerate(gt_bbox_centers):
                if temp_fp_boxes_centers.ndim == 1:
                    temp_fp_boxes_centers = np.expand_dims(temp_fp_boxes_centers, axis=0)
                for f_idx, fp_center in enumerate(temp_fp_boxes_centers):
                    temp_l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)

            temp_gt_r_invalid, temp_fp_r_invalid = np.where(temp_l2_distances_matrix < radius_elimination)

            temp_fp_boxes_match_numpy = np.union1d(temp_fp_boxes_match.numpy(), temp_fp_r_invalid)

        temp_valid_fp_boxes_idx = np.setdiff1d(np.arange(temp_fp_boxes.shape[0]), temp_fp_boxes_match_numpy)
        replaceable_boxes.append(temp_boxes[temp_valid_fp_boxes_idx])
        replaceable_fp_boxes.append(temp_fp_boxes[temp_valid_fp_boxes_idx])
        replaceable_crops.append(temp_crops[temp_valid_fp_boxes_idx])

        boxes_to_replace -= temp_valid_fp_boxes_idx.size

    if replacement_required:
        replaceable_boxes = torch.cat(replaceable_boxes)
        replaceable_fp_boxes = torch.cat(replaceable_fp_boxes)
        replaceable_crops = torch.cat(replaceable_crops)

        boxes = torch.cat((boxes, replaceable_boxes))
        fp_boxes = torch.cat((fp_boxes, replaceable_fp_boxes))
        crops = torch.cat((crops, replaceable_crops))

    if plot:
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes)
        show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
                                   title='xywh')
        fp_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(fp_crops_grid.permute(1, 2, 0))
        plt.show()

    gt_crops_resized = torch.stack(gt_crops_resized)

    # data-augmentation
    if img_transforms:
        out = [img_transforms(image=np.transpose(im, [1, 2, 0])) for im in gt_crops_resized.numpy()]
        out = [torch.from_numpy(o['image']).permute(2, 0, 1) for o in out]
        gt_crops_resized = torch.stack(out)

        out = [img_transforms(image=np.transpose(im, [1, 2, 0])) for im in crops.numpy()]
        out = [torch.from_numpy(o['image']).permute(2, 0, 1) for o in out]
        crops = torch.stack(out)

    if plot:
        gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

        gt_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

    gt_patches_and_labels = {'patches': gt_crops_resized, 'labels': torch.ones(size=(gt_crops_resized.shape[0],))}
    fp_patches_and_labels = {'patches': crops, 'labels': torch.zeros(size=(crops.shape[0],))}

    return gt_patches_and_labels, fp_patches_and_labels


def patches_and_labels_with_anchors_different_crop_track_threshold(
        image, bounding_box_size, annotations, frame_number, num_patches=None,
        new_shape=None,
        use_generated=True, radius_elimination=None, plot=False,
        only_long_trajectories=False,
        track_length_threshold=60, img_transforms=None, additional_w=None,
        additional_h=None,
        aspect_ratios=(), scales=(), track_length_threshold_for_random_crops=30):
    original_shape = (image.shape[2], image.shape[3]) if image.ndim == 4 else (image.shape[1], image.shape[2])
    new_shape = original_shape if new_shape is None else original_shape

    if only_long_trajectories:
        if use_generated:
            frame_annotation = get_generated_frame_annotations(annotations, frame_number)
            track_annotations = get_generated_track_annotations_for_frame(annotations, frame_number)
            track_lengths = np.array([t.shape[0] for t in track_annotations])

            feasible_track_length = track_lengths > track_length_threshold
            feasible_frame_annotations = frame_annotation[feasible_track_length]
            gt_boxes = torch.from_numpy(feasible_frame_annotations[:, 1:5].astype(np.int))
            generated_track_idx = feasible_frame_annotations[:, 0]
            gt_bbox_centers = feasible_frame_annotations[:, 7:9]

            feasible_track_length_for_random_crops = track_lengths > track_length_threshold_for_random_crops
            feasible_frame_annotations_for_random_crops = frame_annotation[feasible_track_length_for_random_crops]
            gt_boxes_for_random_crops = torch.from_numpy(
                feasible_frame_annotations_for_random_crops[:, 1:5].astype(np.int))
            generated_track_idx_for_random_crops = feasible_frame_annotations_for_random_crops[:, 0]
            gt_bbox_centers_for_random_crops = feasible_frame_annotations_for_random_crops[:, 7:9]
        else:
            return NotImplemented
    else:
        if use_generated:
            frame_annotation = get_generated_frame_annotations(annotations, frame_number)
            gt_boxes = torch.from_numpy(frame_annotation[:, 1:5].astype(np.int))
            generated_track_idx = frame_annotation[:, 0]
            gt_bbox_centers = frame_annotation[:, 7:9]
        else:
            frame_annotation = get_frame_annotations_and_skip_lost(annotations, frame_number)
            gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                                original_scale=original_shape,
                                                                new_scale=new_shape, return_track_id=False,
                                                                tracks_with_annotations=True)
            gt_boxes = torch.from_numpy(gt_annotations[:, :-1])

    if frame_annotation.size == 0 or feasible_frame_annotations.size == 0:
        return {}, {}

    gt_boxes_xywh = torchvision.ops.box_convert(gt_boxes, 'xyxy', 'xywh')
    gt_boxes_xywh = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w)) for b in gt_boxes_xywh]

    new_gt_boxes_xywh_w = []
    new_gt_boxes_xywh_h = []

    gt_boxes_cxcywh = torchvision.ops.box_convert(gt_boxes, 'xyxy', 'cxcywh')
    if len(aspect_ratios) != 0 and len(scales) != 0:
        for scale in scales:
            for ratio in aspect_ratios:
                for box in gt_boxes_cxcywh:
                    _x, _y, _w, _h = box
                    adjusted_box = torch.tensor((_x, _y, int(_w * ratio * scale), int(_h * (1 / ratio) * scale)))
                    new_gt_boxes_xywh_w.append(adjusted_box)
                    adjusted_box = torch.tensor((_x, _y, int(_w * 1 / ratio * scale), int(_h * ratio) * scale))
                    new_gt_boxes_xywh_h.append(adjusted_box)

    new_gt_boxes_xywh_w = torchvision.ops.box_convert(torch.stack(new_gt_boxes_xywh_w), 'cxcywh', 'xywh').int()
    new_gt_boxes_xywh_w = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w))
                           for b in new_gt_boxes_xywh_w]

    new_gt_boxes_xywh_h = torchvision.ops.box_convert(torch.stack(new_gt_boxes_xywh_h), 'cxcywh', 'xywh').int()
    new_gt_boxes_xywh_h = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w))
                           for b in new_gt_boxes_xywh_h]
    _gt_crops_w = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in new_gt_boxes_xywh_w]
    _gt_crops_h = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in new_gt_boxes_xywh_h]

    _gt_crops_w_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                           for c in _gt_crops_w if c.shape[1] != 0 and c.shape[2] != 0]
    _gt_crops_h_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                           for c in _gt_crops_h if c.shape[1] != 0 and c.shape[2] != 0]

    # k = 0
    # for i in range(40):
    #     for j in range(4):
    #         try:
    #             # ax[i, j].axis('off')
    #             fig, ax = plt.subplots(1, 2)
    #
    #             ax[0].set_title(f'{k}')
    #             # ax[0].imshow(_gt_crops_w[k].permute(1, 2, 0))
    #             ax[0].imshow(_gt_crops_w_resized[k].permute(1, 2, 0))
    #
    #             ax[1].set_title(f'{k}')
    #             # ax[1].imshow(_gt_crops_h[k].permute(1, 2, 0))
    #             ax[1].imshow(_gt_crops_h_resized[k].permute(1, 2, 0))
    #
    #             k += 1
    #             plt.show()
    #
    #         except IndexError:
    #             continue

    gt_boxes_xywh = torch.stack(gt_boxes_xywh)

    gt_crops = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in gt_boxes_xywh]
    gt_crops_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                        for c in gt_crops if c.shape[1] != 0 and c.shape[2] != 0]

    gt_crops_resized.extend(_gt_crops_w_resized)
    gt_crops_resized.extend(_gt_crops_h_resized)

    num_patches = (len(gt_crops) + len(_gt_crops_w) + len(_gt_crops_h)) if num_patches is None else num_patches
    crops, boxes = sample_random_crops(image, bounding_box_size, num_patches)

    # correct mapping now
    boxes_t = [torch.tensor((b[1], b[0], b[2], b[3])) for b in boxes]
    boxes_t = torch.stack(boxes_t)
    fp_boxes = torchvision.ops.box_convert(boxes_t, 'xywh', 'xyxy')

    # fp_boxes = torchvision.ops.box_convert(boxes, 'xywh', 'xyxy') - incorrect or let crop random switch it

    # boxes_iou = torchvision.ops.box_iou(gt_boxes, fp_boxes)
    boxes_iou = torchvision.ops.box_iou(gt_boxes_for_random_crops, fp_boxes)
    gt_box_match, fp_boxes_match = torch.where(boxes_iou)

    fp_boxes_match_numpy = fp_boxes_match.numpy()

    l2_distances_matrix = np.zeros(shape=(gt_boxes_for_random_crops.shape[0], fp_boxes.shape[0]))
    if radius_elimination is not None:
        fp_boxes_centers = np.stack([get_bbox_center(fp_box) for fp_box in fp_boxes.numpy()]).squeeze()

        # for g_idx, gt_center in enumerate(gt_bbox_centers):
        for g_idx, gt_center in enumerate(gt_bbox_centers_for_random_crops):
            if fp_boxes_centers.ndim == 1:
                fp_boxes_centers = np.expand_dims(fp_boxes_centers, axis=0)
            for f_idx, fp_center in enumerate(fp_boxes_centers):
                l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)

        # l2_distances_matrix = l2_distances_matrix[:gt_bbox_centers.shape[0], ...]
        l2_distances_matrix = l2_distances_matrix[:gt_bbox_centers_for_random_crops.shape[0], ...]
        gt_r_invalid, fp_r_invalid = np.where(l2_distances_matrix < radius_elimination)

        fp_boxes_match_numpy = np.union1d(fp_boxes_match.numpy(), fp_r_invalid)

    valid_fp_boxes_idx = np.setdiff1d(np.arange(fp_boxes.shape[0]), fp_boxes_match_numpy)

    # overlapping_boxes = [fp_boxes[fp_boxes_match], gt_boxes[gt_box_match]]
    overlapping_boxes = [fp_boxes[fp_boxes_match], gt_boxes_for_random_crops[gt_box_match]]

    # if plot:
    # show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, overlapping_boxes=overlapping_boxes)
    # show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, title='xyxy',
    #                            overlapping_boxes=overlapping_boxes)
    # show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes_for_random_crops, title='xyxy',
    #                            overlapping_boxes=overlapping_boxes)
    # show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
    #                            title='xywh')
    # gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
    # plt.imshow(gt_crops_grid.permute(1, 2, 0))
    # plt.show()

    # fp_crops_grid = torchvision.utils.make_grid(crops)
    # plt.imshow(fp_crops_grid.permute(1, 2, 0))
    # plt.show()

    fp_boxes = fp_boxes[valid_fp_boxes_idx]
    boxes = boxes[valid_fp_boxes_idx]
    crops = crops[valid_fp_boxes_idx]
    # crops = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in boxes]
    # crops = torch.stack(crops)

    if plot:
        show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
                                   title='xywh')
        # show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, title='xyxy')
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes_for_random_crops, title='xyxy')
        gt_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

    boxes_to_replace = num_patches - fp_boxes.shape[0]
    replaceable_boxes, replaceable_fp_boxes, replaceable_crops = [], [], []
    replacement_required = False

    try:
        replacement_required = replace_overlapping_boxes(
            bounding_box_size, boxes_to_replace,
            gt_bbox_centers_for_random_crops, gt_boxes_for_random_crops, image,
            radius_elimination, replaceable_boxes, replaceable_crops,
            replaceable_fp_boxes, replacement_required)
    except TimeoutException:
        logger.warning(f'Replacement timed-out : {REPLACEMENT_TIMEOUT}!! Skipping frame')
        return {}, {}

    if replacement_required:
        replaceable_boxes = torch.cat(replaceable_boxes)
        replaceable_fp_boxes = torch.cat(replaceable_fp_boxes)
        replaceable_crops = torch.cat(replaceable_crops)

        boxes = torch.cat((boxes, replaceable_boxes))
        fp_boxes = torch.cat((fp_boxes, replaceable_fp_boxes))
        crops = torch.cat((crops, replaceable_crops))

    if plot:
        # show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes)
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes_for_random_crops)
        show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
                                   title='xywh')
        # fp_crops_grid = torchvision.utils.make_grid(crops)
        # plt.imshow(fp_crops_grid.permute(1, 2, 0))
        # plt.show()

    gt_crops_resized = torch.stack(gt_crops_resized)

    # data-augmentation
    if img_transforms:
        out = [img_transforms(image=np.transpose(im, [1, 2, 0])) for im in gt_crops_resized.numpy()]
        out = [torch.from_numpy(o['image']).permute(2, 0, 1) for o in out]
        gt_crops_resized = torch.stack(out)

        out = [img_transforms(image=np.transpose(im, [1, 2, 0])) for im in crops.numpy()]
        out = [torch.from_numpy(o['image']).permute(2, 0, 1) for o in out]
        crops = torch.stack(out)

    if plot:
        gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

        gt_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

    gt_patches_and_labels = {'patches': gt_crops_resized, 'labels': torch.ones(size=(gt_crops_resized.shape[0],))}
    fp_patches_and_labels = {'patches': crops, 'labels': torch.zeros(size=(crops.shape[0],))}

    return gt_patches_and_labels, fp_patches_and_labels


@timeout_decorator.timeout(seconds=REPLACEMENT_TIMEOUT, timeout_exception=TimeoutException)
def replace_overlapping_boxes(bounding_box_size, boxes_to_replace, gt_bbox_centers_for_random_crops,
                              gt_boxes_for_random_crops, image, radius_elimination, replaceable_boxes,
                              replaceable_crops, replaceable_fp_boxes, replacement_required):
    while boxes_to_replace != 0:
        replacement_required = True
        temp_crops, temp_boxes = sample_random_crops(image, bounding_box_size, boxes_to_replace)
        boxes_temp = [torch.tensor((b[1], b[0], b[2], b[3])) for b in temp_boxes]
        boxes_temp = torch.stack(boxes_temp)
        temp_fp_boxes = torchvision.ops.box_convert(boxes_temp, 'xywh', 'xyxy')

        # temp_boxes_iou = torchvision.ops.box_iou(gt_boxes, temp_fp_boxes)
        temp_boxes_iou = torchvision.ops.box_iou(gt_boxes_for_random_crops, temp_fp_boxes)
        temp_gt_box_match, temp_fp_boxes_match = torch.where(temp_boxes_iou)

        temp_fp_boxes_match_numpy = temp_fp_boxes_match.numpy()
        if radius_elimination:
            # temp_l2_distances_matrix = np.zeros(shape=(gt_boxes.shape[0], temp_fp_boxes.shape[0]))
            temp_l2_distances_matrix = np.zeros(shape=(gt_boxes_for_random_crops.shape[0], temp_fp_boxes.shape[0]))
            temp_fp_boxes_centers = np.stack([get_bbox_center(fp_box) for fp_box in temp_fp_boxes.numpy()]).squeeze()

            # for g_idx, gt_center in enumerate(gt_bbox_centers):
            for g_idx, gt_center in enumerate(gt_bbox_centers_for_random_crops):
                if temp_fp_boxes_centers.ndim == 1:
                    temp_fp_boxes_centers = np.expand_dims(temp_fp_boxes_centers, axis=0)
                for f_idx, fp_center in enumerate(temp_fp_boxes_centers):
                    try:
                        temp_l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)
                    except TypeError:
                        raise TimeoutException

            temp_gt_r_invalid, temp_fp_r_invalid = np.where(temp_l2_distances_matrix < radius_elimination)

            temp_fp_boxes_match_numpy = np.union1d(temp_fp_boxes_match.numpy(), temp_fp_r_invalid)

        temp_valid_fp_boxes_idx = np.setdiff1d(np.arange(temp_fp_boxes.shape[0]), temp_fp_boxes_match_numpy)
        replaceable_boxes.append(temp_boxes[temp_valid_fp_boxes_idx])
        replaceable_fp_boxes.append(temp_fp_boxes[temp_valid_fp_boxes_idx])
        replaceable_crops.append(temp_crops[temp_valid_fp_boxes_idx])

        boxes_to_replace -= temp_valid_fp_boxes_idx.size
    return replacement_required


def patches_and_labels_with_anchors_v1(image, bounding_box_size, annotations, frame_number, num_patches=None,
                                       new_shape=None,
                                       use_generated=True, radius_elimination=None, plot=False,
                                       only_long_trajectories=False,
                                       track_length_threshold=60, img_transforms=None, additional_w=None,
                                       additional_h=None,
                                       aspect_ratios=(), scales=()):
    original_shape = (image.shape[2], image.shape[3]) if image.ndim == 4 else (image.shape[1], image.shape[2])
    new_shape = original_shape if new_shape is None else original_shape

    if only_long_trajectories:
        if use_generated:
            frame_annotation = get_generated_frame_annotations(annotations, frame_number)
            track_annotations = get_generated_track_annotations_for_frame(annotations, frame_number)
            track_lengths = np.array([t.shape[0] for t in track_annotations])
            feasible_track_length = track_lengths > track_length_threshold
            feasible_frame_annotations = frame_annotation[feasible_track_length]
            gt_boxes = torch.from_numpy(feasible_frame_annotations[:, 1:5].astype(np.int))
            generated_track_idx = feasible_frame_annotations[:, 0]
            gt_bbox_centers = feasible_frame_annotations[:, 7:9]
        else:
            return NotImplemented
    else:
        if use_generated:
            frame_annotation = get_generated_frame_annotations(annotations, frame_number)
            gt_boxes = torch.from_numpy(frame_annotation[:, 1:5].astype(np.int))
            generated_track_idx = frame_annotation[:, 0]
            gt_bbox_centers = frame_annotation[:, 7:9]
        else:
            frame_annotation = get_frame_annotations_and_skip_lost(annotations, frame_number)
            gt_annotations, gt_bbox_centers = scale_annotations(frame_annotation,
                                                                original_scale=original_shape,
                                                                new_scale=new_shape, return_track_id=False,
                                                                tracks_with_annotations=True)
            gt_boxes = torch.from_numpy(gt_annotations[:, :-1])

    if frame_annotation.size == 0 or feasible_frame_annotations.size == 0:
        return {}, {}

    gt_boxes_xywh = torchvision.ops.box_convert(gt_boxes, 'xyxy', 'xywh')
    gt_boxes_xywh = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w)) for b in gt_boxes_xywh]

    new_gt_boxes_xywh_w = []
    new_gt_boxes_xywh_h = []

    gt_boxes_cxcywh = torchvision.ops.box_convert(gt_boxes, 'xyxy', 'cxcywh')
    if len(aspect_ratios) != 0 and len(scales) != 0:
        for scale in scales:
            for ratio in aspect_ratios:
                for box in gt_boxes_cxcywh:
                    _x, _y, _w, _h = box
                    adjusted_box = torch.tensor((_x, _y, int(_w * ratio * scale), int(_h * (1 / ratio) * scale)))
                    new_gt_boxes_xywh_w.append(adjusted_box)
                    adjusted_box = torch.tensor((_x, _y, int(_w * 1 / ratio * scale), int(_h * ratio) * scale))
                    new_gt_boxes_xywh_h.append(adjusted_box)

    new_gt_boxes_xywh_w = torchvision.ops.box_convert(torch.stack(new_gt_boxes_xywh_w), 'cxcywh', 'xywh').int()
    new_gt_boxes_xywh_w = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w))
                           for b in new_gt_boxes_xywh_w]

    new_gt_boxes_xywh_h = torchvision.ops.box_convert(torch.stack(new_gt_boxes_xywh_h), 'cxcywh', 'xywh').int()
    new_gt_boxes_xywh_h = [torch.tensor((b[1], b[0], b[2] + additional_h, b[3] + additional_w))
                           for b in new_gt_boxes_xywh_h]
    _gt_crops_w = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in new_gt_boxes_xywh_w]
    _gt_crops_h = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in new_gt_boxes_xywh_h]

    _gt_crops_w_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                           for c in _gt_crops_w if c.shape[1] != 0 and c.shape[2] != 0]
    _gt_crops_h_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                           for c in _gt_crops_h if c.shape[1] != 0 and c.shape[2] != 0]

    # k = 0
    # for i in range(40):
    #     for j in range(4):
    #         try:
    #             # ax[i, j].axis('off')
    #             fig, ax = plt.subplots(1, 2)
    #
    #             ax[0].set_title(f'{k}')
    #             # ax[0].imshow(_gt_crops_w[k].permute(1, 2, 0))
    #             ax[0].imshow(_gt_crops_w_resized[k].permute(1, 2, 0))
    #
    #             ax[1].set_title(f'{k}')
    #             # ax[1].imshow(_gt_crops_h[k].permute(1, 2, 0))
    #             ax[1].imshow(_gt_crops_h_resized[k].permute(1, 2, 0))
    #
    #             k += 1
    #             plt.show()
    #
    #         except IndexError:
    #             continue

    gt_boxes_xywh = torch.stack(gt_boxes_xywh)

    gt_crops = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in gt_boxes_xywh]
    gt_crops_resized = [tvf.resize(c, [bounding_box_size, bounding_box_size])
                        for c in gt_crops if c.shape[1] != 0 and c.shape[2] != 0]

    gt_crops_resized.extend(_gt_crops_w_resized)
    gt_crops_resized.extend(_gt_crops_h_resized)

    num_patches = (len(gt_crops) + len(_gt_crops_w) + len(_gt_crops_h)) if num_patches is None else num_patches
    crops, boxes = sample_random_crops(image, bounding_box_size, num_patches, return_correct_box=True)

    # correct mapping now
    # boxes_t = [torch.tensor((b[1], b[0], b[2], b[3])) for b in boxes]
    # boxes_t = torch.stack(boxes_t)
    fp_boxes = torchvision.ops.box_convert(boxes, 'xywh', 'xyxy')

    # fp_boxes = torchvision.ops.box_convert(boxes, 'xywh', 'xyxy') - incorrect or let crop random switch it

    boxes_iou = torchvision.ops.box_iou(gt_boxes, fp_boxes)
    gt_box_match, fp_boxes_match = torch.where(boxes_iou)

    fp_boxes_match_numpy = fp_boxes_match.numpy()

    l2_distances_matrix = np.zeros(shape=(fp_boxes.shape[0], fp_boxes.shape[0]))
    if radius_elimination is not None:
        fp_boxes_centers = np.stack([get_bbox_center(fp_box) for fp_box in fp_boxes.numpy()]).squeeze()

        for g_idx, gt_center in enumerate(gt_bbox_centers):
            if fp_boxes_centers.ndim == 1:
                fp_boxes_centers = np.expand_dims(fp_boxes_centers, axis=0)
            for f_idx, fp_center in enumerate(fp_boxes_centers):
                l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)

        l2_distances_matrix = l2_distances_matrix[:gt_bbox_centers.shape[0], ...]
        gt_r_invalid, fp_r_invalid = np.where(l2_distances_matrix < radius_elimination)

        fp_boxes_match_numpy = np.union1d(fp_boxes_match.numpy(), fp_r_invalid)

    valid_fp_boxes_idx = np.setdiff1d(np.arange(fp_boxes.shape[0]), fp_boxes_match_numpy)

    overlapping_boxes = [fp_boxes[fp_boxes_match], gt_boxes[gt_box_match]]

    if plot:
        # show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, overlapping_boxes=overlapping_boxes)
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, title='xyxy',
                                   overlapping_boxes=overlapping_boxes)
        # show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
        #                            title='xywh')
        gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

        fp_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(fp_crops_grid.permute(1, 2, 0))
        plt.show()

    fp_boxes = fp_boxes[valid_fp_boxes_idx]
    boxes = boxes[valid_fp_boxes_idx]
    crops = crops[valid_fp_boxes_idx]
    # crops = [tvf.crop(image, top=b[0], left=b[1], width=b[2], height=b[3]) for b in boxes]
    # crops = torch.stack(crops)

    if plot:
        show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
                                   title='xywh')
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes, title='xyxy')
        gt_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

    boxes_to_replace = num_patches - fp_boxes.shape[0]
    replaceable_boxes, replaceable_fp_boxes, replaceable_crops = [], [], []
    replacement_required = False

    while boxes_to_replace != 0:
        replacement_required = True
        temp_crops, temp_boxes = sample_random_crops(image, bounding_box_size, boxes_to_replace,
                                                     return_correct_box=True)
        # boxes_temp = [torch.tensor((b[1], b[0], b[2], b[3])) for b in temp_boxes]
        # boxes_temp = torch.stack(boxes_temp)
        temp_fp_boxes = torchvision.ops.box_convert(temp_boxes, 'xywh', 'xyxy')

        temp_boxes_iou = torchvision.ops.box_iou(gt_boxes, temp_fp_boxes)
        temp_gt_box_match, temp_fp_boxes_match = torch.where(temp_boxes_iou)

        temp_fp_boxes_match_numpy = temp_fp_boxes_match.numpy()
        if radius_elimination:
            temp_l2_distances_matrix = np.zeros(shape=(gt_boxes.shape[0], temp_fp_boxes.shape[0]))
            temp_fp_boxes_centers = np.stack([get_bbox_center(fp_box) for fp_box in temp_fp_boxes.numpy()]).squeeze()

            for g_idx, gt_center in enumerate(gt_bbox_centers):
                if temp_fp_boxes_centers.ndim == 1:
                    temp_fp_boxes_centers = np.expand_dims(temp_fp_boxes_centers, axis=0)
                for f_idx, fp_center in enumerate(temp_fp_boxes_centers):
                    temp_l2_distances_matrix[g_idx, f_idx] = np.linalg.norm((gt_center - fp_center), ord=2, axis=-1)

            temp_gt_r_invalid, temp_fp_r_invalid = np.where(temp_l2_distances_matrix < radius_elimination)

            temp_fp_boxes_match_numpy = np.union1d(temp_fp_boxes_match.numpy(), temp_fp_r_invalid)

        temp_valid_fp_boxes_idx = np.setdiff1d(np.arange(temp_fp_boxes.shape[0]), temp_fp_boxes_match_numpy)
        replaceable_boxes.append(temp_boxes[temp_valid_fp_boxes_idx])
        replaceable_fp_boxes.append(temp_fp_boxes[temp_valid_fp_boxes_idx])
        replaceable_crops.append(temp_crops[temp_valid_fp_boxes_idx])

        boxes_to_replace -= temp_valid_fp_boxes_idx.size

    if replacement_required:
        replaceable_boxes = torch.cat(replaceable_boxes)
        replaceable_fp_boxes = torch.cat(replaceable_fp_boxes)
        replaceable_crops = torch.cat(replaceable_crops)

        boxes = torch.cat((boxes, replaceable_boxes))
        fp_boxes = torch.cat((fp_boxes, replaceable_fp_boxes))
        crops = torch.cat((crops, replaceable_crops))

    if plot:
        show_image_with_crop_boxes(image.permute(1, 2, 0), fp_boxes, gt_boxes)
        show_image_with_crop_boxes(image.permute(1, 2, 0), boxes, gt_boxes_xywh, xywh_mode_v2=False, xyxy_mode=False,
                                   title='xywh')
        fp_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(fp_crops_grid.permute(1, 2, 0))
        plt.show()

    gt_crops_resized = torch.stack(gt_crops_resized)

    # data-augmentation
    if img_transforms:
        out = [img_transforms(image=np.transpose(im, [1, 2, 0])) for im in gt_crops_resized.numpy()]
        out = [torch.from_numpy(o['image']).permute(2, 0, 1) for o in out]
        gt_crops_resized = torch.stack(out)

        out = [img_transforms(image=np.transpose(im, [1, 2, 0])) for im in crops.numpy()]
        out = [torch.from_numpy(o['image']).permute(2, 0, 1) for o in out]
        crops = torch.stack(out)

    if plot:
        gt_crops_grid = torchvision.utils.make_grid(gt_crops_resized)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

        gt_crops_grid = torchvision.utils.make_grid(crops)
        plt.imshow(gt_crops_grid.permute(1, 2, 0))
        plt.show()

    gt_patches_and_labels = {'patches': gt_crops_resized, 'labels': torch.ones(size=(gt_crops_resized.shape[0],))}
    fp_patches_and_labels = {'patches': crops, 'labels': torch.zeros(size=(crops.shape[0],))}

    return gt_patches_and_labels, fp_patches_and_labels


def test_patches_and_labels(video_path, frame_number, bounding_box_size, annotations, num_patches=None,
                            use_generated=True, plot=False, only_long_trajectories=False, img_transforms=None,
                            additional_w=None, additional_h=None, aspect_ratios=(), scales=(),
                            track_length_threshold_for_random_crops=30):
    frame = extract_frame_from_video(video_path, frame_number)
    frame = torch.from_numpy(frame).permute(2, 0, 1)
    # patches_and_labels(frame, bounding_box_size, annotations, frame_number, num_patches,
    #                    use_generated=use_generated, radius_elimination=100, plot=plot,
    #                    only_long_trajectories=only_long_trajectories, img_transforms=img_transforms,
    #                    additional_w=additional_w, additional_h=additional_h)
    # patches_and_labels_with_anchors(frame, bounding_box_size, annotations, frame_number, num_patches,
    #                                 use_generated=use_generated, radius_elimination=150, plot=plot,
    #                                 only_long_trajectories=only_long_trajectories, img_transforms=img_transforms,
    #                                 additional_w=additional_w, additional_h=additional_h, scales=scales,
    #                                 aspect_ratios=aspect_ratios)
    patches_and_labels_with_anchors_different_crop_track_threshold(
        frame, bounding_box_size, annotations, frame_number, num_patches,
        use_generated=use_generated, radius_elimination=150, plot=plot,
        only_long_trajectories=only_long_trajectories, img_transforms=img_transforms,
        additional_w=additional_w, additional_h=additional_h, scales=scales,
        aspect_ratios=aspect_ratios,
        track_length_threshold_for_random_crops=track_length_threshold_for_random_crops)
    print()


if __name__ == '__main__':
    use_generated_annotations = True
    box_size = 50
    num_patch = 10

    v_root_path = '../Datasets/SDD/videos/'
    annotation_root_path = '../Datasets/SDD/annotations/'
    generated_annotation_root_path = '../Plots/baseline_v2/v0/'
    v_clz = SDDVideoClasses.DEATH_CIRCLE
    v_num = 3

    v_path = f'{v_root_path}{v_clz.value}/video{v_num}/video.mov'
    annotation_path = f'{annotation_root_path}{v_clz.value}/video{v_num}/annotation_augmented.csv'
    generated_annotations = pd.read_csv(f'{generated_annotation_root_path}{v_clz.value}{v_num}/'
                                        f'csv_annotation/generated_annotations.csv')
    f_num = 100

    annotations_df = read_annotation_file(annotation_path)

    img_t = aug_transforms.Compose([
        aug_transforms.HorizontalFlip(p=0.4),
        aug_transforms.VerticalFlip(p=0.4),
        aug_transforms.Rotate(p=0.4),
        aug_transforms.RandomBrightnessContrast(p=0.2),
        aug_transforms.ShiftScaleRotate(p=0.4)
    ], p=0.7)

    test_patches_and_labels(v_path, f_num, box_size,
                            generated_annotations if use_generated_annotations else annotations_df,
                            use_generated=use_generated_annotations, plot=True, only_long_trajectories=True,
                            img_transforms=img_t, additional_w=20, additional_h=20, aspect_ratios=(0.75, 0.5),
                            scales=(2.0,))

    # img_path = '../../Datasets/SDD/annotations/deathCircle/video4/reference.jpg'
    # image = tio.read_image(img_path)
    # # batched
    # image = image.unsqueeze(0).repeat(3, 1, 1, 1)

    # image_paths = ['../../Datasets/SDD/annotations/little/video1/reference.jpg',
    #                '../../Datasets/SDD/annotations/little/video2/reference.jpg']
    # image = [tio.read_image(img_path) for img_path in image_paths]
    # image = torch.stack(image)
    #
    # out_crops, out_boxes = sample_random_crops(image, box_size, 10)
    #
    # for crop_idx in range(out_crops.shape[0]):
    #     # show_crop(image.permute(1, 2, 0), out_crops[crop_idx].permute(1, 2, 0), [out_boxes[crop_idx]])
    #     show_crop(image[0].permute(1, 2, 0), out_crops[crop_idx][0].permute(1, 2, 0), [out_boxes[crop_idx]])
    #     show_crop(image[1].permute(1, 2, 0), out_crops[crop_idx][1].permute(1, 2, 0), [out_boxes[crop_idx]])
    #     print()

    # fp_boxes_numpy = fp_boxes.numpy()
    # for fp_idx, (fp_box, (c_x, c_y)) in enumerate(zip(fp_boxes_numpy, gt_bbox_centers)):
    #     is_any_corner_inside = []
    #     x1, y1, x2, y2 = fp_box
    #     is_any_corner_inside.append(is_point_inside_circle(c_x, c_y, radius_elimination, x1, y1))
    #     is_any_corner_inside.append(is_point_inside_circle(c_x, c_y, radius_elimination, x1, y2))
    #     is_any_corner_inside.append(is_point_inside_circle(c_x, c_y, radius_elimination, x2, y1))
    #     is_any_corner_inside.append(is_point_inside_circle(c_x, c_y, radius_elimination, x2, y2))
    #     is_any_corner_inside = np.array(is_any_corner_inside)
    #
    #     if is_any_corner_inside.any() and not np.isin(fp_box, fp_boxes_numpy):
    #         fp_boxes_radius_elimination_idx.append(fp_idx)
