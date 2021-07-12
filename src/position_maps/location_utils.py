import albumentations as A
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmdet.models.utils.gaussian_target import get_local_maximum


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