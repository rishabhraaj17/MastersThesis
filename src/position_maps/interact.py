import albumentations as A
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.feature import blob_log
from torchvision.transforms import ToPILImage

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import get_dataset
from utils import generate_position_map, overlay_images


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

    split_name: str = 'splits_v3' if get_generated else 'splits_v1'
    dataset = get_dataset(SDDVideoClasses.DEATH_CIRCLE, video_number=2, mode=NetworkMode.TRAIN,
                          meta_label=SDDVideoDatasets.DEATH_CIRCLE, get_generated=get_generated,
                          split_name=split_name)
    tracks = dataset.tracks.reshape((-1, 8))
    tracks = np.stack((tracks[..., 0], tracks[..., 5], tracks[..., -2], tracks[..., -1])).T
    tracks_df = pd.DataFrame(tracks, columns=['track_id', 'frame_number', 'x', 'y'])
    current_agents = tracks_df[(tracks_df.track_id.isin([i for i in range(21)])) & (tracks_df.frame_number == 0)].values
    trajectory_locations = current_agents[:, 2:]

    original_shape = meta_list[0]['original_shape']
    trajectory_map = generate_position_map([original_shape[-2], original_shape[-1]],
                                           trajectory_locations,
                                           sigma=1.5 * 4,
                                           heatmap_shape=None,
                                           return_combined=True, hw_mode=True)

    superimposed_image = overlay_images(transformer=ToPILImage(), background=rgb_img[0],
                                        overlay=torch.from_numpy(trajectory_map).unsqueeze(0))
    superimposed_image_flip = overlay_images(transformer=ToPILImage(), background=rgb_img[0],
                                             overlay=torch.from_numpy(scaled_detected_maps[0]).unsqueeze(0))
    print()


if __name__ == '__main__':
    sample_path = '../../Plots/proposed_method/v0/position_maps/' \
                  'PositionMapUNetHeatmapSegmentation_BinaryFocalLossWithLogits/' \
                  'version_424798/epoch=61-step=72787/sample.pt'
    sample = torch.load(sample_path)
    rgb_im, heat_mask, pred_mask, m = sample['rgb'], sample['mask'], sample['out'], sample['meta']
    create_mixture_of_gaussians(masks=pred_mask, blob_overlap=0.2, blob_threshold=0.2,
                                get_generated=False, meta_list=m, rgb_img=rgb_im)  # these params look good
