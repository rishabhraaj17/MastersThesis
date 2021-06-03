import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.feature import blob_log

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import get_dataset
from utils import generate_position_map


def plot_to_debug(im):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(16, 8))
    axs.imshow(im)

    plt.show()


def extract_agents_locations(blob_threshold, mask, objectness_threshold):
    mask = mask.sigmoid()
    mask[mask < objectness_threshold] = 0.
    blobs = []
    for m in mask:
        blobs.append(blob_log(m.squeeze(0).round(), threshold=blob_threshold))
    return blobs, mask


def create_mixture_of_gaussians(mask, objectness_threshold: float = 0.5, blob_threshold: float = 0.45,
                                blob_overlap: float = 0.5):
    blobs, mask = extract_agents_locations(blob_threshold, mask, objectness_threshold)
    # verify
    detected_maps = []
    for blob in blobs:
        locations = blob[:, :2]
        rolled = np.rollaxis(locations, -1).tolist()
        locations_x, locations_y = rolled[1], rolled[0]
        locations = np.stack([locations_x, locations_y]).T
        detected_maps.append(generate_position_map([mask.shape[-2], mask.shape[-1]], locations, sigma=1.5,
                                                   heatmap_shape=None,
                                                   return_combined=True, hw_mode=True))
    # verify ends
    dataset = get_dataset(SDDVideoClasses.DEATH_CIRCLE, video_number=2, mode=NetworkMode.TRAIN,
                          meta_label=SDDVideoDatasets.DEATH_CIRCLE, get_generated=True)
    print()


if __name__ == '__main__':
    sample_path = '../../Plots/proposed_method/v0/position_maps/' \
                  'PositionMapUNetHeatmapSegmentation_BinaryFocalLossWithLogits/' \
                  'version_424798/epoch=61-step=72787/sample.pt'
    sample = torch.load(sample_path)
    rgb_img, heat_mask, pred_mask, meta = sample['rgb'], sample['mask'], sample['out'], sample['meta']
    create_mixture_of_gaussians(mask=pred_mask, blob_overlap=0.2, blob_threshold=0.2)  # these params look good
