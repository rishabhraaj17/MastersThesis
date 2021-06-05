import albumentations as A
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.feature import blob_log
import torch.distributions as D
from torchvision.transforms import ToPILImage

from average_image.constants import SDDVideoClasses
from baselinev2.utils import get_generated_frame_annotations
from gmm import GaussianMixture
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


if __name__ == '__main__':
    sample_path = '../../Plots/proposed_method/v0/position_maps/' \
                  'PositionMapUNetHeatmapSegmentation_BinaryFocalLossWithLogits/' \
                  'version_424798/epoch=61-step=72787/sample.pt'
    sample = torch.load(sample_path)
    rgb_im, heat_mask, pred_mask, m = sample['rgb'], sample['mask'], sample['out'], sample['meta']
    create_mixture_of_gaussians(masks=pred_mask, blob_overlap=0.2, blob_threshold=0.2,
                                get_generated=False, meta_list=m, rgb_img=rgb_im,
                                video_class=SDDVideoClasses.DEATH_CIRCLE, video_number=2)  # these params look good
