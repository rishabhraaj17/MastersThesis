import os
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from log import get_logger
from interplay import setup_frame_only_dataset
from interplay_utils import frames_only_collate_fn
from location_utils import locations_from_heatmaps, prune_locations_proximity_based, \
    get_adjusted_object_locations
from utils import heat_map_collate_fn, ImagePadder
from train import build_model

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


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


def viz_pred(heat_map, frame, show=False):
    fig, ax = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
    ax[0].imshow(heat_map)
    ax[1].imshow(frame)

    ax[0].set_title('Predictions')
    ax[1].set_title('RGB')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def viz_locations_for_3_heads(head0, head1, head2, image, frame_numbers, show=False):
    fig, ax = plt.subplots(1, 3, sharex='none', sharey='none', figsize=(16, 10))
    h0_axis, h1_axis, h2_axis = ax

    h0_axis.set_title('Head 0')
    h1_axis.set_title('Head 1')
    h2_axis.set_title('Head 2')

    num_frames = [loc.frame_number for loc in head0.locations]
    random_frame = np.random.choice(frame_numbers, 1, replace=False).item()

    h0_axis.imshow(image[frame_numbers.index(random_frame)])
    h1_axis.imshow(image[frame_numbers.index(random_frame)])
    h2_axis.imshow(image[frame_numbers.index(random_frame)])

    frame_num_idx = [idx for idx, loc in enumerate(head0.locations) if loc.frame_number == random_frame]
    if len(frame_num_idx) > 1:
        frame_num_idx = [frame_num_idx[0]]

    frame_num_idx = frame_num_idx[0]

    h0_axis.plot(head0.locations[frame_num_idx].scaled_locations[:, 0],
                 head0.locations[frame_num_idx].scaled_locations[:, 1],
                 'o', markerfacecolor='r', markeredgecolor='k', markersize=4)
    h1_axis.plot(head1.locations[frame_num_idx].scaled_locations[:, 0],
                 head1.locations[frame_num_idx].scaled_locations[:, 1],
                 'o', markerfacecolor='r', markeredgecolor='k', markersize=4)
    h2_axis.plot(head2.locations[frame_num_idx].scaled_locations[:, 0],
                 head2.locations[frame_num_idx].scaled_locations[:, 1],
                 'o', markerfacecolor='r', markeredgecolor='k', markersize=4)

    plt.suptitle(f"Frame: {random_frame}")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


@hydra.main(config_path="config", config_name="config")
def extract_locations(cfg):
    logger.info(f'Extract Locations...')
    logger.info(f'Setting up DataLoader and Model...')

    # adjust config here
    cfg.device = 'cpu'  # 'cuda:0'
    cfg.single_video_mode.enabled = True  # for now we work on single video
    cfg.preproccesing.pad_factor = 8
    cfg.frame_rate = 30.
    cfg.video_based.enabled = False

    cfg.eval.objectness.prune_radius = 8

    if cfg.single_video_mode.enabled:
        # config adapt
        cfg.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
        cfg.single_video_mode.video_numbers_to_use = [[2]]
        cfg.desired_pixel_to_meter_ratio_rgb = 0.07
        cfg.desired_pixel_to_meter_ratio = 0.07

        train_dataset = setup_frame_only_dataset(cfg)
        val_dataset = None
    else:
        return NotImplemented

    # position map model config
    cfg.model = 'DeepLabV3Plus'
    position_model = build_model(cfg, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=None,
                                 additional_loss_functions=None, collate_fn=heat_map_collate_fn,
                                 desired_output_shape=None)

    # load model
    checkpoint_path = f'{cfg.interplay_v0.use_pretrained.checkpoint.root}' \
                      f'{cfg.interplay_v0.use_pretrained.checkpoint.path}' \
                      f'{cfg.interplay_v0.use_pretrained.checkpoint.version}/checkpoints/'
    checkpoint_files = os.listdir(checkpoint_path)

    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

    checkpoint_file = checkpoint_path + checkpoint_files[-cfg.interplay_v0.use_pretrained.checkpoint.top_k]

    logger.info(f'Loading weights from: {checkpoint_file}')
    load_dict = torch.load(checkpoint_file, map_location=cfg.device)

    position_model.load_state_dict(load_dict['state_dict'], strict=False)

    position_model.to(cfg.device)
    position_model.eval()

    train_loader = DataLoader(train_dataset, batch_size=cfg.interplay_v0.batch_size, shuffle=False,
                              num_workers=cfg.interplay_v0.num_workers, collate_fn=frames_only_collate_fn,
                              pin_memory=False, drop_last=cfg.interplay_v0.drop_last)

    head0_locations, head1_locations, head2_locations = \
        Locations(locations=[]), Locations(locations=[]), Locations(locations=[])
    for t_idx, data in enumerate(tqdm(train_loader)):
        frames, meta = data

        padder = ImagePadder(frames.shape[-2:], factor=cfg.preproccesing.pad_factor)
        frames = padder.pad(frames)[0]
        frames = frames.to(cfg.device)

        with torch.no_grad():
            pred_position_maps = position_model(frames)

        pred_object_locations = locations_from_heatmaps(
            frames, cfg.interplay_v0.objectness.kernel,
            cfg.interplay_v0.objectness.loc_cutoff,
            cfg.interplay_v0.objectness.marker_size, pred_position_maps,
            vis_on=False)

        for h_idx, head_loc in enumerate(pred_object_locations):
            selected_head_predictions = pred_position_maps[h_idx]

            selected_locations, loc_obj = [], []
            for s_loc, m, head_pred in zip(head_loc, meta, selected_head_predictions):
                frame_num = m['item']
                pruned_locations, pruned_locations_idx = prune_locations_proximity_based(
                    s_loc.cpu().numpy(), cfg.eval.objectness.prune_radius)
                pruned_locations = torch.from_numpy(pruned_locations)
                selected_locations.append(pruned_locations)

                pred_object_locations_scaled, heat_maps_gt_scaled = get_adjusted_object_locations(
                    [pruned_locations.cpu()], head_pred.unsqueeze(0).cpu(), [m])

                loc_obj.append(Location(frame_number=frame_num, locations=s_loc.cpu().numpy(),
                                        pruned_locations=pruned_locations.cpu().numpy(),
                                        scaled_locations=np.stack(pred_object_locations_scaled).squeeze()))

            if h_idx == 0:
                head0_locations.locations.extend(loc_obj)
            elif h_idx == 1:
                head1_locations.locations.extend(loc_obj)
            elif h_idx == 2:
                head2_locations.locations.extend(loc_obj)
            else:
                logger.warning('More than expected heads')

        viz_locations_for_3_heads(
            head0_locations, head1_locations, head2_locations,
            interpolate(frames, size=meta[0]['original_shape']).squeeze(0).permute(0, 2, 3, 1),
            frame_numbers=[m['item'] for m in meta], show=False)

    save_dict = {
        'locations': ExtractedLocations(
            video_class=getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]),
            video_numbers=cfg.single_video_mode.video_numbers_to_use[0][0],
            shape=meta[0]['downscale_shape'],
            scaled_shape=meta[0]['original_shape'],
            head0=head0_locations,
            head1=head1_locations,
            head2=head2_locations,
            padded_shape=(frames.shape[-2], frames.shape[-1])
        )
    }
    filename = 'extracted_locations.pt'
    save_path = os.path.join(os.getcwd(),
                             f'ExtractedLocations'
                             f'/{getattr(SDDVideoClasses, cfg.single_video_mode.video_classes_to_use[0]).name}'
                             f'/{cfg.single_video_mode.video_numbers_to_use[0][0]}/')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path + filename)
    logger.info(f"Saved trajectories at {save_path}{filename}")


if __name__ == '__main__':
    extract_locations()
