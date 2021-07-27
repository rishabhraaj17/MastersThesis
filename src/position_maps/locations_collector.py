import os
import warnings
from pathlib import Path

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
from interplay import setup_frame_only_dataset, setup_frame_only_dataset_flexible
from interplay_utils import frames_only_collate_fn
from location_utils import locations_from_heatmaps, prune_locations_proximity_based, \
    get_adjusted_object_locations
from location_utils import Location, Locations, ExtractedLocations
from utils import heat_map_collate_fn, ImagePadder
from train import build_model

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)

# gates (0,1,2) -
#f {ROOT}wandb/run-20210722_141934-8rgel6j7/files/PositionMap/8rgel6j7
#f {ROOT}wandb/run-20210722_102800-cz42lfgm/files/PositionMap/cz42lfgm

ROOT = '/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/src/position_maps/logs/'
MODEL_PATH_MAPPING = {
    SDDVideoClasses.DEATH_CIRCLE: {
        0: f'{ROOT}lightning_logs/version_509506/',
        1: f'{ROOT}lightning_logs/version_509506/',
        2: f'{ROOT}lightning_logs/version_509506/',
        3: f'{ROOT}lightning_logs/version_511116/',
        4: f'{ROOT}lightning_logs/version_509506/'
    },
    SDDVideoClasses.GATES: {
        0: f'{ROOT}wandb/run-20210722_141934-8rgel6j7/files/PositionMap/8rgel6j7',
        1: f'{ROOT}wandb/run-20210722_141934-8rgel6j7/files/PositionMap/8rgel6j7',
        2: f'{ROOT}wandb/run-20210722_141934-8rgel6j7/files/PositionMap/8rgel6j7',
        3: f'{ROOT}lightning_logs/version_516516/',
        4: f'{ROOT}wandb/run-20210722_104027-y1v1m659/files/PositionMap/y1v1m659',
        5: f'{ROOT}lightning_logs/version_516516/',
        6: f'{ROOT}lightning_logs/version_516516/',
        7: f'{ROOT}wandb/run-20210722_104027-y1v1m659/files/PositionMap/y1v1m659',
        8: f'{ROOT}wandb/run-20210722_104027-y1v1m659/files/PositionMap/y1v1m659'
    },
    SDDVideoClasses.HYANG: {
        0: f'{ROOT}wandb/run-20210723_120209-3mjnkhii/files/PositionMap/3mjnkhii',
        1: f'{ROOT}wandb/run-20210723_161846-3gh1f2yo/files/PositionMap/3gh1f2yo',
        2: f'{ROOT}wandb/run-20210727_141237-13buko8r/files/PositionMap/13buko8r',
        3: f'{ROOT}wandb/run-20210727_141237-13buko8r/files/PositionMap/13buko8r',
        4: f'{ROOT}wandb/run-20210723_162311-cto6qmqr/files/PositionMap/cto6qmqr',
        5: f'{ROOT}wandb/run-20210724_005133-2i25j4c8/files/PositionMap/2i25j4c8',
        6: f'{ROOT}wandb/run-20210727_141237-13buko8r/files/PositionMap/13buko8r',
        7: f'{ROOT}wandb/run-20210723_120209-3mjnkhii/files/PositionMap/3mjnkhii',
        8: f'{ROOT}wandb/run-20210723_120209-3mjnkhii/files/PositionMap/3mjnkhii',
        9: f'{ROOT}wandb/run-20210723_120209-3mjnkhii/files/PositionMap/3mjnkhii',
        10: f'{ROOT}wandb/run-20210727_141237-13buko8r/files/PositionMap/13buko8r',
        11: f'{ROOT}wandb/run-20210727_141237-13buko8r/files/PositionMap/13buko8r',
        12: f'{ROOT}wandb/run-20210727_141237-13buko8r/files/PositionMap/13buko8r',
        13: f'{ROOT}wandb/run-20210727_141237-13buko8r/files/PositionMap/13buko8r',
        14: f'{ROOT}wandb/run-20210727_141237-13buko8r/files/PositionMap/13buko8r'
    },
    SDDVideoClasses.LITTLE: {
        0: f'{ROOT}wandb/run-20210721_174114-2pz3vle2/files/PositionMap/2pz3vle2',
        1: f'{ROOT}wandb/run-20210721_174114-2pz3vle2/files/PositionMap/2pz3vle2',
        2: f'{ROOT}wandb/run-20210721_174114-2pz3vle2/files/PositionMap/2pz3vle2',
        3: f'{ROOT}wandb/run-20210721_174114-2pz3vle2/files/PositionMap/2pz3vle2'
    },
    SDDVideoClasses.NEXUS: {
        0: f'{ROOT}wandb/run-20210722_110438-102uqaua/files/PositionMap/102uqaua',
        1: f'{ROOT}wandb/run-20210722_110438-102uqaua/files/PositionMap/102uqaua',
        2: f'{ROOT}wandb/run-20210722_110438-102uqaua/files/PositionMap/102uqaua',
        3: 'BAD_DATASET',
        4: 'BAD_DATASET',
        5: 'BAD_DATASET',
        6: f'{ROOT}wandb/run-20210724_111012-5ebp711y/files/PositionMap/5ebp711y',
        7: f'{ROOT}wandb/run-20210724_111012-5ebp711y/files/PositionMap/5ebp711y',
        8: f'{ROOT}wandb/run-20210724_111012-5ebp711y/files/PositionMap/5ebp711y',
        9: f'{ROOT}wandb/run-20210722_110438-102uqaua/files/PositionMap/102uqaua',
        10: f'{ROOT}wandb/run-20210722_110438-102uqaua/files/PositionMap/102uqaua',
        11: f'{ROOT}wandb/run-20210722_110438-102uqaua/files/PositionMap/102uqaua'
    },
    SDDVideoClasses.QUAD: {
        0: f'{ROOT}wandb/run-20210722_104400-3e76emv1/files/PositionMap/3e76emv1',
        1: f'{ROOT}wandb/run-20210722_104400-3e76emv1/files/PositionMap/3e76emv1',
        2: f'{ROOT}wandb/run-20210722_104400-3e76emv1/files/PositionMap/3e76emv1',
        3: f'{ROOT}wandb/run-20210722_104400-3e76emv1/files/PositionMap/3e76emv1'
    },
    SDDVideoClasses.BOOKSTORE: {
        0: f'{ROOT}wandb/run-20210726_120031-22yt7yw9/files/PositionMap/22yt7yw9',
        1: f'{ROOT}wandb/run-20210726_120031-22yt7yw9/files/PositionMap/22yt7yw9',
        2: f'{ROOT}wandb/run-20210724_201415-2d6us9kr/files/PositionMap/2d6us9kr',
        3: f'{ROOT}wandb/run-20210726_120031-22yt7yw9/files/PositionMap/22yt7yw9',
        4: f'{ROOT}wandb/run-20210726_120031-22yt7yw9/files/PositionMap/22yt7yw9',
        5: f'{ROOT}wandb/run-20210726_120031-22yt7yw9/files/PositionMap/22yt7yw9',
        6: f'{ROOT}wandb/run-20210727_125014-35bomshn/files/PositionMap/35bomshn'
    },
    SDDVideoClasses.COUPA: {
        0: f'{ROOT}wandb/run-20210724_225713-3fojot5z/files/PositionMap/3fojot5z',
        1: f'{ROOT}wandb/run-20210725_160043-3taaf9g5/files/PositionMap/3taaf9g5',
        2: f'{ROOT}wandb/run-20210725_160043-3taaf9g5/files/PositionMap/3taaf9g5',
        3: f'{ROOT}wandb/run-20210725_160043-3taaf9g5/files/PositionMap/3taaf9g5'
    }
}


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

                scaled_locs = np.stack(pred_object_locations_scaled).squeeze() \
                    if len(pred_object_locations_scaled) != 0 else np.zeros((0, 2))
                loc_obj.append(Location(frame_number=frame_num, locations=s_loc.cpu().numpy(),
                                        pruned_locations=pruned_locations.cpu().numpy(),
                                        scaled_locations=scaled_locs))

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


def extract_locations_core(cfg, position_model, train_loader):
    meta, frames = None, None
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

                scaled_locs = np.stack(pred_object_locations_scaled).squeeze() \
                    if len(pred_object_locations_scaled) != 0 else np.zeros((0, 2))
                loc_obj.append(Location(frame_number=frame_num, locations=s_loc.cpu().numpy(),
                                        pruned_locations=pruned_locations.cpu().numpy(),
                                        scaled_locations=scaled_locs))

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
    return frames, head0_locations, head1_locations, head2_locations, meta


@hydra.main(config_path="config", config_name="config")
def extract_locations_for_all(cfg):
    logger.info(f'Extract Locations...')

    # adjust config here
    cfg.device = 'cpu'  # 'cuda:0'
    cfg.single_video_mode.enabled = True  # for now we work on single video
    cfg.preproccesing.pad_factor = 8
    cfg.frame_rate = 30.
    cfg.video_based.enabled = False

    cfg.eval.objectness.prune_radius = 8
    cfg.desired_pixel_to_meter_ratio = 0.07
    cfg.desired_pixel_to_meter_ratio_rgb = 0.07

    # config adapt
    # done
    # video_classes_to_use = [SDDVideoClasses.DEATH_CIRCLE]
    # video_numbers_to_use = [[i for i in range(5)]]

    video_classes_to_use = [
        SDDVideoClasses.GATES,
        SDDVideoClasses.HYANG,
        SDDVideoClasses.LITTLE,
        SDDVideoClasses.NEXUS,
        SDDVideoClasses.QUAD,
        SDDVideoClasses.BOOKSTORE,
        SDDVideoClasses.COUPA]
    video_numbers_to_use = [
        [i for i in range(9)],
        [i for i in range(15)],
        [i for i in range(4)],
        [i for i in range(12) if i not in [3, 4, 5]],
        [i for i in range(4)],
        [i for i in range(7)],
        [i for i in range(4)], ]

    for v_idx, v_clz in enumerate(video_classes_to_use):
        for v_num in video_numbers_to_use[v_idx]:
            logger.info(f'Setting up dataloader and model for {v_clz.name} - {v_num}')

            train_dataset = setup_frame_only_dataset_flexible(cfg, video_class=[v_clz.name], video_number=[[v_num]])
            val_dataset = None

            # position map model config
            cfg.model = 'DeepLabV3Plus'
            position_model = build_model(cfg, train_dataset=train_dataset, val_dataset=val_dataset, loss_function=None,
                                         additional_loss_functions=None, collate_fn=heat_map_collate_fn,
                                         desired_output_shape=None)

            # load model
            checkpoint_path = os.path.join(MODEL_PATH_MAPPING[v_clz][v_num], 'checkpoints/')
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

            frames, head0_locations, head1_locations, head2_locations, meta = extract_locations_core(
                cfg, position_model, train_loader)

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
    # extract_locations()
    extract_locations_for_all()
