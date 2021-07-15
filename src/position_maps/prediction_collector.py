import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import skimage
import torch
import torchvision
from kornia.losses import BinaryFocalLossWithLogits
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

import src_lib.models_hub as hub
from average_image.constants import SDDVideoDatasets, SDDVideoClasses
from average_image.utils import SDDMeta
from baselinev2.exceptions import TimeoutException
from log import get_logger
from src.position_maps.evaluate import setup_single_video_dataset, setup_dataset, get_gt_annotations_for_metrics, \
    get_precision_recall_for_metrics, get_image_array_from_figure, process_numpy_video_frame_to_tensor
from src.position_maps.location_utils import locations_from_heatmaps, get_adjusted_object_locations, \
    prune_locations_proximity_based
from src.position_maps.losses import CenterNetFocalLoss
from src.position_maps.utils import heat_map_temporal_4d_collate_fn, heat_map_collate_fn, ImagePadder, \
    plot_image_with_features

seed_everything(42)
logger = get_logger(__name__)


def adjust_config(cfg):
    cfg.eval.device = 'cpu'

    # Pixel to Meter 
    cfg.eval.desired_pixel_to_meter_ratio_rgb = 0.07
    cfg.eval.desired_pixel_to_meter_ratio = 0.07

    # if single video mode
    cfg.eval.test.single_video_mode.enabled = False
    cfg.eval.test.single_video_mode.video_classes_to_use = ['DEATH_CIRCLE']
    cfg.eval.test.single_video_mode.video_numbers_to_use = [[4]]
    cfg.eval.test.single_video_mode.num_videos = -1
    cfg.eval.test.single_video_mode.multiple_videos = False

    # one video at a time
    cfg.eval.video_class = 'DEATH_CIRCLE'
    cfg.eval.video_meta_class = 'DEATH_CIRCLE'
    cfg.eval.test.video_number_to_use = 4
    cfg.eval.test.num_videos = -1
    cfg.eval.dataset_workers = 12
    cfg.eval.test.multiple_videos = False
    cfg.eval.plot_samples = False
    cfg.eval.frame_rate = 30.
    cfg.eval.resize_transform_only = True

    # Temporal 2D
    cfg.eval.video_based.enabled = False
    cfg.eval.video_based.frames_per_clip = 4
    cfg.eval.video_based.gt_idx = -1

    # Loader
    cfg.eval.batch_size = 4
    cfg.eval.shuffle = False
    cfg.eval.num_workers = 0
    cfg.eval.pin_memory = False
    cfg.eval.drop_last = False

    # model
    cfg.eval.model = 'DeepLabV3Plus'  # DeepLabV3PlusTemporal2D or DeepLabV3Plus

    # checkpoint
    cfg.eval.checkpoint.version = 509506
    cfg.eval.checkpoint.top_k = 1

    # object
    cfg.eval.objectness.kernel = 3
    cfg.eval.objectness.loc_cutoff = 0.05
    cfg.eval.objectness.marker_size = 3
    cfg.eval.objectness.index_select = -1
    cfg.eval.objectness.prune_radius = 8  # need to change per video

    cfg.eval.gt_pred_loc_distance_threshold = 2  # in meters

    # video + plot
    cfg.eval.show_plots = False
    cfg.eval.make_video = False


def area_under_the_pr_curve(precision, recall):
    return abs(np.trapz(precision, recall))


def average_precision(precision, recall):
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def plot_precision_recall_curve(ps, rs, title, outname=None, outdir="./", adjust_ticks=False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    auc = area_under_the_pr_curve(ps, rs)
    ap = average_precision(ps, rs)

    plt.figure(dpi=130)
    plt.plot(rs, ps, label=f"Precision-Recall (AUC={auc:.6}, AP={ap:.6})")

    if adjust_ticks:
        plt.xlim([0.0, 1.05])
        plt.xticks(np.arange(11) / 10.0)
        plt.ylim([0.0, 1.05])
        plt.yticks(np.arange(11) / 10.0)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if outname is not None:
        plt.savefig(os.path.join(outdir, outname), dpi=150, facecolor='w', edgecolor='w', orientation='landscape',
                    bbox_inches='tight')
    plt.show()


def plot_precision_vs_recall(ths, ps, rs, title, outname=None, outdir="./", adjust_ticks=False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plt.figure(dpi=130)
    plt.plot(ths, ps, label="Precision")
    plt.plot(ths, rs, label="Recall")

    if adjust_ticks:
        plt.xlim([0.0, 1.05])
        plt.xticks(np.arange(11) / 10.0)
        plt.ylim([0.0, 1.05])
        plt.yticks(np.arange(11) / 10.0)

    plt.xlabel("Threshold")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if outname is not None:
        plt.savefig(os.path.join(outdir, outname), dpi=150, facecolor='w', edgecolor='w', orientation='landscape',
                    bbox_inches='tight')
    plt.show()


def save_predictions_on_disk(cfg, frames_sequence, pred_head_0, pred_head_1, pred_head_2, filename):
    save_path = os.path.join(os.getcwd(),
                             f'HeatMapPredictions'
                             f'/{getattr(SDDVideoClasses, cfg.eval.video_class).name}'
                             f'/{cfg.eval.test.video_number_to_use}/')  # only for single dataset setup
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_dict = {
        'out_head_0': torch.cat(pred_head_0),
        'out_head_1': torch.cat(pred_head_1),
        'out_head_2': torch.cat(pred_head_2),
        'frames_sequence': frames_sequence,
    }
    torch.save(save_dict, save_path + filename)
    logger.info(f"Saved Predictions at {save_path}{filename}")


def join_parts_prediction(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    pred_head_0, pred_head_1, pred_head_2, frames_sequences = [], [], [], []
    for file in files:
        data = torch.load(file)
        out_head_0, out_head_1, out_head_2 = data['out_head_0'], data['out_head_1'], data['out_head_2']
        frames_sequence = data['frames_sequence']

        pred_head_0.append(out_head_0)
        pred_head_1.append(out_head_1)
        pred_head_2.append(out_head_2)
        frames_sequences.append(torch.tensor(frames_sequence))

    save_dict = {
        'out_head_0': torch.cat(pred_head_0),
        'out_head_1': torch.cat(pred_head_1),
        'out_head_2': torch.cat(pred_head_2),
        'frames_sequence': torch.cat(frames_sequences),
    }
    torch.save(save_dict, path + 'predictions.pt')
    logger.info(f"Saved Predictions at {path}predictions.pt")


@hydra.main(config_path="config", config_name="config")
def evaluate_and_store_predicted_maps(cfg):
    adjust_config(cfg)

    logger.info(f'Setting up DataLoader')

    if cfg.eval.test.single_video_mode.enabled:
        train_dataset, test_dataset, target_max_shape = setup_single_video_dataset(cfg)
    else:
        # test_dataset, target_max_shape = setup_multiple_test_datasets(cfg, return_dummy_transform=False)
        test_dataset, _, target_max_shape = setup_dataset(cfg)

    collate_fn = heat_map_temporal_4d_collate_fn if cfg.eval.video_based.enabled else heat_map_collate_fn
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval.batch_size, shuffle=cfg.eval.shuffle,
                             num_workers=cfg.eval.num_workers, collate_fn=collate_fn,
                             pin_memory=cfg.eval.pin_memory, drop_last=cfg.eval.drop_last)

    loss_fn = BinaryFocalLossWithLogits(
        alpha=cfg.eval.loss.bfl.alpha, gamma=cfg.eval.loss.bfl.gamma, reduction=cfg.eval.loss.reduction)
    gauss_loss_fn = [CenterNetFocalLoss()]

    model = getattr(hub, cfg.eval.model)(
        config=cfg, train_dataset=None, val_dataset=None,
        loss_function=loss_fn, collate_fn=heat_map_collate_fn, additional_loss_functions=gauss_loss_fn,
        desired_output_shape=None)

    checkpoint_path = f'{cfg.eval.checkpoint.root}{cfg.eval.checkpoint.path}{cfg.eval.checkpoint.version}/checkpoints/'
    checkpoint_files = os.listdir(checkpoint_path)

    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]

    checkpoint_file = checkpoint_path + checkpoint_files[-cfg.eval.checkpoint.top_k]
    logger.info(f'Loading weights : {checkpoint_file}')

    load_dict = torch.load(checkpoint_file, map_location=cfg.eval.device)
    model.load_state_dict(load_dict['state_dict'])

    model.to(cfg.eval.device)
    model.eval()

    logger.info(f'Starting evaluation for storing heatmaps...')

    total_loss = []

    # chunk size to save
    chunk_size = 3
    chunks = np.linspace(0, len(test_loader) - 1, chunk_size, dtype=np.int32)
    save_iter = chunks[1:]

    pred_head_0, pred_head_1, pred_head_2, frames_sequence = [], [], [], []
    for idx, data in enumerate(tqdm(test_loader)):
        frames, heat_masks, position_map, distribution_map, class_maps, meta = data

        padder = ImagePadder(frames.shape[-2:], factor=cfg.eval.preproccesing.pad_factor)
        frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]
        frames, heat_masks = frames.to(cfg.eval.device), heat_masks.to(cfg.eval.device)

        with torch.no_grad():
            out = model(frames)

        if cfg.eval.video_based.enabled:
            frames = frames[:, -3:, ...]
            heat_masks = heat_masks[:, cfg.eval.video_based.gt_idx, None, ...]

        loss1 = getattr(torch.Tensor, cfg.eval.loss.reduction)(model.calculate_loss(out, heat_masks))
        loss2 = getattr(torch.Tensor, cfg.eval.loss.reduction)(model.calculate_additional_losses(
            out, heat_masks, cfg.eval.loss.gaussian_weight, cfg.eval.loss.apply_sigmoid))
        loss = loss1 + loss2

        total_loss.append(loss.item())

        pred_head_0.append(out[0])
        pred_head_1.append(out[1])
        pred_head_2.append(out[2])
        frames_sequence.extend([m['item'] for m in meta])

        if idx in save_iter:
            save_predictions_on_disk(cfg, frames_sequence, pred_head_0, pred_head_1, pred_head_2,
                                     filename=f'predictions_chunk_size_{chunk_size}_idx{idx}.pt')
            pred_head_0, pred_head_1, pred_head_2, frames_sequence = [], [], [], []

    logger.info(f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                f'Video Number: {cfg.eval.test.video_number_to_use}')
    logger.info(f"Threshold: {cfg.eval.gt_pred_loc_distance_threshold}m | "
                f"Max-Pool kernel size: {cfg.eval.objectness.kernel} | "
                f"Head Used: {cfg.eval.objectness.index_select}")
    logger.info(f"Test Loss: {np.array(total_loss).mean()}")


@hydra.main(config_path="config", config_name="config")
def evaluate_metrics(cfg):
    adjust_config(cfg)

    sdd_meta = SDDMeta(cfg.eval.root + 'H_SDD.txt')

    logger.info(f'Setting up DataLoader')

    if cfg.eval.test.single_video_mode.enabled:
        train_dataset, test_dataset, target_max_shape = setup_single_video_dataset(cfg)
    else:
        # test_dataset, target_max_shape = setup_multiple_test_datasets(cfg, return_dummy_transform=False)
        test_dataset, _, target_max_shape = setup_dataset(cfg)

    collate_fn = heat_map_temporal_4d_collate_fn if cfg.eval.video_based.enabled else heat_map_collate_fn
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval.batch_size, shuffle=cfg.eval.shuffle,
                             num_workers=cfg.eval.num_workers, collate_fn=collate_fn,
                             pin_memory=cfg.eval.pin_memory, drop_last=cfg.eval.drop_last)

    load_path = os.path.join(os.getcwd(),
                             f'HeatMapPredictions'
                             f'/{getattr(SDDVideoClasses, cfg.eval.video_class).name}'
                             f'/{cfg.eval.test.video_number_to_use}/predictions.pt')
    model = torch.load(load_path)  # mock the model
    out_head_0, out_head_1, out_head_2 = model['out_head_0'], model['out_head_1'], model['out_head_2']
    frames_sequence = model['frames_sequence']

    ratio = float(sdd_meta.get_meta(getattr(SDDVideoDatasets, cfg.eval.video_meta_class)
                                    , cfg.eval.test.video_number_to_use)[0]['Ratio'].to_numpy()[0])

    logger.info(f'Starting evaluation for metrics...')

    video_frames = []
    tp_list, fp_list, fn_list = [], [], []

    pred_t_idx = 0
    for idx, data in enumerate(tqdm(test_loader)):
        frames, heat_masks, position_map, distribution_map, class_maps, meta = data

        padder = ImagePadder(frames.shape[-2:], factor=cfg.eval.preproccesing.pad_factor)
        frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]

        out = [
            out_head_0[pred_t_idx: pred_t_idx + cfg.eval.batch_size, ...],
            out_head_1[pred_t_idx: pred_t_idx + cfg.eval.batch_size, ...],
            out_head_2[pred_t_idx: pred_t_idx + cfg.eval.batch_size, ...],
        ]
        frames_seq_stored = frames_sequence[pred_t_idx: pred_t_idx + cfg.eval.batch_size]

        locations = locations_from_heatmaps(frames, cfg.eval.objectness.kernel,
                                            cfg.eval.objectness.loc_cutoff,
                                            cfg.eval.objectness.marker_size, out, vis_on=False)

        # filter out overlapping locations
        selected_locations_pre_pruning = locations[cfg.eval.objectness.index_select]
        selected_locations = []
        for s_loc in selected_locations_pre_pruning:
            pruned_locations, pruned_locations_idx = prune_locations_proximity_based(
                s_loc.numpy(), cfg.eval.objectness.prune_radius)
            selected_locations.append(torch.from_numpy(pruned_locations))

        metrics_out = out[cfg.eval.objectness.index_select]
        blobs_per_image, _ = get_adjusted_object_locations(
            selected_locations, metrics_out, meta)

        for f in range(len(meta)):
            frame_number = meta[f]['item']
            rgb_frame = frames[f].cpu()
            gt_heatmap = heat_masks[f].cpu()
            pred_heatmap = metrics_out.sigmoid()[f].cpu()

            gt_bbox_centers, pred_centers, rgb_frame, supervised_boxes = get_gt_annotations_for_metrics(
                blobs_per_image, cfg, f, frame_number, meta, rgb_frame, test_loader)

            fn, fp, precision, recall, tp = get_precision_recall_for_metrics(cfg, gt_bbox_centers, pred_centers,
                                                                             ratio)
            if cfg.eval.show_plots or cfg.eval.make_video:
                fig = plot_image_with_features(
                    rgb_frame.squeeze(dim=0).permute(1, 2, 0).numpy(), gt_bbox_centers,
                    np.stack(pred_centers), boxes=supervised_boxes,
                    txt=f'Frame Number: {frame_number}\n'
                        f'Agent Count: GT-{len(gt_bbox_centers)} | Pred-{len(pred_centers)}'
                        f'\nPrecision: {precision} | Recall: {recall}',
                    footnote_txt=f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                                 f'Video Number: {cfg.eval.test.video_number_to_use}'
                                 f'\n\nL2 Matching Threshold: '
                                 f'{cfg.eval.gt_pred_loc_distance_threshold}m',
                    video_mode=cfg.eval.make_video,
                    plot_heatmaps=True,
                    gt_heatmap=gt_heatmap.squeeze(dim=0).numpy(),
                    pred_heatmap=pred_heatmap.squeeze(dim=0).numpy())

                if cfg.eval.make_video:
                    video_frame = get_image_array_from_figure(fig)

                    if video_frame.shape[0] != meta[f]['original_shape'][1] \
                            or video_frame.shape[1] != meta[f]['original_shape'][0]:
                        video_frame = skimage.transform.resize(
                            video_frame, (meta[f]['original_shape'][1], meta[f]['original_shape'][0]))
                        video_frame = (video_frame * 255).astype(np.uint8)

                        video_frame = process_numpy_video_frame_to_tensor(video_frame)
                        video_frames.append(video_frame)

            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

        pred_t_idx += cfg.eval.batch_size

    final_precision = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fp_list).sum())
    final_recall = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fn_list).sum())

    logger.info(f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                f'Video Number: {cfg.eval.test.video_number_to_use}')
    logger.info(f"Threshold: {cfg.eval.gt_pred_loc_distance_threshold}m | "
                f"Max-Pool kernel size: {cfg.eval.objectness.kernel} | "
                f"Head Used: {cfg.eval.objectness.index_select}")
    logger.info(f"Precision: {final_precision} | Recall: {final_recall}")

    if cfg.eval.make_video:
        logger.info(f"Writing Video")
        Path(os.path.join(os.getcwd(), 'videos')).mkdir(parents=True, exist_ok=True)
        torchvision.io.write_video(
            f'videos/{getattr(SDDVideoClasses, cfg.eval.video_meta_class).name}_'
            f'{cfg.eval.test.video_number_to_use}_threshold_{cfg.eval.gt_pred_loc_distance_threshold}m_'
            f'max_pool_k_{cfg.eval.objectness.kernel}_head_used_{cfg.eval.objectness.index_select}'
            f'prune_radius_{cfg.eval.objectness.prune_radius}.avi',
            torch.cat(video_frames).permute(0, 2, 3, 1),
            cfg.eval.video_fps)


@hydra.main(config_path="config", config_name="config")
def evaluate_metrics_for_each_threshold(cfg):
    adjust_config(cfg)

    sdd_meta = SDDMeta(cfg.eval.root + 'H_SDD.txt')

    logger.info(f'Setting up DataLoader')

    if cfg.eval.test.single_video_mode.enabled:
        train_dataset, test_dataset, target_max_shape = setup_single_video_dataset(cfg)
    else:
        # test_dataset, target_max_shape = setup_multiple_test_datasets(cfg, return_dummy_transform=False)
        test_dataset, _, target_max_shape = setup_dataset(cfg)

    collate_fn = heat_map_temporal_4d_collate_fn if cfg.eval.video_based.enabled else heat_map_collate_fn
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval.batch_size, shuffle=cfg.eval.shuffle,
                             num_workers=cfg.eval.num_workers, collate_fn=collate_fn,
                             pin_memory=cfg.eval.pin_memory, drop_last=cfg.eval.drop_last)

    load_path = os.path.join(os.getcwd(),
                             f'HeatMapPredictions'
                             f'/{getattr(SDDVideoClasses, cfg.eval.video_class).name}'
                             f'/{cfg.eval.test.video_number_to_use}/predictions.pt')
    model = torch.load(load_path)  # mock the model
    out_head_0, out_head_1, out_head_2 = model['out_head_0'], model['out_head_1'], model['out_head_2']
    frames_sequence = model['frames_sequence']

    ratio = float(sdd_meta.get_meta(getattr(SDDVideoDatasets, cfg.eval.video_meta_class)
                                    , cfg.eval.test.video_number_to_use)[0]['Ratio'].to_numpy()[0])

    logger.info(f'Starting evaluation for metrics...')

    precision_dict, recall_dict = {0: 0.00014344582501905648}, {0: 1.0}

    loc_cutoff = np.linspace(0, 1, 10)[1:]  # since for 0 we know for this seq
    for loc_c in loc_cutoff:
        pred_t_idx = 0
        tp_list, fp_list, fn_list = [], [], []
        for idx, data in enumerate(tqdm(test_loader)):
            frames, heat_masks, position_map, distribution_map, class_maps, meta = data

            padder = ImagePadder(frames.shape[-2:], factor=cfg.eval.preproccesing.pad_factor)
            frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]

            out = [
                out_head_0[pred_t_idx: pred_t_idx + cfg.eval.batch_size, ...],
                out_head_1[pred_t_idx: pred_t_idx + cfg.eval.batch_size, ...],
                out_head_2[pred_t_idx: pred_t_idx + cfg.eval.batch_size, ...],
            ]
            locations = locations_from_heatmaps(frames, cfg.eval.objectness.kernel,
                                                loc_c, cfg.eval.objectness.marker_size, out, vis_on=False)

            # filter out overlapping locations
            selected_locations_pre_pruning = locations[cfg.eval.objectness.index_select]

            # if its empty check for other heads
            if all([s.numel() == 0 for s in selected_locations_pre_pruning]):
                non_empty_predictions = []
                for loc_head in locations:
                    non_empty_count = 0
                    for l_head_out in loc_head:
                        if l_head_out.numel() != 0:
                            non_empty_count += 1
                    non_empty_predictions.append(non_empty_count)

                non_empty_predictions = np.array(non_empty_predictions)
                if non_empty_predictions.any():
                    cfg.eval.objectness.index_select = non_empty_predictions.argmax().item()
                    selected_locations_pre_pruning = locations[cfg.eval.objectness.index_select]
                else:
                    selected_locations_pre_pruning = None

            if selected_locations_pre_pruning is not None:
                selected_locations = []
                for s_loc in selected_locations_pre_pruning:
                    try:
                        pruned_locations, pruned_locations_idx = prune_locations_proximity_based(
                            s_loc.numpy(), cfg.eval.objectness.prune_radius)
                        selected_locations.append(torch.from_numpy(pruned_locations))
                    except TimeoutException:
                        selected_locations.append(s_loc)
                    except ValueError:
                        continue

                metrics_out = out[cfg.eval.objectness.index_select]
                blobs_per_image, _ = get_adjusted_object_locations(
                    selected_locations, metrics_out, meta)

                if len(blobs_per_image) < len(meta):
                    non_empty_idx = []
                    for s_idx, s in enumerate(selected_locations_pre_pruning):
                        if s.numel() != 0:
                            non_empty_idx.append(s_idx)
                    meta = [meta[i] for i in non_empty_idx]

                for f in range(len(meta)):
                    frame_number = meta[f]['item']
                    rgb_frame = frames[f].cpu()
                    gt_heatmap = heat_masks[f].cpu()
                    pred_heatmap = metrics_out.sigmoid()[f].cpu()

                    gt_bbox_centers, pred_centers, rgb_frame, supervised_boxes = get_gt_annotations_for_metrics(
                        blobs_per_image, cfg, f, frame_number, meta, rgb_frame, test_loader)

                    fn, fp, precision, recall, tp = get_precision_recall_for_metrics(cfg, gt_bbox_centers, pred_centers,
                                                                                     ratio)
                    if cfg.eval.show_plots:
                        fig = plot_image_with_features(
                            rgb_frame.squeeze(dim=0).permute(1, 2, 0).numpy(), gt_bbox_centers,
                            np.stack(pred_centers), boxes=supervised_boxes,
                            txt=f'Frame Number: {frame_number}\n'
                                f'Agent Count: GT-{len(gt_bbox_centers)} | Pred-{len(pred_centers)}'
                                f'\nPrecision: {precision} | Recall: {recall}',
                            footnote_txt=f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                                         f'Video Number: {cfg.eval.test.video_number_to_use}'
                                         f'\n\nL2 Matching Threshold: '
                                         f'{cfg.eval.gt_pred_loc_distance_threshold}m',
                            video_mode=False,
                            plot_heatmaps=True,
                            gt_heatmap=gt_heatmap.squeeze(dim=0).numpy(),
                            pred_heatmap=pred_heatmap.squeeze(dim=0).numpy())

                    tp_list.append(tp)
                    fp_list.append(fp)
                    fn_list.append(fn)

            pred_t_idx += cfg.eval.batch_size

        if len(tp_list) != 0 and len(fp_list) != 0 and len(fn_list) != 0:
            final_precision = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fp_list).sum())
            final_recall = np.array(tp_list).sum() / (np.array(tp_list).sum() + np.array(fn_list).sum())
        else:
            final_precision = 0.
            final_recall = 0.

        precision_dict[loc_c] = final_precision
        recall_dict[loc_c] = final_recall

        logger.info(f"Precision: {final_precision} | Recall: {final_recall}")

    plot_precision_vs_recall(
        list(precision_dict.keys()), list(precision_dict.values()), list(recall_dict.values()), 'Precision vs Recall')
    plot_precision_recall_curve(list(precision_dict.values()), list(recall_dict.values()), 'Precision-Recall Curve')
    logger.info(f'Video Class: {getattr(SDDVideoClasses, cfg.eval.video_meta_class).name} | '
                f'Video Number: {cfg.eval.test.video_number_to_use}')
    logger.info(f"Threshold: {cfg.eval.gt_pred_loc_distance_threshold}m | "
                f"Max-Pool kernel size: {cfg.eval.objectness.kernel} | "
                f"Head Used: {cfg.eval.objectness.index_select}")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # evaluate_and_store_predicted_maps()
        # join_parts_prediction(os.path.join(os.getcwd(), f'logs/HeatMapPredictions/DEATH_CIRCLE/4/'))
        # evaluate_metrics()
        evaluate_metrics_for_each_threshold()
