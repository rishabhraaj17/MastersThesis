from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baseline.analyze_features import find_distribution
from baselinev2.config import GENERATED_DATASET_ROOT, SAVE_BASE_PATH, ROOT_PATH, TRAIN_CLASS_FOR_WHOLE, TRAIN_META
from baselinev2.constants import NetworkMode
from baselinev2.notebooks.utils import get_trajectory_splits, get_trajectory_length
from log import initialize_logging, get_logger

matplotlib.style.use('ggplot')

initialize_logging()
logger = get_logger('baselinev2.notebooks.dataset_analysis')


def plot_trajectory_length_histogram(trajectory, ratio, title, save_path=None, bins=None):
    per_step_length, length = get_trajectory_length(trajectory, use_l2=True)
    length *= ratio
    plt.hist(length, bins=bins)
    plt.title(title)

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + f"{title}.png")
        plt.close()
    else:
        plt.show()


def plot_trajectory_length_bar(trajectory, ratio, title, save_path=None, with_scatter=True, binned=True,
                               in_meters=False):
    if trajectory.size == 0:
        binned = False
    per_step_length, length = get_trajectory_length(trajectory, use_l2=True)
    if in_meters:
        length *= ratio
    if binned:
        counts = np.bincount(np.round(length).astype(np.int))
        unique = np.arange(start=0, stop=len(counts))
    else:
        unique, counts = np.unique(length, return_counts=True)
    if with_scatter:
        plt.plot(unique, counts)
    plt.bar(unique, counts)
    plt.title(title)

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + f"{title}.png")
        plt.close()
    else:
        plt.show()


def plot_relative_distances_line_plot(distances, title, save_path=None):
    plt.scatter(distances[..., 0], distances[..., 1])
    median = np.median(distances, axis=0)
    mean = np.mean(distances, axis=0)
    # plt.plot(mean[0], mean[1], 'o', markerfacecolor='aqua', markeredgecolor='k',
    #          markersize=10, markeredgewidth=1)
    # plt.plot(median[0], median[1], 'o', markerfacecolor='green', markeredgecolor='k',
    #          markersize=10, markeredgewidth=1)
    # plt.title(f'{title} | Aqua - Mean')

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + f"{title}.png")
        plt.close()
    else:
        plt.show()


def whole_dataset_analysis(generated_dataset, split, mem_mode, root_path, use_all_splits=False, for_phase2=False,
                           extra_path=''):
    obs_trajectories, pred_trajectories, obs_relative_distances_list, \
    pred_relative_distances_list, to_meter_list = \
        [], [], [], [], []
    full_length_trajectory_list, full_length_distances_list = [], []

    for clz_and_videos, meta in zip(TRAIN_CLASS_FOR_WHOLE, TRAIN_META):
        clz = clz_and_videos.value[0]
        videos = clz_and_videos.value[1]
        for vid in videos:
            logger.info(f'Generating for {clz.name} - Video {vid}')
            if for_phase2:
                plot_path = \
                    f"{ROOT_PATH}Plots/baseline_v2/nn/STATS_NN/{extra_path}{clz.value}{vid}/" \
                    f"{'generated/' if generated_dataset else 'gt/'}{'together' if use_all_splits else split.name}/"
            else:
                plot_path = \
                    f"{ROOT_PATH}Plots/baseline_v2/nn/STATS/{clz.value}{vid}/" \
                    f"{'generated/' if generated_dataset else 'gt/'}{'together' if use_all_splits else split.name}/"

            obs_trajectory, pred_trajectory, obs_relative_distances, pred_relative_distances, to_meter = \
                get_trajectory_splits(video_class=clz, video_number=vid, split=split, root=root_path,
                                      meta_label=meta, mmap_mode=mem_mode, generated=generated_dataset,
                                      use_all_splits=use_all_splits, for_phase2=for_phase2)

            full_length_trajectory = np.concatenate((obs_trajectory, pred_trajectory), axis=1)
            full_length_distances = np.concatenate((obs_relative_distances, pred_relative_distances), axis=1)

            plot_trajectory_length_histogram(obs_trajectory, to_meter, 'Observed Trajectory - Len:8',
                                             save_path=plot_path + 'histogram/')
            plot_trajectory_length_histogram(pred_trajectory, to_meter, 'Prediction Trajectory - Len:12',
                                             save_path=plot_path + 'histogram/')
            plot_trajectory_length_histogram(full_length_trajectory, to_meter, 'Full Length Trajectory - Len:20',
                                             save_path=plot_path + 'histogram/')

            plot_trajectory_length_bar(obs_trajectory, to_meter, 'Observed Trajectory - Len:8',
                                       save_path=plot_path + 'bar/', in_meters=True)
            plot_trajectory_length_bar(pred_trajectory, to_meter, 'Prediction Trajectory - Len:12',
                                       save_path=plot_path + 'bar/', in_meters=True)
            plot_trajectory_length_bar(full_length_trajectory, to_meter, 'Full Length Trajectory - Len:20',
                                       save_path=plot_path + 'bar/', in_meters=True)

            plot_relative_distances_line_plot(obs_relative_distances.reshape(-1, 2), 'Observed:8',
                                              save_path=plot_path + 'distances/')
            plot_relative_distances_line_plot(pred_relative_distances.reshape(-1, 2), 'Prediction:12',
                                              save_path=plot_path + 'distances/')
            plot_relative_distances_line_plot(full_length_distances.reshape(-1, 2), 'Full:20',
                                              save_path=plot_path + 'distances/')

            if obs_trajectory.size != 0:
                obs_trajectories.append(obs_trajectory * to_meter)
            if pred_trajectory.size != 0:
                pred_trajectories.append(pred_trajectory * to_meter)
            if obs_relative_distances.size != 0:
                obs_relative_distances_list.append(obs_relative_distances)
            if pred_relative_distances.size != 0:
                pred_relative_distances_list.append(pred_relative_distances)
            if full_length_trajectory.size != 0:
                full_length_trajectory_list.append(full_length_trajectory * to_meter)
            if full_length_distances.size != 0:
                full_length_distances_list.append(full_length_distances)
            # to_meter_list.append(to_meter)

    obs_trajectories = np.concatenate(obs_trajectories, axis=0)
    pred_trajectories = np.concatenate(pred_trajectories, axis=0)
    obs_relative_distances_list = np.concatenate(obs_relative_distances_list, axis=0)
    pred_relative_distances_list = np.concatenate(pred_relative_distances_list, axis=0)
    full_length_trajectory_list = np.concatenate(full_length_trajectory_list, axis=0)
    full_length_distances_list = np.concatenate(full_length_distances_list, axis=0)

    logger.info(f'Total tracks count: {full_length_trajectory_list.shape}')

    if for_phase2:
        plot_path = f"{ROOT_PATH}Plots/baseline_v2/nn/STATS_NN/{extra_path}/full_dataset/" \
                    f"{'generated/' if generated_dataset else 'gt/'}{'together' if use_all_splits else split.name}/"
    else:
        plot_path = f"{ROOT_PATH}Plots/baseline_v2/nn/STATS/full_dataset/" \
                    f"{'generated/' if generated_dataset else 'gt/'}{'together' if use_all_splits else split.name}/"

    plot_trajectory_length_histogram(obs_trajectories, 1, 'Observed Trajectory - Len:8',
                                     save_path=plot_path + 'histogram/')
    plot_trajectory_length_histogram(pred_trajectories, 1, 'Prediction Trajectory - Len:12',
                                     save_path=plot_path + 'histogram/')
    plot_trajectory_length_histogram(full_length_trajectory_list, 1, 'Full Length Trajectory - Len:20',
                                     save_path=plot_path + 'histogram/')

    plot_trajectory_length_bar(obs_trajectories, 1, 'Observed Trajectory - Len:8',
                               save_path=plot_path + 'bar/', in_meters=True)
    plot_trajectory_length_bar(pred_trajectories, 1, 'Prediction Trajectory - Len:12',
                               save_path=plot_path + 'bar/', in_meters=True)
    plot_trajectory_length_bar(full_length_trajectory_list, 1, 'Full Length Trajectory - Len:20',
                               save_path=plot_path + 'bar/', in_meters=True)

    plot_relative_distances_line_plot(obs_relative_distances_list.reshape(-1, 2), 'Observed:8',
                                      save_path=plot_path + 'distances/')
    plot_relative_distances_line_plot(pred_relative_distances_list.reshape(-1, 2), 'Prediction:12',
                                      save_path=plot_path + 'distances/')
    plot_relative_distances_line_plot(full_length_distances_list.reshape(-1, 2), 'Full:20',
                                      save_path=plot_path + 'distances/')


def whole_dataset_distribution_analysis(generated_dataset, split, mem_mode, root_path, outlier_threshold=150.):
    obs_trajectories, pred_trajectories, obs_relative_distances_list, \
    pred_relative_distances_list, to_meter_list = \
        [], [], [], [], []
    full_length_trajectory_list, full_length_distances_list = [], []

    full_length_distances_without_outliers = []

    for clz_and_videos, meta in zip(TRAIN_CLASS_FOR_WHOLE, TRAIN_META):
        clz = clz_and_videos.value[0]
        videos = clz_and_videos.value[1]
        for vid in videos:
            logger.info(f'Computing for {clz.name} - Video {vid}')
            obs_trajectory, pred_trajectory, obs_relative_distances, pred_relative_distances, to_meter = \
                get_trajectory_splits(video_class=clz, video_number=vid, split=split, root=root_path,
                                      meta_label=meta, mmap_mode=mem_mode, generated=generated_dataset)

            full_length_trajectory = np.concatenate((obs_trajectory, pred_trajectory), axis=1)
            full_length_distances = np.concatenate((obs_relative_distances, pred_relative_distances), axis=1)

            all_idx = np.arange(full_length_distances.shape[0])

            positive_outlier_idx = np.where(full_length_distances > outlier_threshold)[0]
            positive_outlier_idx = np.unique(positive_outlier_idx)
            negative_outlier_idx = np.where(full_length_distances < -outlier_threshold)[0]
            negative_outlier_idx = np.unique(negative_outlier_idx)

            outlier_idx = np.union1d(positive_outlier_idx, negative_outlier_idx)

            inlier_idx = np.setdiff1d(all_idx, outlier_idx)

            full_length_distances_inliers = full_length_distances[inlier_idx]

            if full_length_distances_inliers.size != 0:
                full_length_distances_without_outliers.append(full_length_distances_inliers)
            if obs_trajectory.size != 0:
                obs_trajectories.append(obs_trajectory * to_meter)
            if pred_trajectory.size != 0:
                pred_trajectories.append(pred_trajectory * to_meter)
            if obs_relative_distances.size != 0:
                obs_relative_distances_list.append(obs_relative_distances)
            if pred_relative_distances.size != 0:
                pred_relative_distances_list.append(pred_relative_distances)
            if full_length_trajectory.size != 0:
                full_length_trajectory_list.append(full_length_trajectory * to_meter)
            if full_length_distances.size != 0:
                full_length_distances_list.append(full_length_distances)
            # to_meter_list.append(to_meter)

    obs_trajectories = np.concatenate(obs_trajectories, axis=0)
    pred_trajectories = np.concatenate(pred_trajectories, axis=0)
    obs_relative_distances_list = np.concatenate(obs_relative_distances_list, axis=0)
    pred_relative_distances_list = np.concatenate(pred_relative_distances_list, axis=0)
    full_length_trajectory_list = np.concatenate(full_length_trajectory_list, axis=0)
    full_length_distances_list = np.concatenate(full_length_distances_list, axis=0)
    full_length_distances_without_outliers = np.concatenate(full_length_distances_without_outliers, axis=0)

    plot_path = f"{ROOT_PATH}Plots/baseline_v2/nn/STATS/full_dataset/" \
                f"{'generated/' if generated_dataset else 'gt/'}{split.name}/"

    plot_relative_distances_line_plot(full_length_distances_without_outliers.reshape(-1, 2),
                                      f'Filtered_Full:20_{outlier_threshold}',
                                      save_path=plot_path + 'distances/')

    per_step_length, length = get_trajectory_length(full_length_trajectory_list, use_l2=True)

    b_counts = np.bincount(np.round(length).astype(np.int))
    b_unique = np.arange(start=0, stop=len(b_counts))

    b_dict = {v: k for k, v in zip(b_counts, b_unique)}

    unique, counts = np.unique(length, return_counts=True)

    # df = pd.DataFrame.from_dict(data={k: v for k, v in zip(counts, unique)})  # , columns=['Length', 'Count'])
    u_dict = {v: k for k, v in zip(counts, unique)}

    print()

    # find_distribution(full_length_trajectory_list, f'{split.name}_{"generated" if generated_dataset else "gt"}')
    # find_distribution(full_length_distances_list, f'{split.name}_{"generated" if generated_dataset else "gt"}')


if __name__ == '__main__':
    analyze_whole_dataset = False
    analyze_whole_dataset_for_phase2 = True
    find_distribution_whole_dataset = False

    mem_mode = None
    split = NetworkMode.TRAIN
    generated_dataset = False

    root_path = GENERATED_DATASET_ROOT if generated_dataset else SAVE_BASE_PATH

    if analyze_whole_dataset:
        logger.info('Whole Dataset Analysis')
        whole_dataset_analysis(generated_dataset=generated_dataset, split=split, mem_mode=mem_mode, root_path=root_path,
                               use_all_splits=False)
    elif analyze_whole_dataset_for_phase2:
        logger.info('Whole Dataset Analysis - Phase2')
        whole_dataset_analysis(
            generated_dataset=generated_dataset, split=split, mem_mode=mem_mode,
            root_path='/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/extended_splits/classic_nn_extended_annotations_v1/d2/gan/',
            use_all_splits=True, for_phase2=True, extra_path='d2/gan/')
    elif find_distribution_whole_dataset:
        logger.info('Distribution Analysis')
        whole_dataset_distribution_analysis(generated_dataset=generated_dataset, split=split,
                                            mem_mode=mem_mode, root_path=root_path, outlier_threshold=1000)
    else:
        logger.info('Single sequence Analysis')
        vid_clz = SDDVideoClasses.LITTLE
        vid_clz_meta = SDDVideoDatasets.LITTLE
        vid_number = 3
        split = NetworkMode.VALIDATION
        mem_mode = None

        plot_path = f'{ROOT_PATH}Plots/baseline_v2/nn/STATS/{vid_clz.value}{vid_number}/{split.name}/'

        generated_dataset = False

        plot_path += 'generated/' if generated_dataset else 'gt/'

        root_path = GENERATED_DATASET_ROOT if generated_dataset else SAVE_BASE_PATH

        obs_trajectory, pred_trajectory, obs_relative_distances, pred_relative_distances, to_meter = \
            get_trajectory_splits(video_class=vid_clz, video_number=vid_number, split=split, root=root_path,
                                  meta_label=vid_clz_meta, mmap_mode=mem_mode, generated=generated_dataset)

        full_length_trajectory = np.concatenate((obs_trajectory, pred_trajectory), axis=1)
        full_length_distances = np.concatenate((obs_relative_distances, pred_relative_distances), axis=1)

        plot_trajectory_length_histogram(obs_trajectory, to_meter, 'Observed Trajectory - Len:8',
                                         save_path=plot_path + 'histogram/')
        plot_trajectory_length_histogram(pred_trajectory, to_meter, 'Prediction Trajectory - Len:12',
                                         save_path=plot_path + 'histogram/')
        plot_trajectory_length_histogram(full_length_trajectory, to_meter, 'Full Length Trajectory - Len:20',
                                         save_path=plot_path + 'histogram/')

        plot_trajectory_length_bar(obs_trajectory, to_meter, 'Observed Trajectory - Len:8',
                                   save_path=plot_path + 'bar/', in_meters=True)
        plot_trajectory_length_bar(pred_trajectory, to_meter, 'Prediction Trajectory - Len:12',
                                   save_path=plot_path + 'bar/', in_meters=True)
        plot_trajectory_length_bar(full_length_trajectory, to_meter, 'Full Length Trajectory - Len:20',
                                   save_path=plot_path + 'bar/', in_meters=True)

        plot_relative_distances_line_plot(obs_relative_distances.reshape(-1, 2), 'Observed:8',
                                          save_path=plot_path + 'distances/')
        plot_relative_distances_line_plot(pred_relative_distances.reshape(-1, 2), 'Prediction:12',
                                          save_path=plot_path + 'distances/')
        plot_relative_distances_line_plot(full_length_distances.reshape(-1, 2), 'Full:20',
                                          save_path=plot_path + 'distances/')
