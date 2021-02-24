from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import GENERATED_DATASET_ROOT, SAVE_BASE_PATH, ROOT_PATH
from baselinev2.constants import NetworkMode
from baselinev2.notebooks.utils import get_trajectory_splits, get_trajectory_length
from log import initialize_logging, get_logger

matplotlib.style.use('ggplot')

initialize_logging()
logger = get_logger('baselinev2.notebooks.dataset_analysis')


def plot_trajectory_length_histogram(trajectory, ratio, title, save_path=None, bins=None):
    per_step_length, length = get_trajectory_length(trajectory)
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
    per_step_length, length = get_trajectory_length(trajectory)
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
    plt.plot(mean[0], mean[1], 'o', markerfacecolor='aqua', markeredgecolor='k',
             markersize=10, markeredgewidth=1)
    plt.plot(median[0], median[1], 'o', markerfacecolor='green', markeredgecolor='k',
             markersize=10, markeredgewidth=1)
    plt.title(f'{title} | Aqua - Mean')

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + f"{title}.png")
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    vid_clz = SDDVideoClasses.LITTLE
    vid_clz_meta = SDDVideoDatasets.LITTLE
    vid_number = 3
    split = NetworkMode.TRAIN
    mem_mode = None

    plot_path = f'{ROOT_PATH}Plots/baseline_v2/nn/STATS/{vid_clz.value}{vid_number}/'

    generated_dataset = True

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
