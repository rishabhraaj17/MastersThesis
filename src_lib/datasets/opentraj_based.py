# borrowed from OpenTraj

import glob
import os
import tempfile
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from omegaconf import OmegaConf
from pykalman import KalmanFilter
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from baselinev2.nn.dataset import ConcatenateDataset
from src_lib.datasets.dataset_utils import adjust_dataframe_framerate
from src_lib.datasets.trajectory_stgcnn import TrajectoryDatasetFromFile, SmoothTrajectoryDataset

pd.options.mode.chained_assignment = None  # default='warn'


class KalmanModel:
    def __init__(self, dt, n_dim=2, n_iter=4):
        self.n_iter = n_iter
        self.n_dim = n_dim

        # Const-acceleration Model
        self.A = np.array([[1, dt, dt ** 2],
                           [0, 1, dt],
                           [0, 0, 1]])

        self.C = np.array([[1, 0, 0]])

        self.Q = np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                           [dt ** 4 / 8., dt ** 3 / 3, dt ** 2 / 2],
                           [dt ** 3 / 6., dt ** 2 / 2, dt / 1]]) * 0.5

        # =========== Const-velocity Model ================
        # self.A = [[1, t],
        #           [0, 1]]
        #
        # self.C = [[1, 0]}
        #
        # q = 0.0005
        # self.Q = [[q, 0],
        #           [0, q/10]]
        # =================================================

        r = 1
        self.R = np.array([[r]])

        self.kf = [KalmanFilter(transition_matrices=self.A, observation_matrices=self.C,
                                transition_covariance=self.Q, observation_covariance=self.R) for _ in range(n_dim)]

    def filter(self, measurement):
        filtered_means = []
        for dim in range(self.n_dim):
            f = self.kf[dim].em(measurement[:, dim], n_iter=self.n_iter)
            (filtered_state_means, filtered_state_covariances) = f.filter(measurement[:, dim])
            filtered_means.append(filtered_state_means)
        filtered_means = np.stack(filtered_means)
        return filtered_means[:, :, 0].T, filtered_means[:, :, 1].T

    def smooth(self, measurement):
        smoothed_means = []
        if measurement.shape[0] == 1:
            return measurement, np.zeros((1, 2))
        for dim in range(self.n_dim):
            f = self.kf[dim].em(measurement[:, dim], n_iter=self.n_iter)
            (smoothed_state_means, smoothed_state_covariances) = f.smooth(measurement[:, dim])
            smoothed_means.append(smoothed_state_means)
        smoothed_means = np.stack(smoothed_means)
        return smoothed_means[:, :, 0].T, smoothed_means[:, :, 1].T


class TrajDataset:
    def __init__(self):
        """
        data might include the following columns:
        "scene_id", "frame_id", "agent_id",
         "pos_x", "pos_y"
          "vel_x", "vel_y",
        """
        self.critical_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
        self.data = pd.DataFrame(columns=self.critical_columns)

        # a map from agent_id to a list of [agent_ids] that are annotated as her groupmate
        # if such informatoin is not available the map should be filled with an empty list
        # for each agent_id
        self.groupmates = {}

        # fps is necessary to calc any data related to time (e.g. velocity, acceleration)

        self.title = ''
        self.fps = -1

        # bounding box of trajectories
        #  FixME: bbox should be a function of scene_id
        # self.bbox = pd.DataFrame({'x': [np.nan, np.nan],
        #                           'y': [np.nan, np.nan]},
        #                          index=['min', 'max'])

        # FixMe ?
        #  self.trajectories_lazy = []

    def postprocess(self, fps, sampling_rate=1, use_kalman=False):
        """
        This function should be called after loading the data by loader
        It performs the following steps:
        -: check fps value, should be set and bigger than 0
        -: check critical columns should exist in the table
        -: update data types
        -: fill 'groumates' if they are not set
        -: checks if velocity do not exist, compute it for each agent
        -: compute bounding box of trajectories

        :param fps: video framerate
        :param sampling_rate: if bigger than one, the data needs downsampling,
                              otherwise needs interpolation
        :param use_kalman:  for smoothing agent velocities
        :return: None
        """

        # check
        for critical_column in self.critical_columns:
            if critical_column not in self.data:
                raise ValueError("Error! some critical columns are missing from trajectory dataset!")

        # modify data types
        self.data["frame_id"] = self.data["frame_id"].astype(int)
        if str(self.data["agent_id"].iloc[0]).replace('.', '', 1).isdigit():
            self.data["agent_id"] = self.data["agent_id"].astype(int)
        self.data["pos_x"] = self.data["pos_x"].astype(float)
        self.data["pos_y"] = self.data["pos_y"].astype(float)
        self.data["label"] = self.data["label"].str.lower()  # search with lower-case labels

        # fill scene_id
        if "scene_id" not in self.data:
            self.data["scene_id"] = 0
        self.fps = fps

        # fill timestamps based on frame_id and video_fps
        if "timestamp" not in self.data:
            self.data["timestamp"] = self.data["frame_id"] / fps

        # fill groupmates
        agent_ids = pd.unique(self.data["agent_id"])
        for agent_id in agent_ids:
            if agent_id not in self.groupmates:
                self.groupmates[agent_id] = []

        # down/up sampling frames
        if sampling_rate >= 2:
            # FixMe: down-sampling
            sampling_rate = int(sampling_rate)
            self.data = self.data.loc[(self.data["frame_id"] % sampling_rate) == 0]
            self.data = self.data.reset_index()
        elif sampling_rate < (1 - 1E-2):
            # TODO: interpolation
            pass
        else:
            pass

        # remove the trajectories shorter than 2 frames
        data_grouped = self.data.groupby(["scene_id", "agent_id"])
        single_length_inds = data_grouped.head(1).index[data_grouped.size() < 2]
        self.data = self.data.drop(single_length_inds)

        # fill velocities
        if "vel_x" not in self.data:
            data_grouped = self.data.groupby(["scene_id", "agent_id"])
            dt = data_grouped["timestamp"].diff()

            if (dt > 2).sum():
                print('Warning! too big dt in [%s]' % self.title)

            self.data["vel_x"] = (data_grouped["pos_x"].diff() / dt).astype(float)
            self.data["vel_y"] = (data_grouped["pos_y"].diff() / dt).astype(float)
            nan_inds = np.array(np.nonzero(dt.isnull().to_numpy())).reshape(-1)
            self.data["vel_x"].iloc[nan_inds] = self.data["vel_x"].iloc[nan_inds + 1].to_numpy()
            self.data["vel_y"].iloc[nan_inds] = self.data["vel_y"].iloc[nan_inds + 1].to_numpy()

        # ============================================
        if use_kalman:
            def smooth(group):
                if len(group) < 2: return group
                dt = group["timestamp"].diff().iloc[1]
                kf = KalmanModel(dt, n_dim=2, n_iter=7)
                smoothed_pos, smoothed_vel = kf.smooth(group[["pos_x", "pos_y"]].to_numpy())
                group["pos_x"] = smoothed_pos[:, 0]
                group["pos_y"] = smoothed_pos[:, 1]

                group["vel_x"] = smoothed_vel[:, 0]
                group["vel_y"] = smoothed_vel[:, 1]
                return group

            tqdm.pandas(desc="Smoothing trajectories (%s)" % self.title)
            # print('Smoothing trajectories ...')
            data_grouped = self.data.groupby(["scene_id", "agent_id"])
            self.data = data_grouped.progress_apply(smooth)

        # compute bounding box
        # Warning: the trajectories should belong to the same (physical) scene
        # self.bbox['x']['min'] = min(self.data["pos_x"])
        # self.bbox['x']['max'] = max(self.data["pos_x"])
        # self.bbox['y']['min'] = min(self.data["pos_y"])
        # self.bbox['y']['max'] = max(self.data["pos_y"])

    def interpolate_frames(self, inplace=True):
        """
        Knowing the framerate , the FRAMES that are not annotated will be interpolated.
        :param inplace: Todo
        :return: None
        """
        all_frame_ids = sorted(pd.unique(self.data["frame_id"]))
        if len(all_frame_ids) < 2:
            # FixMe: print warning
            return

        frame_id_A = all_frame_ids[0]
        frame_A = self.data.loc[self.data["frame_id"] == frame_id_A]
        agent_ids_A = frame_A["agent_id"].to_list()
        interp_data = self.data  # "agent_id", "pos_x", "pos_y", "vel_x", "vel_y"
        # df.append([df_try] * 5, ignore_index=True
        for frame_id_B in tqdm(all_frame_ids[1:], desc="Interpolating frames"):
            frame_B = self.data.loc[self.data["frame_id"] == frame_id_B]
            agent_ids_B = frame_B["agent_id"].to_list()

            common_agent_ids = list(set(agent_ids_A) & set(agent_ids_B))
            frame_A_fil = frame_A.loc[frame_A["agent_id"].isin(common_agent_ids)]
            frame_B_fil = frame_B.loc[frame_B["agent_id"].isin(common_agent_ids)]
            for new_frame_id in range(frame_id_A + 1, frame_id_B):
                alpha = (new_frame_id - frame_id_A) / (frame_id_B - frame_id_A)
                new_frame = frame_A_fil.copy()
                new_frame["frame_id"] = new_frame_id
                new_frame["pos_x"] = frame_A_fil["pos_x"].to_numpy() * (1 - alpha) + \
                                     frame_B_fil["pos_x"].to_numpy() * alpha
                new_frame["pos_y"] = frame_A_fil["pos_y"].to_numpy() * (1 - alpha) + \
                                     frame_B_fil["pos_y"].to_numpy() * alpha
                new_frame["vel_x"] = frame_A_fil["vel_x"].to_numpy() * (1 - alpha) + \
                                     frame_B_fil["vel_x"].to_numpy() * alpha
                new_frame["vel_y"] = frame_A_fil["vel_y"].to_numpy() * (1 - alpha) + \
                                     frame_B_fil["vel_y"].to_numpy() * alpha
                if inplace:
                    self.data = self.data.append(new_frame)
                else:
                    self.data = self.data.append(new_frame)  # TODO
            frame_id_A = frame_id_B
            frame_A = frame_B
            agent_ids_A = agent_ids_B
        self.data = self.data.sort_values('frame_id')

    # FixMe: rename to add_row()/add_entry()
    def add_agent(self, agent_id, frame_id, pos_x, pos_y):
        """Add one single data at a specific frame to dataset"""
        new_df = pd.DataFrame(columns=self.critical_columns)
        new_df["frame_id"] = [int(frame_id)]
        new_df["agent_id"] = [int(agent_id)]
        new_df["pos_x"] = [float(pos_x)]
        new_df["pos_y"] = [float(pos_y)]
        self.data = self.data.append(new_df)

    def get_agent_ids(self):
        """:return all agent_id in data table"""
        return pd.unique(self.data["agent_id"])

    def get_trajectories(self, label=""):
        """
        Returns a list of trajectories
        :param label: select agents from a specific class (e.g. pedestrian), ignore if empty
        :return list of trajectories
        """

        trajectories = []
        df = self.data
        if label:
            label_filtered = self.data.groupby("label")
            df = label_filtered.get_group(label.lower())

        return df.groupby(["scene_id", "agent_id"])

    # TODO:
    def get_entries(self, agent_ids=[], frame_ids=[], label=""):
        """
        Returns a list of data entries
        :param agent_ids: select specific agent ids, ignore if empty
        :param frame_ids: select a time interval, ignore if empty  # TODO:
        :param label: select agents from a specific label (e.g. car), ignore if empty # TODO:
        :return list of data entries
        """
        output_table = self.data  # no filter
        if agent_ids:
            output_table = output_table[output_table["agent_id"].isin(agent_ids)]
        if frame_ids:
            output_table = output_table[output_table["frame_id"].isin(frame_ids)]
        return output_table

    def get_frames(self, frame_ids: list = [], scene_ids=[]):
        if not len(frame_ids):
            frame_ids = pd.unique(self.data["frame_id"])
        if not len(scene_ids):
            scene_ids = pd.unique(self.data["scene_id"])

        frames = []
        for scene_id in scene_ids:
            for frame_id in frame_ids:
                frame_df = self.data.loc[(self.data["frame_id"] == frame_id) &
                                         (self.data["scene_id"] == scene_id)]
                # traj_df = self.data.filter()
                frames.append(frame_df)
        return frames

    def apply_transformation(self, tf: np.ndarray, inplace=False):
        """
        :param tf: np.ndarray
            Homogeneous Transformation Matrix,
            3x3 for 2D data
        :param inplace: bool, default False
            If True, do operation inplace
        :return: transformed data table
        """
        if inplace:
            target_data = self.data
        else:
            target_data = self.data.copy()

        # data is 2D
        assert tf.shape == (3, 3)
        tf = tf[:2, :]  # remove the last row
        poss = target_data[["pos_x", "pos_y"]].to_numpy(dtype=np.float)
        poss = np.concatenate([poss, np.ones((len(poss), 1))], axis=1)
        target_data[["pos_x", "pos_y"]] = np.matmul(tf, poss.T).T

        # apply on velocities
        tf[:, -1] = 0  # do not apply the translation element on velocities!
        vels = target_data[["vel_x", "vel_y"]].to_numpy(dtype=np.float)
        vels = np.concatenate([vels, np.ones((len(vels), 1))], axis=1)
        target_data[["vel_x", "vel_y"]] = np.matmul(tf, vels.T).T

        return target_data


def merge_datasets(dataset_list, new_title=[]):
    if len(dataset_list) < 1:
        return TrajDataset()
    elif len(dataset_list) == 1:
        return dataset_list[0]

    merged = dataset_list[0]

    for ii in range(1, len(dataset_list)):
        merged.data = merged.data.append(dataset_list[ii].data)
        if dataset_list[ii].title != merged.title:
            merged.title = merged.title + " + " + dataset_list[ii].title
    if len(new_title):
        merged.title = new_title

    return merged


def load_sdd(path, **kwargs):
    sdd_dataset = TrajDataset()
    sdd_dataset.title = kwargs.get("title", "SDD")

    csv_columns = ["agent_id", "x_min", "y_min", "x_max", "y_max", "frame_id",
                   "lost", "occluded", "generated", "label"]
    scale = kwargs.get("scale", 1)

    # read from csv => fill traj table
    raw_dataset = pd.read_csv(path, sep=" ", header=None, names=csv_columns)
    raw_dataset["pos_x"] = scale * (raw_dataset["x_min"] + raw_dataset["x_max"]) / 2
    raw_dataset["pos_y"] = scale * (raw_dataset["y_min"] + raw_dataset["y_max"]) / 2

    drop_lost_frames = kwargs.get('drop_lost_frames', False)
    if drop_lost_frames:
        raw_dataset = raw_dataset.loc[raw_dataset["lost"] != 1]

    # copy columns
    sdd_dataset.data[["frame_id", "agent_id",
                      "pos_x", "pos_y",
                      # "x_min", "y_min", "x_max", "y_max",
                      "label", "lost", "occluded", "generated"]] = \
        raw_dataset[["frame_id", "agent_id",
                     "pos_x", "pos_y",
                     # "x_min", "y_min", "x_max", "y_max",
                     "label", "lost", "occluded", "generated"]]
    sdd_dataset.data["scene_id"] = kwargs.get("scene_id", 0)

    # calculate velocities + perform some checks
    fps = 30
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    sdd_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return sdd_dataset


def load_sdd_dir(path: str, **kwargs):
    search_filter_str = "**/annotations.txt"
    if not path.endswith("/"):
        search_filter_str = "/" + search_filter_str
    files_list = sorted(glob.glob(path + search_filter_str, recursive=True))
    scales_yaml_file = os.path.join(path, 'estimated_scales.yaml')
    with open(scales_yaml_file, 'r') as f:
        scales_yaml_content = yaml.load(f, Loader=yaml.FullLoader)

    partial_datasets = []
    for file in files_list:
        dir_names = file.split('/')
        scene_name = dir_names[-3]
        scene_video_id = dir_names[-2]
        scale = scales_yaml_content[scene_name][scene_video_id]['scale']

        partial_dataset = load_sdd(file, scale=scale,
                                   scene_id=scene_name + scene_video_id.replace('video', ''))
        partial_datasets.append(partial_dataset.data)

    traj_dataset = TrajDataset()
    traj_dataset.data = pd.concat(partial_datasets)

    fps = 30
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return traj_dataset


TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'prediction_number', 'scene_id'])
TrackRow.__new__.__defaults__ = (None, None, None, None, None, None)


# SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])


class CrowdLoader:
    def __init__(self, homog=[]):
        if len(homog):
            self.homog = homog
        else:
            self.homog = np.eye(3)

    def to_world_coord(self, homog, loc):
        """Given H^-1 and world coordinates, returns (u, v) in image coordinates."""

        locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
        loc_tr = np.transpose(locHomogenous)
        loc_tr = np.matmul(homog, loc_tr)  # to camera frame
        locXYZ = np.transpose(loc_tr / loc_tr[2])  # to pixels (from millimeters)
        return locXYZ[:, :2]

    def crowds_interpolate_person(self, ped_id, person_xyf):
        ## Earlier
        # xs = np.array([x for x, _, _ in person_xyf]) / 720 * 12 # 0.0167
        # ys = np.array([y for _, y, _ in person_xyf]) / 576 * 12 # 0.0208

        ## Pixel-to-meter scale conversion according to
        ## https://github.com/agrimgupta92/sgan/issues/5
        # xs_ = np.array([x for x, _, _ in person_xyf]) * 0.0210
        # ys_ = np.array([y for _, y, _ in person_xyf]) * 0.0239

        xys = self.to_world_coord(self.homog, np.array([[x, y] for x, y, _ in person_xyf]))
        xs, ys = xys[:, 0], xys[:, 1]

        fs = np.array([f for _, _, f in person_xyf])

        kind = 'linear'
        # if len(fs) > 5:
        #    kind = 'cubic'

        x_fn = scipy.interpolate.interp1d(fs, xs, kind=kind)
        y_fn = scipy.interpolate.interp1d(fs, ys, kind=kind)

        frames = np.arange(min(fs) // 10 * 10 + 10, max(fs), 10)

        return [TrackRow(int(f), ped_id, x, y)
                for x, y, f in np.stack([x_fn(frames), y_fn(frames), frames]).T]

    def load(self, filename):
        with open(filename) as annot_file:
            whole_file = annot_file.read()

            pedestrians = []
            current_pedestrian = []

            for line in whole_file.split('\n'):
                if '- Num of control points' in line or \
                        '- the number of splines' in line:
                    if current_pedestrian:
                        pedestrians.append(current_pedestrian)
                    current_pedestrian = []
                    continue

                # strip comments
                if ' - ' in line:
                    line = line[:line.find(' - ')]

                # tokenize
                entries = [e for e in line.split(' ') if e]
                if len(entries) != 4:
                    continue

                x, y, f, _ = entries

                current_pedestrian.append([float(x), float(y), int(f)])

            if current_pedestrian:
                pedestrians.append(current_pedestrian)
        return [row for i, p in enumerate(pedestrians) for row in self.crowds_interpolate_person(i, p)]


def load_crowds(path, **kwargs):
    """
        Note: pass the homography matrix as well
        :param path: string, path to folder
    """

    homog_file = kwargs.get("homog_file", "")
    load_homog = kwargs.get("load_homog", False)
    Homog = (np.loadtxt(homog_file)) if os.path.exists(homog_file) and load_homog else np.eye(3)
    raw_dataset = pd.DataFrame()

    data = CrowdLoader(Homog).load(path)
    raw_dataset["frame_id"] = [data[i].frame for i in range(len(data))]
    raw_dataset["agent_id"] = [data[i].pedestrian for i in range(len(data))]
    raw_dataset["pos_x"] = [data[i].x for i in range(len(data))]
    raw_dataset["pos_y"] = [data[i].y for i in range(len(data))]

    traj_dataset = TrajDataset()

    traj_dataset.title = kwargs.get('title', "Crowds")
    # copy columns
    traj_dataset.data[["frame_id", "agent_id", "pos_x", "pos_y"]] = \
        raw_dataset[["frame_id", "agent_id", "pos_x", "pos_y"]]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', 25)

    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset


def load_eth(path, **kwargs):
    traj_dataset = TrajDataset()

    csv_columns = ["frame_id", "agent_id", "pos_x", "pos_z", "pos_y", "vel_x", "vel_z", "vel_y"]
    # read from csv => fill traj table
    raw_dataset = pd.read_csv(path, sep=r"\s+", header=None, names=csv_columns)

    traj_dataset.title = kwargs.get('title', "no_title")

    # copy columns
    traj_dataset.data[["frame_id", "agent_id",
                       "pos_x", "pos_y",
                       "vel_x", "vel_y"
                       ]] = \
        raw_dataset[["frame_id", "agent_id",
                     "pos_x", "pos_y",
                     "vel_x", "vel_y"
                     ]]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', -1)
    if fps < 0:
        d_frame = np.diff(pd.unique(raw_dataset["frame_id"]))
        fps = d_frame[0] * 2.5  # 2.5 is the common annotation fps for all (ETH+UCY) datasets

    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset


def eth_world_to_image(open_traj_root, traj):
    def world2image(traj_w, H_inv):
        # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
        traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
        # to camera frame
        traj_cam = np.matmul(H_inv, traj_homog)
        # to pixel coords
        traj_uvz = np.transpose(traj_cam / traj_cam[2])
        return traj_uvz[:, :2].astype(int)

    H = (np.loadtxt(os.path.join(open_traj_root, "datasets/ETH/seq_eth/H.txt")))
    H_inv = np.linalg.inv(H)
    return world2image(traj, H_inv)  # TRAJ: Tx2 numpy array


def get_eth_dataset(cfg, split_dataset=False, smooth_trajectories=False, smoother=lambda x: x, threshold=1,
                    world_to_image=False, seq='eth'):
    open_traj_dataset = load_eth(os.path.join(cfg.open_traj_root, f"datasets/ETH/seq_{seq}/obsmat.txt"))
    data_df = open_traj_dataset.data[open_traj_dataset.critical_columns]
    data_df = data_df.sort_values(by=['frame_id']).reset_index()
    data_df: pd.DataFrame = data_df.drop(columns=['index'])

    if world_to_image:
        in_image = eth_world_to_image(cfg.open_traj_root, np.stack((data_df['pos_x'], data_df['pos_y']), axis=-1))
        data_df['pos_x'] = in_image[:, 0]
        data_df['pos_y'] = in_image[:, 1]


    temp_file = tempfile.NamedTemporaryFile(suffix='.txt')
    data_df.to_csv(temp_file, header=False, index=False, sep=' ')
    dataset = TrajectoryDatasetFromFile(
        temp_file, obs_len=cfg.obs_len, pred_len=cfg.pred_len, skip=cfg.skip, min_ped=cfg.min_ped,
        delim=cfg.delim, video_class='eth', video_number=-1, construct_graph=cfg.construct_graph,
        compute_homography=False)

    if smooth_trajectories:
        dataset = SmoothTrajectoryDataset(base_dataset=dataset, smoother=smoother, threshold=threshold)

    if not split_dataset:
        return dataset

    val_dataset_len = round(len(dataset) * cfg.val_ratio)
    train_indices = torch.arange(start=0, end=len(dataset) - val_dataset_len)
    val_indices = torch.arange(start=len(dataset) - val_dataset_len, end=len(dataset))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_ucy_dataset(cfg, split_dataset=False, smooth_trajectories=False, smoother=lambda x: x, threshold=1,
                    world_to_image=False, seq='zara01'):
    annot = os.path.join(cfg.open_traj_root, f'datasets/UCY/{seq}/annotation.vsp')
    H_file = os.path.join(cfg.open_traj_root, f'datasets/UCY/{seq}/H.txt')
    open_traj_dataset = load_crowds(annot, use_kalman=False, homog_file=H_file, load_homog=world_to_image)

    data_df = open_traj_dataset.data[open_traj_dataset.critical_columns]
    data_df = data_df.sort_values(by=['frame_id']).reset_index()
    data_df: pd.DataFrame = data_df.drop(columns=['index'])

    temp_file = tempfile.NamedTemporaryFile(suffix='.txt')
    data_df.to_csv(temp_file, header=False, index=False, sep=' ')
    dataset = TrajectoryDatasetFromFile(
        temp_file, obs_len=cfg.obs_len, pred_len=cfg.pred_len, skip=cfg.skip, min_ped=cfg.min_ped,
        delim=cfg.delim, video_class='eth', video_number=-1, construct_graph=cfg.construct_graph,
        compute_homography=False)

    if smooth_trajectories:
        dataset = SmoothTrajectoryDataset(base_dataset=dataset, smoother=smoother, threshold=threshold)

    if not split_dataset:
        return dataset

    val_dataset_len = round(len(dataset) * cfg.val_ratio)
    train_indices = torch.arange(start=0, end=len(dataset) - val_dataset_len)
    val_indices = torch.arange(start=len(dataset) - val_dataset_len, end=len(dataset))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_single_gt_dataset(cfg, video_class, video_number, split_dataset,
                          smooth_trajectories=False, smoother=lambda x: x, threshold=1,
                          frame_rate=30., time_step=0.4):
    load_path = f"{cfg.root_gt}{getattr(SDDVideoClasses, video_class).value}/video{video_number}/annotations.txt"
    open_traj_dataset = load_sdd(load_path)
    data_df = open_traj_dataset.data[open_traj_dataset.critical_columns]
    data_df = data_df.sort_values(by=['frame_id']).reset_index()
    data_df: pd.DataFrame = data_df.drop(columns=['index'])

    data_df = adjust_dataframe_framerate(data_df, for_gt=True, frame_rate=frame_rate, time_step=time_step)

    temp_file = tempfile.NamedTemporaryFile(suffix='.txt')
    data_df.to_csv(temp_file, header=False, index=False, sep=' ')
    dataset = TrajectoryDatasetFromFile(
        temp_file, obs_len=cfg.obs_len, pred_len=cfg.pred_len, skip=cfg.skip, min_ped=cfg.min_ped,
        delim=cfg.delim, video_class=video_class, video_number=video_number, construct_graph=cfg.construct_graph)

    if smooth_trajectories:
        dataset = SmoothTrajectoryDataset(base_dataset=dataset, smoother=smoother, threshold=threshold)

    if not split_dataset:
        return dataset

    val_dataset_len = round(len(dataset) * cfg.val_ratio)
    train_indices = torch.arange(start=0, end=len(dataset) - val_dataset_len)
    val_indices = torch.arange(start=len(dataset) - val_dataset_len, end=len(dataset))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_multiple_gt_dataset(cfg, split_dataset=True, with_dataset_idx=True,
                            smooth_trajectories=False, smoother=lambda x: x, threshold=1,
                            frame_rate=30., time_step=0.4):
    conf = cfg.tp_module.datasets
    video_classes = conf.video_classes
    video_numbers = conf.video_numbers

    train_datasets, val_datasets = [], []
    for v_idx, video_class in enumerate(tqdm(video_classes)):
        for v_num in video_numbers[v_idx]:
            if split_dataset:
                t_dset, v_dset = get_single_gt_dataset(conf, video_class, v_num, split_dataset,
                                                       smooth_trajectories=smooth_trajectories,
                                                       smoother=smoother, threshold=threshold,
                                                       frame_rate=frame_rate, time_step=time_step)
                train_datasets.append(t_dset)
                val_datasets.append(v_dset)
            else:
                dset = get_single_gt_dataset(cfg, video_class, v_num, split_dataset,
                                             smooth_trajectories=smooth_trajectories,
                                             smoother=smoother, threshold=threshold,
                                             frame_rate=frame_rate, time_step=time_step)
                train_datasets.append(dset)

    if split_dataset:
        return (ConcatenateDataset(train_datasets), ConcatenateDataset(val_datasets)) \
            if with_dataset_idx else (ConcatDataset(train_datasets), ConcatDataset(val_datasets))
    return ConcatenateDataset(train_datasets) if with_dataset_idx else ConcatDataset(train_datasets)


if __name__ == '__main__':
    # get_eth_dataset(OmegaConf.load('../../src/position_maps/config/training/training.yaml').tp_module.datasets,
    #                 world_to_image=True)
    get_ucy_dataset(OmegaConf.load('../../src/position_maps/config/training/training.yaml').tp_module.datasets)
    out = get_multiple_gt_dataset(OmegaConf.load('../../src/position_maps/config/training/training.yaml'))
    print()
