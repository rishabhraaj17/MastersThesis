import math
import os

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# https://github.com/abduallahmohamed/Social-STGCNN/blob/dbbb111a0f645e4002dd4885564da226c5e0b19f/utils.py#L86

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     obs_frames_list, pred_frames_list, non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # Network default input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_frames = torch.cat(obs_frames_list, dim=0).permute(2, 0, 1)
    pred_frames = torch.cat(pred_frames_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_frames, pred_frames, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)


def seq_collate_dict(data):
    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_frames, \
    pred_frames, non_linear_ped, loss_mask, seq_start_end = seq_collate(data)

    return {
        'in_xy': obs_traj, 'in_dxdy': obs_traj_rel[1:, ...],
        'gt_xy': pred_traj, 'gt_dxdy': pred_traj_rel,
        'in_frames': obs_frames, 'gt_frames': pred_frames,
        'non_linear_ped': non_linear_ped, 'loss_mask': loss_mask,
        'seq_start_end': seq_start_end
    }


def seq_collate_with_graphs(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     obs_frames_list, pred_frames_list, non_linear_ped_list, loss_mask_list,
     v_obs_list, A_obs_list, v_pred_list, A_pred_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # Network default input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_frames = torch.cat(obs_frames_list, dim=0).permute(2, 0, 1)
    pred_frames = torch.cat(pred_frames_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)

    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_frames, pred_frames, non_linear_ped,
        loss_mask, v_obs_list, A_obs_list, v_pred_list, A_pred_list, seq_start_end
    ]

    return tuple(out)


def seq_collate_with_graphs_dict(data):
    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_frames, \
    pred_frames, non_linear_ped, loss_mask, v_obs_list, A_obs_list, \
    v_pred_list, A_pred_list, seq_start_end = seq_collate_with_graphs(data)

    return {
        'in_xy': obs_traj, 'in_dxdy': obs_traj_rel[1:, ...],
        'gt_xy': pred_traj, 'gt_dxdy': pred_traj_rel,
        'in_frames': obs_frames, 'gt_frames': pred_frames,
        'non_linear_ped': non_linear_ped, 'loss_mask': loss_mask,
        'v_obs': v_obs_list, 'A_obs': A_obs_list,
        'v_pred': v_pred_list, 'A_pred': A_pred_list,
        'seq_start_end': seq_start_end
    }


def seq_collate_with_dataset_idx(data):
    obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list = [], [], [], []
    obs_frames_list, pred_frames_list, non_linear_ped_list, loss_mask_list = [], [], [], []
    obs_tracks_list, pred_tracks_list = [], []
    dataset_idx = []
    for d in data:
        (obs_seq, pred_seq, obs_seq_rel, pred_seq_rel,
         obs_frames, pred_frames, obs_tracks, pred_tracks, non_linear_ped, loss_mask), d_idx = d[0], d[1]
        obs_seq_list.append(obs_seq)
        pred_seq_list.append(pred_seq)
        obs_seq_rel_list.append(obs_seq_rel)
        pred_seq_rel_list.append(pred_seq_rel)
        obs_frames_list.append(obs_frames)
        pred_frames_list.append(pred_frames)
        obs_tracks_list.append(obs_tracks)
        pred_tracks_list.append(pred_tracks)
        non_linear_ped_list.append(non_linear_ped)
        loss_mask_list.append(loss_mask)
        dataset_idx.append(d_idx)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # Network default input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_frames = torch.cat(obs_frames_list, dim=0).permute(2, 0, 1)
    pred_frames = torch.cat(pred_frames_list, dim=0).permute(2, 0, 1)
    obs_tracks = torch.cat(obs_tracks_list, dim=0).permute(2, 0, 1)
    pred_tracks = torch.cat(pred_tracks_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    dataset_idx = torch.LongTensor(dataset_idx)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_frames, pred_frames, obs_tracks, pred_tracks,
        non_linear_ped, loss_mask, seq_start_end, dataset_idx
    ]

    return tuple(out)


def seq_collate_with_dataset_idx_dict(data):
    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_frames, \
    pred_frames, obs_tracks, pred_tracks, non_linear_ped, loss_mask, \
    seq_start_end, dataset_idx = seq_collate_with_dataset_idx(data)

    return {
        'in_xy': obs_traj, 'in_dxdy': obs_traj_rel[1:, ...],
        'gt_xy': pred_traj, 'gt_dxdy': pred_traj_rel,
        'in_frames': obs_frames, 'gt_frames': pred_frames,
        'in_tracks': obs_tracks, 'gt_tracks': pred_tracks,
        'non_linear_ped': non_linear_ped, 'loss_mask': loss_mask,
        'seq_start_end': seq_start_end, 'dataset_idx': dataset_idx
    }


def anorm(p1, p2):
    norm = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if norm == 0:
        return 0
    return 1 / norm


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='space'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line if i != '']  # a fix done
            data.append(line)
    return np.asarray(data)


class STGCNNTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t', norm_lap_matr=True, construct_graph=False,
            video_class=None, video_number=None):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(STGCNNTrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.construct_graph = construct_graph
        self.video_class = video_class
        self.video_number = video_number

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        frames_list = []
        tracks_list = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                curr_frames = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                curr_tracks = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue

                    curr_ped_frames = np.transpose(curr_ped_seq[:, 0])
                    curr_ped_tracks = np.transpose(curr_ped_seq[:, 1])
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_frames[_idx, :, pad_front:pad_end] = curr_ped_frames
                    curr_tracks[_idx, :, pad_front:pad_end] = curr_ped_tracks
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    frames_list.append(curr_frames[:num_peds_considered])
                    tracks_list.append(curr_tracks[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        frames_list = np.concatenate(frames_list, axis=0)
        tracks_list = np.concatenate(tracks_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.obs_frames = torch.from_numpy(
            frames_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_frames = torch.from_numpy(
            frames_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_tracks = torch.from_numpy(
            tracks_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_tracks = torch.from_numpy(
            tracks_list[:, :, self.obs_len:]).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        if self.construct_graph:
            self.v_obs = []
            self.A_obs = []
            self.v_pred = []
            self.A_pred = []
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]

                v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
                self.v_obs.append(v_.clone())
                self.A_obs.append(a_.clone())
                v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :],
                                      self.norm_lap_matr)
                self.v_pred.append(v_.clone())
                self.A_pred.append(a_.clone())
            pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        if self.construct_graph:
            out = [
                self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                self.obs_frames[start:end, :], self.pred_frames[start:end, :],
                self.obs_tracks[start:end, :], self.pred_tracks[start:end, :],
                self.non_linear_ped[start:end], self.loss_mask[start:end, :],
                self.v_obs[index], self.A_obs[index],
                self.v_pred[index], self.A_pred[index]
            ]
        else:
            out = [
                self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                self.obs_frames[start:end, :], self.pred_frames[start:end, :],
                self.obs_tracks[start:end, :], self.pred_tracks[start:end, :],
                self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            ]
        return out


def read_temp_file(f, delim='space'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(f.name, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line if i != '']  # a fix done
            data.append(line)
    return np.asarray(data)


class TrajectoryDatasetFromFile(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, annotation_file, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t', norm_lap_matr=True, construct_graph=False,
            video_class=None, video_number=None):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDatasetFromFile, self).__init__()

        self.max_peds_in_frame = 0
        self.annotation_file = annotation_file
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.construct_graph = construct_graph
        self.video_class = video_class
        self.video_number = video_number

        all_files = [self.annotation_file]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        frames_list = []
        tracks_list = []
        for path in all_files:
            data = read_temp_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                curr_frames = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                curr_tracks = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue

                    curr_ped_frames = np.transpose(curr_ped_seq[:, 0])
                    curr_ped_tracks = np.transpose(curr_ped_seq[:, 1])
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_frames[_idx, :, pad_front:pad_end] = curr_ped_frames
                    curr_tracks[_idx, :, pad_front:pad_end] = curr_ped_tracks
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    frames_list.append(curr_frames[:num_peds_considered])
                    tracks_list.append(curr_tracks[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        frames_list = np.concatenate(frames_list, axis=0)
        tracks_list = np.concatenate(tracks_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.obs_frames = torch.from_numpy(
            frames_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_frames = torch.from_numpy(
            frames_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_tracks = torch.from_numpy(
            tracks_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_tracks = torch.from_numpy(
            tracks_list[:, :, self.obs_len:]).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        if self.construct_graph:
            self.v_obs = []
            self.A_obs = []
            self.v_pred = []
            self.A_pred = []
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]

                v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
                self.v_obs.append(v_.clone())
                self.A_obs.append(a_.clone())
                v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :],
                                      self.norm_lap_matr)
                self.v_pred.append(v_.clone())
                self.A_pred.append(a_.clone())
            pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        if self.construct_graph:
            out = [
                self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                self.obs_frames[start:end, :], self.pred_frames[start:end, :],
                self.obs_tracks[start:end, :], self.pred_tracks[start:end, :],
                self.non_linear_ped[start:end], self.loss_mask[start:end, :],
                self.v_obs[index], self.A_obs[index],
                self.v_pred[index], self.A_pred[index]
            ]
        else:
            out = [
                self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                self.obs_frames[start:end, :], self.pred_frames[start:end, :],
                self.obs_tracks[start:end, :], self.pred_tracks[start:end, :],
                self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            ]
        return out
