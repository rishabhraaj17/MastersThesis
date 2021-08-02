#
import numpy as np
import torch
from torch import nn as nn

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import get_dataset
from baselinev2.nn.evaluate import filter_on_relative_distances
from baselinev2.notebooks.utils import get_trajectory_length, get_trajectory_length_fast


def rotate(X, center, alpha):
    XX = torch.zeros_like(X)

    XX[:, 0] = (X[:, 0] - center[0]) * np.cos(alpha) + (X[:, 1] - center[1]) * np.sin(alpha) + center[0]
    XX[:, 1] = - (X[:, 0] - center[0]) * np.sin(alpha) + (X[:, 1] - center[1]) * np.cos(alpha) + center[1]

    return XX


def make_mlp(dim_list, activation_list, batch_norm=False, dropout=0):
    """
    Generates MLP network:

    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_list : list, list containing activation function for each layer
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)

    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    index = 0
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        activation = activation_list[index]
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))
        index += 1
    return nn.Sequential(*layers)


# BaseModel
"""
Implementation of the generator and discriminator of Goal GAN. 
Hyperparamters of the models are set in 'config/model/model.yaml' and 'experiment'
"""


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def test(self):
        self.eval()
        self.mode = "test"

    def gen(self):
        self.mode = "gen"
        self.train()

    def equivariance_test(self):
        raise NotImplementedError
        # T, N, D = 8, 10, 2
        #
        # InputVectorDim, InputScalarDim = [2, None]
        # OutputVectorDim, OutputScalarDim = [2, None]
        # InputVectordxdy = (1 - 2 * torch.rand(T - 1, N, D))
        #
        # InputVectorxy = torch.rand(T, N, D)
        #
        # InputScalar = None
        #
        # RotInputVectordxdy = InputVectordxdy * 1
        # RotInputVectorxy = InputVectorxy * 1
        # arbitrary_angle = (-2 * np.pi + 4 * np.pi * torch.rand(1))
        #
        # RotInputVectorxy = RotInputVectorxy.view(-1)
        # RotInputVectordxdy = RotInputVectordxdy.view(-1)
        #
        # for i in range(len(RotInputVectorxy) // 2):
        #     RotInputVectorxy[2 * i: 2 * (i + 1)] = se2.rotation(RotInputVectorxy[2 * i: 2 * (i + 1)], arbitrary_angle)
        #
        # for i in range(len(RotInputVectordxdy) // 2):
        #     RotInputVectordxdy[2 * i: 2 * (i + 1)] = se2.rotation(RotInputVectordxdy[2 * i: 2 * (i + 1)],
        #                                                           arbitrary_angle)
        #
        # RotInputVectorxy = RotInputVectorxy.view((T, N, D))
        # RotInputVectordxdy = RotInputVectordxdy.view((T - 1, N, D))
        #
        # if self.type == "SE2":
        #     z_vec = torch.randn(N, self.encoder_h_g_vec // 2, 1)
        # else:
        #     z_vec = None
        # z_scalar = torch.randn(N, self.noise_scalar)
        # RotInput = {"seq_start_end": [(0, N)], "in_xy": RotInputVectorxy, "in_dxdy": RotInputVectordxdy, "z_vec": z_vec,
        #             "z_scalar": z_scalar}
        # NormalInput = {"seq_start_end": [(0, N)], "in_xy": InputVectorxy, "in_dxdy": InputVectordxdy, "z_vec": z_vec,
        #                "z_scalar": z_scalar}
        # RotOutput = self(RotInput)
        # RotOutputVectorxy, RotOutputVectordxdy = RotOutput.values()
        #
        # Output = self(NormalInput)
        # OutputVectorxy, OutputVectordxdy = Output.values()
        #
        # RotAfterVectorxy, RotAfterVectordxdy = OutputVectorxy * 1., OutputVectordxdy * 1.
        # RotAfterVectorxy = RotAfterVectorxy.view(-1)
        # RotAfterVectordxdy = RotAfterVectordxdy.view(-1)
        #
        # for i in range(len(RotAfterVectorxy) // 2):
        #     RotAfterVectorxy[2 * i: 2 * (i + 1)] = se2.rotation(RotAfterVectorxy[2 * i: 2 * (i + 1)], arbitrary_angle)
        #
        # for i in range(len(RotAfterVectordxdy) // 2):
        #     RotAfterVectordxdy[2 * i: 2 * (i + 1)] = se2.rotation(RotAfterVectordxdy[2 * i: 2 * (i + 1)],
        #                                                           arbitrary_angle)
        #
        # RotAfterVectorxy = RotAfterVectorxy.view((self.pred_len, N, D))
        # RotAfterVectordxdy = RotAfterVectordxdy.view((self.pred_len, N, D))
        #
        # vectorDifferencexy = torch.sum(torch.sqrt((RotAfterVectorxy - RotOutputVectorxy) ** 2))
        # vectorDifferencedxdy = torch.sum(torch.sqrt((RotAfterVectordxdy - RotOutputVectordxdy) ** 2))
        #
        # assert vectorDifferencexy < 1e-4, "Equivarance for vectors xy not given: Difference {}, Module: {}".format(
        #     vectorDifferencexy, self.__class__)
        # assert vectorDifferencedxdy < 1e-4, "Equivarance for vectors dxdey not given: Differnece {}, Module: {}".format(
        #     vectorDifferencedxdy, self.__class__)
        #
        # print("Equivariance holds for {} : Diff xy: {} Diff dxdy: {}".format(self.__class__, vectorDifferencexy,
        #                                                                      vectorDifferencedxdy))


class MotionEncoder(nn.Module):
    """MotionEncoder extracts dynamic features of the past trajectory and consists of an encoding LSTM network"""

    def __init__(self,
                 encoder_h_dim=64,
                 input_dim=2,
                 embedding_dim=16,
                 dropout=0.0):
        """ Initialize MotionEncoder.
        Parameters.
            encoder_h_dim (int) - - dimensionality of hidden state
            input_dim (int) - - input dimensionality of spatial coordinates
            embedding_dim (int) - - dimensionality spatial embedding
            dropout (float) - - dropout in LSTM layer
        """
        super(MotionEncoder, self).__init__()
        self.encoder_h_dim = encoder_h_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        if embedding_dim:
            self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
            self.encoder = nn.LSTM(embedding_dim, encoder_h_dim)
        else:
            self.encoder = nn.LSTM(input_dim, encoder_h_dim)

    def init_hidden(self, batch, obs_traj):

        return (
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj),
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj)
        )

    def forward(self, obs_traj, state_tuple=None):
        """ Calculates forward pass of MotionEncoder
            Parameters:
                obs_traj (tensor) - - Tensor of shape (obs_len, batch, 2)
                state_tuple (tuple of tensors) - - Tuple with hidden state (1, batch, encoder_h_dim) and cell state tensor (1, batch, encoder_h_dim)
            Returns:
                output (tensor) - - Output of LSTM netwok for all time steps (obs_len, batch, encoder_h_dim)
                final_h (tensor) - - Final hidden state of LSTM network (1, batch, encoder_h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        if not state_tuple:
            state_tuple = self.init_hidden(batch, obs_traj)
        if self.embedding_dim:
            obs_traj = self.spatial_embedding(obs_traj)

        output, state = self.encoder(obs_traj, state_tuple)
        final_h = state[0]
        return output, final_h


class BaselineGenerator(BaseModel):
    def __init__(self,

                 embedding_dim_scalars: int = 8,

                 encoder_h_g_scalar: int = 8,
                 decoder_h_g_scalar: int = 16,
                 pred_len: int = 12,
                 noise_scalar=4,  # 4,
                 mlp_vec=64,
                 mlp_scalar=32,
                 POV=False,

                 noise_type="global",  # "global",
                 social_attention=False,
                 social_dim_scalar: int = 10,
                 **kwargs
                 ) -> None:
        super(BaselineGenerator, self).__init__()
        self.pred_len = pred_len
        self.type = "Baseline"
        self.POV = POV
        self.embed_dim_scalar = embedding_dim_scalars
        self.encoder_h_g_scalar = encoder_h_g_scalar

        self.noise_type = noise_type
        assert self.noise_type in ["global",
                                   "local"], "Invalid Noise type! Choose `global` or `local`, not %s" % self.noise_type
        self.decoder_h_g_scalar = decoder_h_g_scalar
        self.social_attention = social_attention
        self.noise_scalar = noise_scalar

        self.social_dim_scalar = social_dim_scalar if self.social_attention else 0

        self.embedding = nn.Linear(in_features=2, out_features=self.embed_dim_scalar)

        self.encoder2decoder = nn.Sequential(
            nn.Linear(in_features=encoder_h_g_scalar + noise_scalar + self.social_dim_scalar, out_features=mlp_scalar),
            nn.Tanh(),
            nn.Linear(in_features=mlp_scalar, out_features=decoder_h_g_scalar),
            nn.Tanh()
        )

        # self.c_noise = nn.Sequential(se2.SE2Layer(in_features = encoder_h_g_vec, out_features = decoder_h_g_vec, in_scalars = encoder_h_g_scalar + noise_dim, out_scalars=decoder_h_g_scalar),
        #                                 se2.NormTanh())

        self.encoder = nn.LSTM(input_size=self.embed_dim_scalar, hidden_size=encoder_h_g_scalar)
        # handled in e2d
        # self.encoder = nn.LSTM(input_size=self.embed_dim_scalar, hidden_size=encoder_h_g_scalar - self.noise_scalar)
        self.decoder = nn.LSTM(input_size=self.embed_dim_scalar, hidden_size=decoder_h_g_scalar)
        self.regressor = nn.Linear(in_features=decoder_h_g_scalar, out_features=2)

        if self.social_attention:
            raise NotImplementedError
            # self.attention_net = SocialAttention(
            #     embed_scalar=self.embed_dim_scalar,
            #     h_scalar=self.encoder_h_g_scalar,
            #     social_dim_scalar=self.social_dim_scalar, )

    def get_theta(self, batch):
        in_xy = batch["in_xy"]
        in_dxdy = batch["in_dxdy"]
        last_vel = in_dxdy[-1]
        y_direction = torch.Tensor([[0, 1]]).to(in_xy)
        last_v = torch.norm(last_vel, p=2, dim=1)
        dot_sum = torch.sum(y_direction * last_vel, 1)

        theta = torch.arccos(dot_sum / torch.maximum(last_v, torch.Tensor([1e-20]).to(in_xy)))
        theta = theta + (last_vel[:, 0] < 0) * 2 * (np.pi - theta)

        return theta.to(in_xy)

    def rotate2POV(self, batch, theta):
        raise NotImplementedError
        # in_xy = batch["in_xy"] * 1.
        # in_dxdy = batch["in_dxdy"] * 1.
        #
        # k = in_xy.size(1)
        # for i in range(k):
        #     # batch["in_xy"][:, i] = rotation(in_xy[:, i], theta[i])
        #     batch["in_dxdy"][:, i] = rotation(in_dxdy[:, i], theta[i]) * 1.
        #
        # return batch

    def rotate2Normal(self, out, theta):
        raise NotImplementedError
        # if self.POV:
        #     k = out.size(1)
        #     for i in range(k):
        #         out[:, i] = rotation(out[:, i], -theta[i]) * 1.
        #
        # return out

    def forward(self, batch):
        # batch = preprocess_dataset_elements(batch, batch_first=False)

        if self.POV:
            theta = self.get_theta(batch)
            batch = self.rotate2POV(batch, theta)
        dxdy = batch["in_dxdy"] * 1.
        T, N, D = dxdy.size()
        # N, T, D = dxdy.size()

        dxdy = dxdy.reshape(T * N, D)

        emb = self.embedding(dxdy)
        emb = emb.view(T, N, -1)
        encoding, hidden_states = self.encoder(emb)

        h, c = hidden_states

        if self.noise_scalar:
            if "z_scalar" in batch:
                z_scalar = batch["z_scalar"]
            else:
                if self.noise_type == "global":
                    rand_numbers = torch.randn(1, len(batch["seq_start_end"]), self.noise_scalar)
                    z_scalar = [rand_numbers[:, i].unsqueeze(1).repeat(1, end - start, 1) for i, (start, end) in
                                enumerate(batch["seq_start_end"])]
                    z_scalar = torch.cat(z_scalar, 1).to(h)

                elif self.noise_type == "local":
                    z_scalar = torch.randn(1, N, self.noise_scalar).to(h)
                else:
                    raise Exception("Invalid Noise type! Choose `global` or `local`, not %s" % self.noise_type)

            h_dec = torch.cat((h, z_scalar), -1)
        else:

            h_dec = h * 1.

        out_xy = []
        out_dxdy = []
        final_pos = (batch["in_xy"][-1] * 1.).unsqueeze(0)
        final_vel = (batch["in_dxdy"][-1] * 1.).unsqueeze(0)
        if self.social_attention:
            social_scalar = []
            for (start, end) in batch["seq_start_end"]:
                s_scalar = self.attention_net(h_scalar=h[0, start:end], end_pos=final_pos[0, start:end])
                social_scalar.append(s_scalar)
            social_scalar = torch.cat(social_scalar).unsqueeze(0)
            h_dec = torch.cat((h_dec, social_scalar), 2)
        if self.noise_scalar or self.social_attention or self.encoder_h_g_scalar != self.decoder_h_g_scalar:
            h = self.encoder2decoder(h_dec)
        # # c_vec, c_scalar = self.c_noise([c_vec,torch.cat((c_scalar, z), -1)] )

        c = torch.zeros(1, N, self.decoder_h_g_scalar).to(h)

        # print(final_pos[0])
        for t in range(self.pred_len):

            final_vel_emb = self.embedding(final_vel)

            final_vec, hidden_states_dec = self.decoder(final_vel_emb, (h, c))
            h, c = hidden_states_dec

            # print((h_vec[0], h_scalar[0]))

            # sdsdsdsd
            final_vel = self.regressor(h)
            # print("final vec size", final_vel.size())
            # print("final_vec", final_vel[0])
            if self.POV:
                dxdy_pred = self.rotate2Normal(final_vel * 1., theta)
            else:
                dxdy_pred = final_vel * 1.
            final_pos = dxdy_pred + final_pos
            # print("final+pos", final_pos[0])

            out_xy.append(final_pos * 1.)

            out_dxdy.append(dxdy_pred * 1.)

        out_xy = torch.cat(out_xy, 0)
        out_dxdy = torch.cat(out_dxdy, 0)

        out = {"out_xy": out_xy, "out_dxdy": out_dxdy}

        # print(out_dxdy)
        return out


class Discriminator(BaseModel):
    """Implementation of discriminator of GOAL GAN

       The model consists out of three main components:
       1. encoder of input trajectory
       2. encoder of
       3. Routing Module with visual soft-attention

       """

    def __init__(self,
                 encoder_h_dim_d=64,
                 embedding_dim_scalars=16,
                 dropout_disc=0.0,
                 mlp_scalar: int = 32,
                 social_attention: bool = False,
                 social_dim_scalar: int = 10,
                 **kwargs
                 ) -> None:

        super().__init__()

        self.__dict__.update(locals())

        self.grad_status = True

        self.embed_dim_scalar = embedding_dim_scalars
        self.social_attention = social_attention
        self.social_dim_scalar = social_dim_scalar if self.social_attention else 0

        self.encoder_observation = MotionEncoder(
            encoder_h_dim=self.encoder_h_dim_d,
            input_dim=2,
            embedding_dim=self.embed_dim_scalar,
            dropout=self.dropout_disc
        )

        self.EncoderPrediction = EncoderPrediction(
            input_dim=2,
            encoder_h_dim_d=self.encoder_h_dim_d + self.social_dim_scalar,
            embedding_dim=self.embed_dim_scalar,
            dropout=self.dropout_disc,
            batch_norm=False,
        )

        if self.social_attention:
            raise NotImplementedError
            # self.attention_net = SocialAttention(
            #     embed_scalar=self.embed_dim_scalar,
            #     h_scalar=self.encoder_h_dim_d,
            #     social_dim_scalar=self.social_dim_scalar)

    def init_c(self, batch_size):
        return torch.zeros((1, batch_size, self.encoder_h_dim_d + self.social_dim_scalar))

    def forward(self, in_xy, in_dxdy, out_xy, out_dxdy, seq_start_end=[], images_patches=None):

        output_h, h = self.encoder_observation(in_dxdy)
        final_pos = in_xy[-1] * 1.
        if self.social_attention:
            social_scalar = []
            for (start, end) in seq_start_end:
                s_scalar = self.attention_net(h_scalar=h[0, start:end], end_pos=final_pos[start:end])
                social_scalar.append(s_scalar)
            social_scalar = torch.cat(social_scalar).unsqueeze(0)
            h = torch.cat((h, social_scalar), 2)

        batch_size = in_xy.size(1)
        c = self.init_c(batch_size).to(in_xy)
        state_tuple = (h, c)

        dynamic_scores = self.EncoderPrediction(out_dxdy, images_patches, state_tuple)

        return dynamic_scores

    def grad(self, status):
        if not self.grad_status == status:
            self.grad_status = status
            for p in self.parameters():
                p.requires_grad = self.grad_status


class EncoderPrediction(nn.Module):
    """Part of Discriminator"""

    def __init__(
            self, input_dim=2,
            encoder_h_dim_d=128,
            embedding_dim=64,
            dropout=0.0,

            batch_norm=False,
            dropout_cnn=0,
            mlp_dim=32,
    ):
        super().__init__()

        self.__dict__.update(locals())
        del self.self

        activation = ['leakyrelu', None]

        self.leakyrelu = nn.LeakyReLU()

        self.encoder = nn.LSTM(self.embedding_dim, self.encoder_h_dim_d, dropout=dropout)

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

        real_classifier_dims = [self.encoder_h_dim_d, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation_list=activation,
            dropout=dropout)

    def init_hidden(self, batch, obs_traj):
        return (torch.zeros(1, batch, self.encoder_h_dim_d).to(obs_traj),
                torch.zeros(1, batch, self.encoder_h_dim_d).to(obs_traj))

    def forward(self, dxdy, img_patch, state_tuple):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        Output:
        - pred_traj_fake_rel: tensor of shape (self.seq_len, batch, 2)
        - pred_traj_fake: tensor of shape (self.seq_len, batch, 2)
        - state_tuple[0]: final hidden state
        """

        embedded_pos = self.spatial_embedding(dxdy).tanh()

        encoder_input = embedded_pos
        output, input_classifier = self.encoder(encoder_input, state_tuple)
        dynamic_score = self.real_classifier(input_classifier[0])
        return dynamic_score


def preprocess_dataset_elements(batch, is_generated=False, batch_first=True, filter_mode=True, moving_only=False,
                                stationary_only=False, threshold=1.0, relative_distance_filter_threshold=100.,
                                return_stationary_objects_idx=True):
    if is_generated:
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, _, _, ratio = \
            batch
    else:
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = batch

    if filter_mode:
        full_xy = torch.cat((in_xy, gt_xy), dim=1)
        full_length_per_step, full_length = get_trajectory_length(full_xy.cpu().numpy(), use_l2=True)
        obs_trajectory_length, obs_trajectory_length_summed = get_trajectory_length(in_xy.cpu().numpy(), use_l2=True)
        gt_trajectory_length, gt_trajectory_length_summed = get_trajectory_length(gt_xy.cpu().numpy(), use_l2=True)

        full_length *= ratio.cpu().numpy()
        obs_trajectory_length_summed *= ratio.cpu().numpy()
        gt_trajectory_length_summed *= ratio.cpu().numpy()

        if moving_only:
            # feasible_idx = np.where(length > threshold)[0]
            obs_feasible_idx = np.where(obs_trajectory_length_summed > threshold)[0]
            gt_feasible_idx = np.where(gt_trajectory_length_summed > threshold)[0]
            # feasible_idx = np.union1d(obs_feasible_idx, gt_feasible_idx)
            feasible_idx = np.intersect1d(obs_feasible_idx, gt_feasible_idx)

            if relative_distance_filter_threshold is not None:
                feasible_idx = filter_on_relative_distances(feasible_idx, gt_uv.cpu(), in_uv.cpu(),
                                                            relative_distance_filter_threshold)

            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = \
                in_xy[feasible_idx], gt_xy[feasible_idx], in_uv[feasible_idx], gt_uv[feasible_idx], \
                in_track_ids[feasible_idx], gt_track_ids[feasible_idx], in_frame_numbers[feasible_idx], \
                gt_frame_numbers[feasible_idx], ratio[feasible_idx]

            if is_generated:
                batch = [in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers,
                         gt_frame_numbers, [], [], ratio]
            else:
                batch = [in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers,
                         gt_frame_numbers, ratio]

        if stationary_only:
            # feasible_idx = np.where(length < threshold)[0]
            obs_feasible_idx = np.where(obs_trajectory_length_summed < threshold)[0]
            gt_feasible_idx = np.where(gt_trajectory_length_summed < threshold)[0]
            # feasible_idx = np.union1d(obs_feasible_idx, gt_feasible_idx)
            feasible_idx = np.intersect1d(obs_feasible_idx, gt_feasible_idx)

            if relative_distance_filter_threshold is not None:
                feasible_idx = filter_on_relative_distances(feasible_idx, gt_uv.cpu(), in_uv.cpu(),
                                                            relative_distance_filter_threshold)

            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = \
                in_xy[feasible_idx], gt_xy[feasible_idx], in_uv[feasible_idx], gt_uv[feasible_idx], \
                in_track_ids[feasible_idx], gt_track_ids[feasible_idx], in_frame_numbers[feasible_idx], \
                gt_frame_numbers[feasible_idx], ratio[feasible_idx]
            if is_generated:
                batch = [in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers,
                         gt_frame_numbers, [], [], ratio]
            else:
                batch = [in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers,
                         gt_frame_numbers, ratio]

        if return_stationary_objects_idx:
            # feasible_idx = np.where(length < threshold)[0]
            obs_feasible_idx = np.where(obs_trajectory_length_summed < threshold)[0]
            gt_feasible_idx = np.where(gt_trajectory_length_summed < threshold)[0]
            # feasible_idx = np.union1d(obs_feasible_idx, gt_feasible_idx)
            feasible_idx = np.intersect1d(obs_feasible_idx, gt_feasible_idx)

            if relative_distance_filter_threshold is not None:
                feasible_idx = filter_on_relative_distances(feasible_idx, gt_uv.cpu(), in_uv.cpu(),
                                                            relative_distance_filter_threshold)

    # todo: it has to be fixed - look at how its done for any working dataset, it needs to how many
    #  peds are there in a sequence
    _len = [len(seq) for seq in in_xy.permute(1, 0, 2)]
    # _len = [len(seq) for seq in in_xy]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = torch.LongTensor([[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])])

    if not batch_first:
        in_xy, gt_xy, in_uv, gt_uv = \
            in_xy.permute(1, 0, 2), gt_xy.permute(1, 0, 2), in_uv.permute(1, 0, 2), gt_uv.permute(1, 0, 2)

    return {'in_xy': in_xy,
            'gt_xy': gt_xy,
            'in_dxdy': in_uv,
            'gt_dxdy': gt_uv,
            'ratio': ratio,
            'seq_start_end': seq_start_end,
            'feasible_idx': feasible_idx
            if (moving_only or stationary_only or return_stationary_objects_idx) and filter_mode else []
            }


def preprocess_dataset_elements_from_dict(
        batch, filter_mode=True, moving_only=False,
        stationary_only=False, threshold=1.0, relative_distance_filter_threshold=100.,
        return_stationary_objects_idx=True):

    in_xy, gt_xy, in_uv, gt_uv = batch['in_xy'], batch['gt_xy'], batch['in_dxdy'], batch['gt_dxdy']
    in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers = \
        batch['in_tracks'], batch['gt_tracks'], batch['in_frames'], batch['gt_frames']
    non_linear_ped, loss_mask, seq_start_end, dataset_idx, ratio = \
        batch['non_linear_ped'], batch['loss_mask'], batch['seq_start_end'], batch['dataset_idx'], batch['ratio']

    if filter_mode:
        full_xy = torch.cat((in_xy, gt_xy), dim=0)
        full_length_per_step, full_length = get_trajectory_length_fast(full_xy.cpu().numpy())
        obs_trajectory_length, obs_trajectory_length_summed = get_trajectory_length_fast(in_xy.cpu().numpy())
        gt_trajectory_length, gt_trajectory_length_summed = get_trajectory_length_fast(gt_xy.cpu().numpy())

        full_length *= ratio.cpu().numpy().squeeze()
        obs_trajectory_length_summed *= ratio.cpu().numpy().squeeze()
        gt_trajectory_length_summed *= ratio.cpu().numpy().squeeze()

        if moving_only:
            # feasible_idx = np.where(length > threshold)[0]
            obs_feasible_idx = np.where(obs_trajectory_length_summed > threshold)[0]
            gt_feasible_idx = np.where(gt_trajectory_length_summed > threshold)[0]
            # feasible_idx = np.union1d(obs_feasible_idx, gt_feasible_idx)
            feasible_idx = np.intersect1d(obs_feasible_idx, gt_feasible_idx)

            if relative_distance_filter_threshold is not None:
                feasible_idx = filter_on_relative_distances(feasible_idx, gt_uv.cpu(), in_uv.cpu(),
                                                            relative_distance_filter_threshold, axis=0)

            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = \
                in_xy[:, feasible_idx, :], gt_xy[:, feasible_idx, :], in_uv[:, feasible_idx, :], gt_uv[:, feasible_idx, :], \
                in_track_ids[:, feasible_idx, :], gt_track_ids[:, feasible_idx, :], in_frame_numbers[:, feasible_idx, :], \
                gt_frame_numbers[:, feasible_idx, :], ratio[feasible_idx]
            non_linear_ped, loss_mask = non_linear_ped[feasible_idx], loss_mask[feasible_idx]

        if stationary_only:
            # feasible_idx = np.where(length < threshold)[0]
            obs_feasible_idx = np.where(obs_trajectory_length_summed < threshold)[0]
            gt_feasible_idx = np.where(gt_trajectory_length_summed < threshold)[0]
            # feasible_idx = np.union1d(obs_feasible_idx, gt_feasible_idx)
            feasible_idx = np.intersect1d(obs_feasible_idx, gt_feasible_idx)

            if relative_distance_filter_threshold is not None:
                feasible_idx = filter_on_relative_distances(feasible_idx, gt_uv.cpu(), in_uv.cpu(),
                                                            relative_distance_filter_threshold, axis=0)

            in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = \
                in_xy[:, feasible_idx, :], gt_xy[:, feasible_idx, :], in_uv[:, feasible_idx, :], gt_uv[:, feasible_idx, :], \
                in_track_ids[:, feasible_idx, :], gt_track_ids[:, feasible_idx, :], in_frame_numbers[:, feasible_idx, :], \
                gt_frame_numbers[:, feasible_idx, :], ratio[feasible_idx]
            non_linear_ped, loss_mask = non_linear_ped[feasible_idx], loss_mask[feasible_idx]

        if return_stationary_objects_idx:
            # feasible_idx = np.where(length < threshold)[0]
            obs_feasible_idx = np.where(obs_trajectory_length_summed < threshold)[0]
            gt_feasible_idx = np.where(gt_trajectory_length_summed < threshold)[0]
            # feasible_idx = np.union1d(obs_feasible_idx, gt_feasible_idx)
            feasible_idx = np.intersect1d(obs_feasible_idx, gt_feasible_idx)

            if relative_distance_filter_threshold is not None:
                feasible_idx = filter_on_relative_distances(feasible_idx, gt_uv.cpu(), in_uv.cpu(),
                                                            relative_distance_filter_threshold, axis=0)

    return {'in_xy': in_xy,
            'in_dxdy': in_uv,
            'gt_xy': gt_xy,
            'gt_dxdy': gt_uv,
            'in_frames': in_frame_numbers,
            'gt_frames': gt_frame_numbers,
            'in_tracks': in_track_ids,
            'gt_tracks': gt_track_ids,
            'non_linear_ped': non_linear_ped,
            'loss_mask': loss_mask,
            'seq_start_end': seq_start_end,
            'dataset_idx': dataset_idx,
            'ratio': ratio,
            'feasible_idx': feasible_idx
            if (moving_only or stationary_only or return_stationary_objects_idx) and filter_mode else []
            }


if __name__ == '__main__':
    train_set = get_dataset(video_clazz=SDDVideoClasses.LITTLE, video_number=3,
                            mode=NetworkMode.TRAIN, meta_label=SDDVideoDatasets.LITTLE, get_generated=False)
    data = train_set[0:5]
    data = preprocess_dataset_elements(data, batch_first=False)
    in_xy, gt_xy, in_uv, gt_uv, ratio, seq_start_end = \
        data['in_xy'], data['gt_xy'], data['in_dxdy'], data['gt_dxdy'], data['ratio'], data['seq_start_end']
    gen = BaselineGenerator(noise_type='local')
    dis = Discriminator()
    out = gen(data)
    d_out = dis(in_xy, in_uv, out['out_xy'], out['out_dxdy'], seq_start_end)
    print()
