from typing import Tuple, List, Optional, Callable

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset

from baselinev2.stochastic.model_modules import preprocess_dataset_elements
from src_lib.models_hub import Base


class RNNBaseline(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(RNNBaseline, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        net_params = self.config.trajectory_based.rnn.deterministic
        self.pred_len = net_params.pred_len
        self.rnn_type = net_params.rnn_type
        self.batch_first = net_params.batch_first

        rnn = getattr(nn, self.rnn_type)

        self.embed_dim_scalar = net_params.embedding_dim_scalars
        self.encoder_h_g_scalar = net_params.encoder_h_g_scalar

        self.decoder_h_g_scalar = net_params.decoder_h_g_scalar

        self.mlp_scalar = net_params.mlp_scalar

        # Not in use right now
        self.social_attention = net_params.social_attention
        self.social_dim_scalar = net_params.social_dim_scalar if self.social_attention else 0

        self.embedding = nn.Linear(in_features=net_params.in_features, out_features=self.embed_dim_scalar)

        self.encoder2decoder = nn.Sequential(
            nn.Linear(in_features=self.encoder_h_g_scalar + self.social_dim_scalar,
                      out_features=self.mlp_scalar),
            nn.Tanh(),
            nn.Linear(in_features=self.mlp_scalar, out_features=self.decoder_h_g_scalar),
            nn.Tanh()
        )

        self.encoder = rnn(input_size=self.embed_dim_scalar,
                           hidden_size=self.encoder_h_g_scalar, batch_first=self.batch_first)
        self.decoder = rnn(input_size=self.embed_dim_scalar,
                           hidden_size=self.decoder_h_g_scalar, batch_first=self.batch_first)
        self.regressor = nn.Linear(in_features=self.decoder_h_g_scalar, out_features=net_params.out_features)

        if self.social_attention:
            raise NotImplementedError

    def forward(self, batch):
        # batch = preprocess_dataset_elements(batch, batch_first=False)

        dxdy = batch["in_dxdy"] * 1.

        if self.batch_first:
            N, T, D = dxdy.size()
        else:
            T, N, D = dxdy.size()

        dxdy = dxdy.reshape(T * N, D)

        emb = self.embedding(dxdy)
        emb = emb.view(T, N, -1)
        encoding, hidden_states = self.encoder(emb)

        h, c = hidden_states
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
        if self.social_attention or self.encoder_h_g_scalar != self.decoder_h_g_scalar:
            h = self.encoder2decoder(h_dec)

        c = torch.zeros(1, N, self.decoder_h_g_scalar).to(h)

        for t in range(self.pred_len):
            final_vel_emb = self.embedding(final_vel)

            final_vec, hidden_states_dec = self.decoder(final_vel_emb, (h, c))
            h, c = hidden_states_dec

            final_vel = self.regressor(h)

            dxdy_pred = final_vel * 1.
            final_pos = dxdy_pred + final_pos

            out_xy.append(final_pos * 1.)
            out_dxdy.append(dxdy_pred * 1.)

        out_xy = torch.cat(out_xy, 0)
        out_dxdy = torch.cat(out_dxdy, 0)

        out = {"out_xy": out_xy, "out_dxdy": out_dxdy}

        return out


class RNNBaselineGenerator(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(RNNBaselineGenerator, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        net_params = self.config.trajectory_based.rnn.generator
        self.pred_len = net_params.pred_len
        self.rnn_type = net_params.rnn_type
        self.batch_first = net_params.batch_first

        rnn = getattr(nn, self.rnn_type)

        self.embed_dim_scalar = net_params.embedding_dim_scalars
        self.encoder_h_g_scalar = net_params.encoder_h_g_scalar

        self.noise_type = net_params.noise_type
        assert self.noise_type in ["global", "local"], \
            f"Invalid Noise type! Choose `global` or `local`, not {self.noise_type}"
        self.decoder_h_g_scalar = net_params.decoder_h_g_scalar

        self.noise_scalar = net_params.noise_scalar
        self.mlp_scalar = net_params.mlp_scalar

        # Not in use right now
        self.social_attention = net_params.social_attention
        self.social_dim_scalar = net_params.social_dim_scalar if self.social_attention else 0

        self.embedding = nn.Linear(in_features=net_params.in_features, out_features=self.embed_dim_scalar)

        self.encoder2decoder = nn.Sequential(
            nn.Linear(in_features=self.encoder_h_g_scalar + self.noise_scalar + self.social_dim_scalar,
                      out_features=self.mlp_scalar),
            nn.Tanh(),
            nn.Linear(in_features=self.mlp_scalar, out_features=self.decoder_h_g_scalar),
            nn.Tanh()
        )

        self.encoder = rnn(input_size=self.embed_dim_scalar,
                           hidden_size=self.encoder_h_g_scalar, batch_first=self.batch_first)
        self.decoder = rnn(input_size=self.embed_dim_scalar,
                           hidden_size=self.decoder_h_g_scalar, batch_first=self.batch_first)
        self.regressor = nn.Linear(in_features=self.decoder_h_g_scalar, out_features=net_params.out_features)

        if self.social_attention:
            raise NotImplementedError

    def forward(self, batch):
        # batch = preprocess_dataset_elements(batch, batch_first=False)

        dxdy = batch["in_dxdy"] * 1.

        if self.batch_first:
            N, T, D = dxdy.size()
        else:
            T, N, D = dxdy.size()

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

        c = torch.zeros(1, N, self.decoder_h_g_scalar).to(h)

        for t in range(self.pred_len):
            final_vel_emb = self.embedding(final_vel)

            final_vec, hidden_states_dec = self.decoder(final_vel_emb, (h, c))
            h, c = hidden_states_dec

            final_vel = self.regressor(h)

            dxdy_pred = final_vel * 1.
            final_pos = dxdy_pred + final_pos

            out_xy.append(final_pos * 1.)
            out_dxdy.append(dxdy_pred * 1.)

        out_xy = torch.cat(out_xy, 0)
        out_dxdy = torch.cat(out_dxdy, 0)

        out = {"out_xy": out_xy, "out_dxdy": out_dxdy}

        return out


if __name__ == '__main__':
    m = RNNBaseline(OmegaConf.load('../../../src/position_maps/config/model/model.yaml'), None, None)
    inp = {
        'in_dxdy': torch.randn((7, 2, 2)),
        'in_xy': torch.randn((8, 2, 2))
    }
    o = m(inp)
    print()
