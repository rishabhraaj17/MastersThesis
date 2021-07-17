from typing import Callable, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorchvideo.layers import PositionalEncoding
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub import Base


class TransformerMotionEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super(TransformerMotionEncoder, self).__init__()

        self.config = config
        net_params = self.config.trajectory_based.transformer.encoder

        self.embedding = nn.Sequential(
            nn.Linear(in_features=net_params.in_features, out_features=net_params.d_model // 2),
            nn.ReLU(),
            nn.Linear(in_features=net_params.d_model // 2, out_features=net_params.d_model)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=net_params.d_model,
                nhead=net_params.nhead,
                dim_feedforward=net_params.dim_feedforward,
                dropout=net_params.dropout,
                activation=net_params.activation
            ),
            num_layers=net_params.num_layers,
            norm=nn.LayerNorm(net_params.d_model) if net_params.norm is not None else None
        )
        self.positional_encoding = PositionalEncoding(embed_dim=net_params.d_model, seq_len=net_params.seq_len)

    def forward(self, x):
        in_xy, in_dxdy = x['in_xy'], x['in_dxdy']
        out = self.embedding(in_dxdy)

        # add positional encoding
        # (S, B, E) -> (B, S, E)
        out = out.permute(1, 0, 2)
        out = self.positional_encoding(out)
        # (B, S, E) -> (S, B, E)
        out = out.permute(1, 0, 2)

        out = self.encoder(out)
        return out


class TransformerMotionDecoder(nn.Module):
    def __init__(self, config: DictConfig, return_raw_logits=False):
        super(TransformerMotionDecoder, self).__init__()

        self.config = config
        net_params = self.config.trajectory_based.transformer.decoder

        self.seq_len = net_params.seq_len

        self.embedding = nn.Sequential(
            nn.Linear(in_features=net_params.in_features, out_features=net_params.d_model // 2),
            nn.ReLU(),
            nn.Linear(in_features=net_params.d_model // 2, out_features=net_params.d_model)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=net_params.d_model,
                nhead=net_params.nhead,
                dim_feedforward=net_params.dim_feedforward,
                dropout=net_params.dropout,
                activation=net_params.activation
            ),
            num_layers=net_params.num_layers,
            norm=nn.LayerNorm(net_params.d_model) if net_params.norm is not None else None
        )
        self.projector = nn.Sequential(
            nn.Linear(in_features=net_params.d_model, out_features=net_params.d_model // 2),
            nn.ReLU(),
            nn.Linear(in_features=net_params.d_model // 2, out_features=net_params.out_features)
        )
        self.positional_encoding = PositionalEncoding(embed_dim=net_params.d_model, seq_len=self.seq_len)

        self.return_raw_logits = return_raw_logits

    def forward(self, x, encoder_out):
        in_xy, in_dxdy = x['in_xy'], x['in_dxdy']

        last_obs_pos = in_xy[-1, ...]
        last_obs_vel = self.embedding(in_dxdy[-1, None, ...])

        d_o = last_obs_vel.clone()

        for _ in range(self.seq_len):

            # add positional encoding
            # (S, B, E) -> (B, S, E)
            d_o = d_o.permute(1, 0, 2)
            d_o = self.positional_encoding(d_o)
            # (B, S, E) -> (S, B, E)
            d_o = d_o.permute(1, 0, 2)

            d_o = self.decoder(d_o, encoder_out)
            # for one ts autoregressive comment line below
            d_o = torch.cat((last_obs_vel, d_o))

        if self.return_raw_logits:
            return d_o[1:, ...]

        pred_dxdy = self.projector(d_o[1:, ...])
        out_xy = []
        for pred_vel in pred_dxdy:
            last_obs_pos += pred_vel
            out_xy.append(last_obs_pos)

        out = {
            'out_xy': torch.stack(out_xy),
            'out_dxdy': pred_dxdy
        }
        return out


class TrajectoryTransformer(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None):
        super(TrajectoryTransformer, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.encoder = TransformerMotionEncoder(self.config)
        self.decoder = TransformerMotionDecoder(self.config)

    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(x, enc_out)
        return dec_out

    def _one_step(self, batch):
        target = batch['gt_xy']

        out = self(batch)
        pred = out['out_xy']

        loss = self.calculate_loss(pred, target)
        return loss

    def calculate_loss(self, pred, target):
        return torch.linalg.norm((pred - target), ord=2, dim=0).mean(dim=0).mean()


class TransformerMotionGenerator(nn.Module):
    def __init__(self, config: DictConfig):
        super(TransformerMotionGenerator, self).__init__()
        self.config = config
        net_params = self.config.trajectory_based.transformer.encoder

        self.motion_encoder = TransformerMotionEncoder(config=self.config)
        self.noise_embedding = nn.Sequential(
            nn.Linear(in_features=net_params.d_model * 2, out_features=net_params.d_model)
        )

    def forward(self, x):
        out = self.motion_encoder(x)
        out = torch.cat((out, torch.randn_like(out)), dim=-1)
        out = self.noise_embedding(out)
        return out


class TransformerMotionDiscriminator(nn.Module):
    def __init__(self, config: DictConfig):
        super(TransformerMotionDiscriminator, self).__init__()
        self.config = config

        self.motion_encoder = TransformerMotionEncoder(self.config)
        self.motion_decoder = TransformerMotionDecoder(self.config, return_raw_logits=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.config.trajectory_based.transformer.decoder.d_model,
                      out_features=self.config.trajectory_based.transformer.decoder.d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.config.trajectory_based.transformer.decoder.d_model // 2,
                      out_features=1))

    def forward(self, x):
        enc_out = self.motion_encoder(x)
        dec_out = self.motion_decoder(x, enc_out)
        out = self.classifier(dec_out.mean(0))  # mean over all time-steps - can take 1st or last ts as well?
        return out


if __name__ == '__main__':
    # todo: add Positional Encoding
    conf = OmegaConf.load('../../../src/position_maps/config/model/model.yaml')
    inp = {
        'in_dxdy': torch.randn((7, 2, 2)),
        'in_xy': torch.randn((8, 2, 2)),
        'gt_xy': torch.randn((12, 2, 2))
    }
    m = TransformerMotionDiscriminator(conf)
    o = m(inp)
    # m = TrajectoryTransformer(conf, None, None)
    # o = m._one_step(inp)
    print()
