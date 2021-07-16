from typing import Callable, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
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

    def forward(self, x):
        in_xy, in_dxdy = x['in_xy'], x['in_dxdy']
        out = self.embedding(in_dxdy)
        out = self.encoder(out)
        return out
    
    
class TransformerMotionDecoder(nn.Module):
    def __init__(self, config: DictConfig):
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

    def forward(self, x, encoder_out):
        in_xy, in_dxdy = x['in_xy'], x['in_dxdy']

        last_obs_pos = in_xy[-1, ...]
        last_obs_vel = self.embedding(in_dxdy[-1, None, ...])

        d_o = last_obs_vel.clone()

        for _ in range(self.seq_len):
            d_o = self.decoder(d_o, encoder_out)
            # for one ts autoregressive comment line below
            d_o = torch.cat((last_obs_vel, d_o))

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


if __name__ == '__main__':
    conf = OmegaConf.load('../../../src/position_maps/config/model/model.yaml')
    inp = {
        'in_dxdy': torch.randn((7, 2, 2)),
        'in_xy': torch.randn((8, 2, 2)),
        'gt_xy': torch.randn((12, 2, 2))
    }
    m = TrajectoryTransformer(conf, None, None)
    o = m._one_step(inp)

    # d_inp = torch.randn((1, 2, 64))
    # d_o = d_inp.clone()
    # for i in range(12):
    #     d_o = d_m(d_o, o)
    #
    #     # variant 1 - simple autoregressive but one ts only
    #     # d_inp = d_o - no need self assigned
    #
    #     # variant 2 - take the inp always and take last n-1 output and feed it back
    #     # d_o = torch.cat((d_inp, d_o))

    # variant - 1
    # loss = pairwise_distance(torch.randn(1, 2, 64).view(2, -1), d_o.view(2, -1)).mean()

    # variant - 2
    # loss = pairwise_distance(torch.randn(12, 2, 64).view(2, -1), d_o[1:, ...].view(2, -1)).mean()

    # loss.backward()
    print()
