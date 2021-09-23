from typing import Optional, Callable, List, Tuple

import torch
import torchvision
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf
from pytorchvideo.layers import PositionalEncoding
from torch import nn
from torch.utils.data import Dataset

from src_lib.models_hub import Base
from src_lib.models_hub.trajectory.modules.frozen_backbone import ResNetBackbone


class VisualMotionEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super(VisualMotionEncoder, self).__init__()
        self.config = config
        net_params = self.config.trajectory_based.images.encoder

        self.embedding = nn.Sequential(
            nn.Linear(in_features=net_params.in_features, out_features=net_params.d_model // 2),
            nn.ReLU(),
            nn.Linear(in_features=net_params.d_model // 2, out_features=net_params.d_model // 2)
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

        self.noise_scalar = net_params.noise_scalar
        self.noise_embedding = nn.Sequential(
            nn.Linear(in_features=net_params.d_model + self.noise_scalar, out_features=net_params.mlp_scalar),
            getattr(nn, net_params.noise_activation)(),
            nn.Linear(in_features=net_params.mlp_scalar, out_features=net_params.d_model),
            getattr(nn, net_params.noise_activation)()
        )

    def forward(self, x, global_features, local_features):
        in_xy, in_dxdy = x['in_xy'], x['in_dxdy']
        out = self.embedding(in_dxdy)

        out = torch.cat((local_features, out), dim=-1)
        out = torch.cat((global_features.unsqueeze(0).repeat(1, out.shape[1], 1), out), dim=0)

        # add positional encoding
        # (S, B, E) -> (B, S, E)
        out = out.permute(1, 0, 2)
        out = self.positional_encoding(out)
        # (B, S, E) -> (S, B, E)
        out = out.permute(1, 0, 2)

        out = self.encoder(out)

        seq_len, batch_size, _ = out.shape
        out = torch.cat((out, torch.randn((seq_len, batch_size, self.noise_scalar)).to(out)), dim=-1)
        if self.noise_scalar:
            out = self.noise_embedding(out)
        return out


class VisualMotionDecoder(nn.Module):
    def __init__(self, config: DictConfig):
        super(VisualMotionDecoder, self).__init__()
        self.config = config
        net_params = self.config.trajectory_based.images.decoder

        self.seq_len = net_params.seq_len

        self.embedding = nn.Sequential(
            nn.Linear(in_features=net_params.in_features, out_features=net_params.d_model // 2),
            nn.ReLU(),
            nn.Linear(in_features=net_params.d_model // 2, out_features=net_params.d_model // 2)
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

    def forward(self, x, encoder_out, global_features, local_features):
        in_xy, in_dxdy = x['in_xy'], x['in_dxdy']

        last_obs_vel = self.embedding(in_dxdy[-1, None, ...])
        motion_fused_features = torch.cat((local_features[-1, None, ...], last_obs_vel), dim=-1)
        motion_fused_features = torch.cat((global_features.unsqueeze(0).repeat(1, encoder_out.shape[1], 1),
                                           motion_fused_features), dim=0)

        d_o = motion_fused_features.clone()

        for _ in range(self.seq_len // 2 - 1):
            # add positional encoding
            # (S, B, E) -> (B, S, E)
            d_o = d_o.permute(1, 0, 2)
            d_o = self.positional_encoding(d_o)
            # (B, S, E) -> (S, B, E)
            d_o = d_o.permute(1, 0, 2)

            d_o = self.decoder(d_o, encoder_out)
            d_o = torch.cat((motion_fused_features, d_o))

        out = d_o[2:, ...]
        return out


class AgentCollector(Base):
    def __init__(
            self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
            encoder: nn.Module, decoder: nn.Module, global_image_module: nn.Module,
            local_image_module: nn.Module, desired_output_shape: Tuple[int, int] = None,
            loss_function: nn.Module = None, additional_loss_functions: List[nn.Module] = None,
            collate_fn: Optional[Callable] = None, ):
        super(AgentCollector, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        net_params = self.config.trajectory_based.images
        if encoder is None:
            encoder = VisualMotionEncoder(config=self.config)
        self.encoder = encoder

        if decoder is None:
            decoder = VisualMotionDecoder(config=self.config)
        self.decoder = decoder

        if global_image_module is None:
            global_image_module = ResNetBackbone.from_config(
                config=self.config, resnet_depth=34, froze_bn=True, out_channel=net_params.globall.out_channels)
        self.global_image_module = global_image_module

        if local_image_module is None:
            local_image_module = ResNetBackbone.from_config(
                config=self.config, resnet_depth=18, froze_bn=False, out_channel=net_params.local.out_channels)
        self.local_image_module = local_image_module

    @classmethod
    def from_config(
            cls, config: DictConfig, train_dataset: Dataset = None, val_dataset: Dataset = None,
            encoder: nn.Module = None, decoder: nn.Module = None,
            global_image_module: nn.Module = None, local_image_module: nn.Module = None,
            desired_output_shape: Tuple[int, int] = None, additional_loss_functions: nn.Module = None,
            loss_function: nn.Module = None, collate_fn: Optional[Callable] = None):
        return AgentCollector(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            encoder=encoder, decoder=decoder, global_image_module=global_image_module,
            local_image_module=local_image_module, desired_output_shape=desired_output_shape,
            loss_function=loss_function, additional_loss_functions=additional_loss_functions,
            collate_fn=collate_fn
        )

    @staticmethod
    def get_patches(frames, locations):
        w, h = 64, 64
        patches = []
        for idx, loc in enumerate(locations):
            boxes_cxcywh = torch.stack([torch.tensor([x, y, w, h]) for (x, y) in loc])
            boxes_ijwh = torchvision.ops.box_convert(boxes_cxcywh, 'cxcywh', 'xywh')
            patches.append(torch.stack([tvf.F.crop(frames[idx], i, j, h, w)
                                        for (i, j, w, h) in boxes_ijwh.to(dtype=torch.uint8).tolist()]))
        return torch.stack(patches)

    def forward(self, x):
        global_features = self.global_image_module(x['global_image'])
        patches = self.get_patches(x['frames'], x['locations'])
        seq_len, patch_num, c, h, w = patches.shape

        patches = patches.view(-1, c, h, w)
        local_features = self.local_image_module(patches)
        local_features = local_features.view(seq_len, patch_num, -1)

        encoder_out = self.encoder(x, global_features, local_features)
        decoder_out = self.decoder(x, encoder_out, global_features, local_features)

        d_seq_len, batch_size, _ = decoder_out.shape

        decoder_out = decoder_out.view(d_seq_len, batch_size, 1, h, w)
        return decoder_out


if __name__ == '__main__':
    conf = OmegaConf.merge(OmegaConf.load('../../../src/position_maps/config/model/model.yaml'),
                           OmegaConf.load('../../../src/position_maps/config/training/training.yaml'))
    inp = {
        'in_dxdy': torch.randn((7, 2, 2)),
        'in_xy': torch.randn((8, 2, 2)),
        'gt_xy': torch.randn((12, 2, 2)),
        'gt_dxdy': torch.randn((12, 2, 2)),
        'ratio': torch.tensor([1, 1, 1]),
        'global_image': torch.randn((1, 3, 840, 720)),
        'frames': torch.randn((7, 3, 840, 720)),
        'locations': torch.cat((torch.randint(high=700, size=(2, 1)), torch.randint(high=800, size=(2, 1))),
                               dim=1).unsqueeze(0).repeat(7, 1, 1),
    }
    m = AgentCollector.from_config(config=conf)
    o = m(inp)
    print()
