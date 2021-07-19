from typing import Callable, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorchvideo.layers import PositionalEncoding
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from baselinev2.stochastic.losses import cal_ade, cal_fde
from src_lib.models_hub import Base, BaseGAN


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
        self.positional_encoding = PositionalEncoding(embed_dim=net_params.d_model, seq_len=self.seq_len)

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
        ade, fde = self.calculate_metrics(pred, target, self.config.tp_module.metrics.mode)
        return loss, ade, fde

    @staticmethod
    def calculate_metrics(pred, target, mode='sum'):
        ade = cal_ade(target, pred, mode=mode)
        fde = cal_fde(target, pred, mode=mode)
        return ade, fde

    def calculate_loss(self, pred, target):
        return torch.linalg.norm((pred - target), ord=2, dim=0).mean(dim=0).mean()

    def training_step(self, batch, batch_idx):
        loss, ade, fde = self._one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/ade', ade, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/fde', fde, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ade, fde = self._one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/ade', ade, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/fde', fde, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.config.tp_module.loader.batch_size,
            shuffle=self.config.tp_module.loader.shuffle, num_workers=self.config.tp_module.loader.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.tp_module.loader.pin_memory,
            drop_last=self.config.tp_module.loader.drop_last)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.tp_module.loader.batch_size * self.config.tp_module.loader.val_batch_size_factor,
            shuffle=False, num_workers=self.config.tp_module.loader.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.config.tp_module.loader.pin_memory,
            drop_last=self.config.tp_module.loader.drop_last)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.tp_module.optimizer.lr,
                               weight_decay=self.config.tp_module.optimizer.weight_decay,
                               amsgrad=self.config.tp_module.optimizer.amsgrad)
        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(opt,
                                               patience=self.config.tp_module.scheduler.patience,
                                               verbose=self.config.tp_module.scheduler.verbose,
                                               factor=self.config.tp_module.scheduler.factor,
                                               min_lr=self.config.tp_module.scheduler.min_lr),
                'monitor': self.config.tp_module.scheduler.monitor,
                'interval': self.config.tp_module.scheduler.interval,
                'frequency': self.config.tp_module.scheduler.frequency
            }]
        return [opt], schedulers


class TransformerMotionGenerator(nn.Module):
    def __init__(self, config: DictConfig):
        super(TransformerMotionGenerator, self).__init__()
        self.config = config
        net_params = self.config.trajectory_based.transformer.encoder

        self.motion_encoder = TransformerMotionEncoder(config=self.config)
        self.noise_embedding = nn.Sequential(
            nn.Linear(in_features=net_params.d_model * 2, out_features=net_params.d_model)
        )
        self.motion_decoder = TransformerMotionDecoder(config=self.config)

    def forward(self, x):
        out = self.motion_encoder(x)
        out = torch.cat((out, torch.randn_like(out)), dim=-1)
        out = self.noise_embedding(out)
        out = self.motion_decoder(x, out)
        return out


class TransformerMotionDiscriminator(nn.Module):
    def __init__(self, config: DictConfig):
        super(TransformerMotionDiscriminator, self).__init__()
        self.config = config

        net_params = self.config.trajectory_based.transformer.discriminator

        self.motion_encoder = TransformerMotionEncoder(self.config)
        
        self.embedding = nn.Sequential(
            nn.Linear(in_features=net_params.in_features, out_features=net_params.d_model // 2),
            nn.ReLU(),
            nn.Linear(in_features=net_params.d_model // 2, out_features=net_params.d_model)
        )
        self.motion_decoder = nn.TransformerDecoder(
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, net_params.d_model))
        self.positional_encoding = PositionalEncoding(embed_dim=net_params.d_model, seq_len=net_params.seq_len + 1)
        # +1 for cls token

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.config.trajectory_based.transformer.decoder.d_model,
                      out_features=self.config.trajectory_based.transformer.decoder.d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.config.trajectory_based.transformer.decoder.d_model // 2,
                      out_features=1))

    def forward(self, x, gt_x):
        enc_out = self.motion_encoder(x)

        gt_x = self.embedding(gt_x)
        
        # add cls token to act as classifier head
        cls_tokens = self.cls_token.expand(-1, gt_x.shape[1], -1)
        gt_x = torch.cat((cls_tokens, gt_x), dim=0)

        # add positional encoding
        # (S, B, E) -> (B, S, E)
        gt_x = gt_x.permute(1, 0, 2)
        gt_x = self.positional_encoding(gt_x)
        # (B, S, E) -> (S, B, E)
        gt_x = gt_x.permute(1, 0, 2)
        
        dec_out = self.motion_decoder(gt_x, enc_out)

        # out = self.classifier(dec_out.mean(0))  # mean over all time-steps - can take 1st or last ts as well?
        out = self.classifier(dec_out[0, ...])  # mean over all time-steps - can take 1st or last ts as well?
        return out


class TrajectoryGANTransformer(BaseGAN):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None,
                 desc_loss_function: nn.Module = nn.BCEWithLogitsLoss(),
                 collate_fn: Optional[Callable] = None):
        super(TrajectoryGANTransformer, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.config = config
        self.desc_loss_function = desc_loss_function

        self.net_params = self.config.trajectory_based.transformer

        self.generator = TransformerMotionGenerator(self.config)
        self.discriminator = TransformerMotionDiscriminator(self.config)

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.net_params.discriminator.lr,
                                    weight_decay=self.net_params.discriminator.weight_decay,
                                    amsgrad=self.net_params.discriminator.weight_decay)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.net_params.generator.lr,
                                   weight_decay=self.net_params.generator.weight_decay,
                                   amsgrad=self.net_params.generator.weight_decay)
        return [opt_disc, opt_gen], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(batch)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(batch)

        return result

    def _disc_step(self, x):
        disc_loss = self._get_disc_loss(x)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_step(self, x):
        gen_loss = self._get_gen_loss(x)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def _get_disc_loss(self, x):
        # Train with real
        real_pred = self.discriminator(x, x['gt_dxdy'])
        real_gt = torch.ones_like(real_pred)
        real_loss = self.desc_loss_function(real_pred, real_gt)

        # Train with fake
        with torch.no_grad():
            fake_pred = self.generator(x)

        fake_pred = self.discriminator(x, fake_pred['out_dxdy'])
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.desc_loss_function(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _get_gen_loss(self, x):
        out = self.generator(x)

        target = x['gt_xy']
        pred = out['out_xy']
        loss = self.calculate_loss(pred, target)

        return loss

    @staticmethod
    def calculate_loss(pred, target):
        return torch.linalg.norm((pred - target), ord=2, dim=0).mean(dim=0).mean()


if __name__ == '__main__':
    conf = OmegaConf.load('../../../src/position_maps/config/model/model.yaml')
    inp = {
        'in_dxdy': torch.randn((7, 2, 2)),
        'in_xy': torch.randn((8, 2, 2)),
        'gt_xy': torch.randn((12, 2, 2)),
        'gt_dxdy': torch.randn((12, 2, 2))
    }
    m = TrajectoryGANTransformer(conf, None, None)
    o = m._get_disc_loss(inp)
    # m = TrajectoryTransformer(conf, None, None)
    # o = m._one_step(inp)
    print()
