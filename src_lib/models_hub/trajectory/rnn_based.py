from typing import Tuple, List, Optional, Callable, Union

import torch
from mmedit.models import GANLoss
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from baselinev2.stochastic.losses import cal_fde, cal_ade
from baselinev2.stochastic.model_modules import preprocess_dataset_elements, make_mlp
from src_lib.models_hub import Base, init_weights


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

    def _one_step(self, batch):
        target = batch['gt_xy']

        out = self(batch)
        pred = out['out_xy']

        loss = self.calculate_loss(pred, target, batch['ratio'])
        ade, fde = self.calculate_metrics(pred, target, self.config.tp_module.metrics.mode)
        return loss, ade, fde

    @staticmethod
    def calculate_metrics(pred, target, mode='sum'):
        ade = cal_ade(target, pred, mode=mode).squeeze()
        fde = cal_fde(target, pred, mode=mode).squeeze()
        return ade, fde

    def calculate_loss(self, pred, target, ratio):
        if self.config.tp_module.metrics.in_meters:
            out = torch.linalg.norm((pred - target), ord=2, dim=-1).mean(dim=0) * ratio.squeeze()
            return out.mean()
        else:
            return torch.linalg.norm((pred - target), ord=2, dim=-1).mean(dim=0).mean()

    def training_step(self, batch, batch_idx):
        loss, ade, fde = self._one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/ade_pixel', ade.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/fde_pixel', fde.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/ade', (ade * batch['ratio'].squeeze()).mean(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/fde', (fde * batch['ratio'].squeeze()).mean(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ade, fde = self._one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/ade_pixel', ade.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/fde_pixel', fde.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/ade', (ade * batch['ratio'].squeeze()).mean(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/fde', (fde * batch['ratio'].squeeze()).mean(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
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


class Discriminator(nn.Module):
    """Implementation of discriminator of GOAL GAN

       The model consists out of three main components:
       1. encoder of input trajectory
       2. encoder of
       3. Routing Module with visual soft-attention

       """

    def __init__(self, config, **kwargs):

        super().__init__()

        self.__dict__.update(locals())

        self.grad_status = True

        self.config = config
        net_params = self.config.trajectory_based.rnn.discriminator

        self.encoder_h_dim_d = net_params.encoder_h_dim_d
        self.embed_dim_scalar = net_params.embedding_dim_scalars
        self.dropout_disc = net_params.dropout
        self.mlp_scalar = net_params.mlp_scalar

        self.batch_first = net_params.batch_first

        self.social_attention = net_params.social_attention
        self.social_dim_scalar = net_params.social_dim_scalar if self.social_attention else 0

        self.encoder_observation = MotionEncoder(self.config)
        self.EncoderPrediction = EncoderPrediction(self.config)

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

        if self.batch_first:
            batch_size = in_xy.size(0)
        else:
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

    def __init__(self, config):
        super().__init__()

        self.config = config
        net_params = self.config.trajectory_based.rnn.discriminator.encoder_prediction

        self.rnn_type = net_params.rnn_type
        self.batch_first = net_params.batch_first

        rnn = getattr(nn, self.rnn_type)

        self.input_dim = net_params.in_features
        self.encoder_h_dim_d = net_params.encoder_h_dim_d
        self.embedding_dim = net_params.embedding_dim
        self.dropout = net_params.dropout

        self.batch_norm = net_params.batch_norm
        self.dropout_cnn = net_params.dropout_cnn
        self.mlp_dim = net_params.mlp_dim

        activation = ['leakyrelu', None]

        self.leakyrelu = nn.LeakyReLU()

        self.encoder = rnn(self.embedding_dim, self.encoder_h_dim_d, dropout=self.dropout, batch_first=self.batch_first)

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

        real_classifier_dims = [self.encoder_h_dim_d, self.mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation_list=activation,
            dropout=self.dropout)

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


class MotionEncoder(nn.Module):
    """MotionEncoder extracts dynamic features of the past trajectory and consists of an encoding LSTM network"""

    def __init__(self, config):
        """ Initialize MotionEncoder.
        Parameters.
            encoder_h_dim (int) - - dimensionality of hidden state
            input_dim (int) - - input dimensionality of spatial coordinates
            embedding_dim (int) - - dimensionality spatial embedding
            dropout (float) - - dropout in LSTM layer
        """
        super(MotionEncoder, self).__init__()

        self.config = config
        net_params = self.config.trajectory_based.rnn.discriminator.motion_encoder

        self.rnn_type = net_params.rnn_type
        self.batch_first = net_params.batch_first

        rnn = getattr(nn, self.rnn_type)

        self.encoder_h_dim = net_params.encoder_h_dim
        self.embedding_dim = net_params.embedding_dim
        self.input_dim = net_params.in_features

        if self.embedding_dim:
            self.spatial_embedding = nn.Linear(self.input_dim, self.embedding_dim)
            self.encoder = rnn(self.embedding_dim, self.encoder_h_dim, batch_first=self.batch_first)
        else:
            self.encoder = rnn(self.input_dim, self.encoder_h_dim, batch_first=self.batch_first)

    def init_hidden(self, batch, obs_traj):
        return (
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj),
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj)
        )

    def forward(self, obs_traj, state_tuple=None):
        """ Calculates forward pass of MotionEncoder
            Parameters:
                obs_traj (tensor) - - Tensor of shape (obs_len, batch, 2)
                state_tuple (tuple of tensors) - - Tuple with hidden state (1, batch, encoder_h_dim) and
                cell state tensor (1, batch, encoder_h_dim)
            Returns:
                output (tensor) - - Output of LSTM netwok for all time steps (obs_len, batch, encoder_h_dim)
                final_h (tensor) - - Final hidden state of LSTM network (1, batch, encoder_h_dim)
        """
        # Encode observed Trajectory
        if self.batch_first:
            batch = obs_traj.size(0)
        else:
            batch = obs_traj.size(1)

        if not state_tuple:
            state_tuple = self.init_hidden(batch, obs_traj)
        if self.embedding_dim:
            obs_traj = self.spatial_embedding(obs_traj)

        output, state = self.encoder(obs_traj, state_tuple)
        final_h = state[0]
        return output, final_h


class RNNGANBaseline(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 additional_loss_functions: List[nn.Module] = None, collate_fn: Optional[Callable] = None,
                 desc_loss_function: nn.Module = None):
        super(RNNGANBaseline, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function,
            additional_loss_functions=additional_loss_functions, collate_fn=collate_fn
        )
        self.generator = RNNBaselineGenerator(config=self.config, train_dataset=None, val_dataset=None)
        self.discriminator = Discriminator(config=self.config)
        if desc_loss_function is None:
            if self.config.trajectory_based.rnn.discriminator.use_gan_loss:
                desc_loss_function = GANLoss(
                    gan_type=self.config.trajectory_based.rnn.discriminator.gan_loss_type,
                    loss_weight=self.config.trajectory_based.rnn.discriminator.gan_loss_weight)
            else:
                desc_loss_function = nn.BCEWithLogitsLoss()
        self.desc_loss_function = desc_loss_function

        self.net_params = self.config.trajectory_based.rnn

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)

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
        self.log("train/discriminator/loss", disc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return disc_loss

    def _gen_step(self, x):
        loss, fake_loss, ade, fde = self._get_gen_loss(x)
        gen_loss = loss + fake_loss
        ade, _ = ade.min(dim=0)
        fde, _ = fde.min(dim=0)
        self.log('train/generator/loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/generator/loss", loss, on_epoch=True)
        self.log("train/generator/adv_loss", fake_loss, on_epoch=True)
        self.log('train/generator/ade_pixel', ade.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/generator/fde_pixel', fde.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/generator/ade', (ade * x['ratio'].squeeze()).mean(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/generator/fde', (fde * x['ratio'].squeeze()).mean(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return gen_loss

    def eval_step(self, x):
        x = self.get_k_batches(x, self.config.tp_module.datasets.batch_multiplier)
        batch_size = x["size"]

        with torch.no_grad():
            self.generator.eval()
            out = self.generator(x)
            self.generator.train()

        target = x['gt_xy']
        pred = out['out_xy']

        loss = self.calculate_loss(pred, target)
        loss = loss.view(self.config.tp_module.datasets.batch_multiplier, -1)
        loss, _ = loss.min(dim=0, keepdim=True)
        if self.config.tp_module.metrics.in_meters:
            loss = loss.squeeze() * x['ratio'].squeeze()
        loss = torch.mean(loss)

        ade, fde = cal_ade(target, pred, mode='raw'), cal_fde(target, pred, mode='raw')
        ade = ade.view(self.config.tp_module.datasets.batch_multiplier, batch_size)
        fde = fde.view(self.config.tp_module.datasets.batch_multiplier, batch_size)

        fde_min, _ = fde.min(dim=0)
        modes_caught = (fde < self.config.tp_module.datasets.mode_dist_threshold).float()

        return loss, modes_caught.mean().item(), ade, fde

    def validation_step(self, batch, batch_idx):
        loss, modes_caught, ade, fde = self.eval_step(batch)
        ade, _ = ade.min(dim=0)
        fde, _ = fde.min(dim=0)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/modes", modes_caught, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/ade_pixel', ade.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/fde_pixel', fde.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/ade', (ade * batch['ratio'].squeeze()).mean(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/fde', (fde * batch['ratio'].squeeze()).mean(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _get_disc_loss(self, x):
        # clamp loss?
        # Train with real
        real_pred = self.discriminator(x['in_xy'], x['in_dxdy'], x['gt_xy'], x['gt_dxdy'])
        real_gt = torch.ones_like(real_pred)
        real_loss = self.calculate_discriminator_loss(pred=real_pred, target=real_gt, is_real=True, is_disc=True)

        # Train with fake
        with torch.no_grad():
            self.generator.eval()
            fake_pred = self.generator(x)
            self.generator.train()

        fake_pred = self.discriminator(x['in_xy'], x['in_dxdy'], fake_pred['out_xy'], fake_pred['out_dxdy'])
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.calculate_discriminator_loss(pred=fake_pred, target=fake_gt, is_real=False, is_disc=True)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _get_gen_loss(self, x):
        x = self.get_k_batches(x, self.config.tp_module.datasets.batch_multiplier)
        batch_size = x["size"]

        out = self.generator(x)

        target = x['gt_xy']
        pred = out['out_xy']

        fake_pred = self.discriminator(x['in_xy'], x['in_dxdy'], out['out_xy'], out['out_dxdy'])
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.calculate_discriminator_loss(pred=fake_pred, target=fake_gt, is_real=False, is_disc=False)

        loss = self.calculate_loss(pred, target)
        loss = loss.view(self.config.tp_module.datasets.batch_multiplier, -1)
        loss, _ = loss.min(dim=0, keepdim=True)
        if self.config.tp_module.metrics.in_meters:
            loss = loss.squeeze() * x['ratio'].squeeze()
        loss = torch.mean(loss)

        ade, fde = cal_ade(target, pred, mode='raw'), cal_fde(target, pred, mode='raw')
        ade = ade.view(self.config.tp_module.datasets.batch_multiplier, batch_size)
        fde = fde.view(self.config.tp_module.datasets.batch_multiplier, batch_size)

        return loss, fake_loss, ade, fde

    def calculate_loss(self, pred, target):
        return self.l2_loss(pred, target)

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

    @staticmethod
    def l2_loss(pred_traj, pred_traj_gt, mode='average', typz="mse"):
        seq_len, batch, _ = pred_traj.size()
        d_Traj = pred_traj_gt - pred_traj

        if typz == "mse":
            loss = torch.linalg.norm(d_Traj, ord=2, dim=-1)
            # loss = torch.norm((d_Traj), 2, -1)
        elif typz == "average":
            loss = ((torch.norm(d_Traj, 2, -1)) + (torch.norm(d_Traj[-1], 2, -1))) / 2.
        else:
            raise AssertionError('Mode {} must be either mse or  average.'.format(typz))

        if mode == 'sum':
            return torch.sum(loss)
        elif mode == 'average':
            return torch.mean(loss, dim=0)
        elif mode == 'raw':
            return loss.sum(dim=0)

    @staticmethod
    def get_k_batches(batch, k):
        new_batch = {}
        for name, data in batch.items():
            if name in ["in_xy", "in_dxdy", "gt_xy", "gt_dxdy"]:
                new_batch[name] = data.repeat(1, k, 1).clone()
            elif name in ["gt_frames", "in_frames", "in_tracks", "gt_tracks"]:
                new_batch[name] = data.squeeze().repeat(k, 1).clone()
            else:
                new_batch[name] = data
        new_batch.update({'size': batch['in_xy'].shape[1]})
        return new_batch

    @staticmethod
    def calculate_metrics(pred, target, mode='sum'):
        ade = cal_ade(target, pred, mode=mode)
        fde = cal_fde(target, pred, mode=mode)
        return ade, fde

    def calculate_discriminator_loss(self, pred, target=None, is_real=True, is_disc=False):
        if self.config.trajectory_based.rnn.discriminator.use_gan_loss:
            return self.desc_loss_function(input=pred, target_is_real=is_real, is_disc=is_disc)
        return self.desc_loss_function(pred, target)


if __name__ == '__main__':
    m = RNNBaselineGenerator(OmegaConf.load('../../../src/position_maps/config/model/model.yaml'), None, None)
    inp = {
        'in_dxdy': torch.randn((7, 2, 2)),
        'in_xy': torch.randn((8, 2, 2))
    }
    o = m(inp)

    m2 = Discriminator(OmegaConf.load('../../../src/position_maps/config/model/model.yaml'))
    o2 = m2(inp['in_xy'], inp['in_dxdy'], o['out_xy'], o['out_dxdy'])
    print()
