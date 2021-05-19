import itertools
import os
from argparse import Namespace
from typing import Optional, Callable

import hydra
import psutil
import torch
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.optim import Adam
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselinev2.exceptions import InvalidFrameException
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.dataset import get_all_dataset, get_all_dataset_test_split
from baselinev2.nn.models import ConstantLinearBaseline
from baselinev2.plot_utils import plot_trajectory_alongside_frame, plot_trajectories, \
    plot_trajectory_alongside_frame_stochastic, plot_and_compare_trajectory_four_way_stochastic
from baselinev2.stochastic.losses import l2_loss, GANLoss, cal_ade, cal_fde, cal_ade_stochastic, cal_fde_stochastic, \
    cal_ade_fde_stochastic, cal_ade_fde_stochastic_worse
from baselinev2.stochastic.model_modules import BaselineGenerator, Discriminator, preprocess_dataset_elements
from baselinev2.stochastic.utils import get_batch_k, re_im
from baselinev2.stochastic.viz import visualize_traj_probabilities
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger('baselinev2.stochastic.model')

seed_everything(42)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)


def get_model(cfg):
    """
    Loads Generator and Discriminator
    """

    generator = BaselineGenerator(**cfg.generator)
    generator.apply(init_weights)
    discriminator = Discriminator(**cfg.discriminator)
    discriminator.apply(init_weights)
    return generator, discriminator


class BaselineGAN(pl.LightningModule):
    def __init__(self, hparams: DictConfig = None, args: Namespace = None, loss_fns=None):
        super().__init__()

        # self.automatic_optimization = False
        self.args = args
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.generator, self.discriminator = get_model(self.hparams)
        print(self.generator)
        print(self.discriminator)
        # init loss functions
        self.loss_fns = loss_fns if loss_fns else {'L2': l2_loss,  # L2 loss
                                                   'ADV': GANLoss(hparams.gan_mode),  # adversarial Loss
                                                   }
        # init loss weights

        self.loss_weights = {'L2': hparams.w_L2,
                             'ADV': hparams.w_ADV,  # adversarial Loss
                             }

        self.current_batch_idx = -1
        self.plot_val = hparams.plot_val

        if self.hparams.batch_size_scheduler:
            self.batch_size = self.hparams.batch_size_scheduler
        else:
            self.batch_size = self.hparams.batch_size
        # self.batch_size = self.hparams.batch_size

        self.gsteps_yet = 0
        self.dsteps_yet = 0

        # batch size scheduler
        self.bs_scheduler_bs = self.hparams.batch_size_scheduler
        self.bs_scheduler_factor = self.hparams.batch_size_scheduler_factor or 2
        self.bs_scheduler_patience = self.hparams.patience or 4
        self.bs_scheduler_max_bs = self.hparams.max_batch_size or 2048
        self.bs_scheduler_mode = self.hparams.batch_size_scheduler_mode or 'min'
        self.bs_scheduler_monitor_val = self.hparams.monitor_val or 'val_loss'
        self.bs_scheduler_cur_metric = False
        self.current_batch_size = self.hparams.batch_size_scheduler

        if self.hparams.patience and self.hparams.batch_size_scheduler_mode:
            self.bs_scheduler_current_count = self.hparams.patience * 1.
            if self.hparams.batch_size_scheduler_mode not in ["min", "max"]:
                assert False, "Variable for mode '{}' not valid".format(self.hparams.batch_size_scheduler_mode)
            if self.hparams.max_batch_size > self.hparams.batch_size_scheduler:
                self.bs_scheduler_active = True
            else:
                self.bs_scheduler_active = False
        else:
            self.bs_scheduler_current_count = self.bs_scheduler_patience * 1.
            if self.bs_scheduler_mode not in ["min", "max"]:
                assert False, "Variable for mode '{}' not valid".format(self.bs_scheduler_mode)
            if self.bs_scheduler_max_bs > self.hparams.batch_size_scheduler:
                self.bs_scheduler_active = True
            else:
                self.bs_scheduler_active = False

    def setup_datasets(self):
        root = self.hparams.unsupervised_root if self.hparams.use_generated_dataset else self.hparams.supervised_root
        split_name = self.hparams.unsupervised_split \
            if self.hparams.use_generated_dataset else self.hparams.supervised_split
        self.train_dset, self.val_dset = get_all_dataset(get_generated=self.hparams.use_generated_dataset, root=root,
                                                         split_name=split_name)

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    def setup_test_dataset(self):
        root = self.hparams.unsupervised_root if self.hparams.use_generated_dataset else self.hparams.supervised_root
        split_name = self.hparams.unsupervised_split \
            if self.hparams.use_generated_dataset else self.hparams.supervised_split
        self.test_dset = get_all_dataset_test_split(self.hparams.use_generated_dataset, root=root,
                                                    split_name=split_name)

    def test_dataloader(self):
        self.setup_test_dataset()

        return DataLoader(

            self.test_dset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        optimizers = self.optimizers()
        batch = preprocess_dataset_elements(batch, batch_first=False, is_generated=self.hparams.use_generated_dataset)

        self.batch_idx = batch_idx
        self.generator.gen()
        self.logger.experiment.add_scalar('train/CPU Usage', psutil.cpu_percent(), self.global_step)

        if self.device.type != 'cpu':
            self.logger.experiment.add_scalar('train/GPU Usage',
                                              torch.cuda.get_device_properties(self.device).total_memory,
                                              self.global_step)
        # self.plot_val = True
        # if self.gsteps and optimizer_idx == 0 and self.current_batch_idx != batch_idx:
        if self.gsteps and self.current_batch_idx != batch_idx:

            output = self.generator_step(batch)

            self.output = output
            self.gsteps_yet += 1
            # return output
        elif self.dsteps and self.current_batch_idx != batch_idx:
            # elif self.dsteps and optimizer_idx == 1 and self.current_batch_idx != batch_idx:

            output = self.discriminator_step(batch)

            self.output = output
            self.dsteps_yet += 1
        # return output
        # else:
        #     return self.output

        # # if self.current_batch_idx != self.batch_idx and (
        # #         ((optimizer_idx == 0) and self.gsteps) or ((optimizer_idx == 1) and self.dsteps)):
        # if self.current_batch_idx != self.batch_idx and (self.gsteps or self.dsteps):
        #     self.output['loss'].backward()

        if self.gsteps and self.current_batch_idx != batch_idx:
            optimizer = optimizers[0]
            optimizer.zero_grad()
            self.manual_backward(self.output['loss'])

            self.gsteps -= 1
            if not self.gsteps:
                if self.discriminator:
                    self.dsteps = self.hparams.d_steps
                else:
                    self.gsteps = self.hparams.g_steps
            self.current_batch_idx = batch_idx
            optimizer.step()

        # update discriminator opt every 4 steps
        if self.dsteps and self.current_batch_idx != batch_idx:
            optimizer = optimizers[1]
            optimizer.zero_grad()
            self.manual_backward(self.output['loss'])

            self.dsteps -= 1
            if not self.dsteps:
                self.gsteps = self.hparams.g_steps
            self.current_batch_idx = batch_idx
            optimizer.step()

    def test(self, batch):
        self.generator.test()
        with torch.no_grad():
            out = self.generator(batch)
        return out

    def forward(self, batch):
        return self.generator(batch)

    def manual_backward(self, loss: Tensor, optimizer: Optional[Optimizer] = None, *args, **kwargs) -> None:
        # if self.current_batch_idx != self.batch_idx and (
        #         ((optimizer_idx == 0) and self.gsteps) or ((optimizer_idx == 1) and self.dsteps)):
        if self.current_batch_idx != self.batch_idx and (self.gsteps or self.dsteps):
            loss.backward()

    def generator_step(self, batch):

        """Generator optimization step.
        Args:
            batch: Batch from the data loader.

        Returns:
            discriminator loss on fake
            norm loss on trajectory
            kl loss
        """

        # init loss and loss dict
        tqdm_dict = {}
        total_loss = 0.

        # ade_sum, fde_sum = [], []
        # ade_sum_pixel, fde_sum_pixel = [], []

        # get k times batch
        batch = get_batch_k(batch, self.hparams.best_k)

        batch_size = batch["size"]

        generator_out = self.generator(batch)

        if self.hparams.absolute:
            l2 = self.loss_fns["L2"](
                batch["gt_xy"],
                generator_out["out_xy"],
                mode='average',
                type="mse")
        else:
            l2 = self.loss_fns["L2"](
                batch["gt_dxdy"],
                generator_out["out_dxdy"],
                mode='raw',
                type="mse")

        ade_error = cal_ade(
            batch["gt_xy"], generator_out["out_xy"], mode='raw'
        )

        fde_error = cal_fde(
            batch["gt_xy"], generator_out["out_xy"], mode='raw'
        )

        ade_error = ade_error.view(self.hparams.best_k, batch_size)

        fde_error = fde_error.view(self.hparams.best_k, batch_size)

        # get pixel ratios
        # ratios = []
        # for img in batch["scene_img"]:
        #     ratios.append(torch.tensor(img["ratio"]))
        # ratios = torch.stack(ratios).to(self.device)

        # for idx, (start, end) in enumerate(batch["seq_start_end"]):
        #     ade_error_sum = torch.sum(ade_error[:, start:end], dim=1)
        #     fde_error_sum = torch.sum(fde_error[:, start:end], dim=1)
        #
        #     ade_sum_scene, id_scene = ade_error_sum.min(dim=0, keepdims=True)
        #     fde_sum_scene, _ = fde_error_sum.min(dim=0, keepdims=True)
        #
        #     ade_sum.append(ade_sum_scene / (self.hparams.pred_len * (end - start)))
        #     fde_sum.append(fde_sum_scene / (end - start))
        #
        #     ade_sum_pixel.append(ade_sum_scene / (self.hparams.pred_len * (end - start) * ratios[idx]))
        #     fde_sum_pixel.append(fde_sum_scene / (ratios[idx] * (end - start)))

        # tqdm_dict["ADE_train"] = torch.mean(torch.stack(ade_sum))
        # tqdm_dict["FDE_train"] = torch.mean(torch.stack(fde_sum))
        #
        # tqdm_dict["ADE_pixel_train"] = torch.mean(torch.stack(ade_sum_pixel))
        # tqdm_dict["FDE_pixel_train"] = torch.mean(torch.stack(fde_sum_pixel))

        tqdm_dict["ADE_train"] = torch.mean(ade_error * batch['ratio'])
        tqdm_dict["FDE_train"] = torch.mean(fde_error * batch['ratio'])

        tqdm_dict["ADE_pixel_train"] = torch.mean(ade_error)
        tqdm_dict["FDE_pixel_train"] = torch.mean(fde_error)

        # count trajectories crashing into the 'wall'
        # if any(batch["occupancy"]):
        #     wall_crashes = [0]  # crashIntoWall(generator_out["out_xy"].detach().cpu(), batch["occupancy"])
        # else:
        #     wall_crashes = [0]
        # tqdm_dict["feasibility_train"] = torch.tensor(1 - np.mean(wall_crashes))

        l2 = l2.view(self.hparams.best_k, -1)

        loss_l2, _ = l2.min(dim=0, keepdim=True)
        loss_l2 = torch.mean(loss_l2)

        loss_l2 = self.loss_weights["L2"] * loss_l2
        tqdm_dict["L2_train"] = loss_l2
        total_loss += loss_l2

        # if self.generator.global_vis_type == "goal":
        #     target_reshaped = batch["prob_mask"][:batch_size].view(batch_size, -1)
        #     output_reshaped = generator_out["y_scores"][:batch_size].view(batch_size, -1)
        #
        #     _, targets = target_reshaped.max(dim=1)
        #
        #     loss_gce = self.loss_weights["GCE"] * self.loss_fns["GCE"](output_reshaped, targets)
        #
        #     total_loss += loss_gce
        #     tqdm_dict["GCE_train"] = loss_gce
        #
        #     final_end = torch.sum(generator_out["out_dxdy"], dim=0, keepdim=True)
        #     final_end_gt = torch.sum(batch["gt_dxdy"], dim=0, keepdim=True)
        #
        #     final_pos = generator_out["final_pos"]
        #
        #     goal_error = self.loss_fns["G"](final_pos.detach(), final_end_gt)
        #     goal_error = goal_error.view(self.hparams.best_k, -1)
        #     _, id_min = goal_error.min(dim=0, keepdim=False)
        #     # id_min*=torch.range(0, len(id_min))*10
        #
        #     final_pos = final_pos.view(self.hparams.best_k, batch["size"], -1)
        #     final_end = final_end.view(self.hparams.best_k, batch["size"], -1)
        #
        #     final_pos = torch.cat([final_pos[id_min[k], k].unsqueeze(0)
        #                            for k in range(final_pos.size(1))]).unsqueeze(0)
        #     final_end = torch.cat([final_end[id_min[k], k].unsqueeze(0)
        #                            for k in range(final_end.size(1))]).unsqueeze(0)
        #
        #     loss_G = self.loss_weights["G"] * torch.mean(self.loss_fns["G"](final_pos.detach(), final_end, mode='raw'))
        #
        #     total_loss += loss_G
        #     tqdm_dict["G_train"] = loss_G

        traj_fake = generator_out["out_xy"][:, :batch_size]
        traj_fake_rel = generator_out["out_dxdy"][:, :batch_size]

        # traj_fake = generator_out["out_xy"]
        # traj_fake_rel = generator_out["out_dxdy"]

        # if self.generator.rm_vis_type == "attention":
        #     image_patches = generator_out["image_patches"][:, :batch_size]
        # else:
        #     image_patches = None

        image_patches = None
        fake_scores = self.discriminator(in_xy=batch["in_xy"][:, :batch_size],
                                         in_dxdy=batch["in_dxdy"][:, :batch_size],
                                         out_xy=traj_fake,
                                         out_dxdy=traj_fake_rel,
                                         images_patches=image_patches)

        # fake_scores = self.discriminator(in_xy=batch["in_xy"],
        #                                  in_dxdy=batch["in_dxdy"],
        #                                  out_xy=traj_fake,
        #                                  out_dxdy=traj_fake_rel,
        #                                  images_patches=None)

        loss_adv = self.loss_weights["ADV"] * self.loss_fns["ADV"](fake_scores, True).clamp(min=0)

        total_loss += loss_adv
        tqdm_dict["ADV_train"] = loss_adv

        tqdm_dict["all_loss"] = total_loss
        for key, loss in tqdm_dict.items():
            self.logger.experiment.add_scalar('train/{}'.format(key), loss, self.global_step)

        return {"loss": total_loss}

    def discriminator_step(self, batch):

        """Discriminator optimization step.

        Args:
            batch: Batch from the data loader.

        Returns:
            discriminator loss on fake
            discriminator loss on real
        """
        # init loss and loss dict
        tqdm_dict = {}
        total_loss = 0.

        self.generator.gen()
        self.discriminator.grad(True)

        with torch.no_grad():
            out = self.generator(batch)

        traj_fake = out["out_xy"]
        traj_fake_rel = out["out_dxdy"]

        # if self.generator.rm_vis_type == "attention":
        #     image_patches = out["image_patches"]
        # else:
        #     image_patches = None

        image_patches = None
        dynamic_fake = self.discriminator(in_xy=batch["in_xy"],
                                          in_dxdy=batch["in_dxdy"],
                                          out_xy=traj_fake,
                                          out_dxdy=traj_fake_rel,
                                          images_patches=image_patches)

        # if self.generator.rm_vis_type == "attention":
        #     image_patches = batch["local_patch"].permute(1, 0, 2, 3, 4)
        # else:
        #     image_patches = None

        image_patches = None
        dynamic_real = self.discriminator(in_xy=batch["in_xy"],
                                          in_dxdy=batch["in_dxdy"],
                                          out_xy=batch["gt_xy"],
                                          out_dxdy=batch["gt_dxdy"],
                                          images_patches=image_patches)

        disc_loss_real_dynamic = self.loss_fns["ADV"](dynamic_real, True).clamp(min=0)
        disc_loss_fake_dynamic = self.loss_fns["ADV"](dynamic_fake, False).clamp(min=0)

        disc_loss = disc_loss_real_dynamic + disc_loss_fake_dynamic

        tqdm_dict = {"D_train": disc_loss,
                     "D_real_train": disc_loss_real_dynamic,
                     "D_fake_train": disc_loss_fake_dynamic}

        for key, loss in tqdm_dict.items():
            self.logger.experiment.add_scalar('train/{}'.format(key), loss, self.global_step)
        return {
            'loss': disc_loss
        }

    """########## VISUALIZATION ##########"""

    def visualize_results(self, batch, out):

        background_image = batch["scene_img"][0]["scaled_image"].copy()

        inp = batch["in_xy"]
        gt = batch["gt_xy"]
        pred = out["out_xy"]
        pred = pred.view(pred.size(0), self.hparams.best_k_val, -1, pred.size(-1))

        y = out["y_map"]
        y_softmax = out["y_softmax"]

        image = visualize_traj_probabilities(
            input_trajectory=inp.cpu()[:, 0],
            gt_trajectory=None,
            prediction_trajectories=pred.cpu()[:, :, 0],
            background_image=background_image,
            img_scaling=self.val_dset.img_scaling,
            scaling_global=self.val_dset.scaling_global,
            grid_size=20,
            y_softmax=y_softmax,
            y=y,
            global_patch=re_im(batch["global_patch"][0]).cpu().numpy(),
            probability_mask=batch["prob_mask"][0][0].cpu().numpy(),
            grid_size_in_global=self.val_dset.grid_size_in_global

        )

        self.logger.experiment.add_image(f'Trajectories', image, self.current_epoch)

    """########## EVAL HELPERS ##########"""

    def eval_step(self, batch, best_k=10):
        batch = preprocess_dataset_elements(batch, batch_first=False, is_generated=self.hparams.use_generated_dataset)

        # ade_sum, fde_sum = [], []
        # ade_sum_pixel, fde_sum_pixel = [], []

        # get pixel ratios
        # ratios = []
        # for img in batch["scene_img"]:
        #     ratios.append(torch.tensor(img["ratio"]))
        # ratios = torch.stack(ratios).to(self.device)

        batch = get_batch_k(batch, best_k)
        batch_size = batch["size"]

        out = self.test(batch)

        if self.plot_val:
            self.plot_val = False
            self.visualize_results(batch, out)

        # FDE and ADE metrics
        ade_error = cal_ade(
            batch["gt_xy"], out["out_xy"], mode='raw'
        )

        fde_error = cal_fde(
            batch["gt_xy"], out["out_xy"], mode='raw'
        )

        ade_error = ade_error.view(best_k, batch_size)

        fde_error = fde_error.view(best_k, batch_size)

        # for idx, (start, end) in enumerate(batch["seq_start_end"]):
        #     ade_error_sum = torch.sum(ade_error[:, start:end], dim=1)
        #     fde_error_sum = torch.sum(fde_error[:, start:end], dim=1)
        #
        #     ade_sum_scene, id_scene = ade_error_sum.min(dim=0, keepdims=True)
        #     fde_sum_scene, _ = fde_error_sum.min(dim=0, keepdims=True)
        #
        #     ade_sum.append(ade_sum_scene / (self.hparams.pred_len * (end - start)))
        #     fde_sum.append(fde_sum_scene / (end - start))

        #     ade_sum_pixel.append(ade_sum_scene / (self.hparams.pred_len * (end - start) * ratios[idx]))
        #     fde_sum_pixel.append(fde_sum_scene / (ratios[idx] * (end - start)))

        # compute Mode Caughts metrics
        fde_min, _ = fde_error.min(dim=0)
        modes_caught = (fde_min < self.hparams.mode_dist_threshold).float()

        # if any(batch["occupancy"]):
        #
        #     wall_crashes = [0]  # crashIntoWall(out["out_xy"].cpu(), batch["occupancy"])
        # else:
        #     wall_crashes = [0]
        # return {"ade": ade_sum, "fde": fde_sum, "ade_pixel": ade_sum_pixel, "fde_pixel": fde_sum_pixel,
        #         "wall_crashes": wall_crashes, "modes_caught": modes_caught}
        return {'ade_pixel': ade_error, 'fde_pixel': fde_error, "modes_caught": modes_caught,
                'ade': ade_error * batch['ratio'], 'fde': fde_error * batch['ratio']}

    def collect_losses(self, outputs, mode="val", plot=True):

        ade = torch.stack(list(itertools.chain(*[x['ade'] for x in outputs]))).mean()
        fde = torch.stack(list(itertools.chain(*[x['fde'] for x in outputs]))).mean()
        ade_pixel = torch.stack(list(itertools.chain(*[x['ade_pixel'] for x in outputs]))).mean()
        fde_pixel = torch.stack(list(itertools.chain(*[x['fde_pixel'] for x in outputs]))).mean()
        mc_metric = torch.stack(list(itertools.chain(*[x["modes_caught"] for x in outputs]))).mean()

        # loss = ((fde + ade) / 2.)
        loss = torch.add(fde, ade).div(2.)
        logs = {'{}_loss'.format(mode): loss, 'ade_{}'.format(mode): ade.item(),
                "fde_{}".format(mode): fde.item(), "ade_pixel_{}".format(mode): ade_pixel.item(),
                "fde_pixel_{}".format(mode): fde_pixel.item(), "modes_c_{}".format(mode): mc_metric}
        # plot val
        if plot:
            for key, loss in logs.items():
                self.logger.experiment.add_scalar('{}/{}'.format(mode, key), loss, self.current_epoch)

        return {'{}_loss'.format(mode): loss, 'progress_bar': logs}

    """########## VALIDATION ##########"""

    def bs_scheduler(self, init_val_loss):

        self.current_batch_size = int(np.minimum(self.current_batch_size * self.bs_scheduler_factor,
                                                 self.bs_scheduler_max_bs))

        # set new batch_size
        self.batch_size = self.current_batch_size
        self.trainer.reset_train_dataloader(self)

        if self.bs_scheduler_monitor_val not in self.trainer.callback_metrics:
            self.trainer.callback_metrics.update({self.bs_scheduler_monitor_val: init_val_loss})

        if not self.bs_scheduler_cur_metric:
            self.bs_scheduler_cur_metric = self.trainer.callback_metrics[self.bs_scheduler_monitor_val]

        if self.bs_scheduler_active:
            if self.bs_scheduler_mode == "min":
                if self.trainer.callback_metrics[self.bs_scheduler_monitor_val] < self.bs_scheduler_cur_metric:
                    self.bs_scheduler_cur_metric = self.trainer.callback_metrics[self.bs_scheduler_monitor_val]
                    self.bs_scheduler_current_count = self.bs_scheduler_patience * 1
                    logger.info(f'Resetting patience to default value, current patience: '
                                f'{self.bs_scheduler_current_count}')
                else:
                    self.bs_scheduler_current_count -= 1
                    logger.info(f'Decreasing patience by 1, current patience: {self.bs_scheduler_current_count}')

            else:
                if self.trainer.callback_metrics[self.bs_scheduler_monitor_val] > self.bs_scheduler_cur_metric:
                    self.bs_scheduler_cur_metric = self.trainer.callback_metrics[self.bs_scheduler_monitor_val]
                    self.bs_scheduler_current_count = self.bs_scheduler_patience * 1
                    logger.info(f'Resetting patience to default value, current patience: '
                                f'{self.bs_scheduler_current_count}')
                else:
                    self.bs_scheduler_current_count -= 1
                    logger.info(f'Decreasing patience by 1, current patience: {self.bs_scheduler_current_count}')

            if self.bs_scheduler_current_count == 0:
                logger.info(f'Increasing batch size from : {self.current_batch_size}')
                self.current_batch_size = int(np.minimum(self.current_batch_size * self.bs_scheduler_factor,
                                                         self.bs_scheduler_max_bs))

                # set new batch_size
                self.batch_size = self.current_batch_size
                self.trainer.reset_train_dataloader(self)
                logger.info("SET BS TO {}".format(self.current_batch_size))
                self.bs_scheduler_current_count = self.bs_scheduler_patience * 1
                if self.current_batch_size >= self.bs_scheduler_max_bs:
                    self.bs_scheduler_active = False

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, self.hparams.best_k_val)

    def validation_epoch_end(self, outputs):
        collected_losses = self.collect_losses(outputs, mode="val")
        # init_val_loss = collected_losses['val_loss']
        # # logger.debug(f'Current val loss in trainer: {self.trainer.callback_metrics[self.bs_scheduler_monitor_val]}')
        # self.bs_scheduler(init_val_loss)
        return collected_losses

    """########## TESTING ##########"""

    def test_step(self, batch, batch_idx):
        output = self.eval_step(batch, self.hparams.best_k_test)
        return output

    def test_epoch_end(self, outputs):
        results = self.collect_losses(outputs, mode="test")

        torch.save(results["progress_bar"], os.path.join(self.logger.log_dir, "results.pt"))

        print(results)
        return results

    """########## OPTIMIZATION ##########"""

    # def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs):
    #     # condition that backward is not called when nth is passed
    #     if self.current_batch_idx != self.batch_idx and (
    #             ((optimizer_idx == 0) and self.gsteps) or ((optimizer_idx == 1) and self.dsteps)):
    #         loss.backward()

    # def optimizer_step(self, epoch: int = None,
    #                    batch_idx: int = None,
    #                    optimizer: Optimizer = None,
    #                    optimizer_idx: int = None,
    #                    optimizer_closure: Optional[Callable] = None,
    #                    on_tpu: bool = None,
    #                    using_native_amp: bool = None,
    #                    using_lbfgs: bool = None, ):
    #     # Step using d_loss or g_loss
    #     # update generator opt every 2 steps
    #     if self.gsteps and optimizer_idx == 0 and self.current_batch_idx != batch_idx:
    #         self.gsteps -= 1
    #         if not self.gsteps:
    #             if self.discriminator:
    #                 self.dsteps = self.hparams.d_steps
    #             else:
    #                 self.gsteps = self.hparams.g_steps
    #         self.current_batch_idx = batch_idx
    #         optimizer.step()
    #         optimizer.zero_grad()
    #
    #     # update discriminator opt every 4 steps
    #     if self.dsteps and optimizer_idx == 1 and self.current_batch_idx != batch_idx:
    #         self.dsteps -= 1
    #         if not self.dsteps:
    #             self.gsteps = self.hparams.g_steps
    #         self.current_batch_idx = batch_idx
    #         optimizer.step()
    #         optimizer.zero_grad()

    def configure_optimizers(self):
        opt_g = Adam(self.generator.parameters(), lr=self.hparams.lr_gen)
        opt_d = Adam(self.discriminator.parameters(), lr=self.hparams.lr_dis)

        # fixme
        # schedulers = []
        # if self.hparams.lr_scheduler_G:
        #     lr_scheduler_G = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler_G)(opt_g)
        #     schedulers.append(lr_scheduler_G)
        # else:
        #     schedulers.append(None)
        #
        # if self.hparams.lr_scheduler_D:
        #     lr_scheduler_D = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler_D)(opt_d)
        #     schedulers.append(lr_scheduler_D)
        # else:
        #     schedulers.append(None)

        self.gsteps = self.hparams.g_steps
        self.dsteps = 0
        self.setup_datasets()
        return [opt_g, opt_d]  # , schedulers


@hydra.main(config_path="config", config_name="config")
def debug_model(cfg):
    # tb_logger = TensorBoardLogger(".", name="{}".format(cfg.dataset_name))
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    from utils import BatchSizeScheduler
    bs_scheduler = BatchSizeScheduler(bs=cfg.batch_size_scheduler,
                                      max_bs=cfg.max_batch_size,
                                      patience=cfg.patience)

    if cfg.warm_restart.enable:
        checkpoint_root_path = f'{cfg.warm_restart.checkpoint.root}{cfg.warm_restart.checkpoint.path}' \
                               f'{cfg.warm_restart.checkpoint.version}/checkpoints/'
        hparams_path = f'{cfg.warm_restart.checkpoint.root}{cfg.warm_restart.checkpoint.path}' \
                       f'{cfg.warm_restart.checkpoint.version}/hparams.yaml'
        model_path = checkpoint_root_path + os.listdir(checkpoint_root_path)[0]
        logger.info(f'Resuming from : {model_path}')
        if cfg.warm_restart.custom_load:
            logger.info(f'Loading weights manually as custom load is {cfg.warm_restart.custom_load}')
            m = BaselineGAN(hparams=cfg)
            load_dict = torch.load(model_path)

            m.load_state_dict(load_dict['state_dict'])
            m.to(cfg.device)
            m.train()
            logger.info(f'Batch Size : {m.batch_size}')
            logger.info(f'LR Discriminator : {cfg.lr_dis}')
            logger.info(f'LR Generator : {cfg.lr_gen}')
        else:
            m = BaselineGAN.load_from_checkpoint(
                checkpoint_path=model_path,
                hparams_file=hparams_path,
                map_location='cuda:0')
            m.batch_size = cfg.batch_size
            logger.info(f'Batch Size : {m.batch_size}')
    else:
        m = BaselineGAN(hparams=cfg)
        # m.setup_datasets()

        # trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
        #                      callbacks=[checkpoint_callback, bs_scheduler],
        #                      fast_dev_run=cfg.trainer.fast_dev_run, automatic_optimization=False,
        #                      num_sanity_val_steps=0)
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                         callbacks=[checkpoint_callback],
                         fast_dev_run=cfg.trainer.fast_dev_run, automatic_optimization=False,
                         num_sanity_val_steps=0)

    # cfg.batch_size *= 2
    # trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
    #                      callbacks=[checkpoint_callback], num_sanity_val_steps=0,
    #                      fast_dev_run=cfg.trainer.fast_dev_run, automatic_optimization=False,
    #                      resume_from_checkpoint='/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/baselinev2/'
    #                                             'stochastic/logs/lightning_logs/version_17/'
    #                                             'checkpoints/epoch=108-step=929574.ckpt')

    trainer.fit(m)
    print()


@torch.no_grad()
def quick_eval():
    version = 1
    epoch = 17
    step = 226835

    base_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/'
    model_path = 'stochastic/' + f'logs/lightning_logs/version_{version}/checkpoints/' \
                                 f'epoch={epoch}-step={step}.ckpt'
    hparam_path = 'stochastic/' + f'logs/lightning_logs/version_{version}/hparams.yaml'
    m = BaselineGAN.load_from_checkpoint(checkpoint_path=model_path, hparams_file=hparam_path, map_location='cuda:0')
    m.setup_test_dataset()
    m.eval()
    loader = DataLoader(m.test_dset, batch_size=1, shuffle=True, num_workers=0)
    for data, dataset_idx in loader:
        batch = preprocess_dataset_elements(data, batch_first=False, is_generated=m.hparams.use_generated_dataset)
        out = m.test(batch)

        obs_traj = batch['in_xy'].squeeze()
        gt_traj = batch['gt_xy'].squeeze()
        pred_traj = out['out_xy'].squeeze()

        frame_num = data[6][0, 0].item()
        track_id = data[4][0, 0].item()

        video_dataset = loader.dataset.datasets[dataset_idx[0].item()]
        video_path = f'{base_path}videos/{video_dataset.video_class.value}/' \
                     f'video{video_dataset.video_number}/video.mov'

        plot_trajectory_alongside_frame(obs_trajectory=obs_traj,
                                        gt_trajectory=gt_traj,
                                        pred_trajectory=pred_traj,
                                        frame_number=frame_num,
                                        track_id=track_id,
                                        frame=extract_frame_from_video(video_path, frame_num))
        print()


@torch.no_grad()
def quick_eval_stochastic(k=10, multi_batch=True, batch_s=32, plot=False, eval_on_gt=True, speedup_factor=1,
                          filter_mode=False, moving_only=False, stationary_only=False, threshold=1.0,
                          relative_distance_filter_threshold=100., device='cuda:0', eval_for_worse=False,
                          plot_4_way=False):
    # UNSUPERVISED
    # version = 5
    # epoch = 64
    # step = 819129

    # version = 18
    # epoch = 109
    # step = 929623

    # version = 0
    # epoch = 209
    # step = 2646419

    # Filtered
    version = 397155
    epoch = 348
    step = 68752

    # SUPERVISED
    # version = 7
    # epoch = 3
    # step = 2091035

    # version = 9
    # epoch = 10
    # step = 6273108

    # version = 12
    # epoch = 17
    # step = 7281281

    base_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/'
    model_path = 'stochastic/' + f'logs/lightning_logs/version_{version}/checkpoints/' \
                                 f'epoch={epoch}-step={step}.ckpt'
    hparam_path = 'stochastic/' + f'logs/lightning_logs/version_{version}/hparams.yaml'

    m = BaselineGAN.load_from_checkpoint(checkpoint_path=model_path, hparams_file=hparam_path, map_location='cuda:0')
    m.hparams.use_generated_dataset = False if eval_on_gt else True
    m.setup_test_dataset()
    m.eval()
    m.to(device)
    m.hparams.num_workers = 0 if plot else 12
    if multi_batch and not plot:
        batch_size = batch_s * speedup_factor
    elif multi_batch and plot:
        batch_size = batch_s
        # filter_mode = False
    else:
        batch_size = 1
    loader = DataLoader(m.test_dset, batch_size=batch_size, shuffle=True,
                        num_workers=m.hparams.num_workers)

    constant_linear_baseline_caller = ConstantLinearBaseline()

    ade_list, fde_list = [], []
    linear_ade_list, linear_fde_list = [], []

    for data, dataset_idx in tqdm(loader):
        data = [d.to(device) for d in data]
        batch = preprocess_dataset_elements(data, batch_first=False, is_generated=m.hparams.use_generated_dataset,
                                            filter_mode=filter_mode, moving_only=moving_only,
                                            stationary_only=stationary_only, threshold=threshold,
                                            relative_distance_filter_threshold=relative_distance_filter_threshold)

        feasible_idx = batch['feasible_idx']

        batch = get_batch_k(batch, k)
        batch_size = batch["size"]

        out = m.test(batch)
        constant_linear_baseline_pred_trajectory, constant_linear_baseline_ade, constant_linear_baseline_fde = \
            constant_linear_baseline_caller.eval(
                obs_trajectory=batch['in_xy'].permute(1, 0, 2).cpu().numpy(),
                obs_distances=batch['in_dxdy'].permute(1, 0, 2).cpu().numpy(),
                gt_trajectory=batch['gt_xy'].permute(1, 0, 2).cpu().numpy()
                , ratio=batch['ratio'].squeeze()[0].unsqueeze(0)
                if k == 1 else batch['ratio'].squeeze()[0])

        constant_linear_baseline_pred_trajectory = \
            torch.from_numpy(constant_linear_baseline_pred_trajectory).permute(1, 0, 2)

        if multi_batch:
            im_idx = np.random.choice(batch_size, 1, replace=False).item()

            obs_traj = batch['in_xy'][:, :batch_size][:, im_idx, ...].squeeze()
            gt_traj = batch['gt_xy'][:, :batch_size][:, im_idx, ...].squeeze()
            pred_traj = out['out_xy'].view(out['out_xy'].shape[0], k,
                                           -1, out['out_xy'].shape[2])[:, :, im_idx, ...].squeeze()
            linear_traj = constant_linear_baseline_pred_trajectory[:, im_idx, ...].squeeze()

            frame_num = data[6][im_idx, 0].item()
            track_id = data[4][im_idx, 0].item()
            ratio = batch['ratio'].squeeze()[0]  # data[-1]

            if plot:
                feasible_idx = batch['feasible_idx']
                dataset_idx = dataset_idx[feasible_idx]
            video_dataset = loader.dataset.datasets[dataset_idx[im_idx].item()]
            video_path = f'{base_path}videos/{video_dataset.video_class.value}/' \
                         f'video{video_dataset.video_number}/video.mov'
            # metrics
            # p_traj = batch['gt_xy'].view(batch['gt_xy'].shape[0], -1, k, batch['gt_xy'].shape[2])
            # p_traj_fake = out['out_xy'].view(out['out_xy'].shape[0], -1, k, out['out_xy'].shape[2])
            # constant_linear_p_traj = constant_linear_baseline_pred_trajectory.view(
            #     constant_linear_baseline_pred_trajectory.shape[0], -1, k,
            #     constant_linear_baseline_pred_trajectory.shape[2])

            p_traj = batch['gt_xy'].view(batch['gt_xy'].shape[0], k, -1, batch['gt_xy'].shape[2])
            p_traj_fake = out['out_xy'].view(out['out_xy'].shape[0], k, -1, out['out_xy'].shape[2])
            constant_linear_p_traj = constant_linear_baseline_pred_trajectory.view(
                constant_linear_baseline_pred_trajectory.shape[0], k, -1,
                constant_linear_baseline_pred_trajectory.shape[2])

            # p_traj = batch['gt_xy'][:, :batch_size].unsqueeze(2).repeat(1, 1, k, 1)
            # p_traj_fake = out['out_xy'][:, :batch_size].unsqueeze(2).repeat(1, 1, k, 1)
            # constant_linear_p_traj = \
            #     constant_linear_baseline_pred_trajectory[:, :batch_size].unsqueeze(2).repeat(1, 1, k, 1)

            # ade = cal_ade_stochastic(p_traj, p_traj_fake, 'mean')
            # fde = cal_fde_stochastic(p_traj, p_traj_fake, 'mean')

            ade, fde, best_idx = cal_ade_fde_stochastic(p_traj, p_traj_fake) \
                if not eval_for_worse else cal_ade_fde_stochastic_worse(p_traj, p_traj_fake)
            linear_ade, linear_fde, linear_best_idx = cal_ade_fde_stochastic(p_traj, constant_linear_p_traj.to(device))

            # meter
            ade *= ratio
            fde *= ratio

            linear_ade *= ratio
            linear_fde *= ratio

            plot_ade = ade.squeeze()[im_idx]
            plot_fde = fde.squeeze()[im_idx]
            plot_best_idx = best_idx.squeeze()[im_idx]

            plot_linear_ade = linear_ade.squeeze()[im_idx]
            plot_linear_fde = linear_fde.squeeze()[im_idx]
            plot_linear_best_idx = linear_best_idx.squeeze()[im_idx]

            ade_list.append(ade.mean().item())
            fde_list.append(fde.mean().item())

            linear_ade_list.append(linear_ade.mean().item())
            linear_fde_list.append(linear_fde.mean().item())
        else:
            obs_traj = batch['in_xy'][:, :batch_size].squeeze()
            gt_traj = batch['gt_xy'][:, :batch_size].squeeze()
            pred_traj = out['out_xy'].squeeze()
            linear_traj = constant_linear_baseline_pred_trajectory.squeeze()

            frame_num = data[6][0, 0].item()
            track_id = data[4][0, 0].item()

            video_dataset = loader.dataset.datasets[dataset_idx[0].item()]
            video_path = f'{base_path}videos/{video_dataset.video_class.value}/' \
                         f'video{video_dataset.video_number}/video.mov'

            # fixme
            plot_ade = 0.
            plot_fde = 0.
            plot_best_idx = 0.

            plot_linear_ade = 0.
            plot_linear_fde = 0.
        if plot:
            try:
                if plot_4_way:
                    plot_and_compare_trajectory_four_way_stochastic(
                        frame=extract_frame_from_video(video_path, frame_num),
                        obs_trajectory=obs_traj.cpu().numpy(),
                        gt_trajectory=gt_traj.cpu().numpy(),
                        model_pred_trajectory=pred_traj.cpu().numpy(),
                        other_pred_trajectory=linear_traj,
                        frame_number=frame_num,
                        track_id=track_id,
                        single_mode=k == 1,
                        best_idx=plot_best_idx.item(),
                        additional_text=f'Model: ADE: {plot_ade.item()} | FDE: {plot_fde.item()}\n'
                                        f'Linear: ADE: {plot_linear_ade.item()} | FDE: {plot_linear_fde.item()}',
                    )
                else:
                    plot_trajectory_alongside_frame_stochastic(obs_trajectory=obs_traj.cpu().numpy(),
                                                               gt_trajectory=gt_traj.cpu().numpy(),
                                                               pred_trajectory=pred_traj.cpu().numpy(),
                                                               frame_number=frame_num,
                                                               track_id=track_id,
                                                               frame=extract_frame_from_video(video_path, frame_num),
                                                               single_mode=k == 1,
                                                               best_idx=plot_best_idx.item(),
                                                               additional_text=f'ADE: {plot_ade} | FDE: {plot_fde}')
            except InvalidFrameException:
                logger.error('Frame not found!')
    print(f'Model: ADE: {np.mean(ade_list).item()} | FDE: {np.mean(fde_list).item()}\n'
          f'Linear: ADE: {np.mean(linear_ade_list).item()} | FDE: {np.mean(linear_fde_list).item()}')


@torch.no_grad()
def quick_eval_stochastic_linear_model_combined(
        k=10, multi_batch=True, batch_s=32, plot=False, eval_on_gt=True, speedup_factor=1,
        filter_mode=False, moving_only=False, stationary_only=False, threshold=1.0,
        relative_distance_filter_threshold=100., device='cuda:0', eval_for_worse=False,
        plot_4_way=False):
    # UNSUPERVISED
    # version = 5
    # epoch = 64
    # step = 819129

    # version = 18
    # epoch = 109
    # step = 929623

    # version = 0
    # epoch = 209
    # step = 2646419

    # Filtered
    version = 397155
    epoch = 348
    step = 68752

    # SUPERVISED
    # version = 7
    # epoch = 3
    # step = 2091035

    # version = 9
    # epoch = 10
    # step = 6273108

    # version = 12
    # epoch = 17
    # step = 7281281

    base_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/'
    model_path = 'stochastic/' + f'logs/lightning_logs/version_{version}/checkpoints/' \
                                 f'epoch={epoch}-step={step}.ckpt'
    hparam_path = 'stochastic/' + f'logs/lightning_logs/version_{version}/hparams.yaml'

    m = BaselineGAN.load_from_checkpoint(checkpoint_path=model_path, hparams_file=hparam_path, map_location='cuda:0')
    m.hparams.use_generated_dataset = False if eval_on_gt else True
    m.setup_test_dataset()
    m.eval()
    m.to(device)
    m.hparams.num_workers = 0 if plot else 12
    if multi_batch and not plot:
        batch_size = batch_s * speedup_factor
    elif multi_batch and plot:
        batch_size = batch_s
        # filter_mode = False
    else:
        batch_size = 1
    loader = DataLoader(m.test_dset, batch_size=batch_size, shuffle=True,
                        num_workers=m.hparams.num_workers)

    constant_linear_baseline_caller = ConstantLinearBaseline()

    ade_list, fde_list = [], []

    for data, dataset_idx in tqdm(loader):
        data = [d.to(device) for d in data]
        batch = preprocess_dataset_elements(data, batch_first=False, is_generated=m.hparams.use_generated_dataset,
                                            filter_mode=filter_mode, moving_only=moving_only,
                                            stationary_only=stationary_only, threshold=threshold,
                                            relative_distance_filter_threshold=relative_distance_filter_threshold)

        stationary_idx = batch['feasible_idx']
        moving_idx = np.setdiff1d(np.arange(batch['in_xy'].shape[1]), stationary_idx)

        linear_batch = {
            'in_xy': batch['in_xy'][:, stationary_idx, ...],
            'gt_xy': batch['gt_xy'][:, stationary_idx, ...],
            'in_dxdy': batch['in_dxdy'][:, stationary_idx, ...],
            'gt_dxdy': batch['gt_dxdy'][:, stationary_idx, ...],
            'ratio': batch['ratio'][stationary_idx],
        }
        model_batch = {
            'in_xy': batch['in_xy'][:, moving_idx, ...],
            'gt_xy': batch['gt_xy'][:, moving_idx, ...],
            'in_dxdy': batch['in_dxdy'][:, moving_idx, ...],
            'gt_dxdy': batch['gt_dxdy'][:, moving_idx, ...],
            'ratio': batch['ratio'][moving_idx],
        }

        linear_batch = get_batch_k(linear_batch, k)
        linear_batch_size = linear_batch["size"]

        model_batch = get_batch_k(model_batch, k)
        model_batch_size = model_batch["size"]

        out = m.test(model_batch)
        constant_linear_baseline_pred_trajectory, constant_linear_baseline_ade, constant_linear_baseline_fde = \
            constant_linear_baseline_caller.eval(
                obs_trajectory=linear_batch['in_xy'].permute(1, 0, 2).cpu().numpy(),
                obs_distances=linear_batch['in_dxdy'].permute(1, 0, 2).cpu().numpy(),
                gt_trajectory=linear_batch['gt_xy'].permute(1, 0, 2).cpu().numpy()
                , ratio=linear_batch['ratio'].squeeze()[0].unsqueeze(0)
                if k == 1 else linear_batch['ratio'].squeeze()[0])

        constant_linear_baseline_pred_trajectory = \
            torch.from_numpy(constant_linear_baseline_pred_trajectory).permute(1, 0, 2)

        if multi_batch:
            im_idx = np.random.choice(model_batch_size, 1, replace=False).item()
            # bypasses the error but samples diff!!
            linear_im_idx = np.random.choice(linear_batch_size, 1, replace=False).item()

            obs_traj = batch['in_xy'][:, :model_batch_size][:, im_idx, ...].squeeze()
            gt_traj = batch['gt_xy'][:, :model_batch_size][:, im_idx, ...].squeeze()
            pred_traj = out['out_xy'].view(out['out_xy'].shape[0], k,
                                           -1, out['out_xy'].shape[2])[:, :, im_idx, ...].squeeze()
            linear_traj = constant_linear_baseline_pred_trajectory[:, linear_im_idx, ...].squeeze()

            frame_num = data[6][im_idx, 0].item()
            track_id = data[4][im_idx, 0].item()
            ratio = batch['ratio'].squeeze()[0]  # data[-1]

            if plot:
                feasible_idx = batch['feasible_idx']
                dataset_idx = dataset_idx[feasible_idx]
            video_dataset = loader.dataset.datasets[dataset_idx[im_idx].item()]
            video_path = f'{base_path}videos/{video_dataset.video_class.value}/' \
                         f'video{video_dataset.video_number}/video.mov'
            # metrics
            # p_traj = batch['gt_xy'].view(batch['gt_xy'].shape[0], -1, k, batch['gt_xy'].shape[2])
            # p_traj_fake = out['out_xy'].view(out['out_xy'].shape[0], -1, k, out['out_xy'].shape[2])
            # constant_linear_p_traj = constant_linear_baseline_pred_trajectory.view(
            #     constant_linear_baseline_pred_trajectory.shape[0], -1, k,
            #     constant_linear_baseline_pred_trajectory.shape[2])

            p_traj = model_batch['gt_xy'].view(model_batch['gt_xy'].shape[0], k, -1, model_batch['gt_xy'].shape[2])
            p_traj_fake = out['out_xy'].view(out['out_xy'].shape[0], k, -1, out['out_xy'].shape[2])

            constant_linear_p_traj = linear_batch['gt_xy'].view(
                linear_batch['gt_xy'].shape[0], k, -1, linear_batch['gt_xy'].shape[2])
            constant_linear_pred_traj = constant_linear_baseline_pred_trajectory.view(
                constant_linear_baseline_pred_trajectory.shape[0], k, -1,
                constant_linear_baseline_pred_trajectory.shape[2])

            # p_traj = batch['gt_xy'][:, :batch_size].unsqueeze(2).repeat(1, 1, k, 1)
            # p_traj_fake = out['out_xy'][:, :batch_size].unsqueeze(2).repeat(1, 1, k, 1)
            # constant_linear_p_traj = \
            #     constant_linear_baseline_pred_trajectory[:, :batch_size].unsqueeze(2).repeat(1, 1, k, 1)

            # ade = cal_ade_stochastic(p_traj, p_traj_fake, 'mean')
            # fde = cal_fde_stochastic(p_traj, p_traj_fake, 'mean')

            ade, fde, best_idx = cal_ade_fde_stochastic(p_traj, p_traj_fake) \
                if not eval_for_worse else cal_ade_fde_stochastic_worse(p_traj, p_traj_fake)
            linear_ade, linear_fde, linear_best_idx = cal_ade_fde_stochastic(
                constant_linear_p_traj, constant_linear_pred_traj.to(device))

            # meter
            ade *= ratio
            fde *= ratio

            linear_ade *= ratio
            linear_fde *= ratio

            combined_ade = torch.cat((ade, linear_ade), dim=-1)
            combined_fde = torch.cat((fde, linear_fde), dim=-1)

            plot_ade = ade.squeeze()[im_idx]
            plot_fde = fde.squeeze()[im_idx]
            plot_best_idx = best_idx.squeeze()[im_idx]

            plot_linear_ade = linear_ade.squeeze()[linear_im_idx]
            plot_linear_fde = linear_fde.squeeze()[linear_im_idx]
            plot_linear_best_idx = linear_best_idx.squeeze()[linear_im_idx]

            # model_ade = ade.mean().item()
            # model_fde = fde.mean().item()
            #
            # linear_ade = linear_ade.mean().item()
            # linear_fde = linear_fde.mean().item()
            #
            # final_ade = (model_ade + linear_ade) / 2
            # final_fde = (model_fde + linear_fde) / 2

            final_ade = combined_ade.mean().item()
            final_fde = combined_fde.mean().item()

            ade_list.append(final_ade)
            fde_list.append(final_fde)
        else:
            return NotImplemented
            # obs_traj = batch['in_xy'][:, :batch_size].squeeze()
            # gt_traj = batch['gt_xy'][:, :batch_size].squeeze()
            # pred_traj = out['out_xy'].squeeze()
            # linear_traj = constant_linear_baseline_pred_trajectory.squeeze()
            #
            # frame_num = data[6][0, 0].item()
            # track_id = data[4][0, 0].item()
            #
            # video_dataset = loader.dataset.datasets[dataset_idx[0].item()]
            # video_path = f'{base_path}videos/{video_dataset.video_class.value}/' \
            #              f'video{video_dataset.video_number}/video.mov'
            #
            # # fixme
            # plot_ade = 0.
            # plot_fde = 0.
            # plot_best_idx = 0.
            #
            # plot_linear_ade = 0.
            # plot_linear_fde = 0.
        if plot:
            try:
                if plot_4_way:
                    plot_and_compare_trajectory_four_way_stochastic(
                        frame=extract_frame_from_video(video_path, frame_num),
                        obs_trajectory=obs_traj.cpu().numpy(),
                        gt_trajectory=gt_traj.cpu().numpy(),
                        model_pred_trajectory=pred_traj.cpu().numpy(),
                        other_pred_trajectory=linear_traj,
                        frame_number=frame_num,
                        track_id=track_id,
                        single_mode=k == 1,
                        best_idx=plot_best_idx.item(),
                        additional_text=f'Model: ADE: {plot_ade.item()} | FDE: {plot_fde.item()}\n'
                                        f'Linear: ADE: {plot_linear_ade.item()} | FDE: {plot_linear_fde.item()}',
                    )
                else:
                    plot_trajectory_alongside_frame_stochastic(obs_trajectory=obs_traj.cpu().numpy(),
                                                               gt_trajectory=gt_traj.cpu().numpy(),
                                                               pred_trajectory=pred_traj.cpu().numpy(),
                                                               frame_number=frame_num,
                                                               track_id=track_id,
                                                               frame=extract_frame_from_video(video_path, frame_num),
                                                               single_mode=k == 1,
                                                               best_idx=plot_best_idx.item(),
                                                               additional_text=f'ADE: {plot_ade} | FDE: {plot_fde}')
            except InvalidFrameException:
                logger.error('Frame not found!')
    print(f'Model+Linear: ADE: {np.mean(ade_list).item()} | FDE: {np.mean(fde_list).item()}\n')


if __name__ == '__main__':
    # debug_model()

    # quick_eval_stochastic(plot=False, eval_on_gt=True, k=10, speedup_factor=32, filter_mode=True, moving_only=False,
    #                       stationary_only=False, eval_for_worse=False, relative_distance_filter_threshold=True)

    quick_eval_stochastic_linear_model_combined(
        plot=False, eval_on_gt=True, k=10, speedup_factor=32, filter_mode=True, moving_only=False,
        stationary_only=False, eval_for_worse=False, relative_distance_filter_threshold=True)

    # quick_eval()

    # On unsupervised

    # filtered data
    # Model+Linear: ADE: 0.5301352841026825 | FDE: 1.1355520253251676 - bad average
    # Model+Linear: ADE: 0.8410700329685178 | FDE: 1.8075439942316476
    # k = 10
    # no filter
    # Model: ADE: 0.8893271395228705 | FDE: 1.9251465165561124
    # moving only
    # Model: ADE: 0.9776758797384191 | FDE: 2.078646834115817
    # stationary only
    # Model: ADE: 0.48235866318575876 | FDE: 1.08501878345889

    # k = 10
    # Model: ADE: 0.9994683927553563 | FDE: 2.0943560366250256

    # k = 1
    # Model: ADE: 2.7966834577314694 | FDE: 5.77561737130756
    # supervised
    # Model: ADE: 2.03455348818243 | FDE: 4.0124918072276134

    # On supervised - k=10
    # All Trajectories
    # Model: ADE: 1.0329569692133145 | FDE: 2.1217076520531304
    # ver 18: Model: ADE: 1.0106120413302706 | FDE: 2.074660060222275
    # lower limit
    # Model: ADE: 4.484831560526955 | FDE: 9.039632317203804
    # supervised
    # Model: ADE: 0.541943404247334 | FDE: 1.0914964632178983
    # ver 12: Model: ADE: 0.5273425247091686 | FDE: 1.0667414188999012

    # moving only - 1.0  + outlier removal
    # Model: ADE: 0.9641407612558273 | FDE: 2.009607732119232
    # ver 18: Model: ADE: 0.9417482699733155 | FDE: 1.9596099997154726
    # ver 18: K=20 - Model: ADE: 0.7974822557212522 | FDE: 1.6561251261760093
    # ver 18: K=50 - Model: ADE: 0.6664677064058206 | FDE: 1.3757458298407648
    # worse
    # Model: ADE: 5.949471143770464 | FDE: 11.776895021153107
    # k=1
    # Model: ADE: 2.6990513349074328 | FDE: 5.464550167791963

    # supervised - k=10 + outlier removal
    # Model: ADE: 0.8907297152484545 | FDE: 1.7420249850288165
    # ver 9: Model: ADE: 0.9237790072096367 | FDE: 1.8031240094930459
    # ver 12: Model: ADE: 0.8537704996521618 | FDE: 1.676240554331468
    # ver 12: K=20: Model: ADE: 0.7161743122533683 | FDE: 1.3945243051053209
    # ver 12: K=50: Model: ADE: 0.5974900534163404 | FDE: 1.1540122663749628

    # stationary only - 1.0  + outlier removal
    # Model: ADE: 0.8614191488931643 | FDE: 1.7247877094923139
    # ver 18: Model: ADE: 0.8427964557216341 | FDE: 1.6837042595976617
    # supervised
    # Model: ADE: 0.05635012733756296 | FDE: 0.0999534727096446
    # new
    # Model: ADE: 0.05652092342092408 | FDE: 0.10063885672264172
    # ver 12: Model: ADE: 0.053335761822049584 | FDE: 0.09534861413238387
