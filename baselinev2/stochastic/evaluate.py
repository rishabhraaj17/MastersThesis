import os

import numpy as np
import torch

from baselinev2.nn.models import ConstantLinearBaseline
from baselinev2.stochastic.losses import cal_ade_fde_stochastic
from baselinev2.stochastic.model import BaselineGAN
from baselinev2.stochastic.model_modules import preprocess_dataset_elements
from baselinev2.stochastic.utils import get_batch_k


def evaluate(data, model_version: int, device: str = 'cuda:0', use_generated_dataset: bool = True,
             filter_mode: bool = True, moving_only: bool = False, stationary_only: bool = False,
             threshold: float = 1.0, relative_distance_filter_threshold: bool = True, k: int = 10,
             constant_linear_baseline: ConstantLinearBaseline = ConstantLinearBaseline()):
    base_path = f'/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/' \
                f'stochastic/logs/lightning_logs/version_{model_version}'
    checkpoint_path = f'{base_path}/checkpoints/'
    checkpoint_file = checkpoint_path + os.listdir(checkpoint_path)[0]
    hparams_file = f'{base_path}/hparams.yaml'

    model = BaselineGAN.load_from_checkpoint(checkpoint_path=checkpoint_file, hparams_file=hparams_file,
                                             map_location=device)
    model.hparams.use_generated_dataset = True if use_generated_dataset else False
    model.eval()
    model.to(device)

    data = [d.to(device) for d in data]

    batch = preprocess_dataset_elements(data, batch_first=False, is_generated=use_generated_dataset,
                                        filter_mode=filter_mode, moving_only=moving_only,
                                        stationary_only=stationary_only, threshold=threshold,
                                        relative_distance_filter_threshold=relative_distance_filter_threshold)

    ratio = batch['ratio'].squeeze()[0]
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

    out = model.test(model_batch)
    constant_linear_baseline_pred_trajectory, constant_linear_baseline_ade, constant_linear_baseline_fde = \
        constant_linear_baseline.eval(
            obs_trajectory=linear_batch['in_xy'].permute(1, 0, 2).cpu().numpy(),
            obs_distances=linear_batch['in_dxdy'].permute(1, 0, 2).cpu().numpy(),
            gt_trajectory=linear_batch['gt_xy'].permute(1, 0, 2).cpu().numpy()
            , ratio=linear_batch['ratio'].squeeze()[0].unsqueeze(0)
            if k == 1 else linear_batch['ratio'].squeeze()[0])

    constant_linear_baseline_pred_trajectory = \
        torch.from_numpy(constant_linear_baseline_pred_trajectory).permute(1, 0, 2)

    p_traj = model_batch['gt_xy'].view(model_batch['gt_xy'].shape[0], k, -1, model_batch['gt_xy'].shape[2])
    p_traj_fake = out['out_xy'].view(out['out_xy'].shape[0], k, -1, out['out_xy'].shape[2])

    constant_linear_p_traj = linear_batch['gt_xy'].view(
        linear_batch['gt_xy'].shape[0], k, -1, linear_batch['gt_xy'].shape[2])
    constant_linear_pred_traj = constant_linear_baseline_pred_trajectory.view(
        constant_linear_baseline_pred_trajectory.shape[0], k, -1,
        constant_linear_baseline_pred_trajectory.shape[2])

    ade, fde, best_idx = cal_ade_fde_stochastic(p_traj, p_traj_fake)
    linear_ade, linear_fde, linear_best_idx = cal_ade_fde_stochastic(
        constant_linear_p_traj, constant_linear_pred_traj.to(device))

    # meter
    ade *= ratio
    fde *= ratio

    linear_ade *= ratio
    linear_fde *= ratio

    combined_ade = torch.cat((ade, linear_ade), dim=-1)
    combined_fde = torch.cat((fde, linear_fde), dim=-1)

    return best_idx

