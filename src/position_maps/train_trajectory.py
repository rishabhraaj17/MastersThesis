import os
import warnings

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import src_lib.models_hub as hub
from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.exceptions import InvalidFrameException
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.models import ConstantLinearBaseline, ConstantLinearBaselineV2
from baselinev2.plot_utils import plot_trajectory_alongside_frame, plot_trajectory_alongside_frame_stochastic
from baselinev2.stochastic.losses import cal_ade, cal_fde, cal_ade_fde_stochastic
from baselinev2.stochastic.model_modules import preprocess_dataset_elements_from_dict
from log import get_logger
from src.position_maps.trajectory_utils import get_multiple_datasets, bezier_smoother, splrep_smoother
from src_lib.datasets.extracted_dataset import get_train_and_val_datasets, extracted_collate
from src_lib.datasets.opentraj_based import get_multiple_gt_dataset
from src_lib.datasets.trajectory_stgcnn import seq_collate_with_dataset_idx_dict, SmoothTrajectoryDataset
from src_lib.models_hub import TransformerNoisyMotionGenerator, TrajectoryGANTransformerV2

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


def setup_foreign_dataset(cfg):
    dataset = None
    return dataset, None


def setup_dataset(cfg):
    if cfg.tp_module.datasets.use_foreign_dataset:
        return setup_foreign_dataset(cfg)
    if cfg.tp_module.datasets.use_standard_dataset:
        if cfg.tp_module.datasets.use_generated:
            train_dataset, val_dataset = get_multiple_datasets(
                cfg=cfg, split_dataset=True, with_dataset_idx=True,
                smooth_trajectories=cfg.tp_module.smooth_trajectories.enabled,
                smoother=bezier_smoother if cfg.tp_module.smooth_trajectories.smoother == 'bezier' else splrep_smoother,
                threshold=cfg.tp_module.smooth_trajectories.min_length,
                from_temp_file=cfg.tp_module.datasets.from_temp_file,
                frame_rate=cfg.tp_module.datasets.frame_rate,
                time_step=cfg.tp_module.datasets.time_step
            )
        else:
            train_dataset, val_dataset = get_multiple_gt_dataset(
                cfg=cfg, split_dataset=True, with_dataset_idx=True,
                smooth_trajectories=cfg.tp_module.smooth_trajectories.enabled,
                smoother=bezier_smoother if cfg.tp_module.smooth_trajectories.smoother == 'bezier' else splrep_smoother,
                threshold=cfg.tp_module.smooth_trajectories.min_length,
                frame_rate=cfg.tp_module.datasets.frame_rate,
                time_step=cfg.tp_module.datasets.time_step)
    else:
        train_dataset, val_dataset = get_train_and_val_datasets(
            video_classes=[getattr(SDDVideoClasses, v_c) for v_c in cfg.tp_module.datasets.train.video_classes],
            video_numbers=cfg.tp_module.datasets.train.video_numbers,
            meta_label=[getattr(SDDVideoDatasets, v_c) for v_c in cfg.tp_module.datasets.train.video_classes],
            val_video_classes=[getattr(SDDVideoClasses, v_c) for v_c in cfg.tp_module.datasets.val.video_classes],
            val_video_numbers=cfg.tp_module.datasets.val.video_numbers,
            val_meta_label=[getattr(SDDVideoDatasets, v_c) for v_c in cfg.tp_module.datasets.val.video_classes],
            get_generated=cfg.tp_module.datasets.use_generated,
            meta_path='../../../Datasets/SDD/H_SDD.txt',
            root='../../../Datasets/SDD/pm_extracted_annotations/'
            if cfg.tp_module.datasets.use_generated else '../../../Datasets/SDD_Features/'
        )
    return train_dataset, val_dataset


def setup_model(cfg, train_dataset, val_dataset):
    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate
    model = getattr(hub, cfg.tp_module.model)(
        config=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
        desired_output_shape=None, loss_function=None,
        additional_loss_functions=None, collate_fn=collate_fn
    )
    return model


@hydra.main(config_path="config", config_name="config")
def train_lightning(cfg):
    logger.info(f"Setting up dataset and model")
    train_dataset, val_dataset = setup_dataset(cfg)

    model = setup_model(cfg, train_dataset, val_dataset)

    if cfg.tp_module.warm_restart.enable:
        model = load_checkpoint(cfg, model)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.monitor,
        save_top_k=cfg.tp_module.trainer.num_checkpoints_to_save,
        mode=cfg.mode,
        verbose=cfg.verbose
    )
    wandb_logger = WandbLogger(project='TrajectoryPredictionBaseline', name=cfg.tp_module.model)
    wandb_logger.log_hyperparams({**cfg.tp_module, **cfg.trajectory_based})
    trainer = Trainer(max_epochs=cfg.tp_module.trainer.max_epochs, gpus=cfg.tp_module.trainer.gpus,
                      fast_dev_run=cfg.tp_module.trainer.fast_dev_run, callbacks=[checkpoint_callback],
                      accelerator=cfg.tp_module.trainer.accelerator, deterministic=cfg.tp_module.trainer.deterministic,
                      replace_sampler_ddp=cfg.tp_module.trainer.replace_sampler_ddp,
                      num_nodes=cfg.tp_module.trainer.num_nodes,
                      gradient_clip_val=cfg.tp_module.trainer.gradient_clip_val,
                      accumulate_grad_batches=cfg.tp_module.trainer.accumulate_grad_batches,
                      logger=wandb_logger)
    trainer.fit(model)


def load_checkpoint(cfg, model):
    if cfg.tp_module.warm_restart.wandb.enabled:
        version_name = f"{cfg.tp_module.warm_restart.wandb.checkpoint.run_name}".split('-')[-1]
        checkpoint_root_path = f'{cfg.tp_module.warm_restart.wandb.checkpoint.root}' \
                               f'{cfg.tp_module.warm_restart.wandb.checkpoint.run_name}' \
                               f'{cfg.tp_module.warm_restart.wandb.checkpoint.tail_path}' \
                               f'{cfg.tp_module.warm_restart.wandb.checkpoint.project_name}/' \
                               f'{version_name}/checkpoints/'
    else:
        checkpoint_root_path = \
            f'{cfg.tp_module.warm_restart.checkpoint.root}{cfg.tp_module.warm_restart.checkpoint.path}' \
            f'{cfg.tp_module.warm_restart.checkpoint.version}/checkpoints/'
    checkpoint_files = os.listdir(checkpoint_root_path)
    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]
    model_path = checkpoint_root_path + checkpoint_files[-cfg.tp_module.warm_restart.checkpoint.top_k]
    logger.info(f'Resuming from : {model_path}')
    if cfg.tp_module.warm_restart.custom_load:
        logger.info(f'Loading weights manually as custom load is {cfg.tp_module.warm_restart.custom_load}')
        load_dict = torch.load(model_path, map_location=cfg.tp_module.device)

        model.load_state_dict(load_dict['state_dict'])
        model.to(cfg.tp_module.device)
        model.train()
    else:
        logger.warning('Not supported, no checkpoint loaded')
    return model


@hydra.main(config_path="config", config_name="config")
def evaluate(cfg):
    logger.info(f"Setting up dataset and model")
    train_dataset, val_dataset = setup_dataset(cfg)

    model = setup_model(cfg, train_dataset, val_dataset)

    # load dict
    model = load_checkpoint(cfg, model)

    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate

    loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False,
                        collate_fn=collate_fn)

    constant_linear_baseline_caller = ConstantLinearBaselineV2()

    ade_list, fde_list, loss_list = [], [], []
    linear_ade_list, linear_fde_list = [], []

    for b_idx, batch in enumerate(tqdm(loader)):
        batch = preprocess_dataset_elements_from_dict(
            batch, filter_mode=True, moving_only=True, stationary_only=False)

        if batch['in_xy'].shape[1] == 0:
            continue

        batch = {k: v.to(cfg.tp_module.device) for k, v in batch.items()}

        target = batch['gt_xy']

        with torch.no_grad():
            out = model(batch)

        constant_linear_baseline_pred_trajectory, constant_linear_baseline_ade, constant_linear_baseline_fde = \
            constant_linear_baseline_caller.eval(
                obs_trajectory=batch['in_xy'].permute(1, 0, 2).cpu().numpy(),
                obs_distances=batch['in_dxdy'].permute(1, 0, 2).cpu().numpy(),
                gt_trajectory=batch['gt_xy'].permute(1, 0, 2).cpu().numpy()
                , ratio=batch['ratio'].squeeze(), batch_size=target.shape[1])

        pred = out['out_xy']

        loss = model.calculate_loss(pred, target, batch['ratio'])
        # ade, fde = model.calculate_metrics(pred, target, model.config.tp_module.metrics.mode)

        obs_separated = batch['in_xy'].view(batch['in_xy'].shape[0], -1, target.shape[1], batch['in_xy'].shape[-1])
        target_separated = target.view(target.shape[0], -1, target.shape[1], target.shape[-1])
        pred_separated = pred.view(pred.shape[0], -1, target.shape[1], pred.shape[-1])

        ade, fde, best_idx = cal_ade_fde_stochastic(target_separated, pred_separated)

        ade *= batch['ratio'].squeeze()
        fde *= batch['ratio'].squeeze()

        ade_list.append(ade.mean().item())
        fde_list.append(fde.mean().item())
        loss_list.append(loss.item())

        linear_ade_list.append(constant_linear_baseline_ade.mean().item())
        linear_fde_list.append(constant_linear_baseline_fde.mean().item())

        dataset_idx = batch['dataset_idx'].item()
        seq_start_end = batch['seq_start_end']
        frame_nums = batch['in_frames']
        track_lists = batch['in_tracks']

        random_trajectory_idx = np.random.choice(frame_nums.shape[1], 1, replace=False).item()

        obs_trajectory = batch['in_xy'][:, random_trajectory_idx, ...]
        gt_trajectory = batch['gt_xy'][:, random_trajectory_idx, ...]
        pred_trajectory = out['out_xy'][:, random_trajectory_idx, ...]

        frame_num = int(frame_nums[:, random_trajectory_idx, ...][0].item())
        track_num = int(track_lists[:, random_trajectory_idx, ...][0].item())

        current_dataset = loader.dataset.datasets[dataset_idx].dataset \
            if cfg.tp_module.datasets.use_standard_dataset else loader.dataset.datasets[dataset_idx]
        if isinstance(current_dataset, SmoothTrajectoryDataset):
            current_dataset = current_dataset.base_dataset

        if cfg.tp_module.datasets.use_standard_dataset:
            video_path = f"{cfg.root}videos/{getattr(SDDVideoClasses, current_dataset.video_class).value}" \
                         f"/video{current_dataset.video_number}/video.mov"
        else:
            video_path = f"{cfg.root}videos/{current_dataset.video_class.value}" \
                         f"/video{current_dataset.video_number}/video.mov"

        if b_idx % 500 == 0:
            try:
                frame = extract_frame_from_video(video_path, frame_num)

                plot_trajectory_alongside_frame(
                    frame, obs_trajectory.cpu(), gt_trajectory.cpu(), pred_trajectory.cpu(),
                    frame_num, track_id=track_num,
                    additional_text=f"ADE: {ade.mean().item()} | FDE: {fde.mean().item()}")
            except InvalidFrameException:
                continue

    loss_list = np.array(loss_list).mean()
    ade_list = np.array(ade_list).mean()
    fde_list = np.array(fde_list).mean()
    linear_ade_list = np.array(linear_ade_list).mean()
    linear_fde_list = np.array(linear_fde_list).mean()

    logger.info(f"Loss: {loss_list}\nADE: {ade_list} | FDE: {fde_list}\n"
                f"Linear ADE: {linear_ade_list} | Linear FDE: {linear_fde_list}")


@hydra.main(config_path="config", config_name="config")
def evaluate_stochastic(cfg):
    logger.info(f"Stochastic - Setting up dataset and model")
    train_dataset, val_dataset = setup_dataset(cfg)

    model = setup_model(cfg, train_dataset, val_dataset)

    # load dict
    model = load_checkpoint(cfg, model)

    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate

    loader = DataLoader(val_dataset, batch_size=2, shuffle=True, pin_memory=True, drop_last=False,
                        collate_fn=collate_fn)

    constant_linear_baseline_caller = ConstantLinearBaselineV2()

    ade_list, fde_list, loss_list = [], [], []
    linear_ade_list, linear_fde_list = [], []

    for b_idx, batch in enumerate(tqdm(loader)):
        batch = preprocess_dataset_elements_from_dict(batch, filter_mode=False, moving_only=True, stationary_only=False)

        if batch['in_xy'].shape[1] == 0:
            continue

        batch = {k: v.to(cfg.tp_module.device) for k, v in batch.items()}

        batch = model.get_k_batches(batch, model.config.tp_module.datasets.batch_multiplier)
        batch_size = batch["size"]

        with torch.no_grad():
            model.generator.eval()
            out = model.generator(batch)
            model.generator.train()

        constant_linear_baseline_pred_trajectory, constant_linear_baseline_ade, constant_linear_baseline_fde = \
            constant_linear_baseline_caller.eval(
                obs_trajectory=batch['in_xy'].permute(1, 0, 2).cpu().numpy(),
                obs_distances=batch['in_dxdy'].permute(1, 0, 2).cpu().numpy(),
                gt_trajectory=batch['gt_xy'].permute(1, 0, 2).cpu().numpy()
                , ratio=batch['ratio'].squeeze(), batch_size=batch_size)

        target = batch['gt_xy']
        pred = out['out_xy']

        obs_separated = batch['in_xy'].view(batch['in_xy'].shape[0], -1, batch_size, batch['in_xy'].shape[-1])
        target_separated = target.view(target.shape[0], -1, batch_size, target.shape[-1])
        pred_separated = pred.view(pred.shape[0], -1, batch_size, pred.shape[-1])

        loss = model.calculate_loss(pred, target)
        loss = loss.view(model.config.tp_module.datasets.batch_multiplier, -1)
        loss, _ = loss.min(dim=0, keepdim=True)
        loss = torch.mean(loss)

        # ade, fde = cal_ade(target, pred, mode='raw'), cal_fde(target, pred, mode='raw')
        ade, fde, best_idx = cal_ade_fde_stochastic(target_separated, pred_separated)
        # ade = ade.view(model.config.tp_module.datasets.batch_multiplier, batch_size) * batch['ratio'].squeeze()
        # fde = fde.view(model.config.tp_module.datasets.batch_multiplier, batch_size) * batch['ratio'].squeeze()

        # fde, _ = fde.min(dim=0)
        modes_caught = (fde < model.config.tp_module.datasets.mode_dist_threshold).float()

        # ade, ade_min_idx = ade.min(dim=0, keepdim=True)

        ade *= batch['ratio'].squeeze()
        fde *= batch['ratio'].squeeze()

        ade_list.append(ade.mean().item())
        fde_list.append(fde.mean().item())
        loss_list.append(loss.item())

        linear_ade_list.append(constant_linear_baseline_ade.mean().item())
        linear_fde_list.append(constant_linear_baseline_fde.mean().item())

        seq_start_end = batch['seq_start_end']
        frame_nums = batch['in_frames'].view(model.config.tp_module.datasets.batch_multiplier, 8, batch_size)
        track_lists = batch['in_tracks'].view(model.config.tp_module.datasets.batch_multiplier, 8, batch_size)

        random_trajectory_idx = np.random.choice(
            min(batch['dataset_idx'].shape[-1], obs_separated.shape[2]), 1, replace=False).item()

        dataset_idx = batch['dataset_idx'][random_trajectory_idx].item()

        obs_trajectory = obs_separated[:, :, random_trajectory_idx, ...]
        gt_trajectory = target_separated[:, :, random_trajectory_idx, ...]
        pred_trajectory = pred_separated[:, :, random_trajectory_idx, ...]

        frame_num = int(frame_nums[0, random_trajectory_idx, ...][0].item())
        track_num = int(track_lists[0, random_trajectory_idx, ...][0].item())

        current_dataset = loader.dataset.datasets[dataset_idx].dataset \
            if cfg.tp_module.datasets.use_standard_dataset else loader.dataset.datasets[dataset_idx]
        if isinstance(current_dataset, SmoothTrajectoryDataset):
            current_dataset = current_dataset.base_dataset

        if cfg.tp_module.datasets.use_standard_dataset:
            video_path = f"{cfg.root}videos/{getattr(SDDVideoClasses, current_dataset.video_class).value}" \
                         f"/video{current_dataset.video_number}/video.mov"
        else:
            video_path = f"{cfg.root}videos/{current_dataset.video_class.value}" \
                         f"/video{current_dataset.video_number}/video.mov"

        ade = ade.squeeze(0)[random_trajectory_idx]
        fde = fde.squeeze(0)[random_trajectory_idx]

        # single for give data
        obs_trajectory = obs_trajectory[:, 0, :]
        gt_trajectory = gt_trajectory[:, 0, :]

        if b_idx % 500 == 0:
            try:
                frame = extract_frame_from_video(video_path, frame_num)

                plot_trajectory_alongside_frame_stochastic(
                    frame, obs_trajectory.cpu(), gt_trajectory.cpu(), pred_trajectory.cpu(),
                    frame_num, track_id=track_num,
                    best_idx=best_idx.squeeze(0)[random_trajectory_idx],
                    additional_text=f"ADE: {ade.item()} | FDE: {fde.item()}")
            except InvalidFrameException:
                continue

    loss_list = np.array(loss_list).mean()
    ade_list = np.array(ade_list).mean()
    fde_list = np.array(fde_list).mean()
    linear_ade_list = np.array(linear_ade_list).mean()
    linear_fde_list = np.array(linear_fde_list).mean()

    logger.info(f"Loss: {loss_list}\nADE: {ade_list} | FDE: {fde_list}\n"
                f"Linear ADE: {linear_ade_list} | Linear FDE: {linear_fde_list}")


@hydra.main(config_path="config", config_name="config")
def overfit_gan(cfg):
    device = 'cuda:0'
    epochs = 2000
    plot_idx = 100
    batch_size = 4

    logger.info(f"Overfit GAN - Setting up dataset and model")
    train_dataset, val_dataset = setup_dataset(cfg)

    model = setup_model(cfg, train_dataset, val_dataset)
    model.to(device)

    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False,
                        collate_fn=collate_fn)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False,
                            collate_fn=collate_fn)

    opt_disc = torch.optim.Adam(model.discriminator.parameters(), lr=model.net_params.discriminator.lr,
                                weight_decay=model.net_params.discriminator.weight_decay,
                                amsgrad=model.net_params.discriminator.weight_decay)
    opt_gen = torch.optim.Adam(model.generator.parameters(), lr=model.net_params.generator.lr,
                               weight_decay=model.net_params.generator.weight_decay,
                               amsgrad=model.net_params.generator.weight_decay)

    # load states
    # states = torch.load('transformerGANep100_gt_dc24.pt')
    # model.load_state_dict(states['model'])
    # opt_gen.load_state_dict(states['opt_gen'])
    # opt_disc.load_state_dict(states['opt_disc'])
    # sch = ReduceLROnPlateau(opt_gen,
    #                         patience=model.config.tp_module.scheduler.patience,
    #                         verbose=model.config.tp_module.scheduler.verbose,
    #                         factor=model.config.tp_module.scheduler.factor,
    #                         min_lr=model.config.tp_module.scheduler.min_lr)

    train_loss, ade_list, fde_list, adv_loss, disc_adv_loss = [], [], [], [], []
    loss, disc_loss, ade, fde, fake_loss_gen = \
        torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
    g_steps_yet, d_steps_yet = 0, 0
    gen_only_cutoff = 0
    disc_only_cutoff = 0
    opt_switch = 1
    for epoch in range(epochs):
        model.train()
        with tqdm(loader, position=0) as t:
            t.set_description('Epoch %i' % epoch)
            for b_idx, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                # batch_size = batch["size"]

                if opt_switch == 0 and epoch >= gen_only_cutoff:
                    opt_disc.zero_grad()

                    batch = model.get_k_batches(batch, model.config.tp_module.datasets.batch_multiplier)
                    batch_size = batch["size"]

                    real_pred = model.discriminator(batch, batch['gt_dxdy'])
                    real_gt = torch.ones_like(real_pred)
                    real_loss = model.calculate_discriminator_loss(pred=real_pred, target=real_gt, is_real=True,
                                                                   is_disc=True)

                    # Train with fake
                    with torch.no_grad():
                        model.generator.eval()
                        fake_pred = model.generator(batch)
                        model.generator.train()

                    fake_pred = model.discriminator(batch, fake_pred['out_dxdy'])
                    fake_gt = torch.zeros_like(fake_pred)
                    fake_loss = model.calculate_discriminator_loss(pred=fake_pred, target=fake_gt, is_real=False,
                                                                   is_disc=True)

                    disc_loss = real_loss + fake_loss

                    disc_adv_loss.append(disc_loss.item())

                    disc_loss.backward()
                    opt_disc.step()
                    if epoch > disc_only_cutoff:
                        opt_switch = 1
                    d_steps_yet += 1
                elif opt_switch == 1 and epoch >= 0 and not gen_only_cutoff < epoch < disc_only_cutoff:
                    opt_gen.zero_grad()

                    batch = model.get_k_batches(batch, model.config.tp_module.datasets.batch_multiplier)
                    batch_size = batch["size"]

                    out = model.generator(batch)

                    target = batch['gt_xy']
                    pred = out['out_xy']

                    if epoch > gen_only_cutoff:
                        fake_pred = model.discriminator(batch, out['out_dxdy'])
                        fake_gt = torch.zeros_like(fake_pred)
                        fake_loss_gen = model.calculate_discriminator_loss(
                            pred=fake_pred, target=fake_gt, is_real=False, is_disc=False)
                    if epoch <= gen_only_cutoff:
                        fake_loss_gen = fake_loss_gen.to(pred)

                    loss = model.calculate_loss(pred, target)
                    loss = loss.view(model.config.tp_module.datasets.batch_multiplier, -1)
                    loss, _ = loss.min(dim=0, keepdim=True)
                    # loss *= batch['ratio'][0]
                    loss = torch.mean(loss) + fake_loss_gen

                    ade, fde = cal_ade(target, pred, mode='raw'), cal_fde(target, pred, mode='raw')
                    ade = ade.view(model.config.tp_module.datasets.batch_multiplier, batch_size) * batch['ratio'][0]
                    fde = fde.view(model.config.tp_module.datasets.batch_multiplier, batch_size) * batch['ratio'][0]

                    ade, _ = ade.min(dim=0, keepdim=True)
                    fde, _ = fde.min(dim=0, keepdim=True)

                    ade, fde = ade.mean(), fde.mean()

                    train_loss.append(loss.item())
                    ade_list.append(ade.item())
                    fde_list.append(fde.item())
                    adv_loss.append(fake_loss_gen.item())

                    loss.backward()
                    opt_gen.step()
                    if epoch >= gen_only_cutoff:
                        opt_switch = 0
                    g_steps_yet += 1

                t.set_postfix(gen_loss=loss.item(), disc_loss=disc_loss.item(),
                              gen_adv_loss=fake_loss_gen.item(), ade=ade.item(), fde=fde.item(),
                              running_disc_loss=torch.tensor(disc_adv_loss).mean().item(),
                              running_gen_adv_loss=torch.tensor(adv_loss).mean().item(),
                              running_gen_loss=torch.tensor(train_loss).mean().item(),
                              running_ade=torch.tensor(ade_list).mean().item(),
                              running_fde=torch.tensor(fde_list).mean().item(),
                              gen_step_yet=g_steps_yet, disc_step_yet=d_steps_yet)
                t.update()

            if epoch % plot_idx == 0 and epoch != 0:
                with torch.no_grad():
                    model.generator.eval()
                    out = model.generator(batch)
                    model.generator.train()

                target = batch['gt_xy']
                pred = out['out_xy']

                obs_separated = batch['in_xy'].view(batch['in_xy'].shape[0], -1, batch_size, batch['in_xy'].shape[-1])
                target_separated = target.view(target.shape[0], -1, batch_size, target.shape[-1])
                pred_separated = pred.view(pred.shape[0], -1, batch_size, pred.shape[-1])

                loss = model.calculate_loss(pred, target)
                loss = loss.view(model.config.tp_module.datasets.batch_multiplier, -1)
                loss, _ = loss.min(dim=0, keepdim=True)
                loss = torch.mean(loss)

                ade, fde = cal_ade(target, pred, mode='raw'), cal_fde(target, pred, mode='raw')
                ade = ade.view(model.config.tp_module.datasets.batch_multiplier, batch_size) * batch['ratio'][0]
                fde = fde.view(model.config.tp_module.datasets.batch_multiplier, batch_size) * batch['ratio'][0]

                fde, _ = fde.min(dim=0)
                modes_caught = (fde < model.config.tp_module.datasets.mode_dist_threshold).float()

                ade, ade_min_idx = ade.min(dim=0, keepdim=True)

                seq_start_end = batch['seq_start_end']
                frame_nums = batch['in_frames'].view(model.config.tp_module.datasets.batch_multiplier, 8, batch_size)
                track_lists = batch['in_tracks'].view(model.config.tp_module.datasets.batch_multiplier, 8, batch_size)

                random_trajectory_idx = np.random.choice(frame_nums.shape[-1], 1, replace=False).item()

                dataset_idx = batch['dataset_idx'][random_trajectory_idx].item()

                obs_trajectory = obs_separated[:, :, random_trajectory_idx, ...]
                gt_trajectory = target_separated[:, :, random_trajectory_idx, ...]
                pred_trajectory = pred_separated[:, :, random_trajectory_idx, ...]

                frame_num = int(frame_nums[0, random_trajectory_idx, ...][0].item())
                track_num = int(track_lists[0, random_trajectory_idx, ...][0].item())

                current_dataset = loader.dataset.datasets[dataset_idx].dataset \
                    if cfg.tp_module.datasets.use_standard_dataset else loader.dataset.datasets[dataset_idx]
                if isinstance(current_dataset, SmoothTrajectoryDataset):
                    current_dataset = current_dataset.base_dataset

                if cfg.tp_module.datasets.use_standard_dataset:
                    video_path = f"{cfg.root}videos/{getattr(SDDVideoClasses, current_dataset.video_class).value}" \
                                 f"/video{current_dataset.video_number}/video.mov"
                else:
                    video_path = f"{cfg.root}videos/{current_dataset.video_class.value}" \
                                 f"/video{current_dataset.video_number}/video.mov"

                frame = extract_frame_from_video(video_path, frame_num)

                ade = ade.squeeze()[random_trajectory_idx]
                fde = fde.squeeze()[random_trajectory_idx]

                # single for give data
                obs_trajectory = obs_trajectory[:, 0, :]
                gt_trajectory = gt_trajectory[:, 0, :]

                plot_trajectory_alongside_frame_stochastic(
                    frame, obs_trajectory.cpu(), gt_trajectory.cpu(), pred_trajectory.cpu(), frame_num,
                    track_id=track_num, additional_text=f"ADE: {ade.item()} | FDE: {fde.item()}",
                    best_idx=ade_min_idx.squeeze()[random_trajectory_idx].item())


@hydra.main(config_path="config", config_name="config")
def overfit(cfg):
    device = 'cuda:0'
    epochs = 1000
    plot_idx = 100
    batch_size = 8

    logger.info(f"Overfit - Setting up dataset and model")
    train_dataset, val_dataset = setup_dataset(cfg)

    model = setup_model(cfg, train_dataset, val_dataset)
    model.to(device)

    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False,
                        collate_fn=collate_fn)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False,
                            collate_fn=collate_fn)

    opt = torch.optim.Adam(model.parameters(), lr=model.config.tp_module.optimizer.lr,
                           weight_decay=model.config.tp_module.optimizer.weight_decay,
                           amsgrad=model.config.tp_module.optimizer.amsgrad)
    sch = ReduceLROnPlateau(opt,
                            patience=model.config.tp_module.scheduler.patience,
                            verbose=model.config.tp_module.scheduler.verbose,
                            factor=model.config.tp_module.scheduler.factor,
                            min_lr=model.config.tp_module.scheduler.min_lr)

    train_loss, ade_list, fde_list = [], [], []
    for epoch in range(epochs):
        model.train()
        with tqdm(loader, position=0) as t:
            t.set_description('Epoch %i' % epoch)
            for b_idx, batch in enumerate(loader):
                opt.zero_grad()

                # batch = TrajectoryGANTransformerV2.get_k_batches(batch, 10)
                # batch = {k: v.to(device) for k, v in batch.items() if k != 'size'}

                batch = {k: v.to(device) for k, v in batch.items()}
                target = batch['gt_xy']

                out = model(batch)

                pred = out['out_xy']

                loss = model.calculate_loss(pred, target)

                ade, fde = model.calculate_metrics(pred, target, model.config.tp_module.metrics.mode)

                t.set_postfix(loss=loss.item(), ade=ade.item(), fde=fde.item(),
                              running_loss=torch.tensor(train_loss).mean().item(),
                              running_ade=torch.tensor(ade_list).mean().item(),
                              running_fde=torch.tensor(fde_list).mean().item())
                t.update()

                train_loss.append(loss.item())
                ade_list.append(ade.item())
                fde_list.append(fde.item())

                loss.backward()
                opt.step()

            if epoch % plot_idx == 0:
                seq_start_end = batch['seq_start_end']
                frame_nums = batch['in_frames']
                track_lists = batch['in_tracks']

                random_trajectory_idx = np.random.choice(frame_nums.shape[1], 1, replace=False).item()

                obs_trajectory = batch['in_xy'][:, random_trajectory_idx, ...]
                gt_trajectory = batch['gt_xy'][:, random_trajectory_idx, ...]
                pred_trajectory = out['out_xy'][:, random_trajectory_idx, ...]

                frame_num = int(frame_nums[:, random_trajectory_idx, ...][0].item())
                track_num = int(track_lists[:, random_trajectory_idx, ...][0].item())

                dataset_idx = batch['dataset_idx'][random_trajectory_idx].item()
                current_dataset = loader.dataset.datasets[dataset_idx].dataset \
                    if cfg.tp_module.datasets.use_standard_dataset else loader.dataset.datasets[dataset_idx]
                if isinstance(current_dataset, SmoothTrajectoryDataset):
                    current_dataset = current_dataset.base_dataset

                if cfg.tp_module.datasets.use_standard_dataset:
                    video_path = f"{cfg.root}videos/{getattr(SDDVideoClasses, current_dataset.video_class).value}" \
                                 f"/video{current_dataset.video_number}/video.mov"
                else:
                    video_path = f"{cfg.root}videos/{current_dataset.video_class.value}" \
                                 f"/video{current_dataset.video_number}/video.mov"

                frame = extract_frame_from_video(video_path, frame_num)

                if isinstance(model, TransformerNoisyMotionGenerator):
                    preds = []
                    with torch.no_grad():
                        for i in range(10):
                            multi_out = model(batch)
                            preds.append(multi_out['out_xy'][:, random_trajectory_idx, ...])
                        preds = torch.stack(preds, dim=1).cpu()
                        plot_trajectory_alongside_frame_stochastic(
                            frame, obs_trajectory.cpu(), gt_trajectory.cpu(), preds, frame_num, track_id=track_num,
                            additional_text=f"ADE: {ade.item()} | FDE: {fde.item()}")
                else:
                    plot_trajectory_alongside_frame(
                        frame, obs_trajectory.cpu(), gt_trajectory.cpu(), pred_trajectory.detach().cpu(), frame_num,
                        track_id=track_num, additional_text=f"ADE: {ade.item()} | FDE: {fde.item()}")


if __name__ == '__main__':
    # overfit()
    # overfit_gan()
    # evaluate()
    # train_lightning()
    evaluate_stochastic()
