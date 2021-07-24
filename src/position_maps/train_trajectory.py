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
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.plot_utils import plot_trajectory_alongside_frame, plot_trajectory_alongside_frame_stochastic
from baselinev2.stochastic.losses import cal_ade, cal_fde
from log import get_logger
from src.position_maps.trajectory_utils import get_multiple_datasets, bezier_smoother, splrep_smoother
from src_lib.datasets.extracted_dataset import get_train_and_val_datasets, extracted_collate
from src_lib.datasets.opentraj_based import get_multiple_gt_dataset
from src_lib.datasets.trajectory_stgcnn import seq_collate_with_dataset_idx_dict, SmoothTrajectoryDataset
from src_lib.models_hub import TransformerNoisyMotionGenerator

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


def setup_dataset(cfg):
    if cfg.tp_module.datasets.use_standard_dataset:
        if cfg.tp_module.datasets.use_generated:
            train_dataset, val_dataset = get_multiple_datasets(
                cfg=cfg, split_dataset=True, with_dataset_idx=True,
                smooth_trajectories=cfg.tp_module.smooth_trajectories.enabled,
                smoother=bezier_smoother if cfg.tp_module.smooth_trajectories.smoother == 'bezier' else splrep_smoother,
                threshold=cfg.tp_module.smooth_trajectories.min_length)
        else:
            train_dataset, val_dataset = get_multiple_gt_dataset(
                cfg=cfg, split_dataset=True, with_dataset_idx=True,
                smooth_trajectories=cfg.tp_module.smooth_trajectories.enabled,
                smoother=bezier_smoother if cfg.tp_module.smooth_trajectories.smoother == 'bezier' else splrep_smoother,
                threshold=cfg.tp_module.smooth_trajectories.min_length)
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


@hydra.main(config_path="config", config_name="config")
def evaluate(cfg):
    logger.info(f"Setting up dataset and model")
    train_dataset, val_dataset = setup_dataset(cfg)

    model = setup_model(cfg, train_dataset, val_dataset)

    # load dict
    path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/src/position_maps/logs/wandb/' \
           'run-20210720_204724-3ebzenze/files/' \
           'TrajectoryPredictionBaseline/3ebzenze/checkpoints/epoch=46-step=125677.ckpt'
    state_dict = torch.load(path)['state_dict']
    model.load_state_dict(state_dict)

    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate

    loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False,
                        collate_fn=collate_fn)

    for batch in loader:
        target = batch['gt_xy']

        with torch.no_grad():
            out = model(batch)

        pred = out['out_xy']

        loss = model.calculate_loss(pred, target)
        ade, fde = model.calculate_metrics(pred, target, model.config.tp_module.metrics.mode)

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

        frame = extract_frame_from_video(video_path, frame_num)

        plot_trajectory_alongside_frame(
            frame, obs_trajectory, gt_trajectory, pred_trajectory, frame_num, track_id=track_num,
            additional_text=f"ADE: {ade.item()} | FDE: {fde.item()}")


@hydra.main(config_path="config", config_name="config")
def evaluate_stochastic(cfg):
    logger.info(f"Stochastic - Setting up dataset and model")
    train_dataset, val_dataset = setup_dataset(cfg)

    model = setup_model(cfg, train_dataset, val_dataset)

    # load dict
    path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/src/position_maps/logs/wandb/' \
           'run-20210724_011735-83lknvvr/files/' \
           'TrajectoryPredictionBaseline/83lknvvr/checkpoints/epoch=2-step=72530.ckpt'
    state_dict = torch.load(path)['state_dict']
    model.load_state_dict(state_dict)

    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate

    loader = DataLoader(val_dataset, batch_size=2, shuffle=True, pin_memory=True, drop_last=False,
                        collate_fn=collate_fn)

    for batch in loader:
        batch = model.get_k_batches(batch, model.config.tp_module.datasets.batch_multiplier)
        batch_size = batch["size"]

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

        ade, _ = ade.min(dim=0, keepdim=True)

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
            frame, obs_trajectory, gt_trajectory, pred_trajectory, frame_num, track_id=track_num,
            additional_text=f"ADE: {ade.item()} | FDE: {fde.item()}")


@hydra.main(config_path="config", config_name="config")
def overfit_gan(cfg):
    device = 'cuda:0'
    epochs = 1000
    plot_idx = 100
    batch_size = 8

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
    # sch = ReduceLROnPlateau(opt_gen,
    #                         patience=model.config.tp_module.scheduler.patience,
    #                         verbose=model.config.tp_module.scheduler.verbose,
    #                         factor=model.config.tp_module.scheduler.factor,
    #                         min_lr=model.config.tp_module.scheduler.min_lr)

    train_loss, ade_list, fde_list, adv_loss, disc_adv_loss = [], [], [], [], []
    loss, disc_loss, ade, fde, fake_loss = 0, 0, 0, 0, 0
    for epoch in range(epochs):
        model.train()
        opt_switch = 0
        with tqdm(loader, position=0) as t:
            t.set_description('Epoch %i' % epoch)
            for b_idx, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_size = batch["size"]

                if opt_switch == 0:
                    opt_disc.zero_grad()

                    real_pred = model.discriminator(batch, batch['gt_dxdy'])
                    real_gt = torch.ones_like(real_pred)
                    real_loss = model.desc_loss_function(real_pred, real_gt)

                    # Train with fake
                    with torch.no_grad():
                        model.generator.eval()
                        fake_pred = model.generator(batch)
                        model.generator.train()

                    fake_pred = model.discriminator(batch, fake_pred['out_dxdy'])
                    fake_gt = torch.zeros_like(fake_pred)
                    fake_loss = model.desc_loss_function(fake_pred, fake_gt)

                    disc_loss = real_loss + fake_loss

                    disc_adv_loss.append(disc_loss.item())

                    disc_loss.backward()
                    opt_disc.step()
                    opt_switch = 1

                if opt_switch == 1:
                    opt_gen.zero_grad()

                    x = model.get_k_batches(x, model.config.tp_module.datasets.batch_multiplier)
                    batch_size = x["size"]

                    out = model.generator(x)

                    target = x['gt_xy']
                    pred = out['out_xy']

                    fake_pred = model.discriminator(x, out['out_dxdy'])
                    fake_gt = torch.zeros_like(fake_pred)
                    fake_loss = model.desc_loss_function(fake_pred, fake_gt)

                    loss = model.calculate_loss(pred, target)
                    loss = loss.view(model.config.tp_module.datasets.batch_multiplier, -1)
                    loss, _ = loss.min(dim=0, keepdim=True)
                    loss = torch.mean(loss) + fake_loss

                    ade, fde = cal_ade(target, pred, mode='raw'), cal_fde(target, pred, mode='raw')
                    ade = ade.view(model.config.tp_module.datasets.batch_multiplier, batch_size) * x['ratio'][0]
                    fde = fde.view(model.config.tp_module.datasets.batch_multiplier, batch_size) * x['ratio'][0]

                    ade, _ = ade.min(dim=0, keepdim=True)
                    fde, _ = fde.min(dim=0, keepdim=True)

                    train_loss.append(loss.item())
                    ade_list.append(ade.item())
                    fde_list.append(fde.item())
                    adv_loss.append(fake_loss.item())

                    loss.backward()
                    opt_gen.step()
                    opt_switch = 0

                t.set_postfix(gen_loss=loss.item(), disc_loss=disc_loss.item(),
                              gen_adv_loss=fake_loss.item(), ade=ade.item(), fde=fde.item(),
                              running_disc_loss=torch.tensor(disc_adv_loss).mean().item(),
                              running_gen_adv_loss=torch.tensor(adv_loss).mean().item(),
                              running_gen_loss=torch.tensor(train_loss).mean().item(),
                              running_ade=torch.tensor(ade_list).mean().item(),
                              running_fde=torch.tensor(fde_list).mean().item())
                t.update()

            if epoch % plot_idx == 0:
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

                ade, _ = ade.min(dim=0, keepdim=True)

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
                    frame, obs_trajectory, gt_trajectory, pred_trajectory, frame_num, track_id=track_num,
                    additional_text=f"ADE: {ade.item()} | FDE: {fde.item()}")


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
    overfit()
    # overfit_gan()
    # evaluate()
    # train_lightning()
    # evaluate_stochastic()
