import os
import warnings

import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.nn.dataset import ConcatenateDataset
from baselinev2.stochastic.model import BaselineGAN
from log import get_logger
from src.position_maps.interplay_utils import setup_multiple_frame_only_datasets_core
from src.position_maps.train_trajectory import setup_dataset
from src.position_maps.utils import get_scaled_shapes_with_pad_values
from src_lib.datasets.extracted_dataset import extracted_collate, get_train_and_val_datasets, get_test_datasets
from src_lib.datasets.trajectory_stgcnn import seq_collate_with_dataset_idx_dict

warnings.filterwarnings("ignore")

seed_everything(42)
logger = get_logger(__name__)


def setup_frame_only_dataset_flexible(cfg, video_class, video_number):
    df, rgb_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=video_class,
        video_numbers=video_number,
        desired_ratio=cfg.desired_pixel_to_meter_ratio_rgb)
    df_target, target_max_shape = get_scaled_shapes_with_pad_values(
        root_path=cfg.root, video_classes=video_class,
        video_numbers=video_number,
        desired_ratio=cfg.desired_pixel_to_meter_ratio)
    train_dataset = setup_multiple_frame_only_datasets_core(
        cfg=cfg, video_classes_to_use=video_class,
        video_numbers_to_use=video_number,
        num_videos=-1, multiple_videos=False, df=df, df_target=df_target, use_common_transforms=False)
    return train_dataset


def setup_trajectory_dataset(cfg):
    logger.info(f"Setting up trajectory dataset")
    # train_dataset, val_dataset = setup_dataset(cfg)
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
    test_dataset = get_test_datasets(
        video_classes=[getattr(SDDVideoClasses, v_c) for v_c in cfg.tp_module.datasets.test.video_classes],
        video_numbers=cfg.tp_module.datasets.test.video_numbers,
        meta_label=[getattr(SDDVideoDatasets, v_c) for v_c in cfg.tp_module.datasets.test.video_classes],
        get_generated=cfg.tp_module.datasets.use_generated,
        meta_path='../../../Datasets/SDD/H_SDD.txt',
        root='../../../Datasets/SDD/pm_extracted_annotations/'
        if cfg.tp_module.datasets.use_generated else '../../../Datasets/SDD_Features/'
    )
    trajectory_dataset = ConcatenateDataset([train_dataset, val_dataset, test_dataset])
    return trajectory_dataset


def get_trajectory_loader(cfg, trajectory_dataset):
    if cfg.tp_module.datasets.use_standard_dataset:
        collate_fn = seq_collate_with_dataset_idx_dict
    else:
        collate_fn = extracted_collate
    # trajectory_train_loader = DataLoader(
    #     train_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    # trajectory_val_loader = DataLoader(
    #     val_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    trajectory_loader = DataLoader(
        trajectory_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    return trajectory_loader


def setup_trajectory_model():
    # trajectory_model = BaselineGAN(OmegaConf.merge(
    #     OmegaConf.load('../../baselinev2/stochastic/config/model/model.yaml'),
    #     OmegaConf.load('../../baselinev2/stochastic/config/training/training.yaml'),
    #     OmegaConf.load('../../baselinev2/stochastic/config/eval/eval.yaml'),
    # ))
    logger.info(f"Setting up trajectory model")
    version = 12

    base_path = f'../../../baselinev2/stochastic/logs/lightning_logs/version_{version}'
    checkpoint_path = os.path.join(base_path, 'checkpoints/')
    checkpoint_files = os.listdir(checkpoint_path)
    epoch_part_list = [c.split('-')[0] for c in checkpoint_files]
    epoch_part_list = np.array([int(c.split('=')[-1]) for c in epoch_part_list]).argsort()
    checkpoint_files = np.array(checkpoint_files)[epoch_part_list]
    checkpoint_file = checkpoint_path + checkpoint_files[-1]
    hparam_path = os.path.join(base_path, 'hparams.yaml')

    trajectory_model = BaselineGAN.load_from_checkpoint(
        checkpoint_path=checkpoint_file, hparams_file=hparam_path, map_location='cuda:0')

    return trajectory_model


@hydra.main(config_path="config", config_name="config")
def baseline_interplay(cfg):
    trajectory_dataset = setup_trajectory_dataset(cfg)

    trajectory_loader = get_trajectory_loader(cfg, trajectory_dataset)

    trajectory_model = setup_trajectory_model()

    logger.info(f"Setting up video dataset")

    video_train_dataset = setup_frame_only_dataset_flexible(
        cfg=cfg, video_class=cfg.interplay.video_class, video_number=cfg.interplay.video_number)
    print()


if __name__ == '__main__':
    baseline_interplay()
