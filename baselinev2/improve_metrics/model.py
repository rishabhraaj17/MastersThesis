import enum
import warnings
from typing import T, Sequence, List

import hydra
import torch
from torch import nn
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from torch._utils import _accumulate
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, Dataset, Subset, ConcatDataset

from average_image.constants import SDDVideoClasses
from baselinev2.config import MANUAL_SEED
from baselinev2.improve_metrics.dataset import PatchesDataset
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import resize_frames

initialize_logging()
logger = get_logger('baselinev2.improve_metrics.models')

torch.manual_seed(MANUAL_SEED)
VIDEO_CLASS = SDDVideoClasses.DEATH_CIRCLE


class Activations(enum.Enum):
    TANH = 'tanh'
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size=3, batch_norm=False, non_lin=Activations.RELU, dropout=0.,
                 first_block=False, last_block=False, skip_connection=False, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.skip_connection = skip_connection
        self.last_block = last_block
        self.first_block = first_block
        self.Block = nn.Sequential()
        self.Block.add_module("conv", nn.Conv2d(in_channels=input_dim, out_channels=output_dim,
                                                kernel_size=filter_size, stride=stride, padding=padding))
        if batch_norm:
            self.Block.add_module("bn", nn.BatchNorm2d(output_dim))
        if non_lin == Activations.TANH:
            self.Block.add_module("non_linearity", nn.Tanh())
        elif non_lin == Activations.RELU:
            self.Block.add_module("non_linearity", nn.ReLU())
        elif non_lin == Activations.LEAKY_RELU:
            self.Block.add_module("non_linearity", nn.LeakyReLU())
        else:
            assert False, "non_linearity = {} not valid: 'tanh', 'relu', 'leaky_relu'".format(non_lin)

        self.Block.add_module("pool", nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False))
        if dropout > 0:
            self.Block.add_module("drop", nn.Dropout2d(dropout))

    def forward(self, x, ):
        if self.skip_connection:
            if not self.first_block:
                x, skip_con_list = x
            else:
                skip_con_list = []

        x = self.Block(x)
        if self.skip_connection:
            if not self.last_block:
                skip_con_list.append(x)
            x = [x, skip_con_list]

        return x


def make_conv_blocks(input_dim, output_dim, filter_size, stride, padding, batch_norm=False, non_lin=Activations.RELU,
                     dropout=0., first_block=False, last_block=False, skip_connection=False):
    layers = []
    for idx, (out_dim, kernel_shape, s, p) in enumerate(zip(output_dim, filter_size, stride, padding)):
        if idx == 0:
            layers.append(ConvBlock(input_dim=input_dim, output_dim=out_dim, filter_size=kernel_shape,
                                    batch_norm=batch_norm, non_lin=non_lin, dropout=dropout, first_block=first_block,
                                    last_block=last_block, skip_connection=skip_connection, stride=s, padding=p))
        else:
            layers.append(ConvBlock(input_dim=output_dim[idx - 1], output_dim=out_dim, filter_size=kernel_shape,
                                    batch_norm=batch_norm, non_lin=non_lin, dropout=dropout, first_block=first_block,
                                    last_block=last_block, skip_connection=skip_connection, stride=s, padding=p))
    return nn.Sequential(*layers)


def make_classifier_block(in_feat, out_feat, non_lin=Activations.RELU):
    layers = nn.Sequential()
    layers.add_module('adaptive_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
    layers.add_module('flatten', nn.Flatten())
    for idx, o_feat in enumerate(out_feat):
        if idx == 0:
            layers.add_module(f'linear_{idx}', nn.Linear(in_features=in_feat, out_features=o_feat))
        else:
            layers.add_module(f'linear_{idx}', nn.Linear(in_features=out_feat[idx - 1], out_features=o_feat))
        if idx < len(out_feat) - 1:
            if non_lin == Activations.TANH:
                layers.add_module(f'non_lin_{idx}', nn.Tanh())
            elif non_lin == Activations.RELU:
                layers.add_module(f'non_lin_{idx}', nn.ReLU())
            elif non_lin == Activations.LEAKY_RELU:
                layers.add_module(f'non_lin_{idx}', nn.LeakyReLU())
            else:
                assert False, "non_linearity = {} not valid: 'tanh', 'relu', 'leaky_relu'".format(non_lin)

    return layers


def make_datasets(cfg, video_class, return_test_split=False, plot=False):
    dataset = PatchesDataset(root=cfg.dataset.root, video_label=video_class, frames_per_clip=1,
                             num_workers=cfg.dataset.num_workers, num_videos=cfg.dataset.num_videos,
                             video_number_to_use=cfg.dataset.video_to_use, plot=plot,
                             step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                             single_track_mode=False, track_id=5, multiple_videos=cfg.dataset.multiple_videos,
                             use_generated=cfg.use_generated_dataset, merge_annotations=cfg.dataset.merge_annotations)
    total_len = len(dataset)
    train_len = int(total_len * cfg.dataset.ratio.train)
    test_len = total_len - train_len

    train_dataset, test_dataset = random_split(dataset, [train_len, test_len],
                                               generator=torch.Generator().manual_seed(MANUAL_SEED))

    pre_val_split_len = train_len
    train_len = int(pre_val_split_len * cfg.dataset.ratio.train)
    val_len = pre_val_split_len - train_len

    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len],
                                              generator=torch.Generator().manual_seed(MANUAL_SEED))
    if return_test_split:
        return train_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset


def linear_split(dataset: Dataset[T], lengths: Sequence[int]) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> linear_split(range(10), [3, 7])

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.arange(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def make_datasets_linear_split(cfg, video_class, return_test_split=False, plot=False):
    dataset = PatchesDataset(root=cfg.dataset.root, video_label=video_class, frames_per_clip=1,
                             num_workers=cfg.dataset.num_workers, num_videos=cfg.dataset.num_videos,
                             video_number_to_use=cfg.dataset.video_to_use, plot=plot,
                             step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                             single_track_mode=False, track_id=5, multiple_videos=cfg.dataset.multiple_videos,
                             use_generated=cfg.use_generated_dataset, merge_annotations=cfg.dataset.merge_annotations)
    total_len = len(dataset)
    train_len = int(total_len * cfg.dataset.ratio.train)
    test_len = total_len - train_len

    pre_val_split_len = train_len
    train_len = int(pre_val_split_len * cfg.dataset.ratio.train)
    val_len = pre_val_split_len - train_len

    train_dataset, val_dataset, test_dataset = linear_split(dataset, [train_len, val_len, test_len])
    if return_test_split:
        return train_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset


def make_datasets_simple(cfg, video_class, return_test_split=False, plot=False):
    train_datasets = []
    for n in cfg.dataset.num_videos:
        train_datasets.append(PatchesDataset(root=cfg.dataset.root, video_label=video_class, frames_per_clip=1,
                                             num_workers=cfg.dataset.num_workers, num_videos=1,
                                             video_number_to_use=n, plot=plot,
                                             step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                             single_track_mode=False, track_id=5, multiple_videos=False,
                                             use_generated=cfg.use_generated_dataset, merge_annotations=False))
    val_datasets = []
    for n in cfg.dataset.val_num_videos:
        val_datasets.append(PatchesDataset(root=cfg.dataset.root, video_label=video_class, frames_per_clip=1,
                                           num_workers=cfg.dataset.num_workers, num_videos=1,
                                           video_number_to_use=n, plot=plot,
                                           step_between_clips=1, transform=resize_frames, scale=1, frame_rate=30,
                                           single_track_mode=False, track_id=5, multiple_videos=False,
                                           use_generated=cfg.use_generated_dataset, merge_annotations=False))
    train_dataset, val_dataset = ConcatDataset(train_datasets), ConcatDataset(val_datasets)
    return train_dataset, val_dataset


class PersonClassifier(LightningModule):
    def __init__(self, conv_block, classifier_block, train_dataset, val_dataset, batch_size=1, num_workers=0,
                 use_batch_norm=False, shuffle: bool = False, pin_memory: bool = True, lr=1e-5, collate_fn=None):
        super(PersonClassifier, self).__init__()
        self.conv_block = conv_block
        self.classifier_block = classifier_block

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_batch_norm = use_batch_norm
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.lr = lr
        self.collate_fn = collate_fn
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, x):
        out = self.conv_block(x)
        return self.classifier_block(out)

    def one_step(self, batch):
        gt, fp = batch
        patches = torch.cat((gt['patches'], fp['patches']), dim=0)
        labels = torch.cat((gt['labels'], fp['labels']), dim=0).view(-1, 1)
        out = self(patches)

        loss = self.loss_fn(out, labels)

        accuracy = (torch.round(torch.sigmoid(out)).eq(labels)).float().mean()
        return loss, accuracy

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=self.collate_fn,
                          num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size * 2, collate_fn=self.collate_fn,
                          num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.one_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(opt, patience=15, verbose=True, factor=0.2, cooldown=2),
                'monitor': 'val_loss_epoch',
                'interval': 'epoch',
                'frequency': 1
            }]
        return [opt], schedulers


def people_collate_fn(batch):
    gt_patches, fp_patches = [], []
    gt_labels, fp_labels = [], []
    for ele in batch:
        gt, fp = ele
        gt_patches.append(gt['patches'])
        fp_patches.append(fp['patches'])

        gt_labels.append(gt['labels'])
        fp_labels.append(fp['labels'])

    gt_patches = torch.cat(gt_patches, dim=0)
    fp_patches = torch.cat(fp_patches, dim=0)

    gt_labels = torch.cat(gt_labels)
    fp_labels = torch.cat(fp_labels)

    return [{'patches': gt_patches, 'labels': gt_labels}, {'patches': fp_patches, 'labels': fp_labels}]


@hydra.main(config_path="config", config_name="config")
def model_trainer(cfg):
    train_dataset, val_dataset = make_datasets_simple(cfg, VIDEO_CLASS, return_test_split=False, plot=False)
    conv_layers = make_conv_blocks(cfg.input_dim, cfg.out_channels, cfg.kernel_dims, cfg.stride, cfg.padding,
                                   cfg.batch_norm, non_lin=Activations.RELU, dropout=cfg.dropout)
    classifier_layers = make_classifier_block(cfg.in_feat, cfg.out_feat, Activations.RELU)

    model = PersonClassifier(conv_block=conv_layers, classifier_block=classifier_layers,
                             train_dataset=train_dataset, val_dataset=val_dataset,
                             batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=cfg.shuffle,
                             pin_memory=cfg.pin_memory, lr=cfg.lr, collate_fn=people_collate_fn)

    trainer = Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                      fast_dev_run=cfg.trainer.fast_dev_run)
    trainer.fit(model)
    print()


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model_trainer()
