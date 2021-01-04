import argparse
import warnings
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pl_bolts.models.vision.unet import UNet, DoubleConv, Down, Up
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from average_image.constants import SDDVideoClasses
from average_image.utils import rescale_featues
from log import initialize_logging, get_logger
from unsupervised_tp_0.autotrajectory import LambdaLayer, gaussian
from unsupervised_tp_0.convlstm import ConvLSTM, ConvLSTMCell
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames, SDDSimpleDatasetWithAnnotation, \
    resize_frames_with_annotation
from unsupervised_tp_0.train import LOSS_RECONSTRUCTION, get_args_parser

initialize_logging()
logger = get_logger(__name__)


def init_hidden_conv_lstm(batch_size: int, hidden_size: int, im_size: Tuple[int, int], device: torch.device):
    h, w = im_size
    hx = torch.zeros(size=(batch_size, hidden_size, h, w), device=device)
    cx = torch.zeros(size=(batch_size, hidden_size, h, w), device=device)

    torch.nn.init.xavier_normal_(hx)
    torch.nn.init.xavier_normal_(cx)

    return hx, cx


def collate_temporal(batch):
    frames = [batch[0][0], batch[-1][0]]
    centers = [batch[0][3], batch[-1][3]]
    _, _, h0, w0 = frames[0].shape
    _, _, h1, w1 = frames[-1].shape
    masks_0 = generate_position_map((h0, w0), centers[0], sigma=3)
    masks_1 = generate_position_map((h1, w1), centers[-1], sigma=3)
    position_map_0 = np.zeros_like(masks_0[0])
    position_map_1 = np.zeros_like(masks_1[0])
    for mask_0, mask_1 in zip(masks_0, masks_1):
        position_map_0 += mask_0
        position_map_1 += mask_1
    position_map_0 = rescale_featues(position_map_0, max_range=1.0, min_range=0.0)
    position_map_1 = rescale_featues(position_map_1, max_range=1.0, min_range=0.0)
    position_maps = [torch.from_numpy(np.expand_dims(position_map_0, (0, 1))),
                     torch.from_numpy(np.expand_dims(position_map_1, (0, 1)))]
    frames = [torch.cat((frame, position_map), dim=1) for frame, position_map in zip(frames, position_maps)]
    # position_maps = [torch.from_numpy(np.expand_dims(position_map_0, (0, 1))).repeat(1, 3, 1, 1),
    #                  torch.from_numpy(np.expand_dims(position_map_1, (0, 1))).repeat(1, 3, 1, 1)]
    return frames


def generate_position_map(shape, bounding_boxes_centers, sigma: int = 2):
    h, w = shape
    masks = []
    for center in bounding_boxes_centers:
        x, y = center
        masks.append(gaussian(x=x, y=y, height=h, width=w, sigma=sigma))
    return masks


class TemporalUNet(nn.Module):
    def __init__(self, num_classes: int, num_layers: int = 5, features_start: int = 64, bilinear: bool = False,
                 ts: int = 2, in_channels: int = 3):
        super(TemporalUNet, self).__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(in_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        self.conv_lstm_cell = ConvLSTMCell(input_dim=feats, hidden_dim=feats, kernel_size=(3, 3), bias=True)

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)
        self.ts = ts

    def forward(self, x, device):
        # in_img = x[0]
        hx, cx = None, None
        out = []
        for idx in range(self.ts):
            xi = [self.layers[0](x[idx])]
            # xi = [self.layers[0](in_img)]
            # Down path
            for layer in self.layers[1:self.num_layers]:
                xi.append(layer(xi[-1]))
            # ConvLSTM
            if hx is None or cx is None:
                b, c, h, w = xi[-1].shape
                hx, cx = init_hidden_conv_lstm(b, c, (h, w), device=device)
            hx, cx = self.conv_lstm_cell(xi[-1], (hx, cx))
            xi[-1] = hx
            # Up path
            for i, layer in enumerate(self.layers[self.num_layers:-1]):
                xi[-1] = layer(xi[-1], xi[-2 - i])

            xi[-1] = self.layers[-1](xi[-1])
            # in_img = xi[-1].detach()
            out.append(xi[-1])

        return out


class UNetImageReconstruction(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-5, train_loader=None,
                 val_loader=None, num_layers=5, features_start=64, use_billinear=False):
        super(UNetImageReconstruction, self).__init__()
        self.u_net = UNet(num_classes=num_classes, num_layers=num_layers, features_start=features_start,
                          bilinear=use_billinear)

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.lr = lr
        self.save_hyperparameters('lr')

    def forward(self, x):
        return self.u_net(x)

    def _one_step(self, batch):
        frames, _ = batch
        frames = frames.squeeze(1)
        out = self(frames)
        reconstruction_loss = LOSS_RECONSTRUCTION(out, frames)
        return reconstruction_loss

    def training_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        tensorboard_logs = {'val_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = ReduceLROnPlateau(opt, verbose=True, patience=5, min_lr=1e-10)
        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(opt, patience=5, verbose=True, factor=0.1, min_lr=1e-10),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }]
        return [opt], schedulers  # [scheduler]

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.val_loader


class TemporalUNetImageReconstruction(UNetImageReconstruction):
    def __init__(self, num_classes=3, lr=1e-5, train_loader=None,
                 val_loader=None, num_layers=5, features_start=64, use_billinear=False, ts=2,
                 in_channels: int = 3):
        super(TemporalUNetImageReconstruction, self).__init__(num_classes, lr, train_loader, val_loader,
                                                              num_layers, features_start, use_billinear)
        self.u_net = TemporalUNet(num_classes=num_classes, num_layers=num_layers, features_start=features_start,
                                  bilinear=use_billinear, ts=ts, in_channels=in_channels)

    def _one_step(self, batch):
        frames = [frame.to(self.device) for frame in batch]
        out = self.u_net(frames, self.device)
        reconstruction_loss = LOSS_RECONSTRUCTION(out[-1], frames[-1][:, 0:3, ...])
        return reconstruction_loss


def main(args, video_label, train_video_num, val_video_num, inference_mode=False):
    logger.info(f"Setting up DataLoaders...")

    train_dataset = SDDSimpleDataset(root=args.dataset_root, video_label=video_label, frames_per_clip=1,
                                     num_workers=args.data_loader_num_workers,
                                     num_videos=1, video_number_to_use=train_video_num,
                                     step_between_clips=1, transform=resize_frames, scale=0.25, frame_rate=30,
                                     single_track_mode=False, track_id=5, multiple_videos=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory, drop_last=True)

    val_dataset = SDDSimpleDataset(root=args.dataset_root, video_label=video_label, frames_per_clip=1,
                                   num_workers=args.data_loader_num_workers,
                                   num_videos=1, video_number_to_use=val_video_num,
                                   step_between_clips=1, transform=resize_frames, scale=0.25, frame_rate=30,
                                   single_track_mode=False, track_id=5, multiple_videos=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory, drop_last=True)
    logger.info(f"DataLoaders built successfully")

    if inference_mode:
        model = UNetImageReconstruction \
            .load_from_checkpoint('../lightning_logs/version_273142/checkpoints/epoch=23.ckpt')
        logger.info(f"Inference Network: {model.__class__.__name__}")
        logger.info(f"Starting Inference")
        model.eval()
        # frames, _ = next(iter(val_loader))
        skip_c = 0
        for frames in val_loader:
            if skip_c % 72 == 0:
                frames = frames[0].squeeze(1)
                pred = model(frames)
                plot = make_grid(torch.cat((frames, pred)), nrow=2)
                plt.imshow(plot.permute(1, 2, 0).detach().numpy())
                plt.show()
            else:
                skip_c += 1
    else:
        model = UNetImageReconstruction(train_loader=train_loader, val_loader=val_loader, pretrained=False, lr=1e-3)
        logger.info(f"Train Network: {model.__class__.__name__}")
        logger.info(f"Starting Training")
        trainer = pl.Trainer(auto_scale_batch_size=False, gpus=1, max_epochs=args.epochs, accumulate_grad_batches=8)
        trainer.fit(model=model)


def main_temporal(args, video_label, train_video_num, val_video_num, inference_mode=False):
    logger.info(f"Setting up DataLoaders...")

    train_dataset = SDDSimpleDatasetWithAnnotation(root=args.dataset_root, video_label=video_label, frames_per_clip=1,
                                                   num_workers=args.data_loader_num_workers,
                                                   num_videos=1, video_number_to_use=train_video_num,
                                                   step_between_clips=1, transform=resize_frames_with_annotation,
                                                   scale=0.20,
                                                   frame_rate=30,
                                                   single_track_mode=False, track_id=5, multiple_videos=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory, drop_last=True,
                              collate_fn=collate_temporal)

    val_dataset = SDDSimpleDatasetWithAnnotation(root=args.dataset_root, video_label=video_label, frames_per_clip=1,
                                                 num_workers=args.data_loader_num_workers,
                                                 num_videos=1, video_number_to_use=val_video_num,
                                                 step_between_clips=1, transform=resize_frames_with_annotation,
                                                 scale=0.20,
                                                 frame_rate=30,
                                                 single_track_mode=False, track_id=5, multiple_videos=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory, drop_last=True,
                            collate_fn=collate_temporal)
    logger.info(f"DataLoaders built successfully")

    batch = next(iter(train_loader))
    torch.save(batch, 'cat_pm.pt')

    if inference_mode:
        model = TemporalUNetImageReconstruction \
            .load_from_checkpoint('../lightning_logs/version_273142/checkpoints/epoch=23.ckpt')
        logger.info(f"Inference Network: {model.__class__.__name__}")
        logger.info(f"Starting Inference")
        model.eval()
        # frames, _ = next(iter(val_loader))
        skip_c = 0
        for frames in val_loader:
            if skip_c % 72 == 0:
                frames = frames[0].squeeze(1)
                pred = model(frames)
                plot = make_grid(torch.cat((frames, pred)), nrow=2)
                plt.imshow(plot.permute(1, 2, 0).detach().numpy())
                plt.show()
            else:
                skip_c += 1
    else:
        model = TemporalUNetImageReconstruction(train_loader=train_loader, val_loader=val_loader, lr=1e-3,
                                                in_channels=4)
        logger.info(f"Train Network: {model.__class__.__name__}")
        logger.info(f"Starting Training")
        # trainer = pl.Trainer(auto_scale_batch_size=False, gpus=1, max_epochs=args.epochs, accumulate_grad_batches=8)
        trainer = pl.Trainer(auto_scale_batch_size=False, gpus=1, max_epochs=args.epochs, overfit_batches=1,
                             precision=16)
        trainer.fit(model=model)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        vid_label = SDDVideoClasses.LITTLE

        parser_ = argparse.ArgumentParser('Training Script', parents=[get_args_parser()])
        parsed_args = parser_.parse_args()

        # main(parsed_args, video_label=vid_label, inference_mode=True, train_video_num=3,
        #      val_video_num=1)
        #
        # main_temporal(parsed_args, video_label=vid_label, inference_mode=False, train_video_num=3,
        #               val_video_num=1)

        # import torchvision
        # import torch.nn.functional as F
        # import matplotlib.pyplot as plt
        #
        # im = torchvision.io.read_image('overfit_images/old/overfit_unet.jpeg')
        # im = im.unsqueeze(0).float() / 255.0
        # im = F.interpolate(im, scale_factor=0.25, mode='bilinear', align_corners=False)
        #
        # model = UNetImageReconstruction \
        #     .load_from_checkpoint('../lightning_logs/version_273142/checkpoints/epoch=23.ckpt')
        # model.eval()
        # out = model(im)
        # plt.imshow(im.detach().squeeze().permute(1, 2, 0).numpy())
        # plt.show()
        # plt.imshow(out.detach().squeeze().permute(1, 2, 0).numpy())
        # plt.show()

        # import torchvision
        # import matplotlib.pyplot as plt
        #
        # frames = torch.load('overfit_images/frames.pt')
        # # im1 = torchvision.io.read_image('im1.jpeg')
        # # im2 = torchvision.io.read_image('im2.jpeg')
        # # pm = torchvision.io.read_image('p_map.jpeg')
        # # im1, im2, pm = im1.unsqueeze(0).float() / 255.0, im2.unsqueeze(0).float() / 255.0, \
        # #                pm.unsqueeze(0).float() / 255.0
        # #
        # # im1 = torch.cat((im1, pm[:, 0, ...].unsqueeze(1)), dim=1)
        #
        # model = TemporalUNetImageReconstruction \
        #     .load_from_checkpoint('lightning_logs/version_72/checkpoints/epoch=499.ckpt', in_channels=4)
        #
        # model.eval()
        # out = model.u_net(frames, 'cpu')
        # # plt.imshow(frames[0][0, 0:3, ...].detach().squeeze().permute(1, 2, 0).numpy())
        # # plt.show()
        # # plt.imshow(frames[1][0, 0:3, ...].detach().squeeze().permute(1, 2, 0).numpy())
        # # plt.show()
        # # plt.imshow(out[-1].detach().squeeze().permute(1, 2, 0).numpy())
        # # plt.show()
        # #
        # # plt.imshow(out[0].detach().squeeze().permute(1, 2, 0).numpy())
        # # plt.show()

        # from captum.attr import IntegratedGradients
        # from captum.attr import GradientShap
        # from captum.attr import Occlusion
        # from captum.attr import NoiseTunnel
        # from captum.attr import visualization as viz
        #
        # integrated_gradients = IntegratedGradients(model)
        # attributions_ig = integrated_gradients.attribute(im, n_steps=200)
        #
        # default_cmap = LinearSegmentedColormap.from_list('custom blue',
        #                                                  [(0, '#ffffff'),
        #                                                   (0.25, '#000000'),
        #                                                   (1, '#000000')], N=256)
        #
        # _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        #                              np.transpose(im.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        #                              method='heat_map',
        #                              cmap=default_cmap,
        #                              show_colorbar=True,
        #                              sign='positive',
        #                              outlier_perc=1)
