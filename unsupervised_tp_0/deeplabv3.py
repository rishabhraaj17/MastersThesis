import argparse
import warnings
from typing import Union, List

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import pytorch_lightning as pl
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from average_image.constants import SDDVideoClasses
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import SDDSimpleDataset, resize_frames
from unsupervised_tp_0.train import LOSS_RECONSTRUCTION, get_args_parser

initialize_logging()
logger = get_logger(__name__)


class DeepLab(pl.LightningModule):
    def __init__(self, num_classes=3, pretrained=False, lr=1e-5, train_loader=None,
                 val_loader=None):
        super(DeepLab, self).__init__()
        self.deep_lab = deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.lr = lr
        self.save_hyperparameters('lr')

    def forward(self, x):
        return self.deep_lab(x)

    def _one_step(self, batch):
        frames, _ = batch
        frames = frames.squeeze(1)
        out = self(frames)
        reconstruction_loss = LOSS_RECONSTRUCTION(out['out'], frames)
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
        scheduler = ReduceLROnPlateau(opt)
        return [opt], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.val_loader


def main(args, video_label, train_video_num, val_video_num, inference_mode=False):
    logger.info(f"Setting up DataLoaders...")

    train_dataset = SDDSimpleDataset(root=args.dataset_root, video_label=video_label, frames_per_clip=1,
                                     num_workers=args.data_loader_num_workers,
                                     num_videos=1, video_number_to_use=train_video_num,
                                     step_between_clips=1, transform=resize_frames, scale=0.75, frame_rate=30,
                                     single_track_mode=False, track_id=5, multiple_videos=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory, drop_last=True)

    val_dataset = SDDSimpleDataset(root=args.dataset_root, video_label=video_label, frames_per_clip=1,
                                   num_workers=args.data_loader_num_workers,
                                   num_videos=1, video_number_to_use=val_video_num,
                                   step_between_clips=1, transform=resize_frames, scale=0.75, frame_rate=30,
                                   single_track_mode=False, track_id=5, multiple_videos=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory, drop_last=True)
    logger.info(f"DataLoaders built successfully")

    if inference_mode:
        model = DeepLab.load_from_checkpoint('../lightning_logs/version_254348/checkpoints/epoch=20.ckpt',
                                             )
        logger.info(f"Inference Network: {model.__class__.__name__}")
        logger.info(f"Starting Inference")
        model.eval()
        frames, _ = next(iter(val_loader))
        frames = frames.squeeze(1)
        pred = model(frames)
        plot = make_grid(torch.cat((frames, pred['out'])), nrow=2)
        plt.imshow(plot.permute(1, 2, 0).detach().numpy())
        plt.show()
    else:
        model = DeepLab(train_loader=train_loader, val_loader=val_loader, pretrained=False)
        logger.info(f"Train Network: {model.__class__.__name__}")
        logger.info(f"Starting Training")
        trainer = pl.Trainer(auto_scale_batch_size=False, gpus=1, max_epochs=args.epochs)
        trainer.fit(model=model)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        vid_label = SDDVideoClasses.GATES

        parser_ = argparse.ArgumentParser('Training Script', parents=[get_args_parser()])
        parsed_args = parser_.parse_args()

        main(parsed_args, video_label=vid_label, inference_mode=True, train_video_num=3,
             val_video_num=4)
