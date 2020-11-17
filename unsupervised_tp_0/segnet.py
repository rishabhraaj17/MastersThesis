import argparse
import warnings
from typing import Any, Tuple, Union, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import pytorch_lightning as pl

import matplotlib.pyplot as plt

from average_image.constants import SDDVideoClasses
from log import initialize_logging, get_logger
from unsupervised_tp_0.dataset import resize_frames, SDDSimpleDataset
from unsupervised_tp_0.train import LOSS_RECONSTRUCTION, get_args_parser

device = 'cpu'

initialize_logging()
logger = get_logger(__name__)


class BasicBlockSegNet(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, k_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int]],
                 padding: Union[int, Tuple[int]], bias: bool = True, dilation: int = 1, with_bn: bool = True):
        super(BasicBlockSegNet, self).__init__()
        conv = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=k_size, padding=padding,
                         stride=stride, bias=bias, dilation=dilation)
        if with_bn:
            self.layers = nn.Sequential(conv,
                                        nn.BatchNorm2d(n_filters),
                                        nn.ReLU(inplace=True))
        else:
            self.layers = nn.Sequential(conv, nn.ReLU(inplace=True))

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class SegNetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegNetDown2, self).__init__()
        self.conv1 = BasicBlockSegNet(in_size, out_size, 3, 1, 1)
        self.conv2 = BasicBlockSegNet(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegNetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegNetDown3, self).__init__()
        self.conv1 = BasicBlockSegNet(in_size, out_size, 3, 1, 1)
        self.conv2 = BasicBlockSegNet(out_size, out_size, 3, 1, 1)
        self.conv3 = BasicBlockSegNet(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegNetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegNetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = BasicBlockSegNet(in_size, in_size, 3, 1, 1)
        self.conv2 = BasicBlockSegNet(in_size, out_size, 3, 1, 1)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class SegNetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegNetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = BasicBlockSegNet(in_size, in_size, 3, 1, 1)
        self.conv2 = BasicBlockSegNet(in_size, in_size, 3, 1, 1)
        self.conv3 = BasicBlockSegNet(in_size, out_size, 3, 1, 1)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class SegNet(pl.LightningModule):
    def __init__(self, n_classes=3, in_channels=3, is_unpooling=True, lr=1e-5, train_loader=None,
                 val_loader=None):
        super(SegNet, self).__init__()

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.save_hyperparameters('lr')

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = SegNetDown2(self.in_channels, 64)
        self.down2 = SegNetDown2(64, 128)
        self.down3 = SegNetDown3(128, 256)
        self.down4 = SegNetDown3(256, 512)
        self.down5 = SegNetDown3(512, 512)

        self.up5 = SegNetUp3(512, 512)
        self.up4 = SegNetUp3(512, 256)
        self.up3 = SegNetUp3(256, 128)
        self.up2 = SegNetUp2(128, 64)
        self.up1 = SegNetUp2(64, n_classes)

        self.lr = lr

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.val_loader

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.layers, conv_block.conv2.layers]
            else:
                units = [
                    conv_block.conv1.layers,
                    conv_block.conv2.layers,
                    conv_block.conv3.layers,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


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
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory, drop_last=True)
    logger.info(f"DataLoaders built successfully")

    if inference_mode:
        model = SegNet.load_from_checkpoint('../lightning_logs/version_254245/checkpoints/epoch=19.ckpt')
        logger.info(f"Inference Network: {model.__class__.__name__}")
        logger.info(f"Starting Inference")
        model.eval()
        frames, _ = next(iter(train_loader))
        frames = frames.squeeze(1)
        pred = model(frames)
        plot = make_grid(torch.cat((frames, pred)), nrow=2)
        plt.imshow(plot.permute(1, 2, 0).detach().numpy())
        plt.show()
    else:
        model = SegNet(train_loader=train_loader, val_loader=val_loader)
        logger.info(f"Train Network: {model.__class__.__name__}")
        logger.info(f"Starting Training")
        trainer = pl.Trainer(auto_scale_batch_size=False, gpus=1, max_epochs=args.epochs)
        trainer.fit(model=model)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        vid_label = SDDVideoClasses.QUAD

        parser_ = argparse.ArgumentParser('Training Script', parents=[get_args_parser()])
        parsed_args = parser_.parse_args()

        main(parsed_args, video_label=vid_label, inference_mode=False, train_video_num=0,
             val_video_num=1)

    # m = SegNet().to(device)
    # # inp = torch.randn((1, 3, 320, 240))
    # inp = torch.randn((2, 3, 1945, 1422))
    # inp = nn.functional.interpolate(inp, scale_factor=0.25)
    # print(inp.size())
    # o = m(inp)
    # print(o.size())
