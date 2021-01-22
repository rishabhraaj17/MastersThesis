import math
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch_summary import summary
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger(__name__)

CFG = {
    'encoder': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'decoder': [512, 512, 'U', 256, 256, 'U', 256, 256, 'U', 128, 64, 32, 16, 3]
}


def make_layers(cfg, batch_norm=False, encoder=True, last_without_activation=True):
    layers = []
    if encoder:
        in_channels = 3
    else:
        in_channels = 512
    if last_without_activation:
        for v in cfg[:-1]:
            in_channels, layers = core_layers_maker(batch_norm, in_channels, layers, v)
        layers += [nn.Conv2d(in_channels, cfg[-1], kernel_size=3, padding=1)]
    else:
        for v in cfg:
            in_channels, layers = core_layers_maker(batch_norm, in_channels, layers, v)
    return nn.Sequential(*layers)


def core_layers_maker(batch_norm, in_channels, layers, v):
    if v == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    elif v == 'U':
        layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
    else:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
        else:
            layers += [conv2d, nn.LeakyReLU(inplace=True)]
        in_channels = v
    return in_channels, layers


def gaussian(x, y, height, width, sigma=5):
    channel = [math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(height) for c in range(width)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(height, width))
    return channel


def flatten_calculator(x):
    return x.size(1) * x.size(2) * x.size(3)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DynamicPointModeler(pl.LightningModule):
    def __init__(self, forward_dynamic_point_extractor, backward_dynamic_point_extractor, train_loader=None,
                 val_loader=None):
        super(DynamicPointModeler, self).__init__()
        self.encoder = make_layers(CFG['encoder'], batch_norm=True)
        self.forward_dynamic_point_extractor = forward_dynamic_point_extractor
        self.backward_dynamic_point_extractor = backward_dynamic_point_extractor
        self.pre_decoder = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.decoder = make_layers(CFG['decoder'], batch_norm=True, encoder=False)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def forward(self, x):
        ref_image, image_t_minus_one, image_t, image_t_plus_one = x
        bg_features = self.encoder(ref_image)
        h, w = bg_features.size(2), bg_features.size(3)
        forward_input = torch.cat((image_t_minus_one, image_t), dim=0)
        backward_input = torch.cat((image_t, image_t_plus_one), dim=0)
        forward_features, keypoints_forward = self.forward_dynamic_point_extractor(forward_input, h, w)
        backward_features, keypoints_backward = self.backward_dynamic_point_extractor(backward_input, h, w)
        out = torch.cat((bg_features, forward_features, backward_features), dim=1)
        out = self.pre_decoder(out)
        out = self.decoder(out)
        if out.size() != image_t.size():
            out = torch.nn.functional.interpolate(out, size=(image_t.size(2), image_t.size(3)), mode='bilinear')
        return out, keypoints_forward, keypoints_backward


class StaticFeaturesEncoder(pl.LightningModule):
    def __init__(self):
        super(StaticFeaturesEncoder, self).__init__()
        self.layers = make_layers(CFG['encoder'], batch_norm=True)


class DynamicPointExtractor(pl.LightningModule):
    def __init__(self):
        super(DynamicPointExtractor, self).__init__()
        self.encoder = make_layers(CFG['encoder'], batch_norm=True, last_without_activation=False)
        self.post_encoder = nn.Sequential(LambdaLayer(lambda x: x.view(x.size(0), x.size(1), -1)),
                                          nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),
                                          nn.ReLU(inplace=True))
        self.keypoint_extractor = nn.Sequential(nn.Flatten(),
                                                nn.Linear(in_features=32 * 2632, out_features=20480),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(in_features=20480, out_features=5120),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(in_features=5120, out_features=1024),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(in_features=1024, out_features=512),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(in_features=512, out_features=256))

    def forward(self, x, h, w, sigma=1):
        out = self.encoder(x)
        out = self.post_encoder(out)
        out = self.keypoint_extractor(out)
        channels = out.size(1) // 2
        out_ = out.view(out.size(0), channels, 2)
        out_ = out_.mean(dim=0).unsqueeze(dim=0)  # reconfirm
        out = torch.softmax(out_, dim=-1)
        out = out_ - out
        heatmaps = torch.zeros((0, channels, h, w), requires_grad=True)
        for batch_item in out:
            heatmap = torch.zeros((0, h, w))
            for keypoint in batch_item:
                x, y = keypoint.detach()
                x, y = x.item(), y.item()
                out_map = gaussian(x, y, h, w, sigma)
                heatmap = torch.cat((heatmap, torch.from_numpy(out_map).unsqueeze(0)))
            heatmaps = torch.cat((heatmaps, heatmap.unsqueeze(0)))
        return heatmaps, out_


def overfit(model, path, num_epochs, device, writer, opt):
    # ref_image, image_t_minus_one, image_t, image_t_plus_one = extract_overfit_dataset(path)
    ref_image, image_t_minus_one, image_t, image_t_plus_one = load_images()
    loss_fn = torch.nn.functional.mse_loss
    ref_image, image_t_minus_one, image_t, image_t_plus_one = ref_image.float().unsqueeze(0).to(device) / 255.0, \
                                                              image_t_minus_one.float().unsqueeze(0).to(device) / 255.0, \
                                                              image_t.float().unsqueeze(0).to(device) / 255.0, \
                                                              image_t_plus_one.float().unsqueeze(0).to(device) / 255.0
    for epoch in tqdm(range(num_epochs)):
        # forward pass
        pred, keypoints_forward, keypoints_backward = model([ref_image, image_t_minus_one, image_t, image_t_plus_one])
        # loss
        consistency_loss = loss_fn(keypoints_forward, keypoints_backward)
        reconstruction_loss = loss_fn(pred, image_t)
        loss = consistency_loss + reconstruction_loss
        writer.add_scalar('loss', loss.item())
        logger.info(f'Epoch: {epoch}, Loss: {loss.item()}')
        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()


def extract_overfit_dataset(path):
    reader = torchvision.io.read_video(path, start_pts=0, end_pts=1, pts_unit='sec')
    frames = reader[0].permute(0, 3, 1, 2)  # .float() / 255.0
    # frames = torch.nn.functional.interpolate(frames, scale_factor=0.25, mode='bilinear')

    ref_image, image_t_minus_one, image_t, image_t_plus_one = None, None, None, None
    for f_idx, frame in enumerate(frames):
        if f_idx == 0:
            ref_image = frame
        elif f_idx == 1:
            image_t_minus_one = frame
        elif f_idx == 13:
            image_t = frame
        elif f_idx == 25:
            image_t_plus_one = frame
        else:
            continue
    torchvision.io.write_jpeg(ref_image, 'overfit_images/ref_image.jpeg', 100)
    torchvision.io.write_jpeg(image_t, 'overfit_images/image_t.jpeg', 100)
    torchvision.io.write_jpeg(image_t_minus_one, 'overfit_images/image_t_minus_one.jpeg', 100)
    torchvision.io.write_jpeg(image_t_plus_one, 'overfit_images/image_t_plus_one.jpeg', 100)
    return ref_image, image_t_minus_one, image_t, image_t_plus_one


def load_images():
    ref_image = torchvision.io.read_image('overfit_images/ref_image.jpeg')
    image_t = torchvision.io.read_image('overfit_images/image_t.jpeg')
    image_t_minus_one = torchvision.io.read_image('overfit_images/image_t_minus_one.jpeg')
    image_t_plus_one = torchvision.io.read_image('overfit_images/image_t_plus_one.jpeg')
    return ref_image, image_t_minus_one, image_t, image_t_plus_one


if __name__ == '__main__':
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = DynamicPointModeler(forward_dynamic_point_extractor=DynamicPointExtractor(),
                                backward_dynamic_point_extractor=DynamicPointExtractor())

    summary(model, [(3, 356, 486), (3, 356, 486), (3, 356, 486), (3, 356, 486)], device='cpu', batch_size=1)

    # # base_path = "../Datasets/SDD/"
    # base_path = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/"
    # vid_label = SDDVideoClasses.LITTLE
    # video_number = 3
    #
    # video_path = f'{base_path}videos/{vid_label.value}/video{video_number}/video.mov'
    #
    # model.to(device)
    # writer = SummaryWriter()
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # overfit(model=model, path=video_path, num_epochs=1000, device=device, writer=writer, opt=opt)
    # torch.save(model.state_dict(), 'dynamic_keypoint_extractor.pt')
