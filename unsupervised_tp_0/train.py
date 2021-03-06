import argparse
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg11_bn
from torchvision.utils import make_grid
from tqdm import tqdm

from average_image.constants import SDDVideoClasses
from average_image.utils import show_img
from log import get_logger, initialize_logging
from unsupervised_tp_0.dataset import SDDDatasetBuilder, SDDTrainDataset, SDDValidationDataset
from unsupervised_tp_0.model import make_layers, vgg_decoder_arch, UnsupervisedTP, VanillaAutoEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOSS_L2 = torch.nn.MSELoss()
LOSS_RECONSTRUCTION = torch.nn.MSELoss()
# LOSS_KL = torch.nn.KLDivLoss(reduction='batchmean')

OPTIMIZER = torch.optim.Adam

SUMMARY_WRITER = SummaryWriter()

initialize_logging()
logger = get_logger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser('Reconstruction Networks', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--data_loader_num_workers', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)

    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--batch_norm', action='store_true', help='Use batch_norm or not')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained encoder')

    parser.add_argument('--save_path', type=str, default="../Checkpoints/",
                        help="Path to save outputs")
    parser.add_argument('--dataset_root', type=str, default="../Datasets/SDD/",
                        help="Path to dataset root")
    parser.add_argument('--checkpoint_path', type=str,
                        default="../Checkpoints/quad_batch_size8_2020-09-29 22:57:04.020244.pt",
                        help="Path to checkpoint")

    return parser


def train_one_epoch(model, data_loader, optimizer, scale_factor=0.25, log_nth=100):
    model.train()
    train_loss = 0
    i = 0
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        frames, _, _ = batch
        frames = frames.float().squeeze() / 255.0
        frames = frames.to(device)
        frames = F.interpolate(frames, scale_factor=scale_factor)
        # out, pos_map_1, pos_map_2 = model(frames, None)
        out = model(frames)
        # pos_map_1, pos_map_2 = torch.sigmoid(pos_map_1), torch.sigmoid(pos_map_2)   # TODO: Decide a threshold
        # consistency_loss = LOSS_L2(pos_map_1, pos_map_2)
        # reconstruction_loss = LOSS_RECONSTRUCTION(out, frames[1].unsqueeze(0))
        reconstruction_loss = LOSS_RECONSTRUCTION(out, frames)
        # kl_loss = -LOSS_KL(out, frames)

        optimizer.zero_grad()
        # loss = consistency_loss + reconstruction_loss  # + kl_loss
        loss = reconstruction_loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # SUMMARY_WRITER.add_scalar('train_loss/consistency_loss', consistency_loss.item())
        SUMMARY_WRITER.add_scalar('train_loss/reconstruction_loss', reconstruction_loss.item())
        # SUMMARY_WRITER.add_scalar('train_loss/kl_loss', kl_loss.item())
        # SUMMARY_WRITER.add_scalar('train_loss/summed_loss', loss.item())

        # SUMMARY_WRITER.add_image('Position_map_train/1', pos_map_1.detach().cpu().squeeze(), dataformats='HW')
        # SUMMARY_WRITER.add_image('Position_map_train/2', pos_map_2.detach().cpu().squeeze(), dataformats='HW')

        if i % log_nth == 0:
            imgs_in_batch = frames.size(0)
            gt_grid = make_grid(frames, nrow=imgs_in_batch, padding=10)
            reconstructed_grid = make_grid(out, nrow=imgs_in_batch, padding=10)

            SUMMARY_WRITER.add_image('Reconstruction_Train/Image', gt_grid.detach().cpu().squeeze(), dataformats='CHW')
            SUMMARY_WRITER.add_image('Reconstruction_Train/Reconstruction', reconstructed_grid.detach().cpu().squeeze(),
                                     dataformats='CHW')

    return train_loss / i


def validate_one_epoch(model, data_loader, scale_factor=0.25, log_nth=100):
    model.eval()
    val_loss = 0
    i = 0
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        frames, _, _ = batch
        frames = frames.float().squeeze() / 255.0
        frames = frames.to(device)
        frames = F.interpolate(frames, scale_factor=scale_factor)

        with torch.no_grad():
            # out, pos_map_1, pos_map_2 = model(frames, None)
            out = model(frames)
            # pos_map_1, pos_map_2 = torch.sigmoid(pos_map_1), torch.sigmoid(pos_map_2)  # TODO: Decide a threshold
            # consistency_loss = LOSS_L2(pos_map_1, pos_map_2)
            # reconstruction_loss = LOSS_RECONSTRUCTION(out, frames[1].unsqueeze(0))
            reconstruction_loss = LOSS_RECONSTRUCTION(out, frames)
            # kl_loss = LOSS_KL(out, frames)

            # loss = consistency_loss + reconstruction_loss  # + kl_loss
            loss = reconstruction_loss
            val_loss += loss.item()

        # SUMMARY_WRITER.add_scalar('val_loss/consistency_loss', consistency_loss.item())
        SUMMARY_WRITER.add_scalar('val_loss/reconstruction_loss', reconstruction_loss.item())
        # SUMMARY_WRITER.add_scalar('val_loss/kl_loss', kl_loss.item())
        # SUMMARY_WRITER.add_scalar('val_loss/summed_loss', loss.item())

        # SUMMARY_WRITER.add_image('Position_map_val/1', pos_map_1.detach().cpu().squeeze(), dataformats='HW')
        # SUMMARY_WRITER.add_image('Position_map_val/2', pos_map_2.detach().cpu().squeeze(), dataformats='HW')

        if i % log_nth == 0:
            imgs_in_batch = frames.size(0)
            gt_grid = make_grid(frames, nrow=imgs_in_batch, padding=10)
            reconstructed_grid = make_grid(out, nrow=imgs_in_batch, padding=10)

            SUMMARY_WRITER.add_image('Reconstruction_Val/Image', gt_grid.detach().cpu().squeeze(), dataformats='CHW')
            SUMMARY_WRITER.add_image('Reconstruction_Val/Reconstruction', reconstructed_grid.detach().cpu().squeeze(),
                                     dataformats='CHW')

    return val_loss / i


def train(model, train_data_loader, val_data_loader, lr, epochs=50, weight_decay=1e-5, save_path=None,
          scale_factor=0.25, log_nth=1000):
    best_loss = 10000
    optimizer = OPTIMIZER(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info(f"Network Settings - lr: {lr}, weight_decay: {weight_decay}, epochs: {epochs},"
                f" optimizer: {optimizer.__class__.__name__}")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_data_loader, optimizer, scale_factor=scale_factor, log_nth=log_nth)
        val_loss = validate_one_epoch(model, val_data_loader, scale_factor=scale_factor, log_nth=log_nth)

        if val_loss < best_loss:
            if save_path is not None:
                checkpoint = {'lr': lr,
                              'epoch': epoch,
                              'optimizer': optimizer.state_dict(),
                              'model': model.state_dict(),
                              'loss': best_loss}
                torch.save(checkpoint, save_path)

        logger.info(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")


def inference(model, data_loader, scale_factor=0.25, checkpoint_path=None):
    saved_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved_dict['model'])
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        frames, _, _ = batch
        frames = frames.float().squeeze() / 255.0
        frames = frames.to(device)
        frames = F.interpolate(frames, scale_factor=scale_factor)

        with torch.no_grad():
            out = model(frames)
            out = out.permute(0, 2, 3, 1)
            show_img(out[0].cpu())


def main(args, video_label, inference_mode=False):
    logger.info(f"Setting up DataLoaders...")
    save_path = f"{args.save_path}{video_label.value}_batch_size{args.batch_size}_{datetime.now()}.pt"

    sdd_dataset = SDDDatasetBuilder(root=args.dataset_root, video_label=video_label, frames_per_clip=1,
                                    num_workers=args.data_loader_num_workers)

    logger.info(f"Dataset built successfully")
    train_dataset = SDDTrainDataset(sdd_dataset.video_clips, sdd_dataset.samples, sdd_dataset.transform,
                                    sdd_dataset.train_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory, drop_last=True)

    val_dataset = SDDValidationDataset(sdd_dataset.video_clips, sdd_dataset.samples, sdd_dataset.transform,
                                       sdd_dataset.val_indices)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory, drop_last=True)

    logger.info(f"Dataset preparation done")
    encoder = vgg11_bn
    decoder = make_layers(vgg_decoder_arch['A'], batch_norm=args.batch_norm)
    net = VanillaAutoEncoder(encoder, decoder, pretrained=args.pretrained)
    net = net.to(device)

    logger.info(f"Train Network: {net.__class__.__name__}")

    if inference_mode:
        logger.info(f"Starting Inference")
        inference(model=net, data_loader=val_loader, checkpoint_path=args.checkpoint_path)
    else:
        logger.info(f"Starting Training")
        train(model=net, train_data_loader=train_loader, val_data_loader=val_loader, lr=args.lr, epochs=args.epochs,
              save_path=save_path, weight_decay=args.weight_decay)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        vid_label = SDDVideoClasses.QUAD

        parser_ = argparse.ArgumentParser('Training Script', parents=[get_args_parser()])
        args = parser_.parse_args()
        if args.save_path:
            Path(args.save_path).mkdir(parents=True, exist_ok=True)

        main(args, video_label=vid_label, inference_mode=True)
