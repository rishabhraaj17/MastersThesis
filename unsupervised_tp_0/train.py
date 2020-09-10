from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg11_bn
from torchvision.utils import make_grid
from tqdm import tqdm

from constants import SDDVideoClasses
from unsupervised_tp_0.dataset import SDDDatasetBuilder, SDDTrainDataset, SDDValidationDataset
from unsupervised_tp_0.model import make_layers, vgg_decoder_arch, UnsupervisedTP, VanillaAutoEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOSS_L2 = torch.nn.MSELoss()
LOSS_RECONSTRUCTION = torch.nn.MSELoss()
# LOSS_KL = torch.nn.KLDivLoss(reduction='batchmean')

OPTIMIZER = torch.optim.Adam

SUMMARY_WRITER = SummaryWriter()


def train_one_epoch(model, data_loader, optimizer, scale_factor=0.25):
    model.train()
    train_loss = 0
    i = 0
    for i, batch in tqdm(enumerate(data_loader)):
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

        imgs_in_batch = frames.size(0)
        gt_grid = make_grid(frames, nrow=imgs_in_batch, padding=10)
        reconstructed_grid = make_grid(out, nrow=imgs_in_batch, padding=10)

        SUMMARY_WRITER.add_image('Reconstruction_Train/Image', gt_grid.detach().cpu().squeeze(), dataformats='CHW')
        SUMMARY_WRITER.add_image('Reconstruction_Train/Reconstruction', reconstructed_grid.detach().cpu().squeeze(),
                                 dataformats='CHW')

    return train_loss / i


def validate_one_epoch(model, data_loader, scale_factor=0.25):
    model.eval()
    val_loss = 0
    i = 0
    for i, batch in tqdm(enumerate(data_loader)):
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

        imgs_in_batch = frames.size(0)
        gt_grid = make_grid(frames, nrow=imgs_in_batch, padding=10)
        reconstructed_grid = make_grid(out, nrow=imgs_in_batch, padding=10)

        SUMMARY_WRITER.add_image('Reconstruction_Val/Image', gt_grid.detach().cpu().squeeze(), dataformats='CHW')
        SUMMARY_WRITER.add_image('Reconstruction_Val/Reconstruction', reconstructed_grid.detach().cpu().squeeze(),
                                 dataformats='CHW')

    return val_loss / i


def train(model, train_data_loader, val_data_loader, lr, epochs=50, weight_decay=1e-5, save_path=None,
          scale_factor=0.25):
    best_loss = 10000
    optimizer = OPTIMIZER(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_data_loader, optimizer, scale_factor=scale_factor)
        val_loss = validate_one_epoch(model, val_data_loader, scale_factor=scale_factor)

        if val_loss < best_loss:
            if save_path is not None:
                checkpoint = {'lr': lr,
                              'epoch': epoch,
                              'optimizer': optimizer.state_dict(),
                              'model': model.state_dict(),
                              'loss': best_loss}
                torch.save(checkpoint, save_path)

        print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}")


if __name__ == '__main__':
    base_path = "../Datasets/SDD/"
    vid_label = SDDVideoClasses.LITTLE
    num_workers = 6
    pin_memory = False
    lr = 0.001
    batch_size = 2
    save_path = f"../Checkpoints/{datetime.now()}/"

    sdd_dataset = SDDDatasetBuilder(root=base_path, video_label=vid_label, frames_per_clip=1, num_workers=10)

    train_dataset = SDDTrainDataset(sdd_dataset.video_clips, sdd_dataset.samples, sdd_dataset.transform,
                                    sdd_dataset.train_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=True)

    val_dataset = SDDValidationDataset(sdd_dataset.video_clips, sdd_dataset.samples, sdd_dataset.transform,
                                       sdd_dataset.val_indices)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=True)

    encoder = vgg11_bn
    decoder = make_layers(vgg_decoder_arch['A'], batch_norm=True)
    net = VanillaAutoEncoder(encoder, decoder, pretrained=True)
    net = net.to(device)

    train(model=net, train_data_loader=train_loader, val_data_loader=val_loader, lr=lr, epochs=10, save_path=save_path)
