import os
from typing import Optional

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import compute_fde, compute_ade
from baselinev2.config import BASE_PATH, ROOT_PATH
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import get_dataset
from baselinev2.nn.data_utils import extract_frame_from_video
from baselinev2.nn.models import BaselineRNN
from baselinev2.plot_utils import plot_trajectory_alongside_frame


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, checkpoint_root_path: str, video_path: str,
                   plot_path: Optional[str] = None):
    checkpoint_path = checkpoint_root_path + 'checkpoints/'
    checkpoint_file = os.listdir(checkpoint_path)[-1]
    model = model.load_from_checkpoint(
        checkpoint_path=checkpoint_path + checkpoint_file,
        hparams_file=f'{checkpoint_root_path}hparams.yaml',
        map_location=None
    )

    model.eval()

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        in_xy, gt_xy, in_uv, gt_uv, in_track_ids, gt_track_ids, in_frame_numbers, gt_frame_numbers, ratio = data

        total_loss = torch.tensor(data=0, dtype=torch.float32)
        predicted_xy, true_xy = [], []

        # Encoder
        b, seq_len = in_uv.size(0), in_uv.size(1)
        h0, c0 = model.init_hidden_states(b_size=b)
        out = model.pre_encoder(in_uv.view(-1, 2))
        out = F.relu(out.view(seq_len, b, -1))
        out, (h_enc, c_enc) = model.encoder(out, (h0, c0))
        # Decoder
        # Last (x,y) and (u,v) position at T=8
        last_xy = in_xy[:, -1, ...]
        last_uv = in_uv[:, -1, ...]

        h_dec, c_dec = h_enc.squeeze(0), c_enc.squeeze(0)
        for gt_pred_xy in gt_xy.permute(1, 0, 2):
            out = model.pre_decoder(last_uv)
            h_dec, c_dec = model.decoder(out, (h_dec, c_dec))
            pred_uv = model.post_decoder(F.relu(h_dec))
            out = last_xy + (pred_uv * 0.4)
            total_loss += model.center_based_loss_meters(gt_center=gt_pred_xy, pred_center=out, ratio=ratio[0].item())

            predicted_xy.append(out.detach().cpu().numpy())
            true_xy.append(gt_pred_xy.detach().cpu().numpy())

            last_xy = out
            last_uv = pred_uv

        ade = compute_ade(np.stack(predicted_xy), np.stack(true_xy)).item()
        fde = compute_fde(np.stack(predicted_xy), np.stack(true_xy)).item()

        plot_frame_number = in_frame_numbers.squeeze()[0].item()
        plot_track_id = in_track_ids.squeeze()[0].item()
        obs_trajectory = in_xy.squeeze().numpy()
        gt_trajectory = np.stack(true_xy).squeeze()
        pred_trajectory = np.stack(predicted_xy).squeeze()
        all_frame_numbers = torch.cat((in_frame_numbers.squeeze(), gt_frame_numbers.squeeze())).tolist()

        plot_trajectory_alongside_frame(
            frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
            obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
            pred_trajectory=pred_trajectory, frame_number=plot_frame_number,
            track_id=plot_track_id, additional_text=f'Frame Numbers: {all_frame_numbers}\nADE: {ade} | FDE: {fde}',
            save_path=f'{plot_path}{checkpoint_file}/'
        )
        # plot_trajectories_with_frame(
        #     frame=extract_frame_from_video(video_path=video_path, frame_number=plot_frame_number),
        #     obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
        #     pred_trajectory=pred_trajectory, frame_number=plot_frame_number,
        #     track_id=plot_track_id, additional_text=in_frame_numbers.squeeze()
        # )
        # plot_trajectories(
        #     obs_trajectory=obs_trajectory, gt_trajectory=gt_trajectory,
        #     pred_trajectory=pred_trajectory, frame_number=plot_frame_number,
        #     track_id=plot_track_id, additional_text=in_frame_numbers.squeeze())

        # print()


if __name__ == '__main__':
    num_workers = 12
    shuffle = True

    sdd_video_class = SDDVideoClasses.LITTLE
    sdd_meta_class = SDDVideoDatasets.LITTLE
    network_mode = NetworkMode.TRAIN
    sdd_video_number = 3

    path_to_video = f'{BASE_PATH}videos/{sdd_video_class.value}/video{sdd_video_number}/video.mov'

    version = 9

    plot_save_path = f'{ROOT_PATH}Plots/baseline_v2/nn/v{version}/{sdd_video_class.value}{sdd_video_number}/' \
                     f'eval_plots/{network_mode.value}/'

    checkpoint_root_path = f'../baselinev2/lightning_logs/version_{version}/'
    dataset = get_dataset(video_clazz=sdd_video_class, video_number=sdd_video_number, mode=network_mode,
                          meta_label=sdd_meta_class)
    model = BaselineRNN()

    evaluate_model(model=model, data_loader=DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle),
                   checkpoint_root_path=checkpoint_root_path, video_path=path_to_video, plot_path=plot_save_path)
