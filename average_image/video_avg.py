import datetime
import os
from typing import Union, Optional

import torchvision
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bbox_utils import annotations_to_dataframe, get_frame_annotations, add_bbox_to_axes, resize_v_frames
from constants import SDDVideoClasses, OBJECT_CLASS_COLOR_MAPPING
from layers import MinPool2D
from deep_networks_avg import get_vgg_layer_activations, get_resnet_layer_activations, get_densenet_layer_activations,\
    get_densenet_filtered_layer_activations


def plot_and_save(average, reference, mask, save_path: str, num_frames: int, video_label: str, vid_number: int,
                  annotations, save_file_name: str, reference_frame_number: int, pedestrians_only: bool,
                  pooled_spatial_dim=None, min_pool: bool = False, original_spatial_dim=None,
                  min_pool_iterations: int = 1, use_dnn: bool = False, vgg_scale: float = 1, dnn_arch: str = 'vgg'):
    m = 1
    if min_pool:
        m = 3 * min_pool_iterations
    if use_dnn:
        if dnn_arch == 'vgg':
            m = vgg_scale
        else:
            m = vgg_scale * 3

    if average.size(0) < average.size(1):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="all", sharey="all",
                                            figsize=(average.size(1) * m / 100, average.size(0) * m / 100))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="all", sharey="all",
                                            figsize=(average.size(0) * m / 100, average.size(1) * m / 100))
    ax1.imshow(average)
    ax2.imshow(reference)
    m_img = ax3.imshow(mask, cmap='hot')

    add_bbox_to_axes(ax2, annotations=annotations, only_pedestrians=pedestrians_only,
                     original_spatial_dim=original_spatial_dim, pooled_spatial_dim=pooled_spatial_dim,
                     min_pool=min_pool, use_dnn=use_dnn)
    add_bbox_to_axes(ax3, annotations=annotations, only_pedestrians=pedestrians_only,
                     original_spatial_dim=original_spatial_dim, pooled_spatial_dim=pooled_spatial_dim,
                     min_pool=min_pool, use_dnn=use_dnn)

    ax1.set_title("Average")
    ax2.set_title("Reference")
    ax3.set_title("Mask")

    fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}\nNumber of frames: {num_frames}"
                 f"\nReference Frame: {reference_frame_number}\nMin Pool: {use_min_pool}\nDNN: {dnn_arch}"
                 , fontsize=14, fontweight='bold')

    # fig.colorbar(m_img, orientation="horizontal", pad=0.022)

    patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
    fig.legend(handles=patches, loc=2)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    fig.savefig(save_path + save_file_name + ".png")
    plt.show()


def get_frames(file: str, start: Union[int, float], end: Union[int, float]):
    v_frames, a_frames, meta_info = torchvision.io.read_video(filename=file, start_pts=start, end_pts=end,
                                                              pts_unit="sec")
    v_frames = v_frames.float() / 255.0
    return v_frames, a_frames, meta_info


def get_result_triplet(v_frames, reference_frame_number: int):
    average_frame = v_frames.mean(dim=0).detach()
    reference_frame = v_frames[reference_frame_number].detach()

    activations = average_frame - reference_frame
    activations_mask = activations.mean(dim=-1).detach().numpy()

    return average_frame, reference_frame, activations_mask


def min_pool_baseline(v_frames, kernel_size: int, iterations: int = 1):
    min_pool2d = MinPool2D(kernel_size=kernel_size)
    v_frames = min_pool2d(v_frames.permute(0, 3, 1, 2), iterations).permute(0, 2, 3, 1)
    return v_frames


def vgg_activations(v_frames, layer_number: int = 3, scale_factor: Optional[float] = None):
    v_frames = v_frames.permute(0, 3, 1, 2)
    if scale_factor is not None:
        v_frames = resize_v_frames(v_frames=v_frames, scale_factor=scale_factor)
    vgg_out = get_vgg_layer_activations(x=v_frames, layer_number=layer_number)
    return vgg_out.permute(0, 2, 3, 1)


def resnet_activations(v_frames, scale_factor: Optional[float] = None):
    v_frames = v_frames.permute(0, 3, 1, 2)
    if scale_factor is not None:
        v_frames = resize_v_frames(v_frames=v_frames, scale_factor=scale_factor)
    out = get_resnet_layer_activations(x=v_frames)
    return out.permute(0, 2, 3, 1)


def densenet_activations(v_frames, layer_number: int = 3, scale_factor: Optional[float] = None):
    v_frames = v_frames.permute(0, 3, 1, 2)
    if scale_factor is not None:
        v_frames = resize_v_frames(v_frames=v_frames, scale_factor=scale_factor)
    # out = get_densenet_layer_activations(x=v_frames, layer_number=layer_number)
    out = get_densenet_filtered_layer_activations(x=v_frames, layer_number=layer_number)
    return out.permute(0, 2, 3, 1)


if __name__ == '__main__':
    annotation_base_path = "../Datasets/SDD/annotations/"
    video_base_path = "../Datasets/SDD/videos/"
    vid_label = SDDVideoClasses.HYANG
    video_number = 4
    video_file_name = "video.mov"
    annotation_file_name = "annotations.txt"
    fps = 30
    start_sec = 12
    end_sec = 16

    frame_number = 2

    # min_pool
    use_min_pool = False
    min_pool_itrs = 2

    # vgg/resnet/densenet features
    inp_layers = 4  # 4 for densenet, 3 or 6 for vgg
    use_dnn = True
    resnet = True
    densenet = False
    scale_factor = 0.25

    video_file = video_base_path + str(vid_label.value) + "/video" + str(video_number) + "/" + video_file_name
    annotation_file = annotation_base_path + str(vid_label.value) + "/video" + str(video_number) + "/" + \
                      annotation_file_name

    time_adjusted_frame_number = (start_sec * fps) + frame_number
    plot_save_path = "../Plots/outputs/"
    if use_min_pool:
        plot_save_file_name = f"min_pooling/{vid_label.value}_{video_number}_min_pool_{min_pool_itrs}" \
                              f"_{datetime.datetime.now().isoformat()}"
    elif use_dnn:
        if resnet:
            plot_save_file_name = f"resnet_features/{vid_label.value}_{video_number}_resnet_layer1" \
                                  f"_{datetime.datetime.now().isoformat()}"
        elif densenet:
            plot_save_file_name = f"densenet_features/{vid_label.value}_{video_number}_densenet_{inp_layers}" \
                                  f"_{datetime.datetime.now().isoformat()}"
        else:
            plot_save_file_name = f"vgg_features/{vid_label.value}_{video_number}_vgg_{inp_layers}" \
                                  f"_{datetime.datetime.now().isoformat()}"
    else:
        plot_save_file_name = f"{vid_label.value}_{video_number}_{datetime.datetime.now().isoformat()}"

    df = annotations_to_dataframe(annotation_file)
    annotations = get_frame_annotations(df, time_adjusted_frame_number)

    video_frames, _, meta = get_frames(file=video_file, start=start_sec, end=end_sec)
    original_spatial_dim = (video_frames.size(1), video_frames.size(2))
    pooled_spatial_dim = None

    if use_min_pool:
        video_frames = min_pool_baseline(v_frames=video_frames, kernel_size=3, iterations=min_pool_itrs)
        pooled_spatial_dim = (video_frames.size(1), video_frames.size(2))

    if use_dnn:
        if resnet:
            video_frames = resnet_activations(v_frames=video_frames, scale_factor=scale_factor)
        elif densenet:
            video_frames = densenet_activations(v_frames=video_frames, layer_number=inp_layers,
                                                scale_factor=scale_factor)
        else:
            video_frames = vgg_activations(v_frames=video_frames, layer_number=inp_layers, scale_factor=scale_factor)

        pooled_spatial_dim = (video_frames.size(1), video_frames.size(2))

    avg_frame, ref_frame, activation_mask = get_result_triplet(v_frames=video_frames,
                                                               reference_frame_number=frame_number)

    if use_dnn:
        avg_frame, ref_frame, activation_mask = avg_frame.mean(-1).unsqueeze(-1), \
                                                ref_frame.mean(-1).unsqueeze(-1), \
                                                np.expand_dims(activation_mask, axis=-1)

    dnn_arch = None
    if use_dnn:
        if resnet:
            dnn_arch = 'resnet'
        elif densenet:
            dnn_arch = 'densenet'
        else:
            dnn_arch = 'vgg'

    plot_and_save(average=avg_frame, reference=ref_frame, mask=activation_mask, save_path=plot_save_path,
                  num_frames=(end_sec - start_sec) * meta["video_fps"], video_label=str(vid_label.value),
                  vid_number=video_number, annotations=annotations, save_file_name=plot_save_file_name,
                  reference_frame_number=time_adjusted_frame_number, pedestrians_only=False,
                  original_spatial_dim=original_spatial_dim, pooled_spatial_dim=pooled_spatial_dim,
                  min_pool=use_min_pool, min_pool_iterations=min_pool_itrs, use_dnn=use_dnn, vgg_scale=1 / scale_factor,
                  dnn_arch=dnn_arch)

    print(f"Number of frames in batch: {video_frames.size(0)}, Meta: {meta},"
          f" Min Pool Iterations: {min_pool_itrs if use_min_pool else None}\n\n"
          f"{annotations}")
