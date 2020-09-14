import datetime
import os
from typing import Union, Optional

import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

from bbox_utils import annotations_to_dataframe, get_frame_annotations, add_bbox_to_axes, resize_v_frames
from constants import SDDVideoClasses, OBJECT_CLASS_COLOR_MAPPING
from layers import MinPool2D
from deep_networks_avg import get_vgg_layer_activations, get_resnet_layer_activations, get_densenet_layer_activations, \
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


def optical_flow_and_plot(ref_frame, previous_frame, save_path, save_file_name):
    ref_frame = ref_frame.numpy()
    previous_frame = previous_frame.numpy()

    ref_frame = cv.GaussianBlur(ref_frame, (5, 5), 0)
    previous_frame = cv.GaussianBlur(previous_frame, (5, 5), 0)

    prev_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(previous_frame)
    mask[..., 1] = 255
    gray = cv.cvtColor(ref_frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 1, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # ref_frame = torch.from_numpy(ref_frame)
    # previous_frame = torch.from_numpy(previous_frame)
    # rgb = torch.from_numpy(rgb)

    if ref_frame.shape[0] < ref_frame.shape[1]:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="all", sharey="all",
                                            figsize=(ref_frame.shape[1] / 100, ref_frame.shape[0] / 100))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="all", sharey="all",
                                            figsize=(ref_frame.shape[0] / 100, ref_frame.shape[1] / 100))

    ax1.imshow(previous_frame)
    ax2.imshow(ref_frame)
    m_img = ax3.imshow(rgb)

    ax1.set_title("Previous")
    ax2.set_title("Current")
    ax3.set_title("Optical Flow")

    fig.savefig(save_path + save_file_name + "_flow" + ".png")
    plt.show()


def optical_flow_video(video_path, video_label, annotations_df, vid_number, video_out_save_path,
                       start_frame=0, end_frame=None, desired_fps=6):
    cap = cv.VideoCapture(video_path)
    cap_count = 0

    previous = None
    hsv = None

    original_dims = None
    out = None

    for frame_count in tqdm(range(0, end_frame)):
        ret, frame = cap.read()
        if previous is None:
            previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            continue
        if out is None:
            if frame.shape[0] < frame.shape[1]:
                original_dims = (frame.shape[1] / 100, frame.shape[0] / 100)
                out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                     (frame.shape[1], frame.shape[0]))
            else:
                original_dims = (frame.shape[0] / 100, frame.shape[1] / 100)
                out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                     (frame.shape[0], frame.shape[1]))
        if cap_count < start_frame:
            continue

        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # 0-1 normalize
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


        if frame.shape[0] < frame.shape[1]:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="all", sharey="all",
                                                figsize=original_dims)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="all", sharey="all",
                                                figsize=original_dims)

        canvas = FigureCanvas(fig)

        ax1.imshow(previous, cmap='gray')
        ax2.imshow(next, cmap='gray')
        ax3.imshow(rgb)

        # original_spatial_dim = (frame.shape[0], frame.shape[1])
        # annot = get_frame_annotations(annotations_df, frame_count)

        # add_bbox_to_axes(ax1, annotations=annot, only_pedestrians=False,
        #                  original_spatial_dim=original_spatial_dim, pooled_spatial_dim=None,
        #                  min_pool=False, use_dnn=False)
        # add_bbox_to_axes(ax2, annotations=annot, only_pedestrians=False,
        #                  original_spatial_dim=original_spatial_dim, pooled_spatial_dim=None,
        #                  min_pool=False, use_dnn=False)
        # add_bbox_to_axes(ax3, annotations=annot, only_pedestrians=False,
        #                  original_spatial_dim=original_spatial_dim, pooled_spatial_dim=None,
        #                  min_pool=False, use_dnn=False)

        # patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
        # fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}", fontsize=14, fontweight='bold')

        ax1.set_title("Previous Frame")
        ax2.set_title("Current Frame")
        ax3.set_title("Optical Flow")

        canvas.draw()

        buf = canvas.buffer_rgba()
        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
        out.write(out_frame)

        cap_count += 1
        previous = next

    cap.release()
    out.release()


# fixme: not working this way
def min_pool_video(v_frames, kernel_size=3, iterations=1, desired_fps=6, video_out_save_path=None):
    v_frames_pooled = min_pool_baseline(v_frames=v_frames, kernel_size=kernel_size, iterations=iterations)
    pooled_dims = (v_frames_pooled.size(1), v_frames_pooled.size(2))
    v_frames = F.interpolate(v_frames.permute(0, 3, 1, 2), size=pooled_dims).permute(0, 2, 3, 1)

    video_out_frames = None
    for frame in range(v_frames.size(0)):
        cat_frame = torch.stack((v_frames[frame], v_frames_pooled[frame])).permute(0, 3, 1, 2)
        joined_frame = make_grid(cat_frame, nrow=2, padding=5).unsqueeze(dim=0)
        if video_out_frames is None:
            clubbed_h, clubbed_w = joined_frame.size(2), joined_frame.size(3)
            video_out_frames = torch.zeros(size=(0, 3, clubbed_h, clubbed_w))
        video_out_frames = torch.cat((video_out_frames, joined_frame))

    video_out_frames = video_out_frames.permute(0, 3, 2, 1)
    if video_out_frames.size(1) % 2 != 0 or video_out_frames.size(2) % 2 != 0:
        video_out_frames = F.interpolate(v_frames.permute(0, 3, 1, 2), size=(video_out_frames.size(1) + 1,
                                                                             video_out_frames.size(2) + 1)) \
            .permute(0, 2, 3, 1)
    torchvision.io.write_video(video_out_save_path, video_out_frames, fps=desired_fps)


def min_pool_video_opencv(v_frames, kernel_size=3, iterations=1, frame_join_pad=5, desired_fps=6,
                          video_out_save_path=None):
    v_frames_pooled = min_pool_baseline(v_frames=v_frames, kernel_size=kernel_size, iterations=iterations)
    v_frames_pooled = v_frames_pooled.permute(0, 3, 1, 2)
    pooled_dims = (v_frames_pooled.shape[2], v_frames_pooled.shape[3])
    v_frames = F.interpolate(v_frames.permute(0, 3, 1, 2), size=pooled_dims)

    out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                         (pooled_dims[1] * 2 + frame_join_pad * 3, pooled_dims[0] + frame_join_pad * 2))

    for frame in range(v_frames.size(0)):
        cat_frame = torch.stack((v_frames[frame], v_frames_pooled[frame]))
        joined_frame = make_grid(cat_frame, nrow=2, padding=5).permute(1, 2, 0).numpy()
        joined_frame = (joined_frame * 255).astype(np.uint8)
        out.write(joined_frame)

    out.release()


def min_pool_subtracted_img_video_opencv(v_frames, average_frame, start_sec, original_fps, kernel_size=3, iterations=1,
                                         desired_fps=6, video_out_save_path=None, annotations_df=None, show_bbox=False,
                                         video_label=None, vid_number=None):
    average_frame_stacked = average_frame.repeat(v_frames.size(0), 1, 1, 1)
    activation_masks_stacked = (average_frame_stacked - v_frames).mean(dim=-1).unsqueeze(dim=-1)

    v_frames_pooled = min_pool_baseline(v_frames=activation_masks_stacked, kernel_size=kernel_size,
                                        iterations=iterations)
    v_frames_pooled = (v_frames_pooled.permute(0, 3, 1, 2) * 255).int()
    pooled_dims = (v_frames_pooled.shape[2], v_frames_pooled.shape[3])
    v_frames = v_frames.permute(0, 3, 1, 2)

    if average_frame_stacked.shape[1] < average_frame_stacked.shape[2]:
        original_dims = (average_frame_stacked.size(2) / 100, average_frame_stacked.size(1) / 100)
        out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (average_frame_stacked.size(2), average_frame_stacked.size(1)))
    else:
        original_dims = (average_frame_stacked.size(1) / 100, average_frame_stacked.size(2) / 100)
        out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (average_frame_stacked.size(1), average_frame_stacked.size(2)))

    print('Generating Video: ')
    for i, frame in tqdm(enumerate(range(v_frames.size(0))), total=v_frames.size(0)):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='none', sharey='none',
                                       figsize=original_dims)
        canvas = FigureCanvas(fig)
        ax1.imshow(v_frames[frame].permute(1, 2, 0))
        ax2.imshow(v_frames_pooled[frame].permute(1, 2, 0), cmap='gray')

        ax1.set_title(f"Video Frame: {i}")
        ax2.set_title(f"Mask Frame: {i}")

        if show_bbox:
            # sometimes we need +/- 1 for accurate bboxes
            annot = get_frame_annotations(annotations_df, ((start_sec * original_fps) + i - 1))

            add_bbox_to_axes(ax1, annotations=annot, only_pedestrians=False,
                             original_spatial_dim=(average_frame_stacked.size(1), average_frame_stacked.size(2)),
                             pooled_spatial_dim=(pooled_dims[0], pooled_dims[1]),
                             min_pool=False, use_dnn=False, linewidth=0.2)

            add_bbox_to_axes(ax2, annotations=annot, only_pedestrians=False,
                             original_spatial_dim=(average_frame_stacked.size(1), average_frame_stacked.size(2)),
                             pooled_spatial_dim=(pooled_dims[0], pooled_dims[1]),
                             min_pool=True, use_dnn=False, linewidth=0.2)

            patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
            fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}", fontsize=14, fontweight='bold')

        canvas.draw()

        buf = canvas.buffer_rgba()
        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
        out.write(out_frame)

    out.release()


# def min_pool_video_opencv(video_path, num_frames=30, kernel_size=3, iterations=1, frame_join_pad=5, desired_fps=6,
#                           video_out_save_path=None):
#     in_video = cv.VideoCapture(video_path)
#
#     v_frames = None
#     for i in range(num_frames):
#         ret, frame = in_video.read()
#         if v_frames is None:
#             v_frames = np.zeros(shape=(0, frame.shape[0], frame.shape[1], 3))
#         v_frames = np.concatenate((v_frames, np.expand_dims(frame, axis=0)), axis=0)
#
#     v_frames = torch.from_numpy(v_frames)
#     v_frames_pooled = min_pool_baseline(v_frames=v_frames, kernel_size=kernel_size, iterations=iterations)
#     pooled_dims = (v_frames_pooled.size(1), v_frames_pooled.size(2))
#     v_frames_pooled = v_frames_pooled.permute(0, 3, 1, 2)
#     v_frames = F.interpolate(v_frames.permute(0, 3, 1, 2), size=pooled_dims)
#
#     out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
#                          (pooled_dims[1] + frame_join_pad * 3, pooled_dims[0] + frame_join_pad))
#
#     for frame in range(v_frames.size(0)):
#         cat_frame = torch.stack((v_frames[frame], v_frames_pooled[frame]))
#         joined_frame = make_grid(cat_frame/255.0, nrow=2, padding=5).permute(1, 2, 0).numpy()
#         out.write(joined_frame)
#
#     out.release()


def show_img(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    annotation_base_path = "../Datasets/SDD/annotations/"
    video_base_path = "../Datasets/SDD/videos/"
    vid_label = SDDVideoClasses.BOOKSTORE
    video_number = 0
    video_file_name = "video.mov"
    annotation_file_name = "annotations.txt"
    fps = 30
    start_sec = 131
    end_sec = 138

    frame_number = 9
    previous_frame = frame_number - 7

    # min_pool
    use_min_pool = False
    min_pool_itrs = 1

    # vgg/resnet/densenet features
    inp_layers = 4  # 4 for densenet, 3 or 6 for vgg
    use_dnn = False
    resnet = False
    densenet = False
    scale_factor = 0.25

    video_file = video_base_path + str(vid_label.value) + "/video" + str(video_number) + "/" + video_file_name
    annotation_file = annotation_base_path + str(vid_label.value) + "/video" + str(video_number) + "/" + \
                      annotation_file_name

    time_adjusted_frame_number = (start_sec * fps) + frame_number
    plot_save_path = "../Plots/outputs/"
    # if use_min_pool:
    #     plot_save_file_name = f"min_pooling/{vid_label.value}_{video_number}_min_pool_{min_pool_itrs}" \
    #                           f"_{datetime.datetime.now().isoformat()}"
    # elif use_dnn:
    #     if resnet:
    #         plot_save_file_name = f"resnet_features/{vid_label.value}_{video_number}_resnet_layer1" \
    #                               f"_{datetime.datetime.now().isoformat()}"
    #     elif densenet:
    #         plot_save_file_name = f"densenet_features/{vid_label.value}_{video_number}_densenet_{inp_layers}" \
    #                               f"_{datetime.datetime.now().isoformat()}"
    #     else:
    #         plot_save_file_name = f"vgg_features/{vid_label.value}_{video_number}_vgg_{inp_layers}" \
    #                               f"_{datetime.datetime.now().isoformat()}"
    # else:
    #     plot_save_file_name = f"{vid_label.value}_{video_number}_{datetime.datetime.now().isoformat()}"
    #
    # df = annotations_to_dataframe(annotation_file)
    # annotations = get_frame_annotations(df, time_adjusted_frame_number)
    #
    # video_frames, _, meta = get_frames(file=video_file, start=start_sec, end=end_sec)
    # original_spatial_dim = (video_frames.size(1), video_frames.size(2))
    # pooled_spatial_dim = None
    #
    # if use_min_pool:
    #     video_frames = min_pool_baseline(v_frames=video_frames, kernel_size=3, iterations=min_pool_itrs)
    #     pooled_spatial_dim = (video_frames.size(1), video_frames.size(2))
    #
    # if use_dnn:
    #     if resnet:
    #         video_frames = resnet_activations(v_frames=video_frames, scale_factor=scale_factor)
    #     elif densenet:
    #         video_frames = densenet_activations(v_frames=video_frames, layer_number=inp_layers,
    #                                             scale_factor=scale_factor)
    #     else:
    #         video_frames = vgg_activations(v_frames=video_frames, layer_number=inp_layers, scale_factor=scale_factor)
    #
    #     pooled_spatial_dim = (video_frames.size(1), video_frames.size(2))
    #
    # avg_frame, ref_frame, activation_mask = get_result_triplet(v_frames=video_frames,
    #                                                            reference_frame_number=frame_number)
    #
    # min_pool_subtracted_img_video_opencv(v_frames=video_frames, average_frame=avg_frame, kernel_size=3,
    #                                      iterations=min_pool_itrs,
    #                                      video_out_save_path=plot_save_path + f"video_cv1_{vid_label.value}_bbox.avi",
    #                                      annotations_df=df, start_sec=start_sec, original_fps=fps, show_bbox=True,
    #                                      vid_number=video_number, video_label=vid_label.value)

    optical_flow_video(video_path=video_file, video_label=vid_label.value, annotations_df=None, vid_number=video_number,
                       video_out_save_path=plot_save_path + f"video_optical_flow_{vid_label.value}_0.avi",
                       start_frame=0, end_frame=100, desired_fps=6)

    # avg_frame, ref_frame, activation_mask = get_result_triplet(v_frames=video_frames,
    #                                                            reference_frame_number=frame_number)
    # optical_flow_and_plot(ref_frame=ref_frame, previous_frame=video_frames[previous_frame], save_path=plot_save_path,
    #                       save_file_name=plot_save_file_name)
    #
    # if use_dnn:
    #     avg_frame, ref_frame, activation_mask = avg_frame.mean(-1).unsqueeze(-1), \
    #                                             ref_frame.mean(-1).unsqueeze(-1), \
    #                                             np.expand_dims(activation_mask, axis=-1)
    #
    # dnn_arch = None
    # if use_dnn:
    #     if resnet:
    #         dnn_arch = 'resnet'
    #     elif densenet:
    #         dnn_arch = 'densenet'
    #     else:
    #         dnn_arch = 'vgg'
    #
    # plot_and_save(average=avg_frame, reference=ref_frame, mask=activation_mask, save_path=plot_save_path,
    #               num_frames=(end_sec - start_sec) * meta["video_fps"], video_label=str(vid_label.value),
    #               vid_number=video_number, annotations=annotations, save_file_name=plot_save_file_name,
    #               reference_frame_number=time_adjusted_frame_number, pedestrians_only=False,
    #               original_spatial_dim=original_spatial_dim, pooled_spatial_dim=pooled_spatial_dim,
    #               min_pool=use_min_pool, min_pool_iterations=min_pool_itrs, use_dnn=use_dnn, vgg_scale=1 / scale_factor,
    #               dnn_arch=dnn_arch)
    #
    # print(f"Number of frames in batch: {video_frames.size(0)}, Meta: {meta},"
    #       f" Min Pool Iterations: {min_pool_itrs if use_min_pool else None}\n\n"
    #       f"{annotations}")
