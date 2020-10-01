from copy import deepcopy
import datetime
import os
from itertools import cycle
from typing import Union, Optional

import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import cm
from skimage.transform import resize
from skimage.segmentation import quickshift, felzenszwalb
from sklearn import metrics
from sklearn.decomposition import PCA
from torchvision.utils import make_grid
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.cluster import MeanShift, estimate_bandwidth, SpectralClustering, DBSCAN
from skimage.color import rgb2lab
from tqdm import tqdm

from bbox_utils import annotations_to_dataframe, get_frame_annotations, add_bbox_to_axes, resize_v_frames, \
    scale_annotations, CoordinateHolder, CoordinateHolder2
from constants import SDDVideoClasses, OBJECT_CLASS_COLOR_MAPPING, COLORS, COLORS2
from layers import MinPool2D, min_pool2d_numpy, min_pool2d
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


def get_subtracted_img_for_frames(video_path, start_frame=0, end_frame=None):
    video_frames, _, meta = get_frames(file=video_file, start=start_frame, end=end_frame // 30)
    average_frame = video_frames.mean(dim=0)
    average_frame_stacked = average_frame.repeat(video_frames.size(0), 1, 1, 1)
    activation_masks_stacked = (video_frames - average_frame_stacked).mean(dim=-1).unsqueeze(dim=-1)
    return activation_masks_stacked


def optical_flow_subtraction_mask_video(video_path, video_label, annotations_df, vid_number, video_out_save_path,
                                        start_frame=0, end_frame=None, desired_fps=6, min_pool_kernel_size=None):
    subtracted_img = get_subtracted_img_for_frames(video_path, start_frame, end_frame)
    subtracted_img = (subtracted_img * 255).int().squeeze().numpy()
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
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="none", sharey="none",
                                                figsize=original_dims)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="none", sharey="none",
                                                figsize=original_dims)

        canvas = FigureCanvas(fig)

        s_img = subtracted_img[frame_count]
        if min_pool_kernel_size is not None:
            # s_img = min_pool2d_numpy(s_img, kernel_size=min_pool_kernel_size)
            # rgb = min_pool2d_numpy(rgb, kernel_size=min_pool_kernel_size)
            s_img = min_pool2d(torch.from_numpy(s_img).unsqueeze(dim=0).float(), min_pool_kernel_size).squeeze() \
                .int().numpy()
            rgb = min_pool2d(torch.from_numpy(rgb).permute(2, 0, 1).float(), min_pool_kernel_size).permute(1, 2, 0) \
                .int().numpy()

        ax1.imshow(frame)
        ax2.imshow(s_img, cmap='gray')
        ax3.imshow(rgb)

        original_spatial_dim = (frame.shape[0], frame.shape[1])
        pooled_dim = (s_img.shape[0], s_img.shape[1])
        annot = get_frame_annotations(annotations_df, frame_count)

        add_bbox_to_axes(ax1, annotations=annot, only_pedestrians=False,
                         original_spatial_dim=original_spatial_dim, pooled_spatial_dim=None,
                         min_pool=False, use_dnn=False, linewidth=0.2)
        add_bbox_to_axes(ax2, annotations=annot, only_pedestrians=False,
                         original_spatial_dim=original_spatial_dim, pooled_spatial_dim=pooled_dim,
                         min_pool=True, use_dnn=False, linewidth=0.2)
        add_bbox_to_axes(ax3, annotations=annot, only_pedestrians=False,
                         original_spatial_dim=original_spatial_dim, pooled_spatial_dim=pooled_dim,
                         min_pool=True, use_dnn=False, linewidth=0.5)

        patches = [mpatches.Patch(color=val, label=key.value) for key, val in OBJECT_CLASS_COLOR_MAPPING.items()]
        fig.legend(handles=patches, loc=2)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}\nMinPool: "
                     f"{True if min_pool_kernel_size is not None else False}", fontsize=14, fontweight='bold')

        ax1.set_title("Frame")
        ax2.set_title("Subtracted Image")
        ax3.set_title("Optical Flow")

        canvas.draw()

        buf = canvas.buffer_rgba()
        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
        out.write(out_frame)

        cap_count += 1
        previous = next

    cap.release()
    out.release()


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


def get_background_subtraction(video_path, start_frame=0, end_frame=60, method='mog2', history=120,
                               detect_shadows=True):
    cap = cv.VideoCapture(video_path)
    cap_count = 0
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    if method == 'knn':
        algo = cv.createBackgroundSubtractorKNN()
    else:
        # algo = cv.createBackgroundSubtractorMOG2(detectShadows=detect_shadows, history=history)
        algo = cv.createBackgroundSubtractorMOG2(detectShadows=detect_shadows, history=history, varThreshold=80)

    out = None

    for _ in tqdm(range(0, end_frame)):
        ret, frame = cap.read()
        if out is None:
            out = np.zeros(shape=(0, frame.shape[0], frame.shape[1]))

        if cap_count < start_frame:
            continue

        mask = algo.apply(frame)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        out = np.concatenate((out, np.expand_dims(mask, axis=0)), axis=0)
    return out


def mean_shift_clustering(video_path, start_frame=0, end_frame=None, min_pool_kernel_size=5):
    subtracted_img = get_subtracted_img_for_frames(video_path, start_frame, end_frame)
    subtracted_img = min_pool2d(subtracted_img.permute(0, 3, 1, 2), kernel_size=min_pool_kernel_size)
    subtracted_img = (subtracted_img * 255).int().squeeze().numpy()

    data_ = subtracted_img[0]
    data_ = np.abs(data_)

    object_idx = (data_ > 50).nonzero()

    intensities = data_[object_idx[0], object_idx[1]]  # when needed create same size array and copy values at same loc
    # data = np.stack((object_idx[1], object_idx[0], intensities)).transpose()
    data = np.stack((object_idx[1], object_idx[0])).transpose()  # better

    bandwidth = estimate_bandwidth(data, quantile=0.1, n_jobs=8)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique, points_per_cluster = np.unique(labels, return_counts=True)
    cluster_distribution = dict(zip(labels_unique, points_per_cluster))
    n_clusters_ = len(labels_unique)

    print(cluster_distribution)
    print("number of estimated clusters : %d" % n_clusters_)

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=4)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    plt.imshow(data_, cmap='gray')
    for k, col in zip(range(n_clusters_), colors):
        cluster_center = cluster_centers[k]
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=4)

    plt.show()


def mean_shift_clustering_video(video_path, start_frame=0, end_frame=None, min_pool_kernel_size=5,
                                desired_fps=6, video_out_save_path=None, annotations_df=None, show_bbox=False,
                                video_label=None, vid_number=None, background_subtraction=False,
                                background_subtraction_method='mog2'):
    if background_subtraction:
        processed_img = get_background_subtraction(video_path, start_frame, end_frame,
                                                   method=background_subtraction_method)
        img_shape = processed_img.shape
        processed_img = processed_img / 255
        processed_img = min_pool2d(torch.from_numpy(processed_img).unsqueeze(1), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 1
    else:
        processed_img = get_subtracted_img_for_frames(video_path, start_frame, end_frame)
        img_shape = processed_img.size()
        processed_img = min_pool2d(processed_img.permute(0, 3, 1, 2), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 0

    if background_subtraction:
        mthd = background_subtraction_method
    else:
        mthd = 'Subtraction - Average Image'

    if img_shape[1] < img_shape[2]:
        original_dims = (img_shape[2] / 100, img_shape[1] / 100)
        out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (img_shape[2], img_shape[1]))
    else:
        original_dims = (img_shape[1] / 100, img_shape[2] / 100)
        out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (img_shape[1], img_shape[2]))

    for fr in tqdm(range(start_idx, processed_img.shape[0])):
        data_ = processed_img[fr]
        data_ = np.abs(data_)

        object_idx = (data_ > 50).nonzero()

        intensities = data_[
            object_idx[0], object_idx[1]]  # when needed create same size array and copy values at same loc
        # data = np.stack((object_idx[1], object_idx[0], intensities)).transpose()
        data = np.stack((object_idx[1], object_idx[0])).transpose()  # better

        bandwidth = estimate_bandwidth(data, quantile=0.1, n_jobs=8)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique, points_per_cluster = np.unique(labels, return_counts=True)
        cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)

        fig, ax = plt.subplots(1, 1, sharex='none', sharey='none',
                               figsize=original_dims)
        canvas = FigureCanvas(fig)
        ax.imshow(data_, cmap='gray')

        ax.set_title(f"Frame: {fr}")

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            cluster_center = cluster_centers[k]
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=8)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}\nMethod: {mthd}",
                     fontsize=14, fontweight='bold')

        canvas.draw()

        buf = canvas.buffer_rgba()
        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
        out.write(out_frame)
    out.release()


def mean_shift_clustering_with_optical_video(video_path, start_frame=0, end_frame=None, min_pool_kernel_size=5,
                                             desired_fps=6, video_out_save_path=None, annotations_df=None,
                                             show_bbox=False,
                                             video_label=None, vid_number=None, background_subtraction=False,
                                             background_subtraction_method='mog2'):
    if background_subtraction:
        processed_img = get_background_subtraction(video_path, start_frame, end_frame,
                                                   method=background_subtraction_method)
        img_shape = processed_img.shape
        processed_img = processed_img / 255
        processed_img = min_pool2d(torch.from_numpy(processed_img).unsqueeze(1), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 1
    else:
        processed_img = get_subtracted_img_for_frames(video_path, start_frame, end_frame)
        img_shape = processed_img.size()
        processed_img = min_pool2d(processed_img.permute(0, 3, 1, 2), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 0

    if background_subtraction:
        mthd = background_subtraction_method
    else:
        mthd = 'Subtraction - Average Image'

    if img_shape[1] < img_shape[2]:
        original_dims = (img_shape[2] / 100, img_shape[1] / 100)
        out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (img_shape[2], img_shape[1]))
    else:
        original_dims = (img_shape[1] / 100, img_shape[2] / 100)
        out = cv.VideoWriter(video_out_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (img_shape[1], img_shape[2]))

    cap = cv.VideoCapture(video_path)
    cap_count = 0
    original_scale = (img_shape[1], img_shape[2])

    previous = None
    previous_pooled = False
    hsv = None

    gt_bbox_cluster_center_dict = {}

    for fr in tqdm(range(start_idx, processed_img.shape[0])):
        ret, frame = cap.read()
        if previous is None:
            previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            continue

        if cap_count < start_frame:
            continue

        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if not previous_pooled:  # pooling images and then flow calculation kills smaller motion. think!
            previous = (min_pool2d(torch.from_numpy(previous / 255).unsqueeze(0), kernel_size=min_pool_kernel_size)
                        .squeeze() * 255).int().numpy()
            previous_pooled = True
        next = (min_pool2d(torch.from_numpy(next / 255).unsqueeze(0), kernel_size=min_pool_kernel_size).squeeze()
                * 255).int().numpy()

        flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow = min_pool2d(torch.from_numpy(flow).permute(2, 0, 1), kernel_size=min_pool_kernel_size).permute(1, 2, 0)\
        #     .numpy()

        data_ = processed_img[fr]

        object_idx = (data_ > 0).nonzero()
        intensities = data_[object_idx[0], object_idx[1]]

        flow_idx = flow[object_idx[0], object_idx[1]]

        # data = np.stack((object_idx[1], object_idx[0], intensities)).transpose()
        data = np.stack((object_idx[1], object_idx[0], flow_idx[..., 1], flow_idx[..., 0])).transpose()  # better?
        # - spatial cluster location unchanged

        bandwidth = estimate_bandwidth(data, quantile=0.1, n_jobs=8)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        # scaled_cluster_centers = scale_cluster_center(cluster_centers, (data_.shape[0], data_.shape[1]),
        #                                               (frame.shape[0], frame.shape[1]))  # fixme

        labels_unique, points_per_cluster = np.unique(labels, return_counts=True)
        cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)

        annot = get_frame_annotations(annotations_df, frame_number=fr)  # check-out +/- 1
        annot, _ = scale_annotations(annot, original_scale, (data_.shape[0], data_.shape[1]))
        gt_bbox_cluster_center_dict.update({fr: {'gt_bbox': annot,
                                                 'cluster_centers': cluster_centers}})

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='none', sharey='none',
                                       figsize=original_dims)
        canvas = FigureCanvas(fig)
        ax1.imshow(data_, cmap='gray')
        ax2.imshow(resize(frame, output_shape=(data_.shape[0], data_.shape[1])))

        ax1.set_title(f"Processed Frame: {fr}")
        ax2.set_title(f"Frame: {fr}")

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            cluster_center = cluster_centers[k]
            ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=8)
            ax2.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=8)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}\nMethod: {mthd}",
                     fontsize=14, fontweight='bold')

        canvas.draw()

        buf = canvas.buffer_rgba()
        out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
        out.write(out_frame)

        cap_count += 1
        previous = next

    out.release()
    return gt_bbox_cluster_center_dict


# To get 4d cluster centers into 2d, but doesnt seem logical since two spaces are very different
def pca_cluster_center(cluster_centers):
    pca = PCA(n_components=2)
    cc = pca.fit_transform(cluster_centers)
    return cc


def normalize(x):
    max_ = np.max(x)
    min_ = np.min(x)
    return (x - min_) / (max_ - min_), max_, min_


def denormalize(x, max_, min_):
    return x * (max_ - min_) + min_


def compare_precision(pr_results_1, pr_results_2, avg_img):
    method = 'Average Image' if avg_img else 'MOG2'
    width = 0.35
    label_frame = []
    precision_1 = []
    precision_2 = []
    for (frame, result), (frame_, result_) in zip(pr_results_1.items(), pr_results_2.items()):
        label_frame.append(frame)
        precision_1.append(result[frame]['precision'])
        precision_2.append(result_[frame]['precision'])

    x = np.arange(len(label_frame))

    fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width / 2, precision_1, width, label='Without Optical Flow')
    rects1 = ax.bar(x - width / 2, precision_1, width, label='With no min-pool - (x, y, u, v)')
    rects2 = ax.bar(x + width / 2, precision_2, width, label='With 3x min-pool - (x, y, u, v)')
    # rects2 = ax.bar(x + width / 2, precision_2, width, label='With Optical Flow')
    ax.set_title(f'Precision - {method}')
    ax.set_xticks(x)
    ax.set_xticklabels(label_frame)
    ax.legend()

    # fig.tight_layout()

    plt.show()


def compare_recall(pr_results_1, pr_results_2, avg_img):
    method = 'Average Image' if avg_img else 'MOG2'
    width = 0.35
    label_frame = []
    recall_1 = []
    recall_2 = []
    for (frame, result), (frame_, result_) in zip(pr_results_1.items(), pr_results_2.items()):
        label_frame.append(frame)
        recall_1.append(result[frame]['recall'])
        recall_2.append(result_[frame]['recall'])

    x = np.arange(len(label_frame))

    fig, ax = plt.subplots()
    ax.set_title(f'Recall - {method}')
    # rects1 = ax.bar(x - width / 2, recall_1, width, label='Without Optical Flow')
    rects1 = ax.bar(x - width / 2, recall_1, width, label='With no min-pool - (x, y, u, v)')
    rects2 = ax.bar(x + width / 2, recall_2, width, label='With 3x min-pool - (x, y, u, v)')
    # rects2 = ax.bar(x + width / 2, recall_2, width, label='With Optical Flow')
    ax.set_xticks(x)
    ax.set_xticklabels(label_frame)
    ax.legend()

    # fig.tight_layout()

    plt.show()


def mean_shift_clustering_with_optical_frames(video_path, start_frame=0, end_frame=None, min_pool_kernel_size=5,
                                              desired_fps=6, video_out_save_path=None, annotations_df=None,
                                              show_bbox=False,
                                              video_label=None, vid_number=None, background_subtraction=False,
                                              background_subtraction_method='mog2'):
    if background_subtraction:
        processed_img = get_background_subtraction(video_path, start_frame, end_frame,
                                                   method=background_subtraction_method)
        img_shape = processed_img.shape
        processed_img = processed_img / 255
        processed_img = min_pool2d(torch.from_numpy(processed_img).unsqueeze(1), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 1
    else:
        processed_img = get_subtracted_img_for_frames(video_path, start_frame, end_frame)
        img_shape = processed_img.size()
        processed_img = min_pool2d(processed_img.permute(0, 3, 1, 2), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 0

    if background_subtraction:
        mthd = background_subtraction_method
    else:
        mthd = 'Subtraction - Average Image'

    # if img_shape[1] < img_shape[2]:
    #     original_dims = (img_shape[2] / 100 * 2, img_shape[1] / 100 * 2)
    # else:
    #     original_dims = (img_shape[1] / 100 * 2, img_shape[2] / 100 * 2)

    if img_shape[1] < img_shape[2]:
        original_dims = (img_shape[2] / 100, img_shape[1] / 100)
    else:
        original_dims = (img_shape[1] / 100, img_shape[2] / 100)

    cap = cv.VideoCapture(video_path)
    cap_count = 0
    original_scale = (img_shape[1], img_shape[2])
    min_pooled_shape = (processed_img[0].shape[0], processed_img[0].shape[1])

    previous = None
    previous_pooled = False
    hsv = None

    gt_bbox_cluster_center_dict = {}
    pr_for_frames = {}

    for fr in tqdm(range(start_idx, processed_img.shape[0])):
        ret, frame = cap.read()
        if previous is None:
            previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # hsv = np.zeros_like(frame)
            # hsv[..., 1] = 255
            continue

        if cap_count < start_frame:
            continue

        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if not previous_pooled:
            # previous = (min_pool2d(torch.from_numpy(previous / 255).unsqueeze(0), kernel_size=min_pool_kernel_size)
            #             .squeeze() * 255).int().numpy()
            previous = resize(previous, output_shape=min_pooled_shape)
            previous_pooled = True
        # next = (min_pool2d(torch.from_numpy(next / 255).unsqueeze(0), kernel_size=min_pool_kernel_size).squeeze()
        #         * 255).int().numpy()
        next = resize(next, output_shape=min_pooled_shape)

        flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow = min_pool2d(torch.from_numpy(flow).permute(2, 0, 1), kernel_size=min_pool_kernel_size).permute(1, 2, 0)\
        #     .numpy()

        hsv = np.zeros((next.shape[0], next.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # 0-1 normalize
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        data_ = processed_img[fr]
        data_ = np.abs(data_)
        thresholded_img = np.zeros_like(data_)

        object_idx = (data_ > 0).nonzero()  # note: for bg
        intensities = data_[object_idx[0], object_idx[1]]

        flow_idx = flow[object_idx[0], object_idx[1]]
        flow_idx_normalized_0, f_max_0, f_min_0 = normalize(flow_idx[..., 0])
        flow_idx_normalized_1, f_max_1, f_min_1 = normalize(flow_idx[..., 1])

        thresholded_img[object_idx[0], object_idx[1]] = data_[object_idx[0], object_idx[1]]

        object_idx_normalized_0, max_0, min_0 = normalize(object_idx[0])
        object_idx_normalized_1, max_1, min_1 = normalize(object_idx[1])

        # data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx[..., 0],
        #                  flow_idx[..., 1])).transpose()
        # data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx[..., 1],
        #                  flow_idx[..., 0])).transpose()  # better - increased recall in general, sometimes better/bad pr

        data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
                         flow_idx_normalized_0)).transpose()

        # data = np.stack((object_idx[1], object_idx[0], intensities)).transpose()
        # data = np.stack((object_idx[1], object_idx[0])).transpose()  # better
        # data = np.stack((object_idx[1], object_idx[0], flow_idx[..., 0], flow_idx[..., 1])).transpose()

        # data = np.stack((flow_idx[..., 0], flow_idx[..., 1])).transpose()
        # data = flow.reshape(-1, 2)

        bandwidth = estimate_bandwidth(data, quantile=0.1, n_jobs=8)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        cc_0 = denormalize(cluster_centers[..., 0], max_1, min_1)  # maybe for portrait images we need to switch?
        cc_1 = denormalize(cluster_centers[..., 1], max_0, min_0)
        cluster_centers[..., 0] = cc_0
        cluster_centers[..., 1] = cc_1

        # cluster_centers = pca_cluster_center(cluster_centers) # ??
        # scaled_cluster_centers = scale_cluster_center(cluster_centers, (data_.shape[0], data_.shape[1]),
        #                                               (frame.shape[0], frame.shape[1])) # fixme

        labels_unique, points_per_cluster = np.unique(labels, return_counts=True)
        cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        # print(cluster_distribution)

        annot_ = get_frame_annotations(annotations_df, frame_number=fr)  # check-out +/- 1
        annot, _ = scale_annotations(annot_, original_scale, (data_.shape[0], data_.shape[1]))
        gt_bbox_cluster_center_dict.update({fr: {'gt_bbox': annot,
                                                 'cluster_centers': cluster_centers}})
        frame_results = evaluate_clustering_per_frame(fr, {'gt_bbox': annot,
                                                           'cluster_centers': cluster_centers})

        pre_rec = precision_recall(frame_results)
        pr_for_frames.update({fr: pre_rec})

        # fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all',
        #                                figsize=original_dims)
        # ax1.imshow(data_, cmap='gray')
        # ax2.imshow(resize(frame, output_shape=(data_.shape[0], data_.shape[1])))
        #
        # ax1.set_title(f"Processed Frame: {fr}")
        # ax2.set_title(f"Frame: {fr}")
        #
        # add_bbox_to_axes(ax2, annotations=annot_, only_pedestrians=False,
        #                  original_spatial_dim=original_scale,
        #                  pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
        #                  min_pool=True, use_dnn=False)
        #
        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        # for k, col in zip(range(n_clusters_), colors):
        #     cluster_center = cluster_centers[k]
        #     ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=8)
        #     ax2.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=8)

        # all steps
        # fig, axs = plt.subplots(2, 3, sharex='none', sharey='none',
        #                         figsize=original_dims)
        # axs[0, 0].imshow(frame)
        # axs[0, 1].imshow(data_, cmap='gray')
        # axs[0, 2].imshow(thresholded_img, cmap='gray')
        # axs[1, 0].imshow(data_, cmap='gray')
        # axs[1, 1].imshow(resize(frame, output_shape=(data_.shape[0], data_.shape[1])))
        # # axs[1, 2].imshow(thresholded_img, cmap='gray')
        # axs[1, 2].imshow(rgb)
        #
        # axs[0, 0].set_title('Image')
        # axs[0, 1].set_title('Processed Image')
        # axs[0, 2].set_title('Thresholded Image')
        # axs[1, 0].set_title('Clustered - Subtracted')
        # axs[1, 1].set_title('Clustered - Image')
        # axs[1, 2].set_title('Optical Flow')
        # # axs[1, 2].set_title('Clustered - Thresholded')
        #
        # add_bbox_to_axes(axs[0, 0], annotations=annot_, only_pedestrians=False,
        #                  original_spatial_dim=original_scale,
        #                  pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
        #                  min_pool=False, use_dnn=False)
        # add_bbox_to_axes(axs[0, 1], annotations=annot_, only_pedestrians=False,
        #                  original_spatial_dim=original_scale,
        #                  pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
        #                  min_pool=True, use_dnn=False)
        # add_bbox_to_axes(axs[1, 1], annotations=annot_, only_pedestrians=False,
        #                  original_spatial_dim=original_scale,
        #                  pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
        #                  min_pool=True, use_dnn=False)
        #
        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        # for k, col in zip(range(n_clusters_), colors):
        #     cluster_center = cluster_centers[k]
        #     axs[1, 0].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #                    markeredgecolor='k', markersize=8)
        #     axs[1, 1].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #                    markeredgecolor='k', markersize=8)
        #     # axs[1, 2].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #     #                markeredgecolor='k', markersize=8)
        #
        # fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}\nMethod: {mthd}"
        #              f"\nPrecision: {pre_rec[fr]['precision']}\nRecall: {pre_rec[fr]['recall']}",
        #              fontsize=14, fontweight='bold')

        cap_count += 1
        previous = next
        # fig.savefig(video_out_save_path + f"frame_{fr}.png")

    return gt_bbox_cluster_center_dict, pr_for_frames


def mean_shift_clustering_with_min_pooled_optical_frames(video_path, start_frame=0, end_frame=None,
                                                         min_pool_kernel_size=5,
                                                         desired_fps=6, video_out_save_path=None, annotations_df=None,
                                                         show_bbox=False,
                                                         video_label=None, vid_number=None,
                                                         background_subtraction=False,
                                                         background_subtraction_method='mog2'):
    if background_subtraction:
        processed_img = get_background_subtraction(video_path, start_frame, end_frame,
                                                   method=background_subtraction_method, history=120,
                                                   detect_shadows=True)
        img_shape = processed_img.shape
        # processed_img = processed_img / 255
        # processed_img = min_pool2d(torch.from_numpy(processed_img).unsqueeze(1), kernel_size=min_pool_kernel_size)
        # processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 1
    else:
        processed_img = get_subtracted_img_for_frames(video_path, start_frame, end_frame)
        img_shape = processed_img.size()
        processed_img = min_pool2d(processed_img.permute(0, 3, 1, 2), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 0

    if background_subtraction:
        mthd = background_subtraction_method
    else:
        mthd = 'Subtraction - Average Image'

    # if img_shape[1] < img_shape[2]:
    #     original_dims = (img_shape[2] / 100 * 2, img_shape[1] / 100 * 2)
    # else:
    #     original_dims = (img_shape[1] / 100 * 2, img_shape[2] / 100 * 2)

    if img_shape[1] < img_shape[2]:
        original_dims = (img_shape[2] / 100, img_shape[1] / 100)
    else:
        original_dims = (img_shape[1] / 100, img_shape[2] / 100)

    cap = cv.VideoCapture(video_path)
    cap_count = 0
    original_scale = (img_shape[1], img_shape[2])
    min_pooled_shape = (processed_img[0].shape[0], processed_img[0].shape[1])

    previous = None
    previous_pooled = False
    hsv = None

    gt_bbox_cluster_center_dict = {}
    pr_for_frames = {}

    for fr in tqdm(range(start_idx, processed_img.shape[0])):
        ret, frame = cap.read()
        if previous is None:
            previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # hsv = np.zeros_like(frame)
            # hsv[..., 1] = 255
            continue

        if cap_count < start_frame:
            continue

        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if not previous_pooled:
            previous = (min_pool2d(torch.from_numpy(previous / 255).unsqueeze(0), kernel_size=min_pool_kernel_size)
                        .squeeze() * 255).int().numpy()
            # previous = resize(previous, output_shape=min_pooled_shape)
            previous_pooled = True
        next = (min_pool2d(torch.from_numpy(next / 255).unsqueeze(0), kernel_size=min_pool_kernel_size).squeeze()
                * 255).int().numpy()
        # next = resize(next, output_shape=min_pooled_shape)

        flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow = min_pool2d(torch.from_numpy(flow).permute(2, 0, 1), kernel_size=min_pool_kernel_size).permute(1, 2, 0)\
        #     .numpy()

        hsv = np.zeros((next.shape[0], next.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # 0-1 normalize
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        data_ = processed_img[fr]
        data_ = np.abs(data_)
        thresholded_img = np.zeros_like(data_)

        object_idx = (data_ > 50).nonzero()  # note: for bg
        intensities = data_[object_idx[0], object_idx[1]]

        flow_idx = flow[object_idx[0], object_idx[1]]
        flow_idx_normalized_0, f_max_0, f_min_0 = normalize(flow_idx[..., 0])
        flow_idx_normalized_1, f_max_1, f_min_1 = normalize(flow_idx[..., 1])

        thresholded_img[object_idx[0], object_idx[1]] = data_[object_idx[0], object_idx[1]]

        object_idx_normalized_0, max_0, min_0 = normalize(object_idx[0])
        object_idx_normalized_1, max_1, min_1 = normalize(object_idx[1])

        # data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx[..., 0],
        #                  flow_idx[..., 1])).transpose()
        # data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx[..., 1],
        #                  flow_idx[..., 0])).transpose()  # better - increased recall in general, sometimes better/bad pr

        data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
                         flow_idx_normalized_0)).transpose()

        # data = np.stack((object_idx[1], object_idx[0], intensities)).transpose()
        # data = np.stack((object_idx[1], object_idx[0])).transpose()  # better
        # data = np.stack((object_idx[1], object_idx[0], flow_idx[..., 0], flow_idx[..., 1])).transpose()

        # data = np.stack((flow_idx[..., 0], flow_idx[..., 1])).transpose()
        # data = flow.reshape(-1, 2)

        # bandwidth = estimate_bandwidth(data, quantile=0.1, n_jobs=8)
        bandwidth = 0.1
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True, min_bin_freq=3, max_iter=600)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        cc_0 = denormalize(cluster_centers[..., 0], max_1, min_1)  # maybe for portrait images we need to switch?
        cc_1 = denormalize(cluster_centers[..., 1], max_0, min_0)
        cluster_centers[..., 0] = cc_0
        cluster_centers[..., 1] = cc_1

        # cluster_centers = pca_cluster_center(cluster_centers) # ??
        # scaled_cluster_centers = scale_cluster_center(cluster_centers, (data_.shape[0], data_.shape[1]),
        #                                               (frame.shape[0], frame.shape[1])) # fixme

        labels_unique, points_per_cluster = np.unique(labels, return_counts=True)
        cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        # print(cluster_distribution)

        annot_ = get_frame_annotations(annotations_df, frame_number=fr)  # check-out +/- 1
        annot_ = preprocess_annotations(annot_)
        annot, _ = scale_annotations(annot_, original_scale, (data_.shape[0], data_.shape[1]))
        gt_bbox_cluster_center_dict.update({fr: {'gt_bbox': annot,
                                                 'cluster_centers': cluster_centers}})
        frame_results = evaluate_clustering_per_frame(fr, {'gt_bbox': annot,
                                                           'cluster_centers': cluster_centers})

        pre_rec = precision_recall(frame_results)
        pr_for_frames.update({fr: pre_rec})

        # fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all',
        #                                figsize=original_dims)
        # ax1.imshow(data_, cmap='gray')
        # ax2.imshow(resize(frame, output_shape=(data_.shape[0], data_.shape[1])))
        #
        # ax1.set_title(f"Processed Frame: {fr}")
        # ax2.set_title(f"Frame: {fr}")
        #
        # add_bbox_to_axes(ax2, annotations=annot_, only_pedestrians=False,
        #                  original_spatial_dim=original_scale,
        #                  pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
        #                  min_pool=True, use_dnn=False)
        #
        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        # for k, col in zip(range(n_clusters_), colors):
        #     cluster_center = cluster_centers[k]
        #     ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=8)
        #     ax2.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=8)

        # all steps
        fig, axs = plt.subplots(2, 3, sharex='none', sharey='none',
                                figsize=original_dims)
        axs[0, 0].imshow(frame)
        axs[0, 1].imshow(data_, cmap='binary')
        axs[0, 2].imshow(thresholded_img, cmap='binary')
        axs[1, 0].imshow(data_, cmap='binary')
        axs[1, 1].imshow(resize(frame, output_shape=(data_.shape[0], data_.shape[1])))
        axs[1, 2].imshow(thresholded_img, cmap='binary')
        # axs[1, 2].imshow(rgb)

        axs[0, 0].set_title('Image')
        axs[0, 1].set_title('Processed Image')
        axs[0, 2].set_title('Thresholded Image')
        axs[1, 0].set_title('Clustered - Subtracted')
        axs[1, 1].set_title('Clustered - Image')
        # axs[1, 2].set_title('Optical Flow')
        axs[1, 2].set_title('Clustered - Thresholded')

        add_bbox_to_axes(axs[0, 0], annotations=annot_, only_pedestrians=False,
                         original_spatial_dim=original_scale,
                         pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
                         min_pool=False, use_dnn=False)
        add_bbox_to_axes(axs[0, 1], annotations=annot_, only_pedestrians=False,
                         original_spatial_dim=original_scale,
                         pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
                         min_pool=True, use_dnn=False)
        add_bbox_to_axes(axs[1, 1], annotations=annot_, only_pedestrians=False,
                         original_spatial_dim=original_scale,
                         pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
                         min_pool=True, use_dnn=False)

        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        # colors = cycle(COLORS2)
        colors = cm.rainbow(np.linspace(0, 1, 50))
        marker = cycle('o*v^><12348sphH+xXD')
        for k, col, m in zip(range(n_clusters_), colors, marker):
            cluster_center = cluster_centers[k]
            axs[1, 0].plot(cluster_center[0], cluster_center[1], m, markerfacecolor=col,  # 'o'
                           markeredgecolor='k', markersize=8)
            axs[1, 1].plot(cluster_center[0], cluster_center[1], m, markerfacecolor=col,
                           markeredgecolor='k', markersize=8)
            axs[1, 2].plot(cluster_center[0], cluster_center[1], m, markerfacecolor=col,
                           markeredgecolor='k', markersize=8)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}\nMethod: {mthd}"
                     f"\nPrecision: {pre_rec[fr]['precision']}\nRecall: {pre_rec[fr]['recall']}"
                     f"\nClusters Found: {n_clusters_}",
                     fontsize=14, fontweight='bold')

        cap_count += 1
        previous = next
        # fig.savefig(video_out_save_path + f"frame_{fr}.png")

    return gt_bbox_cluster_center_dict, pr_for_frames


def mean_shift_clustering_without_optical_frames(video_path, start_frame=0, end_frame=None, min_pool_kernel_size=5,
                                                 desired_fps=6, video_out_save_path=None, annotations_df=None,
                                                 show_bbox=False,
                                                 video_label=None, vid_number=None, background_subtraction=False,
                                                 background_subtraction_method='mog2'):
    if background_subtraction:
        processed_img = get_background_subtraction(video_path, start_frame, end_frame,
                                                   method=background_subtraction_method)
        img_shape = processed_img.shape
        processed_img = processed_img / 255
        processed_img = min_pool2d(torch.from_numpy(processed_img).unsqueeze(1), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 1
    else:
        processed_img = get_subtracted_img_for_frames(video_path, start_frame, end_frame)
        img_shape = processed_img.size()
        processed_img = min_pool2d(processed_img.permute(0, 3, 1, 2), kernel_size=min_pool_kernel_size)
        processed_img = (processed_img * 255).int().squeeze().numpy()
        start_idx = 0

    if background_subtraction:
        mthd = background_subtraction_method
    else:
        mthd = 'Subtraction - Average Image'

    # if img_shape[1] < img_shape[2]:
    #     original_dims = (img_shape[2] / 100 * 2, img_shape[1] / 100 * 2)
    # else:
    #     original_dims = (img_shape[1] / 100 * 2, img_shape[2] / 100 * 2)

    if img_shape[1] < img_shape[2]:
        original_dims = (img_shape[2] / 100, img_shape[1] / 100)
    else:
        original_dims = (img_shape[1] / 100, img_shape[2] / 100)

    cap = cv.VideoCapture(video_path)
    cap_count = 0
    original_scale = (img_shape[1], img_shape[2])
    min_pooled_shape = (processed_img[0].shape[0], processed_img[0].shape[1])

    previous = None
    previous_pooled = False
    hsv = None

    gt_bbox_cluster_center_dict = {}
    pr_for_frames = {}

    for fr in tqdm(range(start_idx, processed_img.shape[0])):
        ret, frame = cap.read()
        if previous is None:
            previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # hsv = np.zeros_like(frame)
            # hsv[..., 1] = 255
            continue

        if cap_count < start_frame:
            continue

        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if not previous_pooled:
            # previous = (min_pool2d(torch.from_numpy(previous / 255).unsqueeze(0), kernel_size=min_pool_kernel_size)
            #             .squeeze() * 255).int().numpy()
            previous = resize(previous, output_shape=min_pooled_shape)
            previous_pooled = True
        # next = (min_pool2d(torch.from_numpy(next / 255).unsqueeze(0), kernel_size=min_pool_kernel_size).squeeze()
        #         * 255).int().numpy()
        next = resize(next, output_shape=min_pooled_shape)

        flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow = min_pool2d(torch.from_numpy(flow).permute(2, 0, 1), kernel_size=min_pool_kernel_size).permute(1, 2, 0)\
        #     .numpy()

        hsv = np.zeros((next.shape[0], next.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # 0-1 normalize
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        data_ = processed_img[fr]
        data_ = np.abs(data_)
        thresholded_img = np.zeros_like(data_)

        object_idx = (data_ > 0).nonzero()
        intensities = data_[object_idx[0], object_idx[1]]

        flow_idx = flow[object_idx[0], object_idx[1]]
        flow_idx_normalized_0, f_max_0, f_min_0 = normalize(flow_idx[..., 0])
        flow_idx_normalized_1, f_max_1, f_min_1 = normalize(flow_idx[..., 1])

        thresholded_img[object_idx[0], object_idx[1]] = data_[object_idx[0], object_idx[1]]

        object_idx_normalized_0, max_0, min_0 = normalize(object_idx[0])
        object_idx_normalized_1, max_1, min_1 = normalize(object_idx[1])

        # data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx[..., 0],
        #                  flow_idx[..., 1])).transpose()
        # data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx[..., 1],
        #                  flow_idx[..., 0])).transpose()  # better - increased recall in general, sometimes better/bad pr

        # data = np.stack((object_idx_normalized_1, object_idx_normalized_0, flow_idx_normalized_1,
        #                  flow_idx_normalized_0)).transpose()

        # data = np.stack((object_idx[1], object_idx[0], intensities)).transpose()
        data = np.stack((object_idx[1], object_idx[0])).transpose()  # better
        # data = np.stack((object_idx[1], object_idx[0], flow_idx[..., 0], flow_idx[..., 1])).transpose()

        # data = np.stack((flow_idx[..., 0], flow_idx[..., 1])).transpose()
        # data = flow.reshape(-1, 2)

        bandwidth = estimate_bandwidth(data, quantile=0.1, n_jobs=8)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        # cc_0 = denormalize(cluster_centers[..., 0], max_1, min_1)  # maybe for portrait images we need to switch?
        # cc_1 = denormalize(cluster_centers[..., 1], max_0, min_0)
        # cluster_centers[..., 0] = cc_0
        # cluster_centers[..., 1] = cc_1

        # cluster_centers = pca_cluster_center(cluster_centers) # ??
        # scaled_cluster_centers = scale_cluster_center(cluster_centers, (data_.shape[0], data_.shape[1]),
        #                                               (frame.shape[0], frame.shape[1])) # fixme

        labels_unique, points_per_cluster = np.unique(labels, return_counts=True)
        cluster_distribution = dict(zip(labels_unique, points_per_cluster))
        n_clusters_ = len(labels_unique)
        # print(cluster_distribution)

        annot_ = get_frame_annotations(annotations_df, frame_number=fr)  # check-out +/- 1
        annot, _ = scale_annotations(annot_, original_scale, (data_.shape[0], data_.shape[1]))
        gt_bbox_cluster_center_dict.update({fr: {'gt_bbox': annot,
                                                 'cluster_centers': cluster_centers}})
        frame_results = evaluate_clustering_per_frame(fr, {'gt_bbox': annot,
                                                           'cluster_centers': cluster_centers})

        pre_rec = precision_recall(frame_results)
        pr_for_frames.update({fr: pre_rec})

        # fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all',
        #                                figsize=original_dims)
        # ax1.imshow(data_, cmap='gray')
        # ax2.imshow(resize(frame, output_shape=(data_.shape[0], data_.shape[1])))
        #
        # ax1.set_title(f"Processed Frame: {fr}")
        # ax2.set_title(f"Frame: {fr}")
        #
        # add_bbox_to_axes(ax2, annotations=annot_, only_pedestrians=False,
        #                  original_spatial_dim=original_scale,
        #                  pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
        #                  min_pool=True, use_dnn=False)
        #
        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        # for k, col in zip(range(n_clusters_), colors):
        #     cluster_center = cluster_centers[k]
        #     ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=8)
        #     ax2.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=8)

        # all steps
        fig, axs = plt.subplots(2, 3, sharex='none', sharey='none',
                                figsize=original_dims)
        axs[0, 0].imshow(frame)
        axs[0, 1].imshow(data_, cmap='gray')
        axs[0, 2].imshow(thresholded_img, cmap='gray')
        axs[1, 0].imshow(data_, cmap='gray')
        axs[1, 1].imshow(resize(frame, output_shape=(data_.shape[0], data_.shape[1])))
        # axs[1, 2].imshow(thresholded_img, cmap='gray')
        axs[1, 2].imshow(rgb)

        axs[0, 0].set_title('Image')
        axs[0, 1].set_title('Processed Image')
        axs[0, 2].set_title('Thresholded Image')
        axs[1, 0].set_title('Clustered - Subtracted')
        axs[1, 1].set_title('Clustered - Image')
        axs[1, 2].set_title('Optical Flow')
        # axs[1, 2].set_title('Clustered - Thresholded')

        add_bbox_to_axes(axs[0, 0], annotations=annot_, only_pedestrians=False,
                         original_spatial_dim=original_scale,
                         pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
                         min_pool=False, use_dnn=False)
        add_bbox_to_axes(axs[0, 1], annotations=annot_, only_pedestrians=False,
                         original_spatial_dim=original_scale,
                         pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
                         min_pool=True, use_dnn=False)
        add_bbox_to_axes(axs[1, 1], annotations=annot_, only_pedestrians=False,
                         original_spatial_dim=original_scale,
                         pooled_spatial_dim=(data_.shape[0], data_.shape[1]),
                         min_pool=True, use_dnn=False)

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            cluster_center = cluster_centers[k]
            axs[1, 0].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                           markeredgecolor='k', markersize=8)
            axs[1, 1].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                           markeredgecolor='k', markersize=8)
            # axs[1, 2].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            #                markeredgecolor='k', markersize=8)

        fig.suptitle(f"Video Class: {video_label}\nVideo Number: {vid_number}\nMethod: {mthd}"
                     f"\nPrecision: {pre_rec[fr]['precision']}\nRecall: {pre_rec[fr]['recall']}",
                     fontsize=14, fontweight='bold')

        cap_count += 1
        previous = next
        fig.savefig(video_out_save_path + f"frame_{fr}.png")

    return gt_bbox_cluster_center_dict, pr_for_frames


def point_in_rect(point, rect):
    x1, y1, x2, y2 = rect
    x, y = point
    if x1 < x < x2:
        if y1 < y < y2:
            return True
    return False


def evaluate_clustering(gt_bbox_cluster_center_dict_):
    result_dict = {}
    for frame, frame_dict in gt_bbox_cluster_center_dict_.items():
        matched_cluster_centers = []
        matched_annotation = []
        gt_annotation_matched = 0
        cluster_center_in_bbox_count = 0
        total_cluster_centers = frame_dict['cluster_centers'].shape[0]
        total_annot = len(frame_dict['gt_bbox'])
        for annotation in frame_dict['gt_bbox']:
            annt = CoordinateHolder(annotation)
            if annt in matched_annotation:
                continue
            for cluster_center in frame_dict['cluster_centers']:
                cc = CoordinateHolder(cluster_center)
                if cc in matched_cluster_centers:
                    continue
                point = (cluster_center[0], cluster_center[1])
                rect = annotation
                cluster_center_in_box = point_in_rect(point, rect)
                if cluster_center_in_box:
                    cluster_center_in_bbox_count += 1
                    gt_annotation_matched += 1
                    matched_cluster_centers.append(CoordinateHolder(cluster_center))
                    matched_annotation.append(CoordinateHolder(annotation))
                    continue
        result_dict.update({frame: {'gt_annotation_matched': gt_annotation_matched,
                                    'cluster_center_in_bbox_count': cluster_center_in_bbox_count,
                                    'total_cluster_centers': total_cluster_centers,
                                    'total_annotations': total_annot}})
    return result_dict


def preprocess_annotations(annotations):
    ann = []
    for a in annotations:
        if a[6] != 1:
            ann.append(a)
    return ann


def evaluate_clustering_per_frame(frame, frame_dict):  # use this inside above
    result_dict = {}
    matched_cluster_centers = []
    matched_annotation = []
    gt_annotation_matched = 0
    cluster_center_in_bbox_count = 0
    total_cluster_centers = frame_dict['cluster_centers'].shape[0]
    total_annot = len(frame_dict['gt_bbox'])
    cc_cordinate_holder = CoordinateHolder if len(frame_dict['cluster_centers'][0]) == 4 else CoordinateHolder2
    for annotation in frame_dict['gt_bbox']:
        # annt = CoordinateHolder(annotation)
        # if annt in matched_annotation:
        #     continue
        if check_presence(annotation, matched_annotation):
            continue
        for cluster_center in frame_dict['cluster_centers']:
            # cc = cc_cordinate_holder(cluster_center)
            # if cc in matched_cluster_centers:
            #     continue
            if check_presence(cluster_center, matched_cluster_centers):
                continue
            point = (cluster_center[0], cluster_center[1])
            rect = annotation
            cluster_center_in_box = point_in_rect(point, rect)
            if cluster_center_in_box:
                # cluster_center_in_bbox_count += 1
                # gt_annotation_matched += 1
                # matched_cluster_centers.append(cc_cordinate_holder(cluster_center))
                # matched_annotation.append(CoordinateHolder(annotation))
                if not check_presence(cluster_center, matched_cluster_centers):
                    matched_cluster_centers.append(cluster_center)
                if not check_presence(annotation, matched_annotation):
                    # cluster_center_in_bbox_count += 1
                    # gt_annotation_matched += 1
                    matched_annotation.append(annotation)
                continue
    # result_dict.update({frame: {'gt_annotation_matched': gt_annotation_matched,
    #                             'cluster_center_in_bbox_count': cluster_center_in_bbox_count,
    #                             'total_cluster_centers': total_cluster_centers,
    #                             'total_annotations': total_annot}})

    result_dict.update({frame: {'gt_annotation_matched': len(matched_annotation),
                                'cluster_center_in_bbox_count': len(matched_cluster_centers),
                                'total_cluster_centers': total_cluster_centers,
                                'total_annotations': total_annot}})
    return result_dict


def check_presence(item, item_list):
    return any(set(i).intersection(set(item)) for i in item_list)


def precision_recall(result_dict):
    pr_result = {}
    for frame, res in result_dict.items():
        precision = res['cluster_center_in_bbox_count'] / res['total_cluster_centers']
        recall = res['gt_annotation_matched'] / res['total_annotations']
        pr_result.update({frame: {'precision': precision,
                                  'recall': recall}})
    return pr_result


def plot_pr(precision_recall_):
    pass


# not working properly
def scale_cluster_center(cluster_center, current_scale, new_scale):
    new_cc = []
    for cc in cluster_center:
        new_x = (cc[0] / current_scale[0]) * new_scale[0]
        new_y = (cc[1] / current_scale)[1] * new_scale[1]
        new_u = (cc[2] / current_scale[0]) * new_scale[0]
        new_v = (cc[3] / current_scale)[1] * new_scale[1]  # ???
        new_cc.append([new_x, new_y, new_u, new_v])
    return new_cc


def show_img(img):
    plt.imshow(img)
    plt.show()


def clustering_with_optical_flow(video_path, start_frame=0, end_frame=None, min_pool_kernel_size=5):
    cap = cv.VideoCapture(video_path)
    cap_count = 0
    original_scale = None

    previous = None
    previous_pooled = False
    hsv = None

    for fr in tqdm(range(start_frame, end_frame)):
        ret, frame = cap.read()
        if previous is None:
            previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            original_scale = (frame.shape[0], frame.shape[1])
            continue

        if cap_count < start_frame:
            continue

        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv[..., 1] = 255
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # 0-1 normalize
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        pass


if __name__ == '__main__':
    annotation_base_path = "/home/rishabh/TrajectoryPrediction/Datasets/SDD/annotations/"
    video_base_path = "/home/rishabh/TrajectoryPrediction/Datasets/SDD/videos/"
    vid_label = SDDVideoClasses.GATES
    video_number = 4
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
    plot_save_path = "../../Plots/outputs/"
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
    df = annotations_to_dataframe(annotation_file)
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

    # optical_flow_subtraction_mask_video(video_path=video_file, video_label=vid_label.value, annotations_df=df,
    #                                     vid_number=video_number,
    #                                     video_out_save_path=plot_save_path + f"video_optical_flow_{vid_label.value}"
    #                                                                          f"_subtracted_img_with_min_pool_0_bbox.avi",
    #                                     start_frame=0, end_frame=120, desired_fps=6, min_pool_kernel_size=10)

    # mean_shift_clustering(video_path=video_file, start_frame=0, end_frame=90, min_pool_kernel_size=10)

    # mean_shift_clustering_video(video_path=video_file, start_frame=0, end_frame=90, min_pool_kernel_size=3,
    #                             desired_fps=2, video_out_save_path=plot_save_path + f"video_clustering_bg_sub"
    #                                                                                 f"_{vid_label.value}_mog2_3.avi",
    #                             annotations_df=df, video_label=vid_label.value, vid_number=video_number,
    #                             background_subtraction=True, background_subtraction_method='mog2')

    # mean_shift_clustering_video(video_path=video_file, start_frame=0, end_frame=90, min_pool_kernel_size=6,
    #                             desired_fps=2, video_out_save_path=plot_save_path + f"video_clustering_bg_sub"
    #                                                                                 f"_{vid_label.value}_knn_6.avi",
    #                             annotations_df=df, video_label=vid_label.value, vid_number=video_number,
    #                             background_subtraction=True, background_subtraction_method='knn')

    # mean_shift_clustering_with_optical_video(video_path=video_file, start_frame=0, end_frame=90,
    #                                          min_pool_kernel_size=10,
    #                                          desired_fps=2,
    #                                          video_out_save_path=plot_save_path + f"video_clustering_bg_sub_optical_flw"
    #                                                                               f"_{vid_label.value}_mog2_3_exp.avi",
    #                                          annotations_df=df, video_label=vid_label.value, vid_number=video_number,
    #                                          background_subtraction=False, background_subtraction_method='mog2')
    #
    # gt_bbox_cluster_center_dict = mean_shift_clustering_with_optical_video \
    #     (video_path=video_file, start_frame=0,
    #      end_frame=30,
    #      min_pool_kernel_size=10,
    #      desired_fps=2,
    #      video_out_save_path=plot_save_path +
    #                          f"video_clustering_optical_flw "
    #                          f"_{vid_label.value}_min_pool_input_subtraction.avi",
    #      annotations_df=df,
    #      video_label=vid_label.value,
    #      vid_number=video_number,
    #      background_subtraction=False,
    #      background_subtraction_method='mog2')
    #
    # result = evaluate_clustering(gt_bbox_cluster_center_dict)
    #
    # pr_res = precision_recall(result)
    #
    # print(pr_res)

    # gt_bbox_cluster_center_dict, sequence_pr = mean_shift_clustering_with_optical_frames \
    #     (video_path=video_file, start_frame=0,
    #      end_frame=30,
    #      min_pool_kernel_size=3,
    #      desired_fps=2,
    #      video_out_save_path=plot_save_path + 'clustering/' +
    #                          f"clustering_subtracted_video_number_{vid_label.value}"
    #                          f"video_abs{vid_label.value}",
    #      annotations_df=df,
    #      video_label=vid_label.value,
    #      vid_number=video_number,
    #      background_subtraction=True,
    #      background_subtraction_method='mog2')

    gt_bbox_cluster_center_dict_, sequence_pr_ = mean_shift_clustering_with_min_pooled_optical_frames \
        (video_path=video_file, start_frame=0,
         end_frame=60,
         min_pool_kernel_size=10,
         desired_fps=2,
         video_out_save_path='/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Plots/' + 'clustering/' +
                             f"clustering_subtracted_video_number_{vid_label.value}"
                             f"video_abs{vid_label.value}",
         annotations_df=df,
         video_label=vid_label.value,
         vid_number=video_number,
         background_subtraction=False,
         background_subtraction_method='mog2')

    # gt_bbox_cluster_center_dict_, sequence_pr_ = mean_shift_clustering_without_optical_frames \
    #     (video_path=video_file, start_frame=0,
    #      end_frame=30,
    #      min_pool_kernel_size=3,
    #      desired_fps=2,
    #      video_out_save_path=plot_save_path + 'clustering/' +
    #                          f"clustering_subtracted_video_number_{vid_label.value}"
    #                          f"video_abs{vid_label.value}",
    #      annotations_df=df,
    #      video_label=vid_label.value,
    #      vid_number=video_number,
    #      background_subtraction=True,
    #      background_subtraction_method='mog2')

    # compare_precision(pr_results_1=sequence_pr_, pr_results_2=sequence_pr, avg_img=False)
    # compare_recall(pr_results_1=sequence_pr_, pr_results_2=sequence_pr, avg_img=False)

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
