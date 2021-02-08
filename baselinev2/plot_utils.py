from itertools import cycle
from pathlib import Path

from matplotlib import pyplot as plt, patches, lines as mlines


def plot_image(im):
    plt.imshow(im, cmap='gray')
    plt.show()


def plot_image_simple(im, bbox=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.imshow(im, cmap='gray')
    if bbox is not None:
        add_box_to_axes(axs, bbox)
    plt.show()


def plot_image_set_of_boxes(im, bbox1=None, bbox2=None, overlay=True, annotate=None):
    if overlay:
        fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
        axs.imshow(im, cmap='gray')
        if bbox1 is not None or bbox2 is not None:
            if annotate is None:
                add_box_to_axes(axs, bbox1, 'r')
                add_box_to_axes(axs, bbox2, 'aqua')
            else:
                add_box_to_axes_with_annotation(axs, bbox1, annotate[0], 'r')
                add_box_to_axes_with_annotation(axs, bbox2, annotate[1], 'aqua')

            legends_dict = {'r': 'GT Bounding Box',
                            'aqua': 'Generated Bounding Box'}

            legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
            fig.legend(handles=legend_patches, loc=2)
    else:
        fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
        axs[0].imshow(im, cmap='gray')
        axs[1].imshow(im, cmap='gray')
        if bbox1 is not None or bbox2 is not None:
            if annotate is None:
                add_box_to_axes(axs[0], bbox1, 'r')
                add_box_to_axes(axs[1], bbox2, 'aqua')
            else:
                add_box_to_axes_with_annotation(axs[0], bbox1, annotate[0], 'r')
                add_box_to_axes_with_annotation(axs[1], bbox2, annotate[1], 'aqua')

            legends_dict = {'r': 'GT Bounding Box',
                            'aqua': 'Generated Bounding Box'}

            legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
            fig.legend(handles=legend_patches, loc=2)
    plt.show()


def plot_random_legends():
    blue_star = mlines.Line2D([], [], color='white', marker='*', linestyle='None',
                              markersize=10, label='Cluster\'s Centroid', markeredgecolor='black')
    red_square = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                               markersize=10, label='Region of Validity', markerfacecolor='white')
    purple_triangle = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                                    markersize=10, label='Cluster Pool Region', markerfacecolor='white')
    purple_triangle0 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                                     markersize=10, label='Inside Region of Validity')
    purple_triangle1 = mlines.Line2D([], [], color='aqua', marker='o', linestyle='None',
                                     markersize=10, label='New track candidate activations')

    plt.legend(handles=[blue_star, red_square, purple_triangle, purple_triangle0, purple_triangle1])

    plt.show()
    # fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    # legends_dict = {'*': 'Cluster\'s Centroid',
    #                 'g': 'Cluster Pool Region',
    #                 'r': 'Ground Truth Bounding Box'}
    #
    # legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    # fig.legend(handles=legend_patches, loc=2)
    # plt.show()


def plot_features_simple(features, bbox=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=8)
    if bbox is not None:
        add_box_to_axes(axs, bbox)
    plt.show()


def plot_features_with_mask_simple(features, mask, bbox=None):
    fig, axs = plt.subplots(2, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(mask, cmap='gray')
    axs[1].plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=8)
    if bbox is not None:
        add_box_to_axes(axs[0], bbox)
        add_box_to_axes(axs[1], bbox)
    plt.show()


def plot_tracks_with_features(frame_t, frame_t_minus_one, frame_t_plus_one, features_t, features_t_minus_one, file_idx,
                              features_t_plus_one, box_t, box_t_minus_one, box_t_plus_one, frame_number, marker_size=8,
                              annotations=None, additional_text='', video_mode=False, save_path=None, track_id=None):
    fig, axs = plt.subplots(1, 3, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(frame_t_minus_one, cmap='gray')
    axs[1].imshow(frame_t, cmap='gray')
    axs[2].imshow(frame_t_plus_one, cmap='gray')

    axs[0].plot(features_t_minus_one[:, 0], features_t_minus_one[:, 1], 'o',
                markerfacecolor='blue', markeredgecolor='k', markersize=marker_size)
    axs[1].plot(features_t[:, 0], features_t[:, 1], 'o',
                markerfacecolor='blue', markeredgecolor='k', markersize=marker_size)
    axs[2].plot(features_t_plus_one[:, 0], features_t_plus_one[:, 1], 'o',
                markerfacecolor='blue', markeredgecolor='k', markersize=marker_size)

    if annotations is not None:
        add_box_to_axes_with_annotation(axs[0], box_t_minus_one, annotations[0])
        add_box_to_axes_with_annotation(axs[1], box_t, annotations[1])
        add_box_to_axes_with_annotation(axs[2], box_t_plus_one, annotations[2])
    else:
        add_box_to_axes(axs[0], box_t_minus_one)
        add_box_to_axes(axs[1], box_t)
        add_box_to_axes(axs[2], box_t_plus_one)

    axs[0].set_title('T-1')
    axs[1].set_title('T')
    axs[2].set_title('T+1')

    fig.suptitle(f'Track - Past|Present|Future\nFrame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'blue': 'Features'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            save_path = save_path + f'{track_id if track_id is not None else "all"}/'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"{file_idx}_track_plot_frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()


def plot_mask_matching_bbox(mask, bboxes, frame_num, save_path=None):
    fig, axs = plt.subplots(3, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(mask, cmap='gray')
    axs[1].imshow(mask, cmap='gray')
    axs[2].imshow(mask, cmap='gray')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    iou = {}
    dist = {}
    for box, color in zip(bboxes, colors):
        a_box = box[0]
        r_box = box[2]
        iou.update({color: box[3]})
        dist.update({color: box[1]})
        add_one_box_to_axis(axs[0], color, a_box)
        add_one_box_to_axis(axs[0], color, r_box)
        add_one_box_to_axis(axs[1], color, a_box)
        add_one_box_to_axis(axs[2], color, r_box)
    axs[0].set_title('Both')
    axs[1].set_title('GT')
    axs[2].set_title('OF')

    fig.suptitle(f'Frame number: {frame_num}\nIOU: {iou}\nL2: {dist}\n')
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path + f"frame_{frame_num}.png")
        plt.close()
    else:
        plt.show()


def add_one_box_to_axis(axs, color, box):
    rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                             edgecolor=color, fill=False, linewidth=None)
    axs.add_patch(rect)


def plot_features_overlayed_mask_simple(features, mask, bbox=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.imshow(mask, cmap='gray')
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=8)
    if bbox is not None:
        add_box_to_axes(axs, bbox)
    plt.show()


def plot_features_with_mask_and_rgb_simple(features, mask, rgb, bbox=None):
    fig, axs = plt.subplots(3, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(mask, cmap='gray')
    axs[1].plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=8)
    axs[2].imshow(rgb)
    if bbox is not None:
        add_box_to_axes(axs[0], bbox)
        add_box_to_axes(axs[1], bbox)
        add_box_to_axes(axs[2], bbox)
    plt.show()


def plot_features_with_circle(features, features_inside_circle, center, radius, mask=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    circle = plt.Circle((center[0], center[1]), radius, color='green', fill=False)
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8)
    axs.plot(features_inside_circle[:, 0], features_inside_circle[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=8)
    axs.add_artist(circle)
    if mask is not None:
        axs.imshow(mask, 'gray')
    plt.show()


def plot_features(features, features_inside_circle, features_skipped=None, mask=None, cluster_centers=None,
                  marker_size=1, num_clusters=None, frame_number=None, additional_text=None, boxes=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=marker_size)
    axs.plot(features_inside_circle[:, 0], features_inside_circle[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=marker_size)
    if mask is not None:
        axs.imshow(mask, 'gray')
    if features_skipped is not None:
        axs.plot(features_skipped[:, 0], features_skipped[:, 1], 'o', markerfacecolor='aqua',
                 markeredgecolor='k', markersize=marker_size)
    if cluster_centers is not None:
        axs.plot(cluster_centers[:, 0], cluster_centers[:, 1], '*', markerfacecolor='lavender', markeredgecolor='k',
                 markersize=marker_size + 8)
        fig.suptitle(f'Frame: {frame_number} | Clusters Count: {num_clusters}\n {additional_text}')
    if boxes is not None:
        add_box_to_axes(axs, boxes)

    legends_dict = {'yellow': 'Inside Circle',
                    'blue': 'Features',
                    'aqua': 'Skipped Features'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.show()


def plot_features_with_circles(features, features_inside_circle, features_skipped=None, mask=None, cluster_centers=None,
                               marker_size=1, num_clusters=None, frame_number=None, additional_text=None, boxes=None,
                               radius=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=marker_size)
    axs.plot(features_inside_circle[:, 0], features_inside_circle[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=marker_size)
    if mask is not None:
        axs.imshow(mask, 'gray')
    if features_skipped is not None:
        axs.plot(features_skipped[:, 0], features_skipped[:, 1], 'o', markerfacecolor='aqua',
                 markeredgecolor='k', markersize=marker_size)
    if cluster_centers is not None:
        axs.plot(cluster_centers[:, 0], cluster_centers[:, 1], '*', markerfacecolor='lavender', markeredgecolor='k',
                 markersize=marker_size + 8)
        fig.suptitle(f'Frame: {frame_number} | Clusters Count: {num_clusters}\n {additional_text}')
        for c_center in cluster_centers:
            axs.add_artist(plt.Circle((c_center[0], c_center[1]), radius, color='green', fill=False))
    if boxes is not None:
        add_box_to_axes(axs, boxes)
        # for box in boxes:
        #     box_center = get_bbox_center(box).flatten()
        #     axs.add_artist(plt.Circle((box_center[0], box_center[1]), 70.71, color='red', fill=False))

    legends_dict = {'yellow': 'Inside Circle',
                    'blue': 'Features',
                    'aqua': 'Skipped Features'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    plt.show()


def plot_features_with_mask(features, features_inside_circle, center, radius, mask, box=None, m_size=4,
                            current_boxes=None):
    fig, axs = plt.subplots(2, 1, sharex='none', sharey='none', figsize=(12, 10))
    circle = plt.Circle((center[0], center[1]), radius, color='green', fill=False)
    axs[0].plot(features[:, 0], features[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
                markersize=m_size)
    axs[0].plot(features_inside_circle[:, 0], features_inside_circle[:, 1], 'o', markerfacecolor='yellow',
                markeredgecolor='k', markersize=m_size)
    axs[0].add_artist(circle)
    axs[0].imshow(mask, 'binary')
    axs[1].imshow(mask, 'gray')
    if box is not None:
        rect0 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                  edgecolor='r', fill=False, linewidth=None)
        rect1 = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                  edgecolor='r', fill=False, linewidth=None)
        axs[0].add_patch(rect0)
        axs[1].add_patch(rect1)
    if current_boxes is not None:
        add_box_to_axes(axs[0], current_boxes, 'orange')
        add_box_to_axes(axs[1], current_boxes, 'orange')
    plt.show()


def plot_track_history_with_angle_info(img, box, history, direction, frame_number, track_id, additional_text='',
                                       save_path=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    axs.imshow(img, cmap='gray')
    add_box_to_axes(axs, box)
    add_features_to_axis(axs, history, marker_size=1, marker_color='g')

    fig.suptitle(f'Frame: {frame_number} | Track Id: {track_id}'
                 f'\nAngle bw velocity vectors: {direction}\n {additional_text}')

    legends_dict = {'red': 'Bounding Box',
                    'g': 'Track'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path + f"track_id_{track_id}_frame_{frame_number}.png")
        plt.close()
    else:
        plt.show()


def plot_track_history_with_angle_info_with_track_plot(img, box, history, direction, frame_number, track_id,
                                                       additional_text='', save_path=None):
    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))
    axs[0].imshow(img, cmap='gray')
    if len(box) != 0:
        add_one_box_to_axis(axs[0], 'r', box)
        add_one_box_to_axis(axs[1], 'r', box)

    if len(history) != 0:
        add_features_to_axis(axs[0], history, marker_size=1, marker_color='g')
        add_features_to_axis(axs[1], history, marker_size=1, marker_color='g')

    fig.suptitle(f'Frame: {frame_number} | Track Id: {track_id}'
                 f'\nAngle bw velocity vectors: {direction}\n {additional_text}')

    legends_dict = {'red': 'Bounding Box',
                    'g': 'Track'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path + f"track_id_{track_id}_frame_{frame_number}.png")
        plt.close()
    else:
        plt.show()


def plot_one_with_bounding_boxes(img, boxes):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none',
                            figsize=(12, 10))
    axs.imshow(img, cmap='gray')
    for box in boxes:
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                 edgecolor='r', fill=False,
                                 linewidth=None)
        axs.add_patch(rect)
    plt.show()


def plot_two_with_bounding_boxes(img0, boxes0, img1, boxes1, frame_number):
    fig, ax = plt.subplots(1, 2, sharex='none', sharey='none',
                           figsize=(12, 10))
    ax[0].imshow(img0, cmap='gray')
    ax[1].imshow(img1, cmap='gray')
    add_box_to_axes(ax[0], boxes0)
    add_box_to_axes(ax[1], boxes1)
    ax[0].set_title('GT')
    ax[1].set_title('OF')
    fig.suptitle(f'Frame: {frame_number}')
    plt.show()


def plot_two_with_bounding_boxes_and_rgb(img0, boxes0, img1, boxes1, rgb0, rgb1, frame_number, additional_text=None):
    fig, ax = plt.subplots(2, 2, sharex='none', sharey='none', figsize=(12, 10))
    ax[0, 0].imshow(img0, cmap='gray')
    ax[0, 1].imshow(img1, cmap='gray')
    ax[1, 0].imshow(rgb0)
    ax[1, 1].imshow(rgb1)
    add_box_to_axes(ax[0, 0], boxes0)
    add_box_to_axes(ax[0, 1], boxes1)
    add_box_to_axes(ax[1, 0], boxes0)
    add_box_to_axes(ax[1, 1], boxes1)
    ax[0, 0].set_title('GT/FG Mask')
    ax[0, 1].set_title('OF/FG Mask')
    ax[1, 0].set_title('GT/RGB')
    ax[1, 1].set_title('OF/RGB')
    fig.suptitle(f'Frame: {frame_number} | {additional_text}')
    plt.show()


def plot_for_video(gt_rgb, gt_mask, last_frame_rgb, last_frame_mask, current_frame_rgb, current_frame_mask,
                   gt_annotations, last_frame_annotation, current_frame_annotation, new_track_annotation,
                   frame_number, additional_text=None, video_mode=False, original_dims=None, save_path=None):
    fig, ax = plt.subplots(3, 2, sharex='none', sharey='none', figsize=original_dims or (12, 10))
    ax_gt_rgb, ax_gt_mask, ax_last_frame_rgb, ax_last_frame_mask, ax_current_frame_rgb, ax_current_frame_mask = \
        ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[2, 0], ax[2, 1]
    ax_gt_rgb.imshow(gt_rgb)
    ax_gt_mask.imshow(gt_mask, cmap='gray')
    ax_last_frame_rgb.imshow(last_frame_rgb)
    ax_last_frame_mask.imshow(last_frame_mask, cmap='gray')
    ax_current_frame_rgb.imshow(current_frame_rgb)
    ax_current_frame_mask.imshow(current_frame_mask, cmap='gray')

    add_box_to_axes(ax_gt_rgb, gt_annotations)
    add_box_to_axes(ax_gt_mask, gt_annotations)
    add_box_to_axes(ax_last_frame_rgb, last_frame_annotation)
    add_box_to_axes(ax_last_frame_mask, last_frame_annotation)
    add_box_to_axes(ax_current_frame_rgb, current_frame_annotation)
    add_box_to_axes(ax_current_frame_mask, current_frame_annotation)
    add_box_to_axes(ax_current_frame_rgb, new_track_annotation, 'green')
    add_box_to_axes(ax_current_frame_mask, new_track_annotation, 'green')

    ax_gt_rgb.set_title('GT/RGB')
    ax_gt_mask.set_title('GT/FG Mask')
    ax_last_frame_rgb.set_title('(T-1)/RGB')
    ax_last_frame_mask.set_title('(T-1)/FG Mask')
    ax_current_frame_rgb.set_title('(T)/RGB')
    ax_current_frame_mask.set_title('(T)/FG Mask')

    fig.suptitle(f'Frame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'green': 'New track Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()

    return fig


def add_features_to_axis(ax, features, marker_size=8, marker_shape='o', marker_color='blue'):
    ax.plot(features[:, 0], features[:, 1], marker_shape, markerfacecolor=marker_color, markeredgecolor='k',
            markersize=marker_size)


def plot_for_video_current_frame(gt_rgb, current_frame_rgb, gt_annotations, current_frame_annotation,
                                 new_track_annotation, frame_number, additional_text=None, video_mode=False,
                                 original_dims=None, save_path=None, zero_shot=False, box_annotation=None,
                                 generated_track_histories=None, gt_track_histories=None, track_marker_size=1,
                                 return_figure_only=False):
    fig, ax = plt.subplots(1, 2, sharex='none', sharey='none', figsize=original_dims or (12, 10))
    ax_gt_rgb, ax_current_frame_rgb = ax[0], ax[1]
    ax_gt_rgb.imshow(gt_rgb)
    ax_current_frame_rgb.imshow(current_frame_rgb)

    if box_annotation is None:
        add_box_to_axes(ax_gt_rgb, gt_annotations)
        add_box_to_axes(ax_current_frame_rgb, current_frame_annotation)
        add_box_to_axes(ax_current_frame_rgb, new_track_annotation, 'green')
    else:
        add_box_to_axes_with_annotation(ax_gt_rgb, gt_annotations, box_annotation[0])
        add_box_to_axes_with_annotation(ax_current_frame_rgb, current_frame_annotation, box_annotation[1])
        add_box_to_axes_with_annotation(ax_current_frame_rgb, new_track_annotation, [], 'green')

    if gt_track_histories is not None:
        add_features_to_axis(ax_gt_rgb, gt_track_histories, marker_size=track_marker_size, marker_color='g')

    if generated_track_histories is not None:
        add_features_to_axis(ax_current_frame_rgb, generated_track_histories, marker_size=track_marker_size,
                             marker_color='g')

    ax_gt_rgb.set_title('GT')
    ax_current_frame_rgb.set_title('Our Method')

    fig.suptitle(f'{"Zero Shot" if zero_shot else "One Shot"} Version\nFrame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'green': 'New track Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if return_figure_only:
        plt.close()
        return fig

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()

    return fig


def plot_for_video_image_and_box(gt_rgb, gt_annotations, frame_number, additional_text=None, video_mode=False,
                                 original_dims=None, save_path=None, box_annotation=None,
                                 gt_track_histories=None, track_marker_size=1,
                                 return_figure_only=False):
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=original_dims or (12, 10))
    ax.imshow(gt_rgb)

    if box_annotation is None:
        add_box_to_axes(ax, gt_annotations)
    else:
        add_box_to_axes_with_annotation(ax, gt_annotations, box_annotation[0])

    if gt_track_histories is not None:
        add_features_to_axis(ax, gt_track_histories, marker_size=track_marker_size, marker_color='g')

    ax.set_title('GT')

    fig.suptitle(f'Frame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if return_figure_only:
        plt.close()
        return fig

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()

    return fig


def plot_for_video_current_frame_single(gt_rgb, current_frame_rgb, gt_annotations, current_frame_annotation,
                                        new_track_annotation, frame_number, additional_text=None, video_mode=False,
                                        original_dims=None, save_path=None, zero_shot=False, box_annotation=None,
                                        generated_track_histories=None, gt_track_histories=None, track_marker_size=1,
                                        return_figure_only=False):
    fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=original_dims or (12, 10))
    # ax_gt_rgb, ax_current_frame_rgb = ax[0], ax[1]
    ax.imshow(gt_rgb)
    # ax_current_frame_rgb.imshow(current_frame_rgb)

    if box_annotation is None:
        # add_box_to_axes(ax, gt_annotations)
        add_box_to_axes(ax, current_frame_annotation)
        # add_box_to_axes(ax_current_frame_rgb, current_frame_annotation)
        # add_box_to_axes(ax_current_frame_rgb, new_track_annotation, 'green')
    else:
        # add_box_to_axes_with_annotation(ax, gt_annotations, box_annotation[0])
        add_box_to_axes_with_annotation(ax, current_frame_annotation, box_annotation[1])
        # add_box_to_axes_with_annotation(ax_current_frame_rgb, current_frame_annotation, box_annotation[1])
        # add_box_to_axes_with_annotation(ax_current_frame_rgb, new_track_annotation, [], 'green')

    # if gt_track_histories is not None:
    #     add_features_to_axis(ax, gt_track_histories, marker_size=track_marker_size, marker_color='g')

    if generated_track_histories is not None:
        # add_features_to_axis(ax_current_frame_rgb, generated_track_histories, marker_size=track_marker_size,
        #                      marker_color='g')
        add_features_to_axis(ax, generated_track_histories, marker_size=track_marker_size,
                             marker_color='g')

    ax.set_title('Results')
    # ax_current_frame_rgb.set_title('Our Method')

    fig.suptitle(f'{"Zero Shot" if zero_shot else "One Shot"} Version\nFrame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'green': 'New track Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if return_figure_only:
        plt.close()
        return fig

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()

    return fig


def plot_for_one_track(gt_rgb, gt_mask, last_frame_rgb, last_frame_mask, current_frame_rgb, current_frame_mask,
                       gt_annotations, last_frame_annotation, current_frame_annotation, new_track_annotation,
                       frame_number, track_idx, additional_text=None, video_mode=False, original_dims=None,
                       save_path=None):
    fig, ax = plt.subplots(3, 2, sharex='none', sharey='none', figsize=original_dims or (12, 10))
    ax_gt_rgb, ax_gt_mask, ax_last_frame_rgb, ax_last_frame_mask, ax_current_frame_rgb, ax_current_frame_mask = \
        ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[2, 0], ax[2, 1]
    ax_gt_rgb.imshow(gt_rgb)
    ax_gt_mask.imshow(gt_mask, cmap='gray')
    ax_last_frame_rgb.imshow(last_frame_rgb)
    ax_last_frame_mask.imshow(last_frame_mask, cmap='gray')
    ax_current_frame_rgb.imshow(current_frame_rgb)
    ax_current_frame_mask.imshow(current_frame_mask, cmap='gray')

    add_one_box_to_axis(ax_gt_rgb, box=gt_annotations[track_idx], color='r')
    add_one_box_to_axis(ax_gt_mask, box=gt_annotations[track_idx], color='r')
    add_one_box_to_axis(ax_last_frame_rgb, box=last_frame_annotation[track_idx], color='r')
    add_one_box_to_axis(ax_last_frame_mask, box=last_frame_annotation[track_idx], color='r')
    add_one_box_to_axis(ax_current_frame_rgb, box=current_frame_annotation[track_idx], color='r')
    add_one_box_to_axis(ax_current_frame_mask, box=current_frame_annotation[track_idx], color='r')
    if new_track_annotation.size != 0:
        add_one_box_to_axis(ax_current_frame_rgb, box=new_track_annotation[track_idx], color='r')
        add_one_box_to_axis(ax_current_frame_mask, box=new_track_annotation[track_idx], color='r')

    ax_gt_rgb.set_title('GT/RGB')
    ax_gt_mask.set_title('GT/FG Mask')
    ax_last_frame_rgb.set_title('(T-1)/RGB')
    ax_last_frame_mask.set_title('(T-1)/FG Mask')
    ax_current_frame_rgb.set_title('(T)/RGB')
    ax_current_frame_mask.set_title('(T)/FG Mask')

    fig.suptitle(f'Frame: {frame_number}\n{additional_text}')

    legends_dict = {'r': 'Bounding Box',
                    'green': 'New track Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if video_mode:
        plt.close()
    else:
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path + f"frame_{frame_number}.png")
            plt.close()
        else:
            plt.show()

    return fig


def add_box_to_axes(ax, boxes, edge_color='r'):
    for box in boxes:
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                 edgecolor=edge_color, fill=False,
                                 linewidth=None)
        ax.add_patch(rect)


def add_box_to_axes_with_annotation(ax, boxes, annotation, edge_color='r'):
    for a, box in zip(annotation, boxes):
        if box is None:
            continue
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                 edgecolor=edge_color, fill=False,
                                 linewidth=None)
        ax.add_patch(rect)

        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0

        ax.annotate(a, (cx, cy), color='w', weight='bold', fontsize=6, ha='center', va='center')


def plot_processing_steps(xy_cloud, shifted_xy_cloud, xy_box, shifted_xy_box,
                          final_cloud, xy_cloud_current_frame, frame_number, track_id,
                          selected_past, selected_current,
                          true_cloud_key_point=None, shifted_cloud_key_point=None,
                          overlap_threshold=None, shift_corrected_cloud_key_point=None,
                          key_point_criteria=None, shift_correction=None,
                          line_width=None, save_path=None):
    fig, ax = plt.subplots(2, 2, sharex='none', sharey='none', figsize=(14, 12))
    ax1, ax2, ax3, ax4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    # cloud1
    ax1.plot(xy_cloud[:, 0], xy_cloud[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8)
    ax1.plot(selected_past[:, 0], selected_past[:, 1], 'o', markerfacecolor='silver',
             markeredgecolor='k', markersize=8)
    # ax1.plot(true_cloud_key_point[0], true_cloud_key_point[1], '*', markerfacecolor='silver', markeredgecolor='k',
    #          markersize=9)
    rect1 = patches.Rectangle(xy=(xy_box[0], xy_box[1]), width=xy_box[2] - xy_box[0],
                              height=xy_box[3] - xy_box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    # cloud2
    ax2.plot(xy_cloud_current_frame[:, 0], xy_cloud_current_frame[:, 1], 'o', markerfacecolor='magenta',
             markeredgecolor='k', markersize=8)
    ax2.plot(selected_current[:, 0], selected_current[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=8)
    # ax2.plot(shifted_cloud_key_point[0], shifted_cloud_key_point[1], '*', markerfacecolor='yellow',
    #          markeredgecolor='k', markersize=9)
    rect2 = patches.Rectangle(xy=(shifted_xy_box[0], shifted_xy_box[1]), width=shifted_xy_box[2] - shifted_xy_box[0],
                              height=shifted_xy_box[3] - shifted_xy_box[1], fill=False,
                              linewidth=line_width, edgecolor='teal')

    # cloud1 + cloud2
    # cloud1
    ax3.plot(xy_cloud_current_frame[:, 0], xy_cloud_current_frame[:, 1], '*', markerfacecolor='blue',
             markeredgecolor='k', markersize=8)
    rect3 = patches.Rectangle(xy=(xy_box[0], xy_box[1]), width=xy_box[2] - xy_box[0],
                              height=xy_box[3] - xy_box[1], fill=False,
                              linewidth=line_width, edgecolor='r')
    # cloud2
    ax3.plot(shifted_xy_cloud[:, 0], shifted_xy_cloud[:, 1], 'o', markerfacecolor='magenta',
             markeredgecolor='k', markersize=8)
    rect3_shifted = patches.Rectangle(xy=(shifted_xy_box[0], shifted_xy_box[1]),
                                      width=shifted_xy_box[2] - shifted_xy_box[0],
                                      height=shifted_xy_box[3] - shifted_xy_box[1], fill=False,
                                      linewidth=line_width, edgecolor='teal')
    ax3.plot(selected_past[:, 0], selected_past[:, 1], 'o', markerfacecolor='silver',
             markeredgecolor='k', markersize=8)
    ax3.plot(selected_current[:, 0], selected_current[:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='k', markersize=8)
    # ax3.plot(true_cloud_key_point[0], true_cloud_key_point[1], '*', markerfacecolor='silver', markeredgecolor='k',
    #          markersize=9)
    # ax3.plot(shifted_cloud_key_point[0], shifted_cloud_key_point[1], '*', markerfacecolor='yellow',
    #          markeredgecolor='k', markersize=9)

    # cloud1 + cloud2 - final selected cloud
    # cloud1
    ax4.plot(final_cloud[:, 0], final_cloud[:, 1], 'o', markerfacecolor='blue', markeredgecolor='k',
             markersize=8)
    # rect4 = patches.Rectangle(xy=(xy_box[0], xy_box[1]), width=xy_box[2] - xy_box[0],
    #                           height=xy_box[3] - xy_box[1], fill=False,
    #                           linewidth=line_width, edgecolor='r')
    # # cloud2
    # ax4.plot(shifted_xy_cloud[:, 0], shifted_xy_cloud[:, 1], 'o', markerfacecolor='magenta', markeredgecolor='k',
    #          markersize=8)
    rect4_shifted = patches.Rectangle(xy=(shifted_xy_box[0], shifted_xy_box[1]),
                                      width=shifted_xy_box[2] - shifted_xy_box[0],
                                      height=shifted_xy_box[3] - shifted_xy_box[1], fill=False,
                                      linewidth=line_width, edgecolor='teal')
    # ax4.plot(true_cloud_key_point[0], true_cloud_key_point[1], '*', markerfacecolor='silver', markeredgecolor='k',
    #          markersize=9)
    # ax4.plot(shift_corrected_cloud_key_point[0], shift_corrected_cloud_key_point[1], '*', markerfacecolor='plum',
    #          markeredgecolor='k', markersize=9)

    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    ax3.add_patch(rect3)
    # ax4.add_patch(rect4)
    ax3.add_patch(rect3_shifted)
    ax4.add_patch(rect4_shifted)

    # original_error = np.linalg.norm(true_cloud_key_point - shifted_cloud_key_point, 2)
    # optimized_error = np.linalg.norm(true_cloud_key_point - shift_corrected_cloud_key_point, 2)

    ax1.set_title('XY Past')
    ax2.set_title('XY Current')
    ax3.set_title(f'XY Past Shifted + XY Current')
    ax4.set_title(f'Final Selected XY')

    legends_dict = {'blue': 'Points at (T-1)',
                    'magenta': 'Points at T',
                    'r': '(T-1) Bounding Box',
                    'silver': 'Selected XY',
                    # 'plum': f'Shift Corrected {key_point_criteria}',
                    'yellow': 'Selected XY Current',
                    'teal': '(T-1) OF Shifted Bounding Box'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)
    fig.suptitle(f'Frame: {frame_number} | Track Id: {track_id}')  # \nShift Correction: {shift_correction}\n'
    # f'Overlap Threshold: {overlap_threshold}')

    if save_path is None:
        plt.show()
    else:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + f'fig_frame_{frame_number}_track_{track_id}.png')
        plt.close()