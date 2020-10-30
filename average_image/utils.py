import os
from itertools import cycle
from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
from matplotlib import patches, cm
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir

from average_image.bbox_utils import annotations_to_dataframe
from average_image.constants import SDDVideoClasses, SDDVideoDatasets, OBJECT_CLASS_COLOR_MAPPING, ObjectClasses


def show_img(img):
    plt.imshow(img)
    plt.show()


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


def point_in_rect(point, rect):
    x1, y1, x2, y2 = rect
    x, y = point
    if x1 < x < x2:
        if y1 < y < y2:
            return True
    return False


def evaluate_clustering_per_frame_old_(frame, frame_dict):  # use this inside above
    result_dict = {}
    matched_cluster_centers = []
    matched_annotation = []

    total_cluster_centers = frame_dict['cluster_centers'].shape[0]
    total_annot = len(frame_dict['gt_bbox'])

    for annotation in frame_dict['gt_bbox']:
        if check_presence(annotation, matched_annotation):
            continue
        for cluster_center in frame_dict['cluster_centers']:
            if check_presence(cluster_center, matched_cluster_centers):
                continue
            point = (cluster_center[0], cluster_center[1])
            rect = annotation
            cluster_center_in_box = point_in_rect(point, rect)
            if cluster_center_in_box:
                if not check_presence(cluster_center, matched_cluster_centers):
                    matched_cluster_centers.append(cluster_center)
                if not check_presence(annotation, matched_annotation):
                    matched_annotation.append(annotation)
                continue

    result_dict.update({frame: {'gt_annotation_matched': len(matched_annotation),
                                'cluster_center_in_bbox_count': len(matched_cluster_centers),
                                'total_cluster_centers': total_cluster_centers,
                                'total_annotations': total_annot}})
    return result_dict


def evaluate_clustering_non_cc(frame, frame_dict, one_to_one=False):  # use this for clustering that don't have c_center
    result_dict = {}
    matched_cluster_centers = []
    matched_annotation = []

    total_cluster_centers = len(frame_dict['cluster_centers'])
    total_annot = len(frame_dict['gt_bbox'])

    for a_i, annotation in enumerate(frame_dict['gt_bbox']):
        if check_presence(annotation, matched_annotation):
            continue
        for c_i, cluster_center in enumerate(frame_dict['cluster_centers']):
            if check_presence(cluster_center, matched_cluster_centers):
                continue
            point = (cluster_center[0], cluster_center[1])
            rect = annotation
            cluster_center_in_box = point_in_rect(point, rect)
            if cluster_center_in_box:
                if one_to_one:
                    if not check_presence(annotation, matched_annotation):
                        matched_annotation.append(annotation)
                        if not check_presence(cluster_center, matched_cluster_centers):
                            matched_cluster_centers.append(cluster_center)
                else:
                    if not check_presence(cluster_center, matched_cluster_centers):
                        matched_cluster_centers.append(cluster_center)
                    if not check_presence(annotation, matched_annotation):
                        matched_annotation.append(annotation)
                continue

    result_dict.update({frame: {'gt_annotation_matched': len(matched_annotation),
                                'cluster_center_in_bbox_count': len(matched_cluster_centers),
                                'total_cluster_centers': total_cluster_centers,
                                'total_annotations': total_annot}})
    return result_dict


def evaluate_clustering_per_frame(frame, frame_dict, one_to_one=False):  # use this inside above
    result_dict = {}
    matched_cluster_centers = []
    matched_annotation = []

    total_cluster_centers = frame_dict['cluster_centers'].shape[0]
    total_annot = len(frame_dict['gt_bbox'])

    for a_i, annotation in enumerate(frame_dict['gt_bbox']):
        if check_presence(annotation, matched_annotation):
            continue
        for c_i, cluster_center in enumerate(frame_dict['cluster_centers']):
            if check_presence(cluster_center, matched_cluster_centers):
                continue
            point = (cluster_center[0], cluster_center[1])
            rect = annotation
            cluster_center_in_box = point_in_rect(point, rect)
            if cluster_center_in_box:
                if one_to_one:
                    if not check_presence(annotation, matched_annotation):
                        matched_annotation.append(annotation)
                        if not check_presence(cluster_center, matched_cluster_centers):
                            matched_cluster_centers.append(cluster_center)
                else:
                    if not check_presence(cluster_center, matched_cluster_centers):
                        matched_cluster_centers.append(cluster_center)
                    if not check_presence(annotation, matched_annotation):
                        matched_annotation.append(annotation)
                continue

    result_dict.update({frame: {'gt_annotation_matched': len(matched_annotation),
                                'cluster_center_in_bbox_count': len(matched_cluster_centers),
                                'total_cluster_centers': total_cluster_centers,
                                'total_annotations': total_annot}})
    return result_dict


def evaluate_clustering(gt_bbox_cluster_center_dict_):
    result_dict = {}
    for frame, frame_dict in gt_bbox_cluster_center_dict_.items():
        result_d = evaluate_clustering_per_frame(frame=frame, frame_dict=frame_dict)
        result_dict.update(result_d)
    return result_dict


# To get 4d cluster centers into 2d, but doesnt seem logical since two spaces are very different
def pca_cluster_center(cluster_centers):
    pca = PCA(n_components=2)
    cc = pca.fit_transform(cluster_centers)
    return cc


def normalize(x):
    max_ = np.max(x)
    min_ = np.min(x)
    return (x - min_) / (max_ - min_), max_, min_


def rescale_featues(x, max_range, min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range - min_range) / (max_val - min_val) * (x - max_val) + max_range


def denormalize(x, max_, min_):
    return x * (max_ - min_) + min_


def compare_precision(pr_results_1, pr_results_2, avg_img, lab1, lab2):
    method = 'Average Image' if avg_img else 'MOG2'
    width = 0.35
    label_frame = []
    precision_1 = []
    precision_2 = []
    for (frame, result), (frame_, result_) in zip(pr_results_1.items(), pr_results_2.items()):
        label_frame.append(frame)
        precision_1.append(result['precision'])
        precision_2.append(result_['precision'])

    x = np.arange(len(label_frame))

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, precision_1, width, label=lab1)
    rects2 = ax.bar(x + width / 2, precision_2, width, label=lab2)
    ax.set_title(f'Precision - {method}')
    ax.set_xticks(x)
    ax.set_xticklabels(label_frame)
    ax.legend()

    # fig.tight_layout()

    plt.show()


def compare_recall(pr_results_1, pr_results_2, avg_img, lab1, lab2):
    method = 'Average Image' if avg_img else 'MOG2'
    width = 0.35
    label_frame = []
    recall_1 = []
    recall_2 = []
    for (frame, result), (frame_, result_) in zip(pr_results_1.items(), pr_results_2.items()):
        label_frame.append(frame)
        recall_1.append(result['recall'])
        recall_2.append(result_['recall'])

    x = np.arange(len(label_frame))

    fig, ax = plt.subplots()
    ax.set_title(f'Recall - {method}')
    rects1 = ax.bar(x - width / 2, recall_1, width, label=lab1)
    rects2 = ax.bar(x + width / 2, recall_2, width, label=lab2)
    ax.set_xticks(x)
    ax.set_xticklabels(label_frame)
    ax.legend()

    # fig.tight_layout()

    plt.show()


# Only for 1 now
def object_of_interest_mask(img, annotation):
    bbox = annotation[0]
    mask = np.zeros(shape=(img.shape[1], img.shape[2]))
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    return mask


def features_from_crop(feature_img, features, annotation):
    bbox = annotation[0]
    mask = np.zeros_like(feature_img)
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = feature_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return NotImplemented


def plot_precision_recall(pr_results_1, pr_results_2, avg_img, lab1, lab2):
    compare_precision(pr_results_1, pr_results_2, avg_img, lab1, lab2)
    compare_recall(pr_results_1, pr_results_2, avg_img, lab1, lab2)


def precision_recall_one_sequence(results, average_image):
    method = 'Average Image' if average_image else 'MOG2'
    width = 0.35
    label_frame = []
    recall_1 = []
    recall_2 = []
    for frame, result in results.items():
        label_frame.append(frame)
        recall_1.append(results[frame]['precision'])
        recall_2.append(results[frame]['recall'])

    x = np.arange(len(label_frame))

    fig, ax = plt.subplots()
    ax.set_title(f'Precision - Recall -> {method}')
    rects1 = ax.bar(x - width / 2, recall_1, width, label="Precision")
    rects2 = ax.bar(x + width / 2, recall_2, width, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(label_frame)
    ax.legend()

    # fig.tight_layout()

    plt.show()


def cal_centers(b_box):
    x_min = b_box[:, 0]
    y_min = b_box[:, 1]
    x_max = b_box[:, 2]
    y_max = b_box[:, 3]
    x_mid = (x_min + (x_max - x_min) / 2.).astype('int')
    y_mid = (y_min + (y_max - y_min) / 2.).astype('int')

    return np.vstack((x_mid, y_mid)).T


def is_inside_bbox(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min < x < x_max and y_min < y < y_max


def plot_bars_if_inside_bbox(data):
    n = len(data)
    ind = np.arange(n)
    width = 0.25

    true_count = []
    false_count = []

    for d in data:
        true_count.append(d.count(True))
        false_count.append(d.count(False))

    plt.bar(ind, true_count, width, color='r')
    plt.bar(ind + width, false_count, width, color='b')
    plt.ylabel('Centers Count')
    plt.title('Times center falls inside bounding box')
    plt.xticks(ind, ('GT Bbox Center', 'OF (OpenCV)', 'Pred OF', 'Pred GT'))
    plt.legend(labels=['Inside', 'Outside'])
    plt.show()


def renormalize_optical_flow(features, options):
    of_0 = denormalize(features[..., 0], options['f_max_0'], options['f_min_0'])
    of_1 = denormalize(features[..., 1], options['f_max_1'], options['f_min_1'])
    features[..., 0] = of_0
    features[..., 1] = of_1
    return features


def plot_with_bbox(img, box, linewidth=None):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none',
                            figsize=(12, 10))
    axs.imshow(img)
    # for box in bbox:
    rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                             edgecolor=OBJECT_CLASS_COLOR_MAPPING[ObjectClasses.PEDESTRIAN], fill=False,
                             linewidth=linewidth)
    axs.add_patch(rect)
    plt.show()


def add_bbox_to_image(ax, bbox, linewidth=None):
    for box in bbox:
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                 edgecolor=OBJECT_CLASS_COLOR_MAPPING[ObjectClasses.PEDESTRIAN], fill=False,
                                 linewidth=linewidth)
        ax.add_patch(rect)


def add_centers_to_image(ax, center):
    center_x = center[:, 0]
    center_y = center[:, 1]
    ax.plot(center_x, center_y, 'o', markerfacecolor='g', markersize=2, markeredgecolor='k')


def plot_one_with_center(img, ax, bbox, center, lw=None):
    ax.imshow(img)
    add_bbox_to_image(ax, bbox, lw)
    add_centers_to_image(ax, center)


def plot_with_centers(img_tensor, bbox, centers, nrows=2, ncols=2):
    fig, axs = plt.subplots(nrows, ncols, sharex='none', sharey='none',
                            figsize=(12, 10))
    img_idx = [i for i in range(nrows * ncols)]
    k = 0
    for r in range(nrows):
        for c in range(ncols):
            im = (img_tensor[img_idx[k]].permute(1, 2, 0) * 255.0).numpy().astype(np.uint8)
            plot_one_with_center(im, axs[r, c], bbox[img_idx[k]], centers[img_idx[k]], lw=None)
            k += 1

    plt.show()


def plot_with_one_bbox(img, box):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none',
                            figsize=(12, 10))
    axs.imshow(img)
    rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                             edgecolor=OBJECT_CLASS_COLOR_MAPPING[ObjectClasses.PEDESTRIAN], fill=False,
                             linewidth=None)
    axs.add_patch(rect)
    plt.show()


def renormalize_any_cluster(cluster_centers, options):
    out = np.zeros_like(cluster_centers)
    cc_0 = denormalize(cluster_centers[..., 0], options['max_1'], options['min_1'])
    cc_1 = denormalize(cluster_centers[..., 1], options['max_0'], options['min_0'])
    # cluster_centers[..., 0] = cc_0
    # cluster_centers[..., 1] = cc_1
    out[..., 0] = cc_0
    out[..., 1] = cc_1
    return out


def plot_extracted_features(frame_object_list, img=None):
    object_points = []
    for obj in frame_object_list:
        renormalized_ = renormalize_any_cluster(obj.features[:, :2], obj.normalize_params).astype(np.int)
        object_points.append([renormalized_[:, 0], renormalized_[:, 1]])

    if img is not None:
        plt.imshow(img)

    colors = cm.rainbow(np.linspace(0, 1, 20))
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for obj_, col in zip(object_points, colors):
        plt.plot(obj_[0], obj_[1], 'o', markerfacecolor=col, markersize=1, markeredgecolor='k', markeredgewidth=0.2)
        plt.plot(int(obj_[0].mean()), int(obj_[1].mean()), '*', markerfacecolor='r', markersize=5, markeredgecolor='k',
                 markeredgewidth=0.2)

    plt.show()


def plot_points_predicted_and_true(predicted_points, true_points, actual_points=None):
    plt.plot(predicted_points[..., 0], predicted_points[..., 1], 'o', markerfacecolor='orange', markersize=5,
             markeredgecolor='k',
             markeredgewidth=0.2, label='Predicted Position')
    plt.plot((predicted_points[..., 0].mean()), (predicted_points[..., 1].mean()), '*', markerfacecolor='orange',
             markersize=10,
             markeredgecolor='k', markeredgewidth=0.2)

    plt.plot(true_points[..., 0], true_points[..., 1], '^', markerfacecolor='g', markersize=5, markeredgecolor='k',
             markeredgewidth=0.2, label='True Position')
    plt.plot((true_points[..., 0].mean()), (true_points[..., 1].mean()), '*', markerfacecolor='g',
             markersize=10,
             markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        plt.plot(actual_points[..., 0], actual_points[..., 1], 'p', markerfacecolor='blue', markersize=5,
                 markeredgecolor='k',
                 markeredgewidth=0.2, label='Position prior shift')
        plt.plot((actual_points[..., 0].mean()), (actual_points[..., 1].mean()), '*', markerfacecolor='blue',
                 markersize=10,
                 markeredgecolor='k', markeredgewidth=0.2)

    plt.legend(loc="upper left")
    plt.show()


def plot_points_predicted_and_true_center_only(predicted_points, true_points, actual_points=None, img=None):
    if img is not None:
        plt.imshow(img)
    plt.plot((predicted_points[..., 0]), (predicted_points[..., 1]), '*', markerfacecolor='orange',
             markersize=3,
             markeredgecolor='k', markeredgewidth=0.2, label='Predicted Center')

    # plt.plot(true_points[..., 0], true_points[..., 1], '^', markerfacecolor='g', markersize=1, markeredgecolor='k',
    #          markeredgewidth=0.2, label='True Position')
    plt.plot((true_points[..., 0].mean()), (true_points[..., 1].mean()), '*', markerfacecolor='g',
             markersize=3,
             markeredgecolor='k', markeredgewidth=0.2, label='True Position')

    if actual_points is not None:
        plt.plot((actual_points[..., 0].mean()), (actual_points[..., 1].mean()), '*', markerfacecolor='blue',
                 markersize=3,
                 markeredgecolor='k', markeredgewidth=0.2, label='Center prior shift')

    plt.legend(loc="upper left")
    plt.show()


def trajectory_length(trajectory):
    length = len(trajectory)
    trajectory_len = 0
    for i in range(length - 1):
        trajectory_len += np.linalg.norm(trajectory[i] - trajectory[i + 1], 2)
    return trajectory_len


def plot_points_predicted_and_true_center_only_rnn(predicted_points, true_points, actual_points=None, imgs=None):
    fig, axs = plt.subplots(2, 3, sharex='none', sharey='none',
                            figsize=(12, 9))
    if imgs is not None:
        axs[0, 0].imshow(imgs[0])
        axs[0, 1].imshow(imgs[1])
        axs[0, 2].imshow(imgs[2])
        axs[1, 0].imshow(imgs[3])
        axs[1, 1].imshow(imgs[4])
        axs[1, 2].imshow(imgs[0])

        axs[0, 0].set_title('T = 0')
        axs[0, 1].set_title('T = 1')
        axs[0, 2].set_title('T = 2')
        axs[1, 0].set_title('T = 3')
        axs[1, 1].set_title('T = 4')
        axs[1, 2].set_title('Image T = 0')

        m_size = 3
    else:
        m_size = 9

    axs[0, 0].plot((predicted_points[0][..., 0]), (predicted_points[0][..., 1]), '*', markerfacecolor='orange',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    axs[0, 0].plot((true_points[0][..., 0].mean()), (true_points[0][..., 1].mean()), '*', markerfacecolor='g',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs[0, 0].plot((actual_points[0][..., 0].mean()), (actual_points[0][..., 1].mean()), '*',
                       markerfacecolor='blue',
                       markersize=m_size,
                       markeredgecolor='k', markeredgewidth=0.2)

    axs[0, 1].plot((predicted_points[1][..., 0]), (predicted_points[1][..., 1]), '*', markerfacecolor='orange',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    axs[0, 1].plot((true_points[1][..., 0].mean()), (true_points[1][..., 1].mean()), '*', markerfacecolor='g',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs[0, 1].plot((actual_points[1][..., 0].mean()), (actual_points[1][..., 1].mean()), '*',
                       markerfacecolor='blue',
                       markersize=m_size,
                       markeredgecolor='k', markeredgewidth=0.2)

    axs[0, 2].plot((predicted_points[2][..., 0]), (predicted_points[2][..., 1]), '*', markerfacecolor='orange',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    axs[0, 2].plot((true_points[2][..., 0].mean()), (true_points[2][..., 1].mean()), '*', markerfacecolor='g',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs[0, 2].plot((actual_points[2][..., 0].mean()), (actual_points[2][..., 1].mean()), '*',
                       markerfacecolor='blue',
                       markersize=m_size,
                       markeredgecolor='k', markeredgewidth=0.2)

    axs[1, 0].plot((predicted_points[3][..., 0]), (predicted_points[3][..., 1]), '*', markerfacecolor='orange',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    axs[1, 0].plot((true_points[3][..., 0].mean()), (true_points[3][..., 1].mean()), '*', markerfacecolor='g',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs[1, 0].plot((actual_points[3][..., 0].mean()), (actual_points[3][..., 1].mean()), '*',
                       markerfacecolor='blue',
                       markersize=m_size,
                       markeredgecolor='k', markeredgewidth=0.2)

    axs[1, 1].plot((predicted_points[4][..., 0]), (predicted_points[4][..., 1]), '*', markerfacecolor='orange',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    axs[1, 1].plot((true_points[4][..., 0].mean()), (true_points[4][..., 1].mean()), '*', markerfacecolor='g',
                   markersize=m_size,
                   markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs[1, 1].plot((actual_points[4][..., 0].mean()), (actual_points[4][..., 1].mean()), '*',
                       markerfacecolor='blue',
                       markersize=m_size,
                       markeredgecolor='k', markeredgewidth=0.2)

    legends_dict = {'orange': 'Predicted Center',
                    'g': 'Optical Flow Center',
                    'blue': 'Center prior shift'}

    patches = [mpatches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=patches, loc=2)

    plt.show()


def plot_trajectory_rnn(predicted_points, true_points, actual_points=None, imgs=None, gt=False):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none',
                            figsize=(12, 9))
    if imgs is not None:
        axs.imshow(imgs[0])

        axs.set_title('Image')

        m_size = 3
    else:
        m_size = 9

    if gt:
        gt_text = 'GT Trajectory'
    else:
        gt_text = 'Optical Flow Trajectory'

    # axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), '*', markerfacecolor='orange',
    #          markersize=m_size,
    #          markeredgecolor='k', markeredgewidth=0.2)
    #
    # axs.plot((true_points[..., 0]), (true_points[..., 1]), '*', markerfacecolor='g',
    #          markersize=m_size,
    #          markeredgecolor='k', markeredgewidth=0.2)
    #
    # if actual_points is not None:
    #     axs.plot((actual_points[0][..., 0]), (actual_points[0][..., 1]), '*',
    #              markerfacecolor='blue',
    #              markersize=m_size,
    #              markeredgecolor='k', markeredgewidth=0.2)

    axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), color='orange')

    axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), '*', markerfacecolor='red',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    axs.plot((true_points[..., 0]), (true_points[..., 1]), color='g')

    axs.plot((true_points[..., 0]), (true_points[..., 1]), '*', markerfacecolor='aqua',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs.plot((actual_points[0][..., 0]), (actual_points[0][..., 1]), '*',
                 markerfacecolor='blue',
                 markersize=m_size,
                 markeredgecolor='k', markeredgewidth=0.2)

    legends_dict = {'orange': 'Predicted Trajectory',
                    'g': gt_text,
                    'blue': 'Center at T=0'}

    patches = [mpatches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=patches, loc=2)

    plt.show()


def plot_trajectory_rnn_tb(predicted_points, true_points, actual_points=None, imgs=None, gt=False):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none',
                            figsize=(12, 9))
    if imgs is not None:
        axs.imshow(imgs[0])

        axs.set_title('Image')

        m_size = 3
    else:
        m_size = 9

    if gt:
        gt_text = 'GT Trajectory'
    else:
        gt_text = 'Optical Flow Trajectory'

    axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), color='orange')

    axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), '*', markerfacecolor='red',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    axs.plot((true_points[..., 0]), (true_points[..., 1]), color='g')

    axs.plot((true_points[..., 0]), (true_points[..., 1]), '*', markerfacecolor='aqua',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs.plot((actual_points[0][..., 0]), (actual_points[0][..., 1]), '*',
                 markerfacecolor='blue',
                 markersize=m_size,
                 markeredgecolor='k', markeredgewidth=0.2)

    legends_dict = {'orange': 'Predicted Trajectory',
                    'g': gt_text,
                    'blue': 'Center at T=0'}

    patches = [mpatches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=patches, loc=2)

    return fig

def plot_trajectory_rnn_compare(predicted_points, predicted_points_gt, true_points, true_points_of, of_l2, gt_l2,
                                actual_points=None, imgs=None, gt=False, m_ratio=None, save_path=None, show=False):
    fig, axs = plt.subplots(1, 1, sharex='none', sharey='none',
                            figsize=(12, 9))
    if imgs is not None:
        axs.imshow(imgs[0])

        axs.set_title('Image')

        m_size = 3
    else:
        m_size = 9

    # if gt:
    #     gt_text = 'GT Trajectory'
    # else:
    #     gt_text = 'Optical Flow Trajectory'

    axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), color='orange')

    axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), '*', markerfacecolor='red',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    axs.plot((predicted_points_gt[..., 0]), (predicted_points_gt[..., 1]), color='magenta')

    axs.plot((predicted_points_gt[..., 0]), (predicted_points_gt[..., 1]), '*', markerfacecolor='red',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    axs.plot((true_points[..., 0]), (true_points[..., 1]), color='g')

    axs.plot((true_points[..., 0]), (true_points[..., 1]), '*', markerfacecolor='aqua',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    axs.plot((true_points_of[..., 0]), (true_points_of[..., 1]), color='navy')

    axs.plot((true_points_of[..., 0]), (true_points_of[..., 1]), '*', markerfacecolor='aqua',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs.plot((actual_points[0][..., 0]), (actual_points[0][..., 1]), '*',
                 markerfacecolor='blue',
                 markersize=m_size,
                 markeredgecolor='k', markeredgewidth=0.2)

    legends_dict = {'orange': 'OF Predicted Trajectory',
                    'navy': 'OF Trajectory',
                    'magenta': 'GT Predicted Trajectory',
                    'g': 'GT Trajectory',
                    'blue': 'Center at T=0'}

    ade = compute_ade(predicted_points, true_points_of)
    fde = compute_fde(predicted_points, true_points_of)

    ade_gt = compute_ade(predicted_points_gt, true_points)
    fde_gt = compute_fde(predicted_points_gt, true_points)

    fig.suptitle(f'Optical Flow L2 bw points: {of_l2}'
                 f'\nGT L2 bw points: {gt_l2}\n'
                 f'OF - ADE: {ade} |  FDE: {fde}\n'
                 f'GT - ADE: {ade_gt} |  FDE: {fde_gt}\n'
                 f'[m]OF - ADE: {ade * m_ratio} |  FDE: {fde * m_ratio}\n'
                 f'[m]GT - ADE: {ade_gt * m_ratio} |  FDE: {fde_gt * m_ratio}')

    patches = [mpatches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=patches, loc=2)

    if save_path is not None:
        fig.savefig(f"{save_path}.png")

    if show:
        plt.show()


def plot_trajectory_rnn_compare_side_by_side(predicted_points, predicted_points_gt, true_points, true_points_of, of_l2,
                                             gt_l2,
                                             actual_points=None, imgs=None, gt=False, m_ratio=None, save_path=None,
                                             show=False):
    fig, (axs, axs_gt) = plt.subplots(1, 2, sharex='none', sharey='none',
                                      figsize=(12, 9))
    if imgs is not None:
        axs.imshow(imgs[0])

        axs.set_title('OF Based')

        axs_gt.imshow(imgs[0])

        axs_gt.set_title('GT Based')

        m_size = 3
    else:
        m_size = 9

    # if gt:
    #     gt_text = 'GT Trajectory'
    # else:
    #     gt_text = 'Optical Flow Trajectory'

    axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), color='orange')

    axs.plot((predicted_points[..., 0]), (predicted_points[..., 1]), '*', markerfacecolor='red',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    axs_gt.plot((predicted_points_gt[..., 0]), (predicted_points_gt[..., 1]), color='magenta')

    axs_gt.plot((predicted_points_gt[..., 0]), (predicted_points_gt[..., 1]), '*', markerfacecolor='red',
                markersize=m_size,
                markeredgecolor='k', markeredgewidth=0.2)

    axs_gt.plot((true_points[..., 0]), (true_points[..., 1]), color='g')

    axs_gt.plot((true_points[..., 0]), (true_points[..., 1]), '*', markerfacecolor='aqua',
                markersize=m_size,
                markeredgecolor='k', markeredgewidth=0.2)

    axs.plot((true_points_of[..., 0]), (true_points_of[..., 1]), color='navy')

    axs.plot((true_points_of[..., 0]), (true_points_of[..., 1]), '*', markerfacecolor='aqua',
             markersize=m_size,
             markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        axs.plot((actual_points[0][..., 0]), (actual_points[0][..., 1]), '*',
                 markerfacecolor='blue',
                 markersize=m_size,
                 markeredgecolor='k', markeredgewidth=0.2)
        axs_gt.plot((actual_points[0][..., 0]), (actual_points[0][..., 1]), '*',
                    markerfacecolor='blue',
                    markersize=m_size,
                    markeredgecolor='k', markeredgewidth=0.2)

    legends_dict = {'orange': 'OF Predicted Trajectory',
                    'navy': 'OF Trajectory',
                    'magenta': 'GT Predicted Trajectory',
                    'g': 'GT Trajectory',
                    'blue': 'Center at T=0'}

    ade = compute_ade(predicted_points, true_points_of)
    fde = compute_fde(predicted_points, true_points_of)

    ade_gt = compute_ade(predicted_points_gt, true_points)
    fde_gt = compute_fde(predicted_points_gt, true_points)

    fig.suptitle(f'Optical Flow L2 bw points: {of_l2}'
                 f'\nGT L2 bw points: {gt_l2}\n'
                 f'OF - ADE: {ade} |  FDE: {fde}\n'
                 f'GT - ADE: {ade_gt} |  FDE: {fde_gt}\n'
                 f'[m]OF - ADE: {ade * m_ratio} |  FDE: {fde * m_ratio}\n'
                 f'[m]GT - ADE: {ade_gt * m_ratio} |  FDE: {fde_gt * m_ratio}')

    patches = [mpatches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=patches, loc=2)

    if save_path is not None:
        fig.savefig(f"{save_path}.png")

    if show:
        plt.show()


def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[-1] - gt_traj[-1], axis=-1)
    # final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def plot_and_compare_points_predicted_and_true(predicted_points, true_points,
                                               predicted_points_1, true_points_1, actual_points=None,
                                               actual_points_1=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(predicted_points[..., 0], predicted_points[..., 1], 'o', markerfacecolor='orange', markersize=5,
             markeredgecolor='k',
             markeredgewidth=0.2, label='Predicted Position')
    ax1.plot((predicted_points[..., 0].mean()), (predicted_points[..., 1].mean()), '*', markerfacecolor='orange',
             markersize=10,
             markeredgecolor='k', markeredgewidth=0.2)

    ax1.plot(true_points[..., 0], true_points[..., 1], '^', markerfacecolor='g', markersize=5, markeredgecolor='k',
             markeredgewidth=0.2, label='True Position')
    ax1.plot((true_points[..., 0].mean()), (true_points[..., 1].mean()), '*', markerfacecolor='g',
             markersize=10,
             markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        ax1.plot(actual_points[..., 0], actual_points[..., 1], 'p', markerfacecolor='blue', markersize=5,
                 markeredgecolor='k',
                 markeredgewidth=0.2, label='Actual Position')
        ax1.plot((actual_points[..., 0].mean()), (actual_points[..., 1].mean()), '*', markerfacecolor='blue',
                 markersize=10,
                 markeredgecolor='k', markeredgewidth=0.2)

    ax2.plot(predicted_points_1[..., 0], predicted_points_1[..., 1], 'o', markerfacecolor='orange', markersize=5,
             markeredgecolor='k',
             markeredgewidth=0.2)
    ax2.plot((predicted_points_1[..., 0].mean()), (predicted_points_1[..., 1].mean()), '*', markerfacecolor='orange',
             markersize=10,
             markeredgecolor='k', markeredgewidth=0.2)

    ax2.plot(true_points_1[..., 0], true_points_1[..., 1], '^', markerfacecolor='g', markersize=5, markeredgecolor='k',
             markeredgewidth=0.2)
    ax2.plot((true_points_1[..., 0].mean()), (true_points_1[..., 1].mean()), '*', markerfacecolor='g',
             markersize=10,
             markeredgecolor='k', markeredgewidth=0.2)

    if actual_points is not None:
        ax2.plot(actual_points_1[..., 0], actual_points_1[..., 1], 'p', markerfacecolor='blue', markersize=5,
                 markeredgecolor='k',
                 markeredgewidth=0.2)
        ax2.plot((actual_points_1[..., 0].mean()), (actual_points_1[..., 1].mean()), '*', markerfacecolor='blue',
                 markersize=10,
                 markeredgecolor='k', markeredgewidth=0.2)

    # print(f'Predicted Center: {(predicted_points[..., 0].mean()), (predicted_points[..., 1].mean())}, '
    #       f'True Center: {(true_points[..., 0].mean()), (true_points[..., 1].mean())}')

    ax1.set_title('Un-normalized')
    ax2.set_title('Normalized')

    f.legend(loc="upper left")
    plt.show()


def plot_extracted_features_and_verify_flow(features, frames, batch_size=32, normalized=False):
    total_frames = len(features)
    for fr in range(total_frames):
        of_frame_num = (fr + 12) % total_frames

        if of_frame_num < fr:
            break

        current_frame = features[fr]
        current_image = (frames[fr].permute(1, 2, 0) * 255).int().numpy()

        of_frame = features[of_frame_num]
        of_frame_image = (frames[of_frame_num].permute(1, 2, 0) * 255).int().numpy()

        current_x_y = []
        current_u_v = []
        for obj in current_frame:
            if normalized:
                re_normalized_x_y = renormalize_any_cluster(obj.features[:, :2], obj.normalize_params).astype(np.int)
                re_normalized_u_v = renormalize_optical_flow(obj.features[:, 2:4], obj.normalize_params).astype(np.int)
                current_x_y.append([re_normalized_x_y[:, 0], re_normalized_x_y[:, 1]])
                current_u_v.append([re_normalized_u_v[:, 0], re_normalized_u_v[:, 1]])
            else:
                x_y_ = obj.features[:, :2].astype(np.int)
                u_v_ = obj.features[:, 2:4].astype(np.int)
                current_x_y.append([x_y_[:, 0], x_y_[:, 1]])
                current_u_v.append([u_v_[:, 0], u_v_[:, 1]])

        future_of_x_y = []
        for f_obj in of_frame:
            if normalized:
                re_normalized_x_y_f = renormalize_any_cluster(f_obj.features[:, :2], f_obj.normalize_params).astype(
                    np.int)
                future_of_x_y.append([re_normalized_x_y_f[:, 0], re_normalized_x_y_f[:, 1]])
            else:
                x_y_f_ = f_obj.features[:, :2].astype(np.int)
                future_of_x_y.append([x_y_f_[:, 0], x_y_f_[:, 1]])

        fig, axs = plt.subplots(1, 3, sharex='none', sharey='none',
                                figsize=(12, 10))
        axs[0].imshow(current_image)
        axs[1].imshow(of_frame_image)
        axs[2].imshow(of_frame_image)

        axs[0].set_title(f'Frame {fr}')
        axs[1].set_title('Optical Flow Shifted')
        axs[2].set_title(f'Frame {of_frame_num}')

        colors = cm.rainbow(np.linspace(0, 1, 20))
        for obj_, col in zip(current_x_y, colors):
            axs[0].plot(obj_[0], obj_[1], 'o', markerfacecolor=col, markersize=1, markeredgecolor='k',
                        markeredgewidth=0.2)
            axs[0].plot(int(obj_[0].mean()), int(obj_[1].mean()), '*', markerfacecolor='r', markersize=5,
                        markeredgecolor='k',
                        markeredgewidth=0.2)

        shifted_x_y = []
        for xy, uv in zip(current_x_y, current_u_v):  # check which indices to sum
            x_ = xy[0] + uv[0]
            y_ = xy[1] + uv[1]
            shifted_x_y.append([x_, y_])

        for obj_, col in zip(shifted_x_y, colors):
            axs[1].plot(obj_[0], obj_[1], 'o', markerfacecolor=col, markersize=1, markeredgecolor='k',
                        markeredgewidth=0.2)
            axs[1].plot(int(obj_[0].mean()), int(obj_[1].mean()), '*', markerfacecolor='r', markersize=5,
                        markeredgecolor='k',
                        markeredgewidth=0.2)

        for obj_, col in zip(future_of_x_y, colors):
            axs[2].plot(obj_[0], obj_[1], 'o', markerfacecolor=col, markersize=1, markeredgecolor='k',
                        markeredgewidth=0.2)
            axs[2].plot(int(obj_[0].mean()), int(obj_[1].mean()), '*', markerfacecolor='r', markersize=5,
                        markeredgecolor='k',
                        markeredgewidth=0.2)

        plt.show()


def plot_points_only(x, y):
    plt.plot(x, y, 'o', markerfacecolor='r', markersize=5, markeredgecolor='k', markeredgewidth=0.2)
    plt.plot(x.mean(), y.mean(), 'x', markerfacecolor='b', markersize=8, markeredgecolor='k',
             markeredgewidth=0.2)
    plt.show()


class SDDAnnotations(object):
    def __init__(self, root: str, video_label: SDDVideoClasses, transform: Optional[Any] = None):
        _mid_path = video_label.value
        annotation_path = root + "annotations/" + _mid_path

        classes = list(sorted(list_dir(annotation_path),
                              key=lambda x: int(x[-1]) if len(x) == 6 else int(x[-2:])))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(directory=annotation_path, class_to_idx=class_to_idx,
                                    extensions=('txt',))
        self.annotation_list = [x[0] for x in self.samples]
        self.annotation_list_idx = [x[1] for x in self.samples]
        self.transform = transform

        meta_file = 'H_SDD.txt'
        self.sdd_meta = SDDMeta(root + meta_file)

    def augment_annotations(self):
        for annotation_path in self.annotation_list:
            df = annotations_to_dataframe(annotation_path)
            bbox = df[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            centers = cal_centers(bbox)
            df['bbox_center_x'] = centers[:, 0]
            df['bbox_center_y'] = centers[:, 1]
            save_path = os.path.join(os.path.split(annotation_path)[0], 'annotation_augmented.csv')
            df.to_csv(save_path)
            print(f'Generated for {annotation_path}')


class SDDMeta(object):
    def __init__(self, path):
        self.path = path
        columns = ['File', 'Dataset', 'Nr', 'Version', 'Feet', 'Inch', 'Meters', 'Pixel X', 'Pixel Y',
                   'Pixel Dist', 'Average', 'Diff', 'Ratio']
        self.data = pd.DataFrame(pd.read_csv(path, sep='\t', header=None).values[1:], columns=columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        return self.data.loc[item]

    def get_meta(self, dataset: SDDVideoDatasets, sequence: int, version: str = 'A'):
        where_dataset = self.data['Dataset'] == dataset.value
        where_sequence = self.data['Nr'] == str(sequence)
        where_version = self.data['Version'] == version

        out = self.data.loc[where_dataset & where_sequence & where_version]
        return out, out.to_numpy()

    def get_new_scale(self, img_path, dataset: SDDVideoDatasets, sequence: int, desired_ratio: float = 0.1,
                      version: str = 'A'):
        ratio = self.get_meta(dataset, sequence, version)[0]['Ratio']
        im: Image = Image.open(img_path)
        w, h = im.width, im.height
        out_w, out_h = self.calculate_scale(w, h, ratio, desired_ratio)
        return out_w, out_h

    def get_new_scale_from_img(self, img, dataset: SDDVideoDatasets, sequence: int, desired_ratio: float = 0.1,
                               version: str = 'A'):
        ratio = float(self.get_meta(dataset, sequence, version)[0]['Ratio'].to_numpy()[0])
        w, h = img.shape[0], img.shape[1]
        out_w, out_h = self.calculate_scale(w, h, ratio, desired_ratio)
        return int(out_w), int(out_h)  # Quantization

    @staticmethod
    def calculate_scale(w, h, ratio, desired_ratio):
        img_x = w * ratio
        img_y = h * ratio
        out_w = img_x / desired_ratio
        out_h = img_y / desired_ratio
        return out_w, out_h

    def test_scale_logic(self):
        ratio = 0.03976968
        desired_ratio = 0.1
        w, h = 1325, 1973
        out_w, out_h = self.calculate_scale(w, h, ratio, desired_ratio)
        print(out_w, out_h)
        out_w, out_h = self.calculate_scale(out_w, out_h, desired_ratio, ratio)
        print(out_w, out_h)


class AgentFeatures(object):
    def __init__(self, features, track_id, frame_number, normalize_params, optical_flow_frame_num, bbox_center, bbox,
                 cluster_centers=None, cluster_labels=None):
        super(AgentFeatures, self).__init__()
        self.frame_number = frame_number
        self.features = features
        self.track_id = track_id
        self.normalize_params = normalize_params
        self.cluster_centers = cluster_centers
        self.cluster_labels = cluster_labels
        self.optical_flow_frame_num = optical_flow_frame_num
        self.bbox_center = bbox_center
        self.bbox = bbox

    def __repr__(self):
        pass  # auto-printing??
        #  print(f"Frame: {self.frame_number}, Track ID: {self.track_id}, Features: {self.features.shape}")

    def __eq__(self, other):
        return self.track_id == other.track_id


class BasicTrainData(object):
    def __init__(self, frame, track_id, pair_0_features, pair_1_features, bbox_center_t0, bbox_center_t1,
                 frame_t0, frame_t1, bbox_t0, bbox_t1):
        super(BasicTrainData, self).__init__()
        self.frame = frame
        self.track_id = track_id
        self.pair_0_features = pair_0_features
        self.pair_1_features = pair_1_features
        self.bbox_center_t0 = bbox_center_t0
        self.bbox_center_t1 = bbox_center_t1
        self.frame_t0 = frame_t0
        self.frame_t1 = frame_t1
        self.bbox_t0 = bbox_t0
        self.bbox_t1 = bbox_t1

    def __eq__(self, other):
        return self.track_id == other.track_id


class BasicTestData(object):
    def __init__(self, frame, track_id, pair_0_features, pair_1_features, pair_0_normalize, pair_1_normalize):
        super(BasicTestData, self).__init__()
        self.frame = frame
        self.track_id = track_id
        self.pair_0_features = pair_0_features
        self.pair_1_features = pair_1_features
        self.pair_1_normalize = pair_1_normalize
        self.pair_0_normalize = pair_0_normalize


if __name__ == '__main__':
    pd.options.display.max_columns = None
    p = '../Datasets/SDD/H_SDD.txt'
    meta = SDDMeta(p)
    m = meta.get_meta(SDDVideoDatasets.LITTLE, 0)
    print(m[0]['Ratio'].to_numpy())
    # print(float(m[1][-1, -1]))
    # # print(meta.get_meta(SDDVideoClasses.GATES, 0, 'A')[0])
    # meta.test_scale_logic()

    # im_path = '/home/rishabh/TrajectoryPrediction/Datasets/SDD/annotations/gates/video0/reference.jpg'
    #
    # im_ = Image.open(im_path)
    # print(im_.info)

    # root_ = '../Datasets/SDD/'
    # vid_label = SDDVideoClasses.QUAD
    # sdd_annotations = SDDAnnotations(root_, vid_label)
    # # sdd_annotations.augment_annotations()
    # # dff = pd.read_csv('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/annotations'
    # #                   '/little/video0/annotation_augmented.csv')
    # # dff.drop(dff.columns[[0]], axis=1)
    print()
