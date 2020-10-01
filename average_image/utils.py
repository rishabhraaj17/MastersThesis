import os
from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
from matplotlib import patches
from sklearn.decomposition import PCA
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir

from bbox_utils import annotations_to_dataframe
from constants import SDDVideoClasses, SDDVideoDatasets, OBJECT_CLASS_COLOR_MAPPING, ObjectClasses


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


def renormalize_features(features, options):
    # todo: verify
    cc_0 = denormalize(features[..., 0], options['max_0'], options['min_0'])
    cc_1 = denormalize(features[..., 1], options['max_1'], options['min_1'])
    of_0 = denormalize(features[..., 0], options['f_max_0'], options['f_min_0'])
    of_1 = denormalize(features[..., 1], options['f_max_1'], options['f_min_1'])
    features[..., 0] = cc_0
    features[..., 1] = cc_1
    features[..., 2] = of_0
    features[..., 3] = of_1
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


if __name__ == '__main__':
    pd.options.display.max_columns = None
    p = '../Datasets/SDD/H_SDD.txt'
    # meta = SDDMeta(p)
    # m = meta.get_meta(SDDVideoDatasets.LITTLE, 0)
    # print(m[0]['Ratio'].to_numpy())
    # print(float(m[1][-1, -1]))
    # # print(meta.get_meta(SDDVideoClasses.GATES, 0, 'A')[0])
    # meta.test_scale_logic()

    # im_path = '/home/rishabh/TrajectoryPrediction/Datasets/SDD/annotations/gates/video0/reference.jpg'
    #
    # im_ = Image.open(im_path)
    # print(im_.info)

    root_ = '../Datasets/SDD/'
    vid_label = SDDVideoClasses.QUAD
    sdd_annotations = SDDAnnotations(root_, vid_label)
    # sdd_annotations.augment_annotations()
    # dff = pd.read_csv('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/annotations'
    #                   '/little/video0/annotation_augmented.csv')
    # dff.drop(dff.columns[[0]], axis=1)
    print()
