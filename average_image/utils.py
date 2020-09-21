import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


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


def evaluate_clustering_per_frame(frame, frame_dict):  # use this inside above
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
        precision_1.append(result[frame]['precision'])
        precision_2.append(result_[frame]['precision'])

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
        recall_1.append(result[frame]['recall'])
        recall_2.append(result_[frame]['recall'])

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


def plot_precision_recall(pr_results_1, pr_results_2, avg_img, lab1, lab2):
    compare_precision(pr_results_1, pr_results_2, avg_img, lab1, lab2)
    compare_recall(pr_results_1, pr_results_2, avg_img, lab1, lab2)
