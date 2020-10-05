from typing import List, Optional, Tuple

import numpy as np
import torch.nn.functional as F
import pandas as pd
import matplotlib.patches as patches

from average_image.constants import ObjectClasses, ANNOTATION_COLUMNS, OBJECT_CLASS_COLOR_MAPPING


def annotations_to_dataframe(file: str):
    df = pd.read_csv(file, sep=" ", header=None)
    df.columns = ANNOTATION_COLUMNS
    return df


def get_frame_annotations(df: pd.DataFrame, frame_number: int):
    idx: pd.DataFrame = df.loc[df["frame"] == frame_number]
    return idx.to_numpy()


def get_frame_by_track_annotations(df: pd.DataFrame, frame_number: int, track_id: int):
    filtered_by_track_id = df.loc[df["track_id"] == track_id]
    idx: pd.DataFrame = filtered_by_track_id.loc[filtered_by_track_id["frame"] == frame_number]
    return idx.to_numpy()


def add_bbox_to_axes(ax, annotations: List, show_lost: bool = False, only_pedestrians: bool = True,
                     original_spatial_dim=None, linewidth=None,
                     pooled_spatial_dim=None, min_pool: bool = False, use_dnn: bool = False):
    for annotation in annotations:
        x, y, width, height = bbox_to_matplotlib_representation(annotation[1:5],
                                                                original_spatial_dim=original_spatial_dim,
                                                                pooled_spatial_dim=pooled_spatial_dim,
                                                                min_pool=min_pool, use_dnn=use_dnn)
        if annotation[6] == 1 and not show_lost:
            continue
        if only_pedestrians:
            if annotation[-1] == ObjectClasses.PEDESTRIAN.value:
                rect = patches.Rectangle(xy=(x, y), width=width, height=height,
                                         edgecolor=OBJECT_CLASS_COLOR_MAPPING[ObjectClasses.PEDESTRIAN], fill=False,
                                         linewidth=linewidth)
                ax.add_patch(rect)
        else:
            rect = patches.Rectangle(xy=(x, y), width=width, height=height,
                                     edgecolor=OBJECT_CLASS_COLOR_MAPPING[ObjectClasses(annotation[-1])],
                                     fill=False, linewidth=linewidth)
            ax.add_patch(rect)


def bbox_to_matplotlib_representation(coordinates: List, original_spatial_dim=None,
                                      pooled_spatial_dim=None, min_pool: bool = False, use_dnn: bool = False):
    x_min, y_min, x_max, y_max = coordinates
    x, y, w, h = x_min, y_min, (x_max - x_min), (y_max - y_min)
    if min_pool or use_dnn:
        h, w, x, y = _process_scale(h, pooled_spatial_dim, original_spatial_dim, w, x, y)
    return x, y, w, h


def scale_annotations_(annotations, original_scale, new_scale):
    scaled_annotations = []
    for annot in annotations:
        x_min, y_min, x_max, y_max = annot[1:5]
        x, y, w, h = x_min, y_min, (x_max - x_min), (y_max - y_min)
        h, w, x, y = _process_scale(h, new_scale, original_scale, w, x, y)
        scaled_annotations.append([x, y, x + w, y + h])
    return scaled_annotations


def scale_annotations(annotations, original_scale, new_scale, return_track_id=False, tracks_with_annotations=False):
    bbox = annotations[:, 1:5].astype(np.int)
    x_min, y_min, x_max, y_max = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x, y, w, h = x_min, y_min, (x_max - x_min), (y_max - y_min)
    h, w, x, y = _process_scale(h, new_scale, original_scale, w, x, y)
    x_min_, y_min_, x_max_, y_max_ = x, y, x + w, y + h
    if tracks_with_annotations:
        scaled_annotations = np.vstack((x_min_, y_min_, x_max_, y_max_, annotations[:, 0].astype(np.int))).T
    else:
        scaled_annotations = np.vstack((x_min_, y_min_, x_max_, y_max_)).T
    scaled_centers = cal_centers(scaled_annotations)
    if return_track_id:
        track_id = annotations[:, 0].astype(np.int)
        return scaled_annotations, scaled_centers, track_id
    return scaled_annotations, scaled_centers


def cal_centers(b_box):
    x_min = b_box[:, 0]
    y_min = b_box[:, 1]
    x_max = b_box[:, 2]
    y_max = b_box[:, 3]
    x_mid = (x_min + (x_max - x_min) / 2.).astype('int')
    y_mid = (y_min + (y_max - y_min) / 2.).astype('int')

    return np.vstack((x_mid, y_mid)).T


def _process_scale(h, new_scale, original_scale, w, x, y):
    o_w, o_h = original_scale
    c_w, c_h = new_scale
    x = (c_w * x) // o_w
    y = (c_h * y) // o_h
    w = (c_w * w) // o_w
    h = (c_h * h) // o_h
    return h, w, x, y


def resize_v_frames(v_frames, scale_factor: float = 0.5, size: Optional[Tuple] = None):
    return F.interpolate(input=v_frames, size=size, scale_factor=scale_factor, mode='bilinear')


def preprocess_annotations(annotations):
    ann = []
    for a in annotations:
        if a[6] != 1:
            ann.append(a.tolist())
    return ann


class CoordinateHolder(object):
    def __init__(self, coordinate):
        super(CoordinateHolder, self).__init__()
        self.a = coordinate[0]
        self.b = coordinate[1]
        self.c = coordinate[2]
        self.d = coordinate[3]

    def __eq__(self, other):
        if not isinstance(other, CoordinateHolder):
            return NotImplemented

        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d


class CoordinateHolder2(object):
    def __init__(self, coordinate):
        super(CoordinateHolder2, self).__init__()
        self.a = coordinate[0]
        self.b = coordinate[1]

    def __eq__(self, other):
        if not isinstance(other, CoordinateHolder2):
            return NotImplemented

        return self.a == other.a and self.b == other.b


if __name__ == '__main__':
    a = CoordinateHolder((100, 35, 89, 95))
    b = CoordinateHolder((10, 6, 2, 1))
    c = CoordinateHolder((9, 4, 8, 9))

    l = [a, b]
    print(a in l)
