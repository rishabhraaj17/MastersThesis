from typing import List, Optional, Tuple

import torch.nn.functional as F
import pandas as pd
import matplotlib.patches as patches

from constants import ObjectClasses, ANNOTATION_COLUMNS, OBJECT_CLASS_COLOR_MAPPING


def annotations_to_dataframe(file: str):
    df = pd.read_csv(file, sep=" ", header=None)
    df.columns = ANNOTATION_COLUMNS
    return df


def get_frame_annotations(df: pd.DataFrame, frame_number: int):
    idx: pd.DataFrame = df.loc[df["frame"] == frame_number]
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
        o_w, o_h = original_spatial_dim
        c_w, c_h = pooled_spatial_dim
        x = (c_w * x) // o_w
        y = (c_h * y) // o_h
        w = (c_w * w) // o_w
        h = (c_h * h) // o_h
    return x, y, w, h


def scale_annotations(annotations, original_scale, new_scale):
    scaled_annotations = []
    for annot in annotations:
        x_min, y_min, x_max, y_max = annot[1:5]
        x, y, w, h = x_min, y_min, (x_max - x_min), (y_max - y_min)
        o_w, o_h = original_scale
        c_w, c_h = new_scale
        x = (c_w * x) // o_w
        y = (c_h * y) // o_h
        w = (c_w * w) // o_w
        h = (c_h * h) // o_h
        scaled_annotations.append([x, y, x+w, y+h])
    return scaled_annotations


def resize_v_frames(v_frames, scale_factor: float = 0.5, size: Optional[Tuple] = None):
    return F.interpolate(input=v_frames, size=size, scale_factor=scale_factor, mode='bilinear')


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
    a = CoordinateHolder((1, 3, 8, 9))
    b = CoordinateHolder((10, 6, 2, 1))
    c = CoordinateHolder((9, 4, 8, 9))

    l = [a, b]
    print(c in l)
