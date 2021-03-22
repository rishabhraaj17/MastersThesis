import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
import torch.nn as nn
import itertools
import importlib
from argparse import Namespace
import os.path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import json
import random
import os
import pytorch_lightning as pl

from log import initialize_logging, get_logger

initialize_logging()
logger = get_logger(__name__)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def distanceP2W(point, wall):
    p0 = np.array([wall[0], wall[1]])
    p1 = np.array([wall[2], wall[3]])

    d = p1 - p0
    ymp0 = point - p0
    t = np.dot(d, ymp0) / np.dot(d, d)
    if t > 0.0 and t < 1.:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
        npw = normalize(cross - point)

    else:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
        npw = normalize(cross - point) * 0

    return dist, npw


def image_json(scene, json_path, scaling=1):
    json_path = os.path.join(json_path, "{}_seg.json".format(scene))

    wall_labels = ["lawn", "building", "car", "roundabout"]

    walls = []
    wall_points = []

    start_end_points = {}
    decisionZone = {}
    directionZone = {}

    nr_start_end = 0

    with open(json_path) as json_file:

        data = json.load(json_file)
        for p in data["shapes"]:
            label = p["label"]

            if label in wall_labels:

                points = np.array(p["points"]).astype(int)

                points = order_clockwise(points)
                for i in np.arange(len(points)):
                    j = (i + 1) % len(points)

                    p1 = points[i]
                    p2 = points[j]

                    concat = np.concatenate((p1, p2))
                    walls.append(scaling * concat)

                wall_points.append([p * scaling for p in points])
            elif "StartEndZone" in label:
                id = int(label.split("_")[-1])
                start_end_points[nr_start_end] = {"point": scaling * np.array(p["points"]),
                                                  "id": id}
                nr_start_end += 1
            elif "decisionZone" in label:
                id = int(label.split("_")[-1])
                decisionZone[id] = scaling * np.array(p["points"])

            elif "directionZone" in label:
                id = int(label.split("_")[-1])
                directionZone[id] = scaling * np.array(p["points"])

    return walls, wall_points, start_end_points, decisionZone, directionZone


# order points clockwise

def order_clockwise(point_array, orientation=np.array([1, 0])):
    center = np.mean(point_array, axis=0)
    directions = point_array - center

    angles = []
    for d in directions:
        t = np.arctan2(d[1], d[0])
        angles.append(t)
    point_array = [x for _, x in sorted(zip(angles, point_array))]

    return point_array


def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            break

    return random_point


def get_batch_k(batch, k):
    new_batch = {}
    for name, data in batch.items():

        if name in ["global_patch", "prob_mask"]:

            new_batch[name] = data.repeat(k, 1, 1, 1).clone()
        elif name in ["local_patch"]:

            new_batch[name] = data.repeat(k, 1, 1, 1, 1).clone()
        elif name in ["scene_img", "occupancy", "walls"]:

            new_batch[name] = data * k
        elif name not in ["size", "scene_nr", "scene", "img", "cropped_img", "seq_start_end"]:
            new_batch[name] = data.repeat(1, k, 1).clone()


        else:
            new_batch[name] = data

    new_batch.update({'size': batch['in_xy'].shape[1]})
    return new_batch


def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def re_im(img):
    """ Rescale images """
    img = (img + 1) / 2.
    return img


class BatchSizeScheduler(pl.callbacks.base.Callback):
    """
    Implmentation of a BatchSize Scheduler following the paper
    'Don't Decay the Learning Rate, Increase the Batch Size'
    (https://arxiv.org/abs/1711.00489)
    The scheduler increases the batchsize if the validation loss does not decrease.
    """

    def __init__(self, bs=4, factor=2, patience=3, max_bs=64, mode="min", monitor_val="val_loss"):
        """
        :param bs: initial batch size
        :param factor: factor by which current batch size is increased
        :param patience: duration in which loss does not have to decrease
        :param max_bs: maximum batch size
        :param mode: considering 'min' or 'max' for 'monitor_val'
        :param monitor_val: considered loss for scheduler
        """

        self.factor = factor
        self.patience = patience
        self.max_bs = max_bs
        self.current_count = patience * 1.
        self.cur_metric = False
        self.monitor_metric = monitor_val
        self.cur_bs = bs
        if mode not in ["min", "max"]:
            assert False, "Variable for mode '{}' not valid".format(mode)
        self.mode = mode
        if max_bs > bs:
            self.active = True
        else:
            self.active = False

    def on_validation_epoch_end(self, trainer, pl_module):

        self.cur_bs = int(np.minimum(self.cur_bs * self.factor, self.max_bs))

        # set new batch_size
        pl_module.batch_size = self.cur_bs
        trainer.reset_train_dataloader(pl_module)

        if not self.cur_metric:
            self.cur_metric = trainer.callback_metrics[self.monitor_metric]

        if self.active:
            if self.mode == "min":
                if trainer.callback_metrics[self.monitor_metric] < self.cur_metric:
                    self.cur_metric = trainer.callback_metrics[self.monitor_metric]
                    self.current_count = self.patience * 1
                    logger.info(f'Resetting patience to default value, current patience: {self.current_count}')
                else:
                    self.current_count -= 1
                    logger.info(f'Decreasing patience by 1, current patience: {self.current_count}')

            else:
                if trainer.callback_metrics[self.monitor_metric] > self.cur_metric:
                    self.cur_metric = trainer.callback_metrics[self.monitor_metric]
                    self.current_count = self.patience * 1
                    logger.info(f'Resetting patience to default value, current patience: {self.current_count}')
                else:
                    self.current_count -= 1
                    logger.info(f'Decreasing patience by 1, current patience: {self.current_count}')

            if self.current_count == 0:
                logger.info(f'Increasing batch size from : {self.cur_bs}')
                self.cur_bs = int(np.minimum(self.cur_bs * self.factor, self.max_bs))

                # set new batch_size
                pl_module.batch_size = self.cur_bs
                trainer.reset_train_dataloader(pl_module)
                logger.info("SET BS TO {}".format(self.cur_bs))
                self.current_count = self.patience * 1
                if self.cur_bs >= self.max_bs:
                    self.active = False
