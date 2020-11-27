from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import is_inside_bbox, compute_ade, compute_fde, \
    compute_per_stop_de, SDDMeta, plot_track_analysis, plot_violin_plot
from baseline.extract_features import extract_trainable_features_rnn, process_complex_features_rnn
from log import initialize_logging, get_logger
from unsupervised_tp_0.extracted_of_optimization import cost_function
from unsupervised_tp_0.nn_clustering_0 import get_track_info

initialize_logging()
logger = get_logger(__name__)

SAVE_BASE_PATH = "../Datasets/SDD_Features/"
BASE_PATH = "../Datasets/SDD/"
VIDEO_LABEL = SDDVideoClasses.LITTLE
META_VIDEO_LABEL = SDDVideoDatasets.LITTLE
VIDEO_NUMBER = 3
SAVE_PATH = f'{SAVE_BASE_PATH}{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/'
FILE_NAME = 'time_distributed_dict_with_gt_bbox_centers_and_bbox_gt_velocity.pt'
LOAD_FILE = SAVE_PATH + FILE_NAME
ANNOTATIONS_FILE = 'annotation_augmented.csv'
ANNOTATIONS_PATH = f'{BASE_PATH}annotations/{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/'
TIME_STEPS = 8
META_PATH = '../Datasets/SDD/H_SDD.txt'
META = SDDMeta(META_PATH)
DF_SAVE_PATH = f'{SAVE_PATH}analysis/'
DF_FILE_NAME = f'analysis_t{TIME_STEPS}.csv'


def analyze_extracted_features(features_dict: dict, test_mode=False, time_steps=TIME_STEPS, track_info=None, ratio=1.0,
                               num_frames_to_build_bg_sub_model=12, optimized_of=False, top_k=1, alpha=1,
                               weight_points_inside_bbox_more=True):
    train_data = process_complex_features_rnn(features_dict=features_dict, test_mode=test_mode, time_steps=time_steps,
                                              num_frames_to_build_bg_sub_model=num_frames_to_build_bg_sub_model)

    x_, y_, frame_info, track_id_info, bbox_center_x, bbox_center_y, bbox_x, bbox_y, gt_velocity_x, gt_velocity_y = \
        extract_trainable_features_rnn(train_data)
    of_track_analysis = {}
    of_track_analysis_df = None
    for features_x, features_y, features_f_info, features_t_info, features_b_c_x, features_b_c_y, features_b_x, \
        features_b_y in tqdm(zip(x_, y_, frame_info, track_id_info, bbox_center_x,
                                 bbox_center_y, bbox_x, bbox_y)):
        unique_tracks = np.unique(features_t_info)
        current_track = unique_tracks[0]
        of_inside_bbox_list, of_track_list, gt_track_list, of_ade_list, of_fde_list, of_per_stop_de = \
            [], [], [], [], [], []
        for feature_x, feature_y, f_info, t_info, b_c_x, b_c_y, b_x, b_y in zip(features_x, features_y, features_f_info,
                                                                                features_t_info, features_b_c_x,
                                                                                features_b_c_y,
                                                                                features_b_x, features_b_y):
            of_flow = feature_x[:, :2] + feature_x[:, 2:4]
            if optimized_of:
                of_flow_center = of_flow.mean(0)
            else:
                _, of_flow_top_k = cost_function(of_flow, bbox_center_y, top_k=top_k, alpha=alpha, bbox=bbox_y,
                                                 weight_points_inside_bbox_more=weight_points_inside_bbox_more)
                of_flow_center = of_flow_top_k[0]
            of_inside_bbox = is_inside_bbox(of_flow_center, b_y)
            of_inside_bbox_list.append(of_inside_bbox)

            of_track_list.append(of_flow_center)
            gt_track_list.append(b_c_y)

        of_ade = compute_ade(np.stack(of_track_list), np.stack(gt_track_list))
        of_fde = compute_fde(np.stack(of_track_list), np.stack(gt_track_list))
        of_ade_list.append(of_ade.item() * ratio)
        of_fde_list.append(of_fde.item() * ratio)

        per_stop_de = compute_per_stop_de(np.stack(of_track_list), np.stack(gt_track_list))
        of_per_stop_de.append(per_stop_de)

        if len(unique_tracks) == 1:
            d = {'track_id': current_track,
                 'of_inside_bbox_list': of_inside_bbox_list,
                 'ade': of_ade.item() * ratio,
                 'fde': of_fde.item() * ratio,
                 'per_stop_de': [p * ratio for p in per_stop_de]}
            if of_track_analysis_df is None:
                of_track_analysis_df = pd.DataFrame(data=d)
            else:
                temp_df = pd.DataFrame(data=d)
                of_track_analysis_df = of_track_analysis_df.append(temp_df, ignore_index=False)
            # of_track_analysis.update({current_track: {
            #     'of_inside_bbox_list': of_inside_bbox_list,
            #     'ade': of_ade.item() * ratio,
            #     'fde': of_fde.item() * ratio,
            #     'per_stop_de': [p * ratio for p in per_stop_de]}})
        else:
            logger.info(f'Found multiple tracks! - {unique_tracks}')

    return of_track_analysis_df


def parse_df_analysis(in_df, save_path=None):
    t_id_list, ade_list, fde_list = [], [], []
    inside_bbox_list, per_stop_de_list, inside_bbox, per_stop_de = [], [], [], []
    inside_bbox_count, outside_bbox_count = [], []
    t_id, ade, fde = None, None, None
    for idx, (index, row) in enumerate(tqdm(in_df.iterrows())):
        if idx == 0:
            t_id = row['track_id']
            ade = row['ade']
            fde = row['fde']
        if row['Unnamed: 0'] == 0:
            if idx != 0:
                t_id_list.append(t_id)
                ade_list.append(ade)
                fde_list.append(fde)
                inside_bbox_list.append(inside_bbox)
                per_stop_de_list.append(per_stop_de)
                inside_bbox_count.append(inside_bbox.count(True))
                outside_bbox_count.append(inside_bbox.count(False))
                if idx % 99 == 0:
                    plot_track_analysis(t_id, ade, fde, inside_bbox, per_stop_de, save_path + 'plots/', idx)
                # plot_track_analysis(t_id, ade, fde, inside_bbox, per_stop_de, save_path+'plots/', idx)
                inside_bbox, per_stop_de = [], []
            t_id = row['track_id']
            ade = row['ade']
            fde = row['fde']
            inside_bbox.append(row['of_inside_bbox_list'])
            per_stop_de.append(row['per_stop_de'])
        else:
            inside_bbox.append(row['of_inside_bbox_list'])
            per_stop_de.append(row['per_stop_de'])
    plot_violin_plot(ade_list, fde_list, save_path)
    in_count = sum(inside_bbox_count)
    out_count = sum(outside_bbox_count)
    print(f'% inside = {(in_count / (in_count + out_count)) * 100}')
    return t_id_list, ade_list, fde_list, inside_bbox_list, per_stop_de_list


def analyze(save=False, optimized_of=True, top_k=1, alpha=1, weight_points_inside_bbox_more=True):
    features = torch.load(LOAD_FILE)
    annotation = get_track_info(ANNOTATIONS_PATH + ANNOTATIONS_FILE)
    pixel_to_meter_ratio = float(META.get_meta(META_VIDEO_LABEL, VIDEO_NUMBER)[0]['Ratio'].to_numpy()[0])
    analysis_data = analyze_extracted_features(features, time_steps=TIME_STEPS, track_info=annotation,
                                               ratio=pixel_to_meter_ratio, optimized_of=optimized_of, top_k=top_k,
                                               alpha=alpha, weight_points_inside_bbox_more=
                                               weight_points_inside_bbox_more)
    if DF_SAVE_PATH and save:
        Path(DF_SAVE_PATH).mkdir(parents=True, exist_ok=True)
        analysis_data.to_csv(DF_SAVE_PATH + DF_FILE_NAME)
    return analysis_data


def parse_analysis(df):
    if isinstance(df, str):
        df = pd.read_csv(DF_SAVE_PATH + DF_FILE_NAME)
    parsed_data = parse_df_analysis(df, DF_SAVE_PATH)
    return parsed_data


def main(save=True, optimized_of=True, top_k=1, alpha=1, weight_points_inside_bbox_more=True):
    df = analyze(save=save, optimized_of=optimized_of, top_k=top_k, alpha=alpha,
                 weight_points_inside_bbox_more=weight_points_inside_bbox_more)
    parsed_data = parse_analysis(df)
    logger.info("Analysis Completed!")


if __name__ == '__main__':
    main(optimized_of=False, top_k=1, alpha=1, weight_points_inside_bbox_more=True)
