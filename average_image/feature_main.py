import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bbox_utils import annotations_to_dataframe
from constants import SDDVideoClasses
from feature_extractor import MOG2
from utils import precision_recall_one_sequence


def mog2_optical_flow_clustering_per_frame(mog2, video_label, vid_number, frames_out_save_path, annotations_df,
                                           plot_scale_factor: int = 1, plot: bool = True,
                                           precision_recall: bool = False, with_optical_flow: bool = True):
    activations = mog2.get_activations(history=100, detect_shadows=True, var_threshold=100)
    gt_bbox_cluster_center_dict, pr_for_frames = mog2.get_per_frame_results(processed_data=activations,
                                                                            video_label=video_label,
                                                                            video_number=vid_number,
                                                                            frames_out_save_path=frames_out_save_path,
                                                                            annotations_df=annotations_df,
                                                                            plot_scale_factor=plot_scale_factor,
                                                                            plot=plot,
                                                                            with_optical_flow=with_optical_flow)
    if precision_recall:
        precision_recall_one_sequence(pr_for_frames, average_image=False)


def mog2_optical_flow_clustering_video(mog2, video_label, vid_number, video_out_save_path, annotations_df,
                                       plot_scale_factor: int = 1,
                                       precision_recall: bool = False, with_optical_flow: bool = True):
    activations = mog2.get_activations(history=100, detect_shadows=True, var_threshold=100)
    gt_bbox_cluster_center_dict, pr_for_frames = mog2.make_video(processed_data=activations,
                                                                 video_label=video_label,
                                                                 video_number=vid_number,
                                                                 video_out_save_path=video_out_save_path,
                                                                 annotations_df=annotations_df,
                                                                 plot_scale_factor=plot_scale_factor,
                                                                 with_optical_flow=with_optical_flow)
    if precision_recall:
        precision_recall_one_sequence(pr_for_frames, average_image=False)


def evaluate(mog2, num_frames, video_label, vid_number, frames_out_save_path, annotations_df,
             plot_scale_factor: int = 1, plot: bool = False, evaluation_save_path_: str = None,
             precision_recall: bool = False, with_optical_flow: bool = True, history: int = 120,
             detect_shadows: bool = True, var_threshold: int = 100):
    gt_bbox_cluster_center_dict, pr_for_frames = mog2.evaluate_sequence(num_frames=num_frames,
                                                                        video_label=video_label,
                                                                        video_number=vid_number,
                                                                        frames_out_save_path=frames_out_save_path,
                                                                        annotations_df=annotations_df,
                                                                        plot_scale_factor=plot_scale_factor,
                                                                        plot=plot,
                                                                        with_optical_flow=with_optical_flow,
                                                                        history=history,
                                                                        detect_shadows=detect_shadows,
                                                                        var_threshold=var_threshold)

    if evaluation_save_path_ is not None:
        result_dict = {'video_label': vid_label.value,
                       'video_number': vid_number,
                       'method': 'MOG2',
                       'history': history,
                       'var_threshold': var_threshold,
                       'optical_flow': with_optical_flow,
                       'precision_recall': pr_for_frames}
        torch.save(result_dict, f"{evaluation_save_path_}.pt")
    if precision_recall:
        precision_recall_one_sequence(pr_for_frames, average_image=False)


def post_evaluation_tb(eval_file):
    results = torch.load(eval_file)
    writer = SummaryWriter(comment=os.path.split(eval_file)[0])

    for f_no, metric in tqdm(results['precision_recall'].items()):
        writer.add_scalar('Precision', metric['precision'], global_step=f_no)
        writer.add_scalar('Recall', metric['recall'], global_step=f_no)


def post_evaluation(eval_file):
    results = torch.load(eval_file)
    frame_list = []
    precision_list = []
    recall_list = []

    for f_no, metric in tqdm(results['precision_recall'].items()):
        frame_list.append(f_no)
        precision_list.append(metric['precision'])
        recall_list.append(metric['recall'])

    print(f"Video : {results['video_label']}, Number: {results['video_number']}")
    total_precision = np.array(precision_list)
    total_recall = np.array(recall_list)
    print(f"Average Precision: {total_precision.sum()/total_precision.shape[0]}"
          f"\nAverage Recall: {total_recall.sum()/total_recall.shape[0]}")


if __name__ == '__main__':
    annotation_base_path = "/home/rishabh/TrajectoryPrediction/Datasets/SDD/annotations/"
    video_base_path = "/home/rishabh/TrajectoryPrediction/Datasets/SDD/videos/"
    vid_label = SDDVideoClasses.HYANG
    video_number = 9
    video_file_name = "video.mov"
    annotation_file_name = "annotations.txt"
    base_save_path = "../Plots/outputs/"

    video_file_path = f"{video_base_path}{str(vid_label.value)}/video{str(video_number)}/{video_file_name}"
    annotation_file_path = f"{annotation_base_path}{str(vid_label.value)}/video{str(video_number)}/" \
                           f"{annotation_file_name}"

    df = annotations_to_dataframe(annotation_file_path)
    frames_save_path = f"{base_save_path}clustering/per_frame_video_label_{vid_label.value}_video_number_" \
                       f"{video_number}_"
    video_save_path = f"{base_save_path}video_label_{vid_label.value}_video_number_" \
                      f"{video_number}.avi"
    evaluation_save_path = f"{base_save_path}clustering/video_label_{vid_label.value}_video_number_" \
                           f"{video_number}_evaluation_1"
    evaluation_save_path_last = f"{base_save_path}clustering/video_label_{vid_label.value}_video_number_" \
                                f"{video_number}_evaluation"

    mog2_ = MOG2(video_path=video_file_path, start_frame=0, end_frame=90)

    # TODO: verify (x,y) once again
    # mog2_optical_flow_clustering_per_frame(mog2=mog2_, video_label=vid_label.value, vid_number=video_number,
    #                                        frames_out_save_path=frames_save_path, annotations_df=df,
    #                                        plot_scale_factor=1, plot=False, precision_recall=True,
    #                                        with_optical_flow=True)

    # mog2_optical_flow_clustering_video(mog2=mog2_, video_label=vid_label.value, vid_number=video_number,
    #                                    video_out_save_path=video_save_path, annotations_df=df,
    #                                    plot_scale_factor=1, precision_recall=False,
    #                                    with_optical_flow=True)

    frames_count = 550
    evaluate(mog2=mog2_, num_frames=frames_count, video_label=vid_label.value, vid_number=video_number,
             frames_out_save_path=frames_save_path, annotations_df=df,
             plot_scale_factor=1, plot=False, precision_recall=True,
             with_optical_flow=True, evaluation_save_path_=evaluation_save_path, history=frames_count,
             detect_shadows=True, var_threshold=100)

    # post_evaluation(f"{base_save_path}clustering/video_label_deathCircle_video_number_2_evaluation_1.pt")
