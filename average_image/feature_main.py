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


if __name__ == '__main__':
    annotation_base_path = "/home/rishabh/TrajectoryPrediction/Datasets/SDD/annotations/"
    video_base_path = "/home/rishabh/TrajectoryPrediction/Datasets/SDD/videos/"
    vid_label = SDDVideoClasses.GATES
    video_number = 4
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

    mog2_ = MOG2(video_path=video_file_path, start_frame=0, end_frame=90)

    # TODO: verify (x,y) once again
    mog2_optical_flow_clustering_per_frame(mog2=mog2_, video_label=vid_label.value, vid_number=video_number,
                                           frames_out_save_path=frames_save_path, annotations_df=df,
                                           plot_scale_factor=1, plot=False, precision_recall=True,
                                           with_optical_flow=True)

    # mog2_optical_flow_clustering_video(mog2=mog2_, video_label=vid_label.value, vid_number=video_number,
    #                                    video_out_save_path=video_save_path, annotations_df=df,
    #                                    plot_scale_factor=1, precision_recall=False,
    #                                    with_optical_flow=True)
