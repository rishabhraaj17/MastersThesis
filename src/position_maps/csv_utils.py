import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches, lines


def csv_merge(default_path, boosted_path, nn_path, save_root_path, dump_as_markdown=True):
    default_df = pd.read_csv(default_path)
    boosted_df = pd.read_csv(boosted_path)
    nn_df = pd.read_csv(nn_path)

    out_df = nn_df[['class', 'number']]
    out_df['class'] = out_df.loc[:, 'class'].str.lower()

    # out_df['precision'] = default_df.loc[:, 'precision'].round(3)
    # out_df['classifier_precision'] = boosted_df.loc[:, 'precision'].round(3)
    # out_df['nn_precision'] = nn_df.loc[:, 'precision'].round(3)
    #
    # out_df['recall'] = default_df.loc[:, 'recall'].round(3)
    # out_df['classifier_recall'] = boosted_df.loc[:, 'recall'].round(3)
    # out_df['nn_recall'] = nn_df.loc[:, 'recall'].round(3)
    #
    # out_df['ade'] = default_df.loc[:, 'ade'].round(3)
    # out_df['classifier_ade'] = boosted_df.loc[:, 'ade'].round(3)
    # out_df['nn_ade'] = nn_df.loc[:, 'ade'].round(3)
    #
    # out_df['fde'] = default_df.loc[:, 'fde'].round(3)
    # out_df['classifier_fde'] = boosted_df.loc[:, 'fde'].round(3)
    # out_df['nn_fde'] = nn_df.loc[:, 'fde'].round(3)

    out_df['vanilla_precision'] = default_df.loc[:, 'precision'].round(3)
    out_df['classic_precision'] = boosted_df.loc[:, 'precision'].round(3)
    out_df['nn_precision'] = nn_df.loc[:, 'precision'].round(3)

    out_df['vanilla_recall'] = default_df.loc[:, 'recall'].round(3)
    out_df['classic_recall'] = boosted_df.loc[:, 'recall'].round(3)
    out_df['nn_recall'] = nn_df.loc[:, 'recall'].round(3)

    # out_df['vanilla_ade'] = default_df.loc[:, 'ade'].round(3)
    # out_df['classic_ade'] = boosted_df.loc[:, 'ade'].round(3)
    # out_df['nn_ade'] = nn_df.loc[:, 'ade'].round(3)
    #
    # out_df['vanilla_fde'] = default_df.loc[:, 'fde'].round(3)
    # out_df['classic_fde'] = boosted_df.loc[:, 'fde'].round(3)
    # out_df['nn_fde'] = nn_df.loc[:, 'fde'].round(3)


    # out_df['neighbourhood_radius'] = nn_df['neighbourhood_radius']

    # out_df.to_latex(f'{save_root_path}final_traj_metrics_for_all_steps.tex', index=False)
    out_df.to_latex(f'{save_root_path}final_detection_metrics_for_all_steps.tex', index=False)
    out_df.to_csv(f'{save_root_path}final_metrics_for_all_steps.csv', index=False)

    if dump_as_markdown:
        save_as_markdown(
            out_df,
            path=f'{save_root_path}final_metrics_for_all_steps.md')
    else:
        save_as_tex(
            out_df,
            path=f'{save_root_path}final_metrics_for_all_steps.tex')


def save_as_markdown(out_df, path):
    series_list = []
    for r_idx, row in out_df.iterrows():
        row_numpy = row.values

        precision_winner = np.argmax(row_numpy[2:5])
        row[2 + precision_winner] = f"**{row[2 + precision_winner]}**"

        recall_winner = np.argmax(row_numpy[5:8])
        row[5 + recall_winner] = f"**{row[5 + recall_winner]}**"

        ade_winner = np.argmin(row_numpy[8:11])
        row[8 + ade_winner] = f"**{row[8 + ade_winner]}**"

        fde_winner = np.argmin(row_numpy[11:14])
        row[11 + fde_winner] = f"**{row[11 + fde_winner]}**"

        series_list.append(row.values)

    out_df_bold = pd.DataFrame(series_list, columns=out_df.columns.values)
    out_df_bold.to_markdown(
        path,
        index=False)


def save_as_tex(out_df, path):
    series_list = []
    for r_idx, row in out_df.iterrows():
        row_numpy = row.values

        precision_winner = np.argmax(row_numpy[2:5])
        row[2 + precision_winner] = f"\textbf{{{row[2 + precision_winner]}}}"

        recall_winner = np.argmax(row_numpy[5:8])
        row[5 + recall_winner] = f"\textbf{{{row[5 + recall_winner]}}}"

        ade_winner = np.argmin(row_numpy[8:11])
        row[8 + ade_winner] = f"\textbf{{{row[8 + ade_winner]}}}"

        fde_winner = np.argmin(row_numpy[11:14])
        row[11 + fde_winner] = f"\textbf{{{row[11 + fde_winner]}}}"

        series_list.append(row.values)

    out_df_bold = pd.DataFrame(series_list, columns=out_df.columns.values)

    out_df_bold.to_latex(
        path,
        index=False, multirow=True)


def pm_annotation_versions_diff_csv(
        version0_path, version1_path, save_root_path, dump_as_markdown=True):
    version0_df = pd.read_csv(version0_path)
    version1_df = pd.read_csv(version1_path)

    out_df = version0_df[['class', 'number']]
    out_df['class'] = version0_df.loc[:, 'class'].str.lower()


    out_df['v0_precision'] = version0_df.loc[:, 'precision'].round(3)
    out_df['v1_precision'] = version1_df.loc[:, 'precision'].round(3)

    out_df['v0_recall'] = version0_df.loc[:, 'recall'].round(3)
    out_df['v1_recall'] = version1_df.loc[:, 'recall'].round(3)

    # out_df['v0_ade'] = version0_df.loc[:, 'ade'].round(3)
    # out_df['v1_ade'] = version1_df.loc[:, 'ade'].round(3)
    #
    # out_df['v0_fde'] = version0_df.loc[:, 'fde'].round(3)
    # out_df['v1_fde'] = version1_df.loc[:, 'fde'].round(3)

    # out_df['neighbourhood_radius'] = version0_df['neighbourhood_radius']

    # out_df.to_latex(f'{save_root_path}final_final_approach_traj.tex', index=False)
    out_df.to_latex(f'{save_root_path}final_final_approach_detections.tex', index=False)

    if dump_as_markdown:
        pm_annotation_versions_diff_save_as_markdown(
            out_df,
            path=f'{save_root_path}pm_annotation_versions_diff.md')
    else:
        return NotImplemented
        save_as_tex(
            out_df,
            path=f'{save_root_path}pm_annotation_versions_diff.tex')


def pm_annotation_versions_diff_save_as_markdown(out_df, path):
    series_list = []
    for r_idx, row in out_df.iterrows():
        row_numpy = row.values

        precision_winner = np.argmax(row_numpy[2:4])
        row[2 + precision_winner] = f"**{row[2 + precision_winner]}**"

        recall_winner = np.argmax(row_numpy[4:6])
        row[4 + recall_winner] = f"**{row[4 + recall_winner]}**"

        ade_winner = np.argmin(row_numpy[6:8])
        row[6 + ade_winner] = f"**{row[6 + ade_winner]}**"

        fde_winner = np.argmin(row_numpy[8:10])
        row[8 + fde_winner] = f"**{row[8 + fde_winner]}**"

        series_list.append(row.values)

    out_df_bold = pd.DataFrame(series_list, columns=out_df.columns.values)
    out_df_bold.to_markdown(
        path,
        index=False)


def load_df_and_diff_save_as_markdown_for_precision_recall(out_df_path, path):
    out_df = pd.read_csv(out_df_path)
    series_list = []
    for r_idx, row in out_df.iterrows():
        row_numpy = row.values

        precision_winner = np.argmax(row_numpy[2:4])
        row[2 + precision_winner] = f"**{row[2 + precision_winner]}**"

        recall_winner = np.argmax(row_numpy[4:6])
        row[4 + recall_winner] = f"**{row[4 + recall_winner]}**"
        series_list.append(row.values)

    out_df_bold = pd.DataFrame(series_list, columns=out_df.columns.values)
    out_df_bold.to_markdown(
        path,
        index=False)


def sdd_stats_from_website():
    columns = ['Scenes', 'Videos', 'Bicyclist', 'Pedestrian', 'Skateboarder', 'Cart', 'Car', 'Bus']
    scenes = ['gates', 'little', 'nexus', 'coupa', 'bookstore', 'deathCircle', 'quad', 'hyang']
    videos = [9, 4, 12, 4, 7, 5, 4, 15]
    bicyclist = [51.94, 56.04, 4.22, 18.89, 32.89, 56.30, 12.50, 27.68]
    pedes = [43.36, 42.46, 64.02, 80.61, 63.94, 33.13, 87.50, 70.01]
    skateb = [2.55, 0.67, 0.60, 0.17, 1.63, 2.33, 0, 1.29]
    cart = [0.29, 0, 0.40, 0.17, 0.34, 3.10, 0, 0.43]
    car = [1.08, 0.17, 29.51, 0.17, 0.83, 4.71, 0, 0.50]
    bus = [0.78, 0.67, 1.25, 0, 0.37, 0.42, 0, 0.09]

    df = pd.DataFrame.from_dict(
        {
            columns[0]: scenes,
            columns[1]: videos,
            columns[2]: bicyclist,
            columns[3]: pedes,
            columns[4]: skateb,
            columns[5]: cart,
            columns[6]: car,
            columns[7]: bus,
        }
    )
    df.to_latex('/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/sdd_stats.tex')


def compare_crop_classifiers():
    old_path = "../../Datasets/SDD/classic_nn_extracted_annotations_v1/metrics_2m.csv"
    new_path = "../../Datasets/SDD/classic_nn_extracted_annotations_new_v1/metrics_2m.csv"

    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    out_df = old_df[['class', 'number']]

    out_df['classic_precision'] = old_df.loc[:, 'classic_precision']
    out_df['nn_precision_v0'] = old_df.loc[:, 'classic_nn_precision']
    out_df['nn_precision_v1'] = new_df.loc[:, 'classic_nn_precision']

    out_df['p_diff_v0_v1'] = old_df.loc[:, 'classic_nn_precision'] - new_df.loc[:, 'classic_nn_precision']
    out_df['p_diff_classic_v0'] = old_df.loc[:, 'classic_precision'] - old_df.loc[:, 'classic_nn_precision']
    out_df['p_diff_classic_v1'] = old_df.loc[:, 'classic_precision'] - new_df.loc[:, 'classic_nn_precision']

    out_df['classic_recall'] = old_df.loc[:, 'classic_recall']
    out_df['nn_recall_v0'] = old_df.loc[:, 'classic_nn_recall']
    out_df['nn_recall_v1'] = new_df.loc[:, 'classic_nn_recall']

    out_df['r_diff_v0_v1'] = old_df.loc[:, 'classic_nn_recall'] - new_df.loc[:, 'classic_nn_recall']
    out_df['r_diff_classic_v0'] = old_df.loc[:, 'classic_recall'] - old_df.loc[:, 'classic_nn_recall']
    out_df['r_diff_classic_v1'] = old_df.loc[:, 'classic_recall'] - new_df.loc[:, 'classic_nn_recall']

    out_df['neighbourhood_radius'] = old_df.loc[:, 'neighbourhood_radius']

    series_list = []
    for r_idx, row in out_df.iterrows():
        row_numpy = row.values

        precision_winner = np.argmax(row_numpy[3:5])
        row[3 + precision_winner] = f"**{row[3 + precision_winner]}**"

        # recall_winner = np.argmax(row_numpy[6:8])
        # row[6 + recall_winner] = f"**{row[6 + recall_winner]}**"
        recall_winner = np.argmax(row_numpy[9:11])
        row[9 + recall_winner] = f"**{row[9 + recall_winner]}**"
        series_list.append(row.values)

    out_df_bold = pd.DataFrame(series_list, columns=out_df.columns.values)
    out_df_bold.to_markdown(
        "../../Datasets/SDD/classic_nn_extracted_annotations_new_v1/metrics_2m_extended.md",
        index=False)


def plot_random_legends():
    blue_star = lines.Line2D([], [], color='white', marker='*', linestyle='None',
                             markersize=10, label='Cluster\'s Centroid', markeredgecolor='black')
    red_circle = lines.Line2D([], [], color='red', marker='o', linestyle='None',
                              markersize=10, label='Region of Validity', markerfacecolor='white')
    purple_triangle = lines.Line2D([], [], color='green', marker='o', linestyle='None',
                                   markersize=10, label='Cluster Pool Region', markerfacecolor='white')
    purple_triangle0 = lines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                                    markersize=10, label='Inside Region of Validity')
    purple_triangle1 = lines.Line2D([], [], color='aqua', marker='o', linestyle='None',
                                    markersize=10, label='New track candidate activations')
    # red_square = lines.Line2D([], [], color='red', marker='s', linestyle='None',
    #                           markersize=10, label='Ground Truth Bounding Box', markerfacecolor='white')
    # aqua_square = lines.Line2D([], [], color='aqua', marker='s', linestyle='None',
    #                           markersize=10, label='Generated Bounding Box', markerfacecolor='white')

    # blue_square = lines.Line2D([], [], color='blue', marker='s', linestyle='None',
    #                           markersize=10, label='Points at (T-1)', markerfacecolor='white')
    # magenta_square = lines.Line2D([], [], color='magenta', marker='s', linestyle='None',
    #                           markersize=10, label='Points at T', markerfacecolor='white')
    # red_square = lines.Line2D([], [], color='red', marker='s', linestyle='None',
    #                           markersize=10, label='(T-1) Shifted Bounding Box', markerfacecolor='white')
    # gray_square = lines.Line2D([], [], color='gray', marker='s', linestyle='None',
    #                           markersize=10, label='Selected XY', markerfacecolor='white')
    # yellow_square = lines.Line2D([], [], color='yellow', marker='s', linestyle='None',
    #                           markersize=10, label='Selected XY Current', markerfacecolor='white')
    # green_square = lines.Line2D([], [], color='green', marker='s', linestyle='None',
    #                           markersize=10, label='(T-1) OF Shifted Bounding Box', markerfacecolor='white')

    # red_square = lines.Line2D([], [], color='red', marker='s', linestyle='None',
    #                           markersize=10, label='Pedestrian', markerfacecolor='white')
    # green_square = lines.Line2D([], [], color='green', marker='s', linestyle='None',
    #                           markersize=10, label='Bus', markerfacecolor='white')
    # black_square = lines.Line2D([], [], color='black', marker='s', linestyle='None',
    #                           markersize=10, label='Biker', markerfacecolor='white')
    # white_square = lines.Line2D([], [], color='white', marker='s', linestyle='None',
    #                           markersize=10, label='Car', markerfacecolor='white')
    # purple_square = lines.Line2D([], [], color='purple', marker='s', linestyle='None',
    #                           markersize=10, label='Skater', markerfacecolor='white')
    # aqua_square = lines.Line2D([], [], color='aqua', marker='s', linestyle='None',
    #                           markersize=10, label='Cart', markerfacecolor='white')

    blue_circle = lines.Line2D([], [], color='blue', marker='s', linestyle='None',
                              markersize=10, label='Observed', markerfacecolor='white')
    green_circle = lines.Line2D([], [], color='green', marker='s', linestyle='None',
                              markersize=10, label='Predicted', markerfacecolor='white')
    red_circle = lines.Line2D([], [], color='red', marker='s', linestyle='None',
                              markersize=10, label='Ground Truth', markerfacecolor='white')
    # aqua_circle = lines.Line2D([], [], color='aqua', marker='o', linestyle='None',
    #                           markersize=10, label='Last Observed Position', markerfacecolor='white')

    # plt.legend(handles=[blue_star, red_circle, purple_triangle, purple_triangle0, purple_triangle1, red_square])
    plt.legend(handles=[blue_star, red_circle, purple_triangle, purple_triangle0, purple_triangle1])

    plt.show()
    print()
    # fig, axs = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(12, 10))
    # legends_dict = {'*': 'Cluster\'s Centroid',
    #                 'g': 'Cluster Pool Region',
    #                 'r': 'Ground Truth Bounding Box'}
    #
    # legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    # fig.legend(handles=legend_patches, loc=2)
    # plt.show()


if __name__ == '__main__':
    # sdd_stats_from_website()
    # plot_random_legends()
    # d_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/generated_annotations/metrics_2m.csv'
    # b_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/' \
    #          'filtered_generated_annotations/metrics_2m.csv'
    # n_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/' \
    #          'SDD/classic_nn_extended_annotations_v1/d2/simple/metrics_2m.csv'
    # csv_merge(d_path, b_path, n_path,
    #           save_root_path='/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/',
    #           dump_as_markdown=True)

    v0_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/' \
             'pm_extracted_annotations_v0/metrics.csv'
    v1_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/' \
             'pm_extracted_annotations/metrics_2m.csv'
    pm_annotation_versions_diff_csv(v0_path, v1_path,
                                    '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/')

    compare_crop_classifiers()
    load_df_and_diff_save_as_markdown_for_precision_recall(
        '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD'
        '/classic_nn_extracted_annotations_v1/metrics_2m.csv',
        '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/classic_nn_extracted_annotations_v1/'
        'metrics_2m.md'
    )
