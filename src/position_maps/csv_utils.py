import numpy as np
import pandas as pd


def csv_merge(default_path, boosted_path, nn_path, save_root_path, dump_as_markdown=True):
    default_df = pd.read_csv(default_path)
    boosted_df = pd.read_csv(boosted_path)
    nn_df = pd.read_csv(nn_path)

    out_df = nn_df[['class', 'number']]

    out_df['precision'] = default_df.loc[:, 'precision']
    out_df['classifier_precision'] = boosted_df.loc[:, 'precision']
    out_df['nn_precision'] = nn_df.loc[:, 'precision']

    out_df['recall'] = default_df.loc[:, 'recall']
    out_df['classifier_recall'] = boosted_df.loc[:, 'recall']
    out_df['nn_recall'] = nn_df.loc[:, 'recall']

    out_df['ade'] = default_df.loc[:, 'ade']
    out_df['classifier_ade'] = boosted_df.loc[:, 'ade']
    out_df['nn_ade'] = nn_df.loc[:, 'ade']

    out_df['fde'] = default_df.loc[:, 'fde']
    out_df['classifier_fde'] = boosted_df.loc[:, 'fde']
    out_df['nn_fde'] = nn_df.loc[:, 'fde']

    out_df['neighbourhood_radius'] = nn_df['neighbourhood_radius']

    if dump_as_markdown:
        save_as_markdown(
            out_df,
            path=f'{save_root_path}metrics_for_all_steps.md')
    else:
        save_as_tex(
            out_df,
            path=f'{save_root_path}metrics_for_all_steps.tex')


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


if __name__ == '__main__':
    d_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/generated_annotations/metrics_2m.csv'
    b_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/' \
             'filtered_generated_annotations/metrics_2m.csv'
    n_path = '/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/pm_extracted_annotations/metrics.csv'
    csv_merge(d_path, b_path, n_path,
              save_root_path='/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/')
