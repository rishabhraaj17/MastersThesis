import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import patches
from tqdm import tqdm

import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
from statsmodels import datasets as sm_datasets
import matplotlib
import matplotlib.pyplot as plt

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from average_image.utils import is_inside_bbox, compute_ade, compute_fde, \
    compute_per_stop_de, SDDMeta, plot_track_analysis, plot_violin_plot
from baseline.extract_features import extract_trainable_features_rnn, process_complex_features_rnn
from log import initialize_logging, get_logger
from baseline.extracted_of_optimization import cost_function
from unsupervised_tp_0.nn_clustering_0 import get_track_info

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

initialize_logging()
logger = get_logger(__name__)

SAVE_BASE_PATH = "../Datasets/SDD_Features/"
BASE_PATH = "../Datasets/SDD/"
VIDEO_LABEL = SDDVideoClasses.LITTLE
META_VIDEO_LABEL = SDDVideoDatasets.LITTLE
VIDEO_NUMBER = 3
SAVE_PATH = f'{SAVE_BASE_PATH}{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/'
# FILE_NAME = 'time_distributed_dict_with_gt_bbox_centers_and_bbox_gt_velocity.pt'
T_STEPS = 20
FILE_NAME = f'time_distributed_velocity_features_with_frame_track_rnn_bbox_gt_centers_and_bbox_' \
            f'center_based_gt_velocity_of_optimized_t{T_STEPS}.pt'
LOAD_FILE = SAVE_PATH + FILE_NAME
ANNOTATIONS_FILE = 'annotation_augmented.csv'
ANNOTATIONS_PATH = f'{BASE_PATH}annotations/{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/'
TIME_STEPS = 100
OPTIMIZED_OF = True
META_PATH = '../Datasets/SDD/H_SDD.txt'
META = SDDMeta(META_PATH)
DF_SAVE_PATH = f'{SAVE_PATH}analysis2/'
DF_FILE_NAME = f'optimized_of_analysis_t{TIME_STEPS}.csv' if OPTIMIZED_OF else f'analysis_t{TIME_STEPS}.csv'


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    # DISTRIBUTIONS = [
    #     st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
    #     st.cosine,
    #     st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife,
    #     st.fisk,
    #     st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
    #     st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
    #     st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma,
    #     st.invgauss,
    #     st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l,
    #     st.levy_stable,
    #     st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
    #     st.ncf,
    #     st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
    #     st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
    #     st.tukeylambda,
    #     st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    # ]

    DISTRIBUTIONS = [
        st.beta, st.betaprime, st.cauchy, st.chi, st.chi2,
        st.cosine,
        st.expon, st.exponnorm, st.exponweib, st.exponpow,
        st.frechet_r, st.frechet_l,
        st.gamma,
        st.invgauss,
        st.johnsonsb, st.johnsonsu, st.laplace,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm,
        st.pearson3, st.powerlaw, st.powerlognorm,
        st.tukeylambda,
        st.uniform
    ]

    # DISTRIBUTIONS = [getattr(st, distname) for distname in _distn_names]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in tqdm(DISTRIBUTIONS):

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    # end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return best_distribution.name, best_params


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def find_distribution(data, data_type):
    # Load data from statsmodels datasets
    # data = pd.Series(sm_datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
    # data = pd.Series(sm_datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
    data = pd.Series(data.ravel())

    # Plot for comparison
    plt.figure(figsize=(12, 8))
    # ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
    # ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color='b')
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5,
                   color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])
    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    best_dist = getattr(st, best_fit_name)

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(f'All Fitted Distributions - {data_type}')
    ax.set_xlabel(u'Velocity')
    ax.set_ylabel('Frequency')

    # Make PDF with best params
    pdf = make_pdf(best_dist, best_fit_params)

    # Display
    plt.figure(figsize=(12, 8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax.set_title(u'Best fit distribution \n' + dist_str)
    ax.set_xlabel(u'Velocity')
    ax.set_ylabel('Frequency')

    plt.show()


def plot_object_tracks(shifted_points_gt, true_points, frames, track_ids, center_shifted, center_true, bbox_shifted,
                       shifted_points_of, bbox_true, line_width=None, plot_save_path=None):
    rows = T_STEPS // 5
    k = 0
    fig, ax = plt.subplots(rows, 5, sharex='none', sharey='none', figsize=(24 * rows, 6 * rows))
    for r in range(rows):
        for c in range(5):
            if rows == 1:
                ax[c].plot(true_points[k][:, 0], true_points[k][:, 1], 'o', markerfacecolor='blue',
                           markeredgecolor='k', markersize=8)
                ax[c].plot(shifted_points_gt[k][:, 0], shifted_points_gt[k][:, 1], 'o', markerfacecolor='magenta',
                           markeredgecolor='k', markersize=8)
                ax[c].plot(shifted_points_of[k][:, 0], shifted_points_of[k][:, 1], 'o', markerfacecolor='pink',
                           markeredgecolor='k', markersize=8)
                ax[c].plot(center_true[k][0], center_true[k][1], '*', markerfacecolor='yellow',
                           markeredgecolor='k', markersize=9)
                ax[c].plot(center_shifted[k][0], center_shifted[k][1], '*', markerfacecolor='orange',
                           markeredgecolor='k', markersize=9)
                rect_true = patches.Rectangle(xy=(bbox_true[k][0], bbox_true[k][1]),
                                              width=bbox_true[k][2] - bbox_true[k][0],
                                              height=bbox_true[k][3] - bbox_true[k][1], fill=False,
                                              linewidth=line_width, edgecolor='g')
                rect_shifted = patches.Rectangle(xy=(bbox_shifted[k][0], bbox_shifted[k][1]),
                                                 width=bbox_shifted[k][2] - bbox_shifted[k][0],
                                                 height=bbox_shifted[k][3] - bbox_shifted[k][1], fill=False,
                                                 linewidth=line_width, edgecolor='r')
                ax[c].set_title(f'Frame: {frames[k]} | TrackId: {track_ids[k]}')
                ax[c].add_patch(rect_true)
                ax[c].add_patch(rect_shifted)
            else:
                ax[r, c].plot(true_points[k][:, 0], true_points[k][:, 1], 'o', markerfacecolor='blue',
                              markeredgecolor='k', markersize=8)
                ax[r, c].plot(shifted_points_gt[k][:, 0], shifted_points_gt[k][:, 1], 'o', markerfacecolor='magenta',
                              markeredgecolor='k', markersize=8)
                ax[r, c].plot(shifted_points_of[k][:, 0], shifted_points_of[k][:, 1], 'o', markerfacecolor='pink',
                              markeredgecolor='k', markersize=8)
                ax[r, c].plot(center_true[k][0], center_true[k][1], '*', markerfacecolor='yellow',
                              markeredgecolor='k', markersize=9)
                ax[r, c].plot(center_shifted[k][0], center_shifted[k][1], '*', markerfacecolor='orange',
                              markeredgecolor='k', markersize=9)
                rect_true = patches.Rectangle(xy=(bbox_true[k][0], bbox_true[k][1]),
                                              width=bbox_true[k][2] - bbox_true[k][0],
                                              height=bbox_true[k][3] - bbox_true[k][1], fill=False,
                                              linewidth=line_width, edgecolor='g')
                rect_shifted = patches.Rectangle(xy=(bbox_shifted[k][0], bbox_shifted[k][1]),
                                                 width=bbox_shifted[k][2] - bbox_shifted[k][0],
                                                 height=bbox_shifted[k][3] - bbox_shifted[k][1], fill=False,
                                                 linewidth=line_width, edgecolor='r')
                ax[r, c].set_title(f'Frame: {frames[k]} | TrackId: {track_ids[k]}')
                ax[r, c].add_patch(rect_true)
                ax[r, c].add_patch(rect_shifted)
            k += 1

    legends_dict = {'blue': 'Points at T',
                    'magenta': '(T-1) GT Shifted points at T',
                    'pink': '(T-1) OF Shifted points at T',
                    'g': 'T Bounding Box',
                    'r': 'T-1 Bounding Box',
                    'yellow': 'T Bbox Center',
                    'orange': 'T-1 Bbox Center'}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)
    fig.suptitle(f'Object Features movement')

    if plot_save_path is None:
        plt.show()
    else:
        Path(plot_save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_save_path + f'fig_tracks_{frames[0]}{"_".join(track_ids)}.png')
        plt.close()


def analyze_train_data_distribution(ratio=1.0, optimized_of=False, top_k=1, alpha=1,
                                    weight_points_inside_bbox_more=True):
    features = torch.load(LOAD_FILE)
    center_data_list = features['center_based']

    c_data_gt_vel = []
    c_data_of_vel = []
    for center_data in tqdm(center_data_list):
        c_data = np.array(center_data)
        try:
            of_val_data = c_data[:, 2]
            rolled = np.rollaxis(of_val_data, -1).tolist()
            of_data_x, of_data_y = rolled[1], rolled[0]
            of_data = np.stack([of_data_x, of_data_y]).T
            of_val_sign = np.sign(of_data)
            gt_data = c_data[:, 4] * of_val_sign
            of_data = of_data / 0.4
            c_data_gt_vel.append(gt_data)
            c_data_of_vel.append(of_data)
            # c_data_gt_vel.append(c_data[:, 4])
            # c_data_of_vel.append(c_data[:, 2] / 0.4)
        except IndexError:
            print(c_data.shape)

    c_data_gt_vel = filter(lambda x: x.shape[0] == 20, c_data_gt_vel)
    c_data_gt_vel = list(c_data_gt_vel)

    c_data_of_vel = filter(lambda x: x.shape[0] == 20, c_data_of_vel)
    c_data_of_vel = list(c_data_of_vel)

    c_data_gt_vel = np.stack(c_data_gt_vel)
    c_data_of_vel = np.stack(c_data_of_vel)

    find_distribution(c_data_gt_vel, 'Ground Truth')
    # find_distribution(c_data_of_vel, 'Optical Flow')


def analyze_train_data_via_plots(ratio=1.0, optimized_of=False, top_k=1, alpha=1, weight_points_inside_bbox_more=True):
    features = torch.load(LOAD_FILE)
    x_, y_, frames_, track_ids_, center_x_, center_y_, bbox_x_, bbox_y_, center_data_ = \
        features['x'], features['y'], features['frames'], features['track_ids'], features['bbox_center_x'], \
        features['bbox_center_y'], features['bbox_x'], features['bbox_y'], features['center_based']

    for x, y, frames, track_ids, center_x, center_y, bbox_x, bbox_y, center_data in tqdm(
            zip(x_, y_, frames_, track_ids_, center_x_, center_y_, bbox_x_, bbox_y_, center_data_),
            total=len(x_)
    ):
        shifted_points_gt = []
        shifted_points_of = []
        true_points = []
        for feat_x, feat_y, frame, track_id, cx, cy, bx, by, feat_center in zip(
                x, y, frames, track_ids, center_x, center_y, bbox_x, bbox_y, center_data
        ):
            center_xy, center_true_uv, center_past_uv, shifted_point, gt_past_velocity = \
                feat_center[0, :], feat_center[1, :], feat_center[2, :], feat_center[3, :], feat_center[4, :]
            gt_past_displacement = gt_past_velocity * 0.4
            # !!swap is necessary!!
            center_past_uv[0], center_past_uv[1] = center_past_uv[1], center_past_uv[0]
            # !!sign is wrong!!
            correct_sign = np.sign(center_past_uv)
            gt_past_displacement = gt_past_displacement * correct_sign
            shifted_points_gt.append(feat_x[:, :2] + np.expand_dims(gt_past_displacement, axis=0))
            shifted_points_of.append(feat_x[:, :2] + np.expand_dims(center_past_uv, axis=0))
            true_points.append(feat_y[:, :2])
        plot_object_tracks(shifted_points_gt=shifted_points_gt, true_points=true_points, frames=frames,
                           shifted_points_of=shifted_points_of, track_ids=track_ids, center_shifted=center_x,
                           center_true=center_y, bbox_shifted=bbox_x, bbox_true=bbox_y)


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
                                 bbox_center_y, bbox_x, bbox_y), total=len(x_)):
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
                _, of_flow_top_k = cost_function(of_flow, b_c_y, top_k=top_k, alpha=alpha, bbox=b_y,
                                                 weight_points_inside_bbox_more=weight_points_inside_bbox_more)
                of_flow_center = of_flow_top_k[0]
            else:
                of_flow_center = of_flow.mean(0)
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
    plot_folder = 'of_optimized_plots/' if OPTIMIZED_OF else 'plots/'
    # plot_folder = 'non_bbox_of_optimized_plots/' if OPTIMIZED_OF else 'plots/'
    for idx, (index, row) in enumerate(tqdm(in_df.iterrows(), total=len(in_df))):
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
                if idx % 999 == 0:
                    plot_track_analysis(t_id, ade, fde, inside_bbox, per_stop_de, save_path + plot_folder, idx)
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
    if OPTIMIZED_OF:
        save_path = save_path + 'of_optimized'
        # save_path = save_path + 'non_bbox_of_optimized'
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
    parsed_data = parse_df_analysis(df, DF_SAVE_PATH)
    return parsed_data


def main(save=True, optimized_of=True, top_k=1, alpha=1, weight_points_inside_bbox_more=True, from_file=True,
         process_both=False):
    if from_file:
        logger.info(f'Loading saved data - {DF_FILE_NAME} - for parsing!')
        df = pd.read_csv(DF_SAVE_PATH + DF_FILE_NAME)
        logger.info('Parsing ...')
        parsed_data = parse_analysis(df)
        logger.info("Analysis Completed!")
    else:
        logger.info('Computing ...')
        df = analyze(save=save, optimized_of=optimized_of, top_k=top_k, alpha=alpha,
                     weight_points_inside_bbox_more=weight_points_inside_bbox_more)
        if process_both:
            logger.info('Parsing ...')
            parsed_data = parse_analysis(df)
        logger.info("Computation Completed! Load file for parsing")


if __name__ == '__main__':
    # main(optimized_of=OPTIMIZED_OF, top_k=1, alpha=1, weight_points_inside_bbox_more=True, save=True,
    #      from_file=True, process_both=False)
    analyze_train_data_distribution()
    # analyze_train_data_via_plots()
