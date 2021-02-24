import torch

from average_image.constants import SDDVideoDatasets, SDDVideoClasses
from average_image.utils import SDDMeta
from baselinev2.constants import STEP

SEVER_MODE = False

LOCAL_PATH = "../"
SERVER_PATH = "/usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/"

if SEVER_MODE:
    ROOT_PATH = SERVER_PATH
else:
    ROOT_PATH = LOCAL_PATH

CLUSTERING_TIMEOUT = 90
BATCH_CHECKPOINT = 50
RESUME_MODE = False
TIMEOUT_MODE = True
CSV_MODE = False

META_PATH = f'{ROOT_PATH}Datasets/SDD/H_SDD.txt'
DATASET_META = SDDMeta(META_PATH)
META_LABEL = SDDVideoDatasets.NEXUS

VIDEO_LABEL = SDDVideoClasses.NEXUS
VIDEO_NUMBER = 3

SAVE_BASE_PATH = f"{ROOT_PATH}Datasets/SDD_Features/"

BASE_PATH = f"{ROOT_PATH}Datasets/SDD/"

EXECUTE_STEP = STEP.GENERATE_ANNOTATIONS

version = 0
video_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/zero_shot/'
plot_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/'
features_save_path = f'{ROOT_PATH}Plots/baseline_v2/v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/'
GENERATED_DATASET_ROOT = f'{ROOT_PATH}Plots/baseline_v2/v{version}/'

CLASSIC_MODE = True
NN_FOLDER = 'nn_v0/'
BASELINE_FOLDER = 'baseline_v2/'

FOLDER_NAME = BASELINE_FOLDER if CLASSIC_MODE else NN_FOLDER

SAVE_PATH = f'{SAVE_BASE_PATH}{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/{FOLDER_NAME}'
VIDEO_SAVE_PATH = f'{ROOT_PATH}Plots/{FOLDER_NAME}v{version}/{VIDEO_LABEL.value}{VIDEO_NUMBER}/'

ANNOTATION_BASE_PATH = f'{BASE_PATH}annotations/{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/'
VIDEO_BASE_PATH = f'{BASE_PATH}videos/{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/'

VIDEO_PATH = f'{VIDEO_BASE_PATH}video.mov'

ANNOTATION_TXT_PATH = f'{ANNOTATION_BASE_PATH}annotations.txt'
ANNOTATION_CSV_PATH = f'{ANNOTATION_BASE_PATH}annotation_augmented.csv'

SPLIT_ANNOTATION_SAVE_PATH = f'{SAVE_BASE_PATH}{VIDEO_LABEL.value}/video{VIDEO_NUMBER}/splits/'

SDD_VIDEO_ROOT_PATH = f'{BASE_PATH}videos/'
SDD_ANNOTATIONS_ROOT_PATH = f'{BASE_PATH}annotations/'

SDD_VIDEO_CLASSES_LIST = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
                          SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS,
                          SDDVideoClasses.QUAD]
SDD_PER_CLASS_VIDEOS_LIST = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8],
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3],
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3]]

SDD_VIDEO_META_CLASSES_LIST_FOR_NN = [SDDVideoDatasets.BOOKSTORE, SDDVideoDatasets.COUPA, SDDVideoDatasets.DEATH_CIRCLE,
                                      SDDVideoDatasets.GATES, SDDVideoDatasets.HYANG, SDDVideoDatasets.LITTLE,
                                      SDDVideoDatasets.NEXUS, SDDVideoDatasets.QUAD]

SDD_VIDEO_CLASSES_LIST_FOR_NN = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
                                 SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE,
                                 SDDVideoClasses.NEXUS, SDDVideoClasses.QUAD]

SDD_PER_CLASS_VIDEOS_LIST_FOR_NN = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3]]

# SDD_VIDEO_CLASSES_RESUME_LIST = [SDDVideoClasses.DEATH_CIRCLE, SDDVideoClasses.GATES, SDDVideoClasses.HYANG,
#                                  SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS, SDDVideoClasses.QUAD]
# SDD_PER_CLASS_VIDEOS_RESUME_LIST = [[2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8],
#                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3],
#                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3]]

SDD_VIDEO_CLASSES_RESUME_LIST = [SDDVideoClasses.DEATH_CIRCLE,
                                 SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE,
                                 SDDVideoClasses.NEXUS,
                                 SDDVideoClasses.QUAD]
SDD_PER_CLASS_VIDEOS_RESUME_LIST = [[2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3]]

GENERATE_BUNDLED_ANNOTATIONS = False
BUNDLED_ANNOTATIONS_VIDEO_CLASSES_LIST = \
    [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE, SDDVideoClasses.GATES,
     SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS, SDDVideoClasses.QUAD]
BUNDLED_ANNOTATIONS_PER_CLASSES_VIDEO_LIST = \
    [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8],
     [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3], [0, 1, 2, 4, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3]]
# BUNDLED_ANNOTATIONS_VIDEO_CLASSES_LIST = [SDDVideoClasses.HYANG, SDDVideoClasses.NEXUS]
# BUNDLED_ANNOTATIONS_PER_CLASSES_VIDEO_LIST = [[0, 2], [3]]

# SDD_VIDEO_CLASSES_RESUME_LIST = [SDDVideoClasses.LITTLE]
# SDD_PER_CLASS_VIDEOS_RESUME_LIST = [[0]]

# NN Configs ###############################################################################################

TRAIN_SPLIT_PERCENTAGE = 0.6
VALIDATION_SPLIT_PERCENTAGE = 0.15
TEST_SPLIT_PERCENTAGE = 0.25

TIME_STEPS = 5
NUM_WORKERS = 12
BATCH_SIZE = 256
LR = 1e-3
NUM_EPOCHS = 500
OVERFIT = False

USE_BATCH_NORM = False
GT_BASED = False
CENTER_BASED = True
SAME_INIT = False

EVAL_VERSION = 2
EVAL_EPOCH = 2268
EVAL_STEP = 4537

OF_VERSION = 1
GT_VERSION = 0
OF_EPOCH = 363
GT_EPOCH = 88

MANUAL_SEED = 42
GENERATOR_SEED = torch.Generator().manual_seed(MANUAL_SEED)

# LINEAR_CFG = {
#     'encoder': [4, 8, 16, 32, 64],
#     'decoder': [64, 32, 16, 8, 4, 2],
#     'lstm_in': 64,
#     'lstm_encoder': 128
# }

# small network
# LINEAR_CFG = {
#     'encoder': [8],
#     'decoder': [2],
#     'lstm_in': 8,
#     'lstm_encoder': 16
# }
#
# LINEAR_CFG = {
#     'encoder': [32],
#     'decoder': [32, 2],
#     'lstm_in': 32,
#     'lstm_encoder': 64
# }

# Bigger
LINEAR_CFG = {
    'encoder': [32, 64],
    'decoder': [64, 32, 2],
    'lstm_in': 64,
    'lstm_encoder': 128
}

TRAIN_CLASS = SDDVideoClasses.LITTLE
VAL_CLASS = TRAIN_CLASS

TRAIN_VIDEO_NUMBER = 3
VAL_VIDEO_NUMBER = TRAIN_VIDEO_NUMBER

TRAIN_META = SDDVideoDatasets.LITTLE
VAL_META = TRAIN_META

USE_GENERATED_DATA = False  # Use unsupervised trajectories or not
RESUME_TRAINING = False
RESUME_VERSION = 9
CHECKPOINT_ROOT = f'lightning_logs/version_{RESUME_VERSION}/'

OVERFIT_BATCHES = 0.0
LIMIT_BATCHES = (1.0, 1.0)  # (Train, Val)

USE_SOCIAL_LSTM_MODEL = False
USE_FINAL_POSITIONS = False

USE_RELATIVE_VELOCITIES = False

TRAIN_CUSTOM = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LOG_HISTOGRAM = False

# Evaluation
DEBUG_MODE = False
PLOT_MODE = False

EVAL_USE_SOCIAL_LSTM_MODEL = True
EVAL_USE_BATCH_NORM = False

EVAL_USE_FINAL_POSITIONS_SUPERVISED = True
EVAL_USE_FINAL_POSITIONS_UNSUPERVISED = True

EVAL_BATCH_SIZE = 1 if PLOT_MODE else 512
EVAL_WORKERS = 0 if PLOT_MODE else 12
EVAL_SHUFFLE = True

EVAL_TRAIN_CLASS = SDDVideoClasses.LITTLE
EVAL_VAL_CLASS = EVAL_TRAIN_CLASS
EVAL_TEST_CLASS = EVAL_TRAIN_CLASS

EVAL_TRAIN_VIDEO_NUMBER = 3
EVAL_VAL_VIDEO_NUMBER = EVAL_TRAIN_VIDEO_NUMBER
EVAL_TEST_VIDEO_NUMBER = EVAL_TRAIN_VIDEO_NUMBER

EVAL_TRAIN_META = SDDVideoDatasets.LITTLE
EVAL_VAL_META = EVAL_TRAIN_META
EVAL_TEST_META = EVAL_TRAIN_META

GT_CHECKPOINT_VERSION = 14
GT_CHECKPOINT_ROOT_PATH = f'lightning_logs/version_{GT_CHECKPOINT_VERSION}/'

UNSUPERVISED_CHECKPOINT_VERSION = 15
UNSUPERVISED_CHECKPOINT_ROOT_PATH = f'lightning_logs/version_{UNSUPERVISED_CHECKPOINT_VERSION}/'

EVAL_PATH_TO_VIDEO = f'{BASE_PATH}videos/{EVAL_TRAIN_CLASS.value}/video{EVAL_TRAIN_VIDEO_NUMBER}/video.mov'
EVAL_PLOT_PATH = f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/{EVAL_TRAIN_CLASS.value}{EVAL_TRAIN_VIDEO_NUMBER}/' \
                     f'final_eval/gt_{GT_CHECKPOINT_VERSION}_unsupervised_{UNSUPERVISED_CHECKPOINT_VERSION}/'
