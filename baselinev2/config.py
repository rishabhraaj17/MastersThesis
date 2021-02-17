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

USE_BATCH_NORM = True
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

LINEAR_CFG = {
    'encoder': [4, 8, 16, 32, 64],
    'decoder': [64, 32, 16, 8, 4, 2],
    'lstm_in': 64,
    'lstm_encoder': 128
}
