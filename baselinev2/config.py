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

BATCH_CHECKPOINT = 50
RESUME_MODE = False
CSV_MODE = False

META_PATH = f'{ROOT_PATH}Datasets/SDD/H_SDD.txt'
DATASET_META = SDDMeta(META_PATH)
META_LABEL = SDDVideoDatasets.DEATH_CIRCLE

VIDEO_LABEL = SDDVideoClasses.DEATH_CIRCLE
VIDEO_NUMBER = 4

SAVE_BASE_PATH = f"{ROOT_PATH}Datasets/SDD_Features/"

BASE_PATH = f"{ROOT_PATH}Datasets/SDD/"

EXECUTE_STEP = STEP.MINIMAL

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

# NN Configs ###############################################################################################

TRAIN_SPLIT_PERCENTAGE = 0.5
VALIDATION_SPLIT_PERCENTAGE = 0.2
TEST_SPLIT_PERCENTAGE = 0.3

TIME_STEPS = 5
NUM_WORKERS = 10
BATCH_SIZE = 256
LR = 1e-3

USE_BATCH_NORM = False
GT_BASED = False
CENTER_BASED = True
SAME_INIT = False

OF_VERSION = 1
GT_VERSION = 0
OF_EPOCH = 363
GT_EPOCH = 88

MANUAL_SEED = 42
GENERATOR_SEED = torch.Generator().manual_seed(MANUAL_SEED)

LINEAR_CFG = {
    'encoder': [4, 8, 16],
    'decoder': [32, 16, 8, 4, 2]
}
