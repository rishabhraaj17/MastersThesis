import torch

from average_image.constants import SDDVideoDatasets, SDDVideoClasses
from average_image.utils import SDDMeta
from baselinev2.constants import STEP, SDDVideoClassAndNumbers

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

SDD_VIDEO_CLASSES_RESUME_LIST = [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE,
                                 SDDVideoClasses.GATES, SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE,
                                 SDDVideoClasses.NEXUS,
                                 SDDVideoClasses.QUAD]
SDD_PER_CLASS_VIDEOS_RESUME_LIST = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3]]

GENERATE_BUNDLED_ANNOTATIONS = False
# BUNDLED_ANNOTATIONS_VIDEO_CLASSES_LIST = \
#     [SDDVideoClasses.BOOKSTORE, SDDVideoClasses.COUPA, SDDVideoClasses.DEATH_CIRCLE, SDDVideoClasses.GATES,
#      SDDVideoClasses.HYANG, SDDVideoClasses.LITTLE, SDDVideoClasses.NEXUS, SDDVideoClasses.QUAD]
# BUNDLED_ANNOTATIONS_PER_CLASSES_VIDEO_LIST = \
#     [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8],
#      [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3], [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3]]
BUNDLED_ANNOTATIONS_VIDEO_CLASSES_LIST = \
    [SDDVideoClasses.NEXUS]
BUNDLED_ANNOTATIONS_PER_CLASSES_VIDEO_LIST = \
    [[3]]
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
BATCH_SIZE = 32  # 4096  # * 2  # 32  # 1024  # 2048
LR = 1e-3  # 0.0014462413062537917  # 2e-3
NUM_EPOCHS = 1000
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
# smaller
# LINEAR_CFG = {
#     'encoder': [4],
#     'decoder': [2],
#     'lstm_in': 4,
#     'lstm_encoder': 8
# }
#
# LINEAR_CFG = {
#     'encoder': [32],
#     'decoder': [32, 2],
#     'lstm_in': 32,
#     'lstm_encoder': 64
# }
LINEAR_CFG = {
    'encoder': [8, 16, 32],
    'decoder': [32, 16, 2],
    'lstm_in': 32,
    'lstm_encoder': 64
}

# LINEAR_CFG = {
#     'encoder': [8, 16, 32, 64],
#     'decoder': [64, 32, 8, 2],
#     'lstm_in': 64,
#     'lstm_encoder': 128
# }

# # Bigger
# LINEAR_CFG = {
#     'encoder': [32, 64],
#     'decoder': [64, 32, 2],
#     'lstm_in': 64,
#     'lstm_encoder': 128
# }

TRAIN_FOR_WHOLE_CLASS = True

TRAIN_CLASS = SDDVideoClasses.LITTLE
TRAIN_CLASS_FOR_WHOLE = [SDDVideoClassAndNumbers.LITTLE, SDDVideoClassAndNumbers.DEATH_CIRCLE,
                         SDDVideoClassAndNumbers.BOOKSTORE, SDDVideoClassAndNumbers.COUPA,
                         SDDVideoClassAndNumbers.GATES, SDDVideoClassAndNumbers.HYANG,
                         SDDVideoClassAndNumbers.NEXUS, SDDVideoClassAndNumbers.QUAD]
VAL_CLASS = TRAIN_CLASS
VAL_CLASS_FOR_WHOLE = [SDDVideoClassAndNumbers.LITTLE, SDDVideoClassAndNumbers.DEATH_CIRCLE,
                       SDDVideoClassAndNumbers.HYANG,
                       SDDVideoClassAndNumbers.BOOKSTORE, SDDVideoClassAndNumbers.COUPA,
                       SDDVideoClassAndNumbers.GATES, SDDVideoClassAndNumbers.NEXUS, SDDVideoClassAndNumbers.QUAD]

TRAIN_VIDEO_NUMBER = 3
VAL_VIDEO_NUMBER = TRAIN_VIDEO_NUMBER

TRAIN_META = SDDVideoDatasets.LITTLE if not TRAIN_FOR_WHOLE_CLASS else \
    [SDDVideoDatasets.LITTLE, SDDVideoDatasets.DEATH_CIRCLE,
     SDDVideoDatasets.BOOKSTORE, SDDVideoDatasets.COUPA,
     SDDVideoDatasets.GATES, SDDVideoDatasets.HYANG,
     SDDVideoDatasets.NEXUS, SDDVideoDatasets.QUAD]
VAL_META = SDDVideoDatasets.LITTLE if not TRAIN_FOR_WHOLE_CLASS else \
    [SDDVideoDatasets.LITTLE, SDDVideoDatasets.DEATH_CIRCLE, SDDVideoDatasets.HYANG,
     SDDVideoDatasets.BOOKSTORE, SDDVideoDatasets.COUPA,
     SDDVideoDatasets.GATES, SDDVideoDatasets.NEXUS, SDDVideoDatasets.QUAD
     ]

TRAIN_VIDEOS_TO_SKIP = [(), (), (), (), (), (), (), ()]
VAL_VIDEOS_TO_SKIP = [(), (), (), (), (), (), (), ()]

USE_GENERATED_DATA = True  # Use unsupervised trajectories or not
RESUME_TRAINING = False
RESUME_VERSION = 9
CHECKPOINT_ROOT = f'lightning_logs/version_{RESUME_VERSION}/'

OVERFIT_BATCHES = 0.0
LIMIT_BATCHES = (1.0, 1.0)  # (Train, Val)
# Custom Solver
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 300
AMS_GRAD = True

OVERFIT_ELEMENT_COUNT = None
RANDOM_INDICES_IN_OVERFIT_ELEMENTS = True
BATCH_SIZE = OVERFIT_ELEMENT_COUNT if OVERFIT_ELEMENT_COUNT is not None else BATCH_SIZE

USE_SOCIAL_LSTM_MODEL = False
USE_SIMPLE_MODEL = True
USE_GRU = False
USE_FINAL_POSITIONS = False

DROPOUT = None
RNN_DROPOUT = 0
RNN_LAYERS = 1
LEARN_HIDDEN_STATES = False
FEED_MODEL_DISTANCES_IN_METERS = False

USE_RELATIVE_VELOCITIES = False

TRAIN_CUSTOM = True
RESUME_CUSTOM_TRAINING_PATH = None
RESUME_CUSTOM_HPARAM_PATH = f'runs/Feb26_00-37-11_rishabh-Precision-5540baseline/' \
                            f'Feb26_00-37-11_rishabh-Precision-5540baseline_hparams.yaml'
RESUME_ADDITIONAL_EPOCH = 1000
RESUME_FROM_LAST_EPOCH = True

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LOG_HISTOGRAM = False

# Evaluation
DEBUG_MODE = False
EVAL_SINGLE_MODEL = True

PLOT_MODE = False
BEST_MODEL = True  # if False then last epoch model

EVAL_USE_SOCIAL_LSTM_MODEL = False
EVAL_USE_SIMPLE_MODEL = True
EVAL_USE_BATCH_NORM = False

EVAL_USE_FINAL_POSITIONS_SUPERVISED = True
EVAL_USE_FINAL_POSITIONS_UNSUPERVISED = True

EVAL_BATCH_SIZE = 1 if PLOT_MODE else 32
EVAL_WORKERS = 0 if PLOT_MODE else 12
EVAL_SHUFFLE = True

EVAL_FOR_WHOLE_CLASS = False

EVAL_TRAIN_CLASS = SDDVideoClasses.LITTLE if not EVAL_FOR_WHOLE_CLASS else \
    [SDDVideoClassAndNumbers.LITTLE, SDDVideoClassAndNumbers.DEATH_CIRCLE,
     SDDVideoClassAndNumbers.BOOKSTORE, SDDVideoClassAndNumbers.COUPA,
     SDDVideoClassAndNumbers.GATES, SDDVideoClassAndNumbers.HYANG,
     SDDVideoClassAndNumbers.NEXUS, SDDVideoClassAndNumbers.QUAD]
EVAL_VAL_CLASS = EVAL_TRAIN_CLASS
EVAL_TEST_CLASS = EVAL_TRAIN_CLASS

EVAL_TRAIN_VIDEO_NUMBER = 3
EVAL_VAL_VIDEO_NUMBER = EVAL_TRAIN_VIDEO_NUMBER
EVAL_TEST_VIDEO_NUMBER = EVAL_TRAIN_VIDEO_NUMBER

EVAL_TRAIN_META = SDDVideoDatasets.LITTLE if not EVAL_FOR_WHOLE_CLASS else \
    [SDDVideoDatasets.LITTLE, SDDVideoDatasets.DEATH_CIRCLE,
     SDDVideoDatasets.BOOKSTORE, SDDVideoDatasets.COUPA,
     SDDVideoDatasets.GATES, SDDVideoDatasets.HYANG,
     SDDVideoDatasets.NEXUS, SDDVideoDatasets.QUAD]
EVAL_VAL_META = EVAL_TRAIN_META
EVAL_TEST_META = EVAL_TRAIN_META

EVAL_TRAIN_VIDEOS_TO_SKIP = [(), (), (), (), (), (), (), ()]
EVAL_VAL_VIDEOS_TO_SKIP = [(), (), (), (), (), (), (), ()]
EVAL_TEST_VIDEOS_TO_SKIP = [(), (), (), (), (), (), (), ()]

GT_CHECKPOINT_VERSION = 14
GT_CHECKPOINT_ROOT_PATH = f'lightning_logs/version_{GT_CHECKPOINT_VERSION}/'

UNSUPERVISED_CHECKPOINT_VERSION = 15
UNSUPERVISED_CHECKPOINT_ROOT_PATH = f'lightning_logs/version_{UNSUPERVISED_CHECKPOINT_VERSION}/'

SIMPLE_GT_CHECKPOINT_ROOT_PATH = f'runs/Maar_overfit_experiments/full_train/'
SIMPLE_UNSUPERVISED_CHECKPOINT_ROOT_PATH = f'runs/Maar_overfit_experiments/full_train/'

SIMPLE_GT_CHECKPOINT_FILE_PATH = 'element_size_None_random_True_lr_0.001_generated_True_learn_hidden_False' \
                                 '_rnn_layers_1_2021-03-04 13:37:06.911715/element_size_None_random_True_' \
                                 'lr_0.001_generated_True_learn_hidden_False_rnn_layers_1_2021-03-04 13:37:06.911715' \
                                 '_checkpoint.ckpt'
SIMPLE_UNSUPERVISED_CHECKPOINT_FILE_PATH = 'element_size_None_random_True_lr_0.001_generated_True_learn_hidden_False' \
                                 '_rnn_layers_1_2021-03-04 13:37:06.911715/element_size_None_random_True_' \
                                 'lr_0.001_generated_True_learn_hidden_False_rnn_layers_1_2021-03-04 13:37:06.911715' \
                                 '_checkpoint.ckpt'

SINGLE_MODEL_CHECKPOINT_ROOT_PATH = f'runs/Maar_overfit_experiments/full_train/'
SINGLE_MODEL_CHECKPOINT_FILE_PATH = 'element_size_None_random_True_lr_0.001_generated_True_learn_hidden_False' \
                                 '_rnn_layers_1_2021-03-04 13:37:06.911715/element_size_None_random_True_' \
                                 'lr_0.001_generated_True_learn_hidden_False_rnn_layers_1_2021-03-04 13:37:06.911715' \
                                 '_checkpoint.ckpt'

if EVAL_SINGLE_MODEL and USE_SIMPLE_MODEL and not EVAL_FOR_WHOLE_CLASS:
    EVAL_PATH_TO_VIDEO = f'{BASE_PATH}videos/{EVAL_TRAIN_CLASS.value}/video{EVAL_TRAIN_VIDEO_NUMBER}/video.mov'
    EVAL_PLOT_PATH = f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/' \
                     f'{EVAL_TRAIN_CLASS.value}{EVAL_TRAIN_VIDEO_NUMBER}/final_eval/' \
                     f'single_model_{SIMPLE_GT_CHECKPOINT_FILE_PATH[-40:-16]}_'
elif EVAL_SINGLE_MODEL and (not USE_SIMPLE_MODEL or USE_SIMPLE_MODEL) and EVAL_FOR_WHOLE_CLASS:
    EVAL_PATH_TO_VIDEO = ''
    EVAL_PLOT_PATH = f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/' \
                     f'{"_".join([e.value[0].value for e in EVAL_TRAIN_CLASS])}/final_eval/' \
                     f'single_model_{SIMPLE_GT_CHECKPOINT_FILE_PATH[-40:-16]}_'
elif USE_SIMPLE_MODEL and EVAL_FOR_WHOLE_CLASS:
    EVAL_PATH_TO_VIDEO = ''
    EVAL_PLOT_PATH = f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/' \
                     f'{"_".join([e.value[0].value for e in EVAL_TRAIN_CLASS])}/final_eval/' \
                     f'supervised_{SIMPLE_GT_CHECKPOINT_FILE_PATH[-40:-16]}_' \
                     f'unsupervised_{SIMPLE_UNSUPERVISED_CHECKPOINT_FILE_PATH[-40:-16]}'
elif USE_SIMPLE_MODEL and not EVAL_FOR_WHOLE_CLASS:
    EVAL_PATH_TO_VIDEO = f'{BASE_PATH}videos/{EVAL_TRAIN_CLASS.value}/video{EVAL_TRAIN_VIDEO_NUMBER}/video.mov'
    EVAL_PLOT_PATH = f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/' \
                     f'{EVAL_TRAIN_CLASS.value}{EVAL_TRAIN_VIDEO_NUMBER}/final_eval/' \
                     f'supervised_{SIMPLE_GT_CHECKPOINT_FILE_PATH[-40:-16]}_' \
                     f'unsupervised_{SIMPLE_UNSUPERVISED_CHECKPOINT_FILE_PATH[-40:-16]}'
elif not USE_SIMPLE_MODEL and EVAL_FOR_WHOLE_CLASS:
    EVAL_PATH_TO_VIDEO = ''
    EVAL_PLOT_PATH = f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/' \
                     f'{"_".join([e.value[0].value for e in EVAL_TRAIN_CLASS])}/final_eval/' \
                     f'supervised_{SIMPLE_GT_CHECKPOINT_FILE_PATH[-40:-16]}_' \
                     f'unsupervised_{SIMPLE_UNSUPERVISED_CHECKPOINT_FILE_PATH[-40:-16]}'
else:
    EVAL_PATH_TO_VIDEO = f'{BASE_PATH}videos/{EVAL_TRAIN_CLASS.value}/video{EVAL_TRAIN_VIDEO_NUMBER}/video.mov'
    EVAL_PLOT_PATH = f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/{EVAL_TRAIN_CLASS.value}{EVAL_TRAIN_VIDEO_NUMBER}/' \
                     f'final_eval/gt_{GT_CHECKPOINT_VERSION}_unsupervised_{UNSUPERVISED_CHECKPOINT_VERSION}/'


# EVAL_PATH_TO_VIDEO = f'{BASE_PATH}videos/{EVAL_TRAIN_CLASS.value}/video{EVAL_TRAIN_VIDEO_NUMBER}/video.mov' \
#     if not EVAL_FOR_WHOLE_CLASS else ''
# EVAL_PLOT_PATH = f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/{EVAL_TRAIN_CLASS.value}{EVAL_TRAIN_VIDEO_NUMBER}/' \
#                  f'final_eval/gt_{GT_CHECKPOINT_VERSION}_unsupervised_{UNSUPERVISED_CHECKPOINT_VERSION}/' \
#     if not EVAL_FOR_WHOLE_CLASS else f'{ROOT_PATH}Plots/baseline_v2/nn/COMPARE/' \
#                                      f'{"_".join([e.value[0].value for e in EVAL_TRAIN_CLASS])}/final_eval/'


SIMPLE_GT_CHECKPOINT_PATH = SIMPLE_GT_CHECKPOINT_ROOT_PATH + SIMPLE_GT_CHECKPOINT_FILE_PATH
SIMPLE_UNSUPERVISED_CHECKPOINT_PATH = SIMPLE_UNSUPERVISED_CHECKPOINT_ROOT_PATH + \
                                      SIMPLE_UNSUPERVISED_CHECKPOINT_FILE_PATH
SINGLE_MODEL_CHECKPOINT_PATH = SINGLE_MODEL_CHECKPOINT_ROOT_PATH + SINGLE_MODEL_CHECKPOINT_FILE_PATH

EVAL_SIMPLE_MODEL_CONFIG_DICT_GT = {
    'arch_config': LINEAR_CFG,
    'batch_size': 1 if PLOT_MODE else 512,
    'use_batch_norm': False,
    'encoder_lstm_num_layers': 1,
    'decoder_lstm_num_layers': 1,
    'generated_dataset': False,
    'dropout': None,
    'rnn_dropout': 0,
    'use_gru': False,
    'learn_hidden_states': False,
    'feed_model_distances_in_meters': False,
    'relative_velocities': False
}

EVAL_SIMPLE_MODEL_CONFIG_DICT_UNSUPERVISED = {
    'arch_config': LINEAR_CFG,
    'batch_size': 1 if PLOT_MODE else 512,
    'use_batch_norm': False,
    'encoder_lstm_num_layers': 1,
    'decoder_lstm_num_layers': 1,
    'generated_dataset': True,
    'dropout': None,
    'rnn_dropout': 0,
    'use_gru': False,
    'learn_hidden_states': False,
    'feed_model_distances_in_meters': False,
    'relative_velocities': False
}
