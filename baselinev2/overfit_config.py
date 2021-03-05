import torch

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.constants import SDDVideoClassAndNumbers

ROOT_PATH = "../"
BASE_PATH = f"{ROOT_PATH}Datasets/SDD/"


TRAIN_SPLIT_PERCENTAGE = 0.6
VALIDATION_SPLIT_PERCENTAGE = 0.15
TEST_SPLIT_PERCENTAGE = 0.25

TIME_STEPS = 5
NUM_WORKERS = 12
BATCH_SIZE = 32  # 32  # 1024  # 2048
LR = 1e-3  # 1e-3  # 3e-2  # 0.0014462413062537917  # 2e-3
NUM_EPOCHS = 1000
OVERFIT = False
DO_VALIDATION = True

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
                         SDDVideoClassAndNumbers.HYANG]
VAL_CLASS = TRAIN_CLASS
VAL_CLASS_FOR_WHOLE = TRAIN_CLASS_FOR_WHOLE

TRAIN_VIDEO_NUMBER = 3
VAL_VIDEO_NUMBER = TRAIN_VIDEO_NUMBER

TRAIN_META = SDDVideoDatasets.LITTLE if not TRAIN_FOR_WHOLE_CLASS else \
    [SDDVideoDatasets.LITTLE, SDDVideoDatasets.DEATH_CIRCLE, SDDVideoDatasets.HYANG]
VAL_META = TRAIN_META

TRAIN_VIDEOS_TO_SKIP = [(), (), ()]
VAL_VIDEOS_TO_SKIP = [(), (), ()]

USE_GENERATED_DATA = True  # Use unsupervised trajectories or not
RESUME_TRAINING = False
RESUME_VERSION = 9
CHECKPOINT_ROOT = f'lightning_logs/version_{RESUME_VERSION}/'

OVERFIT_BATCHES = 0.0
LIMIT_BATCHES = (1.0, 1.0)  # (Train, Val)
# Custom Solver
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 500
AMS_GRAD = True

OVERFIT_ELEMENT_COUNT = None  # 6144
RANDOM_INDICES_IN_OVERFIT_ELEMENTS = True
# BATCH_SIZE = OVERFIT_ELEMENT_COUNT if OVERFIT_ELEMENT_COUNT is None else 32  # BATCH_SIZE

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
