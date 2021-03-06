# encoding: utf-8
from yacs.config import CfgNode as CN
import multiprocessing

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.DEBUG = True
_C.SEED = 42
_C.VERBOSE = True
_C.MIXED_PRECISION = True

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.IMG_SIZE = 1024
# RandomSizedCrop paramters
_C.INPUT.RSC_MIN_MAX_HEIGHT = (int(_C.INPUT.IMG_SIZE*0.7), int(_C.INPUT.IMG_SIZE*0.7))
_C.INPUT.RSC_HEIGHT = _C.INPUT.IMG_SIZE
_C.INPUT.RSC_WIDTH = _C.INPUT.IMG_SIZE
_C.INPUT.RSC_PROB = 0.5
# HueSaturationValue paramters
_C.INPUT.HSV_H = 0.2
_C.INPUT.HSV_S = 0.2
_C.INPUT.HSV_V = 0.2
_C.INPUT.HSV_PROB = 0.9
# RandomBrightnessContrast paramters
_C.INPUT.BC_B = 0.2
_C.INPUT.BC_C = 0.2
_C.INPUT.BC_PROB = 0.9
# Color paramters
_C.INPUT.COLOR_PROB = 0.9
# Random probability for ToGray
_C.INPUT.TOFGRAY_PROB = 0.01
# Random probability for HorizontalFlip
_C.INPUT.HFLIP_PROB = 0.5
# Random probability for VerticalFlip
_C.INPUT.VFLIP_PROB = 0.5
# Coutout paramters
_C.INPUT.COTOUT_NUM_HOLES = 8
_C.INPUT.COTOUT_MAX_H_SIZE = 64
_C.INPUT.COTOUT_MAX_W_SIZE = 64
_C.INPUT.COTOUT_FILL_VALUE = 0
_C.INPUT.COTOUT_PROB = 0.4

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root dir of dataset
_C.DATASETS.ROOT_DIR = "/home/giorgio/Desktop/Kaggle/osics_pulmonary_kaggle/data"
# Fold to validate
_C.DATASETS.N_FOLDS = 5
# # List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.TRAIN = ()
# # List of the dataset names for testing, as present in paths_catalog.py
# _C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = multiprocessing.cpu_count()

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MODEL_NAME = 'tf_efficientdet_d6'
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.NOMINAL_BATCH_SIZE = 64
_C.SOLVER.SCHEDULER_NAME = "CosineAnnealingWarmRestarts"
#_C.SOLVER.SCHEDULER_NAME = "LambdaLR"
_C.SOLVER.COS_CPOCH = 2
_C.SOLVER.T_MUL = 2

_C.SOLVER.MAX_EPOCHS = 100

_C.SOLVER.MAX_LR = 0.1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005

_C.SOLVER.WARMUP_EPOCHS = 10

_C.SOLVER.TRAIN_CHECKPOINT = False

_C.SOLVER.IMS_PER_BATCH = 2

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 4
_C.TEST.WEIGHT = "/output/best-checkpoint.bin"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "/content/experiments/baseline"
