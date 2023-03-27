from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "default"
_C.REDIRECTOR = True

_C.RELOAD = CN()
_C.RELOAD.TYPE = True
_C.RELOAD.PTH = 'saved_model/'

_C.DISTRIBUTTED = False

# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.TYPE = "cifar"
_C.DATASET.NAME = "cifar100"
_C.DATASET.TRAIN_SPLIT = 'train'
_C.DATASET.VAL_SPLIT = 'test'
_C.DATASET.TEST_SPLIT = 'test'

# ----- BACKBONE BUILDER -----
_C.BACKBONE = CN()
_C.BACKBONE.TYPE = "wideresnet"
_C.BACKBONE.DROP_OUT = 0.0

# ----- TRAIN BUILDER -----
_C.TRAIN = CN()
_C.TRAIN.BN_WD = True
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.MAX_EPOCH = 200
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.CLIP_GRAD = False

_C.TRAIN.DATAAUG = CN()
_C.TRAIN.DATAAUG.TYPE = 'base'
_C.TRAIN.DATAAUG.AUTOAUG_PROB = 0.5

_C.TRAIN.EMA = CN()
_C.TRAIN.EMA.ENABLE = False
_C.TRAIN.EMA.DECAY = 0.9998
_C.TRAIN.EMA.FORCE_CPU = False

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.NESTEROV = True
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4

_C.TRAIN.EPSILON = 10.
_C.TRAIN.CLS_LOSS_WEIGHT = 0.

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = 'multistep'
_C.TRAIN.LR_SCHEDULER.LR_STEP = [60,120,160]
_C.TRAIN.LR_SCHEDULER.LR_FT = 1e-1
_C.TRAIN.LR_SCHEDULER.LR_NEW = 1e-1
_C.TRAIN.LR_SCHEDULER.WMUP_COEF = 0.1
_C.TRAIN.LR_SCHEDULER.FACTOR = 0.2
_C.TRAIN.NON_BLOCKING = True
# ----- INFER BUILDER -----

_C.INFER = CN()
_C.INFER.SAMPLING = False

# ------ visualization ---------
_C.VIS = CN()
_C.VIS.CAM = 'valid'
_C.VIS.TENSORBOARD = CN()
#_C.VIS.TENSORBOARD.ENABLE = True

_C.VIS.VISDOM = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
