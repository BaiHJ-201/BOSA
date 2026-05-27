from yacs.config import CfgNode as CN

_C = CN()
cfg = _C

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CN()

_C.MODEL.ARCH = 'Standard'

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CN()

_C.CORRUPTION.DATASET = 'cifar10'
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]
_C.CORRUPTION.NUM_EX = 10000
_C.CORRUPTION.NUM_CLASS = -1

# ----------------------------- Input options -------------------------- #
_C.INPUT = CN()

_C.INPUT.SIZE = (32, 32)
_C.INPUT.INTERPOLATION = "bilinear"
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# _C.INPUT.TRANSFORMS = ("normalize", )
_C.INPUT.TRANSFORMS = ()

# ----------------------------- loader options -------------------------- #
_C.LOADER = CN()

_C.LOADER.SAMPLER = CN()
_C.LOADER.SAMPLER.TYPE = "sequence"
# _C.LOADER.SAMPLER.GAMMA = 0.001
_C.LOADER.SAMPLER.GAMMA = 0.1
_C.LOADER.SAMPLER.IMB_FACTOR = 1
_C.LOADER.SAMPLER.CLASS_RATIO = "constant"

_C.LOADER.NUM_WORKS = 2

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CN()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 1e-3

_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9
_C.OPTIM.WD = 0.0

# ------------------------------- Testing options --------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64

# ---------------------------------- Misc options --------------------------- #

_C.SEED = 427
_C.OUTPUT_DIR = "./output"
_C.DATA_DIR = "./datasets"
_C.CKPT_DIR = "./ckpt"

_C.LOG_DEST = "log.txt"

_C.BN_ONLY = True
# tta method specific
_C.ADAPTER = CN()

_C.ADAPTER.NAME = "rotta"

_C.ADAPTER.RoTTA = CN()
_C.ADAPTER.RoTTA.MEMORY_SIZE = 64
_C.ADAPTER.RoTTA.UPDATE_FREQUENCY = 64
_C.ADAPTER.RoTTA.NU = 0.001
_C.ADAPTER.RoTTA.ALPHA = 0.05
_C.ADAPTER.RoTTA.LAMBDA_T = 1.0
_C.ADAPTER.RoTTA.LAMBDA_U = 1.0

_C.ADAPTER.TRIBE = CN()
_C.ADAPTER.TRIBE.ETA = 0.005
_C.ADAPTER.TRIBE.GAMMA = 0.0
_C.ADAPTER.TRIBE.LAMBDA = 0.5
_C.ADAPTER.TRIBE.H0 = 0.05

_C.ADAPTER.COTTA = CN()
_C.ADAPTER.COTTA.STEPS = 1
_C.ADAPTER.COTTA.EPISODIC = False
_C.ADAPTER.COTTA.MT_ALPHA = 0.99
_C.ADAPTER.COTTA.RST_M = 0.1
_C.ADAPTER.COTTA.AP = 0.9

_C.ADAPTER.BOSA = CN()
_C.ADAPTER.BOSA.MEMORY_SIZE = 64
_C.ADAPTER.BOSA.UPDATE_FREQUENCY = 64
_C.ADAPTER.BOSA.LAMBDA_T = 1.0
_C.ADAPTER.BOSA.LAMBDA_U = 1.0
_C.ADAPTER.BOSA.EMA_DECAY = 0.999

_C.ADAPTER.ECOTTA = CN()
_C.ADAPTER.ECOTTA.lambda_reg = 0.25
_C.ADAPTER.ECOTTA.e_margin = 0.4

_C.ADAPTER.DAS = CN()
_C.ADAPTER.DAS.e_margin = 0.4

_C.ADAPTER.PALM = CN()
_C.ADAPTER.PALM.BETA3 = 0.5
_C.ADAPTER.PALM.TEMP = 50.0
_C.ADAPTER.PALM.THRESH = 1.0
_C.ADAPTER.PALM.LAMBDA = 0.01

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()
