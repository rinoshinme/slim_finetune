import easydict
import platform
os_type = platform.system()

__C = easydict.EasyDict()
cfg = __C

########################################
# Data Configurations
__C.CLASS_NAMES = ['normal', 'bloody', '']
__C.NUM_CLASSES = len(__C.CLASS_NAMES)

########################################
# Model Configurations
__C.MODEL_NAME = 'ResNetV1_101'
__C.IMAGE_SIZE = 224
__C.IMAGE_CHANNELS = 3
__C.TRAIN_LAYERS = 'DEFAULT'

#########################################
# Training Configurations
if os_type == 'Window' or os_type == 'Darwin':
    __C.BATCH_SIZE = 4
    __C.VALIDATION_SIZE = 100
else:
    __C.BATCH_SIZE = 16
    __C.VALIDATION_SIZE = 3000

__C.LEARNING_RATE = 0.00002
__C.KEEP_PROB = 0.5
__C.WEIGHT_DECAY = 0.001
__C.OPTIMIZER = 'adam'
__C.ACTIVATION_FN = 'relu'  # ['relu', 'leaky', 'elu']

__C.DISPLAY_STEP = 10
__C.EVALUATE_STEP = 100
__C.CHECKPOINT_STEP = 1000
__C.MAX_CHECKPOINTS = 1000
__C.MAX_STEP = 30000

########################################
# Path Configurations [os specific]
if os_type == 'Window':
    __C.OUTPUT_DIR = './checkpoints'
    __C.PRETRAINED_WEIGHT_PATH = ''
    __C.TRAINED_WEIGHT_PATH = ''
    __C.TRAIN_DATASET_PATH = ''
    __C.VAL_DATASET_PATH = ''

elif os_type == 'Linux':
    __C.OUTPUT_DIR = './checkpoints'
    __C.PRETRAINED_WEIGHT_PATH = ''
elif os_type == 'Darwin':
    __C.OUTPUT_DIR = './checkpoints'
    __C.PRETRAINED_WEIGHT_PATH = ''
else:
    raise ValueError("OS type not supported")
