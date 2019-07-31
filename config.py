"""
Global configuration file
"""

import easydict
import tensorflow as tf
import platform

os_type = platform.system()

__C = easydict.EasyDict()
cfg = __C

# POSSIBLE MODIFICAITON NEEDED FOR NEW TRAINING
# 1. CLASS_NAMES
# 2. DATASET_PATH
# 3. MODEL_NAME

###########################################
# Class Configurations
# __C.CLASS_NAMES = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']
# __C.CLASS_NAMES = ['normal', 'riot', 'crash', 'fire',
#                    'army', 'terrorism', 'weapon', 'bloody',
#                    'protest']
__C.CLASS_NAMES = ['normal', 'army', 'bloody', 'crash', 'fire', 'identity',
                   'normal_artificial', 'normal_crowd', 'normal_document',
                   'protest', 'riot', 'terrorism', 'weapon']
__C.NUM_CLASSES = len(__C.CLASS_NAMES)

# Model Configurations
__C.MODEL_NAME = 'ResNetV1_101'  # ['InceptionV3/V4', 'VGG16', 'ResNetV1_50/101', 'DenseNet121', 'MOBILENET_V1/V2']
if __C.MODEL_NAME.startswith('Inception'):
    __C.IMAGE_SIZE = 299
else:
    __C.IMAGE_SIZE = 224
__C.IMAGE_CHANNELS = 3
__C.TRAIN_LAYERS = 'DEFAULT'

# Output Configurations
if os_type == 'Windows':
    __C.OUTPUT_DIR = r'D:\projects\output_finetune'
elif os_type == 'Linux':
    __C.OUTPUT_DIR = '/home/deploy/rinoshinme/projects/output_finetune'
else:
    raise ValueError("OS type not supported")

###########################################
# Training Configurations
__C.TRAIN = easydict.EasyDict()

if os_type == 'Windows':
    __C.TRAIN.PRETRAINED_WEIGHT_PATH = r'D:\library\pretrained_models'
    __C.TRAIN.TRAIN_DATASET_PATH = r'F:\DATASET2019\baokong12_20190703\train.txt'
    __C.TRAIN.VAL_DATASET_PATH = r'F:\DATASET2019\baokong12_20190703\val.txt'
elif os_type == 'Linux':
    __C.TRAIN.PRETRAINED_WEIGHT_PATH = '/home/deploy/rinoshinme/projects/pretrained_models'
    __C.TRAIN.TRAIN_DATASET_PATH = '/home/deploy/rinoshinme/data/baokong13_20190731/train.txt'
    __C.TRAIN.VAL_DATASET_PATH = '/home/deploy/rinoshinme/data/baokong13_20190731/val.txt'
else:
    raise ValueError('OS type not supported')

if os_type == 'Windows':
    __C.TRAIN.BATCH_SIZE = 4
elif os_type == 'Linux':
    __C.TRAIN.BATCH_SIZE = 128
else:
    raise ValueError('OS type not supported')

# Please see log history to choose apporpriate lr steps
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.LEARNING_RATE_POLICY = 'multisteps'  # [fixed, step, exp, inv, multisteps, poly]
__C.TRAIN.LEARNING_RATE_DECAY = 0.2
__C.TRAIN.LEARNING_RATE_STEPVALUES = [7000, 13000, 18000]
__C.TRAIN.LEARNING_RATE_STEP = 10000
__C.TRAIN.LEARNING_RATE_GAMMA = 0.99
__C.TRAIN.LEARNING_RATE_POWER = 2

__C.TRAIN.KEEP_PROB = 0.5  # no dropout in resnet.
__C.TRAIN.WEIGHT_DECAY = 0.001
__C.TRAIN.OPTIMIZER = 'adam'  # ['sgd', 'adam', 'moment']
__C.TRAIN.ACTIVATION_FN = tf.nn.relu  # tf.nn.learky_relu, tf.nn.elu

__C.TRAIN.VALIDATION_SIZE = 3000

# __C.TRAIN.NUM_EPOCHS = 50
__C.TRAIN.DISPLAY_STEP = 10
__C.TRAIN.EVALUATE_STEP = 100
__C.TRAIN.CHECKPOINT_STEP = 1000
__C.TRAIN.MAX_CHECKPOINTS = 100
__C.TRAIN.MAX_STEP = 30000

###########################################
# Test Configurations

__C.TEST = easydict.EasyDict()
# __C.TEST.CLASS_NAMES = ['normal', 'bloody', 'terrorism', 'bomb', 'weapon', 'army']
__C.TEST.CLASS_NAMES = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody',
                        'protest', 'falungong', 'terrorflag', 'privacy']
__C.TEST.NUM_CLASSES = len(__C.CLASS_NAMES)

# Model Configurations
__C.TEST.MODEL_NAME = 'ResNetV1_101'
__C.TEST.IMAGE_SIZE = 224
__C.TEST.IMAGE_CHANNELS = 3

__C.TEST.INPUT_NODE_NAME = 'input/x_input'
__C.TEST.OUTPUT_NODE_NAME = 'resnet_v1_101_1/predictions/Softmax'
__C.TEST.DROPOUT_NODE_NAME = None  # 'input/keep_prob'

# TEST DATASET
if os_type == 'Windows':
    # __C.TEST.TEST_DATASET_PATH = r'D:\data\DATASET2019\baokong12\test.txt'
    # __C.TEST.TEST_DATASET_PATH = r'F:\DATASET2019\baokong12_20190703\test.txt'
    __C.TEST.TEST_DATASET_PATH = r'F:\DATASET2019\baokong09_20190717\test_weapon.txt'
    __C.TEST.BATCH_SIZE = 4
    # __C.TEST.CHECKPOINT_PATH = r'D:\projects\output_finetune\ResNetV1_50\20190626_154551\ckpt\model-8000'
    # __C.TEST.CHECKPOINT_PATH = r'F:\output_finetune\ResNetV1_50\20190702_151234\ckpt\model-11000'
    # __C.TEST.CHECKPOINT_PATH = r'F:\output_finetune\ResNetV1_101\20190708_145053\ckpt\model-1000'
    # __C.TEST.CHECKPOINT_PATH = r'F:\output_finetune\ResNetV1_101\20190710_124127\ckpt\model-2000'
    __C.TEST.CHECKPOINT_PATH = r'F:\output_finetune\ResNetV1_101\20190717_182122\ckpt\model-9000'
elif os_type == 'Linux':
    __C.TEST.TEST_DATASET_PATH = '/home/deploy/rinoshinme/data/violence_data/test.txt'
    __C.TEST.BATCH_SIZE = 64
    __C.TEST.CHECKPOINT_PATH = ''
else:
    raise ValueError('OS type not supported')

__C.TEST.NUM_TEST = 0  # number of samples for testing, not necessary to be test set size
