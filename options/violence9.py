from easydict import EasyDict
import tensorflow as tf
import platform

os_type = platform.system()

# cfg_violence = dict({
#     'model_name': 'ResNetV1_101',
#     'class_names': ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest'],
#     'ng_indices': [0, 1, 1, 1, 0, 1, 0, 1, 1],
#     'num_classes': 9,
#     'image_size': 224,
#     # 9 category model
#     'ckpt_path': r'D:\incoming\ResNetV1_101_20190719_132123\ckpt\model-25000'
# })


def basic_options():
    cfg = EasyDict()

    # Dataset
    cfg.CLASS_NAMES = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']
    cfg.NUM_CLASSES = len(cfg.CLASS_NAMES)
    
    # Model
    cfg.MODEL_NAME = 'ResNetV1_101'
    cfg.IMAGE_SIZE = 224
    cfg.IMAGE_CHANNELS = 3

    return cfg


def train_options():
    cfg = basic_options()
    cfg.PHASE = 'train'
    
    # Data ----------------------------------------------------
    if os_type == 'Windows':
        cfg.TRAIN_DATASET_PATH = 'data/train.txt'
        cfg.VAL_DATASET_PATH = 'data/test.txt'
        cfg.BATCH_SIZE = 2
        cfg.VALIDATION_SIZE = 10
    elif os_type == 'Linux':
        cfg.TRAIN_DATASET_PATH = 'data/train.txt'
        cfg.VAL_DATASET_PATH = 'data/test.txt'
        cfg.BATCH_SIZE = 128
        cfg.VALIDATION_SIZE = 2000
    else:
        raise ValueError("OS type not supported")
    
    # Model -----------------------------------------------------
    cfg.TRAIN_LAYERS = 'default'
    
    # OPTIMIZATION --------------------------------------------------
    cfg.LEARNING_RATE = 0.00002
    cfg.LEARNING_RATE_POLICY = 'fixed'
    cfg.LEARNING_RATE_DECAY = 0.2
    cfg.LEARNING_RATE_STEPVALUES = [7000, 13000, 18000]
    cfg.LEARNING_RATE_STEP = 10000
    cfg.LEARNING_RATE_GAMMA = 0.99
    cfg.LEARNING_RATE_POWER = 2

    cfg.KEEP_PROB = 0.5
    cfg.WEIGHT_DECAY = 0.001
    cfg.OPTIMIZER = 'adam'
    cfg.ACTIVATION_FN = tf.nn.relu

    # LOGGING ---------------------------------------------------------
    cfg.DISPLAY_STEP = 10
    cfg.EVALUATE_STEP = 100
    cfg.CHECKPOINT_STEP = 1000
    cfg.MAX_CHECKPOINTS = 100
    cfg.MAX_STEP = 30000

    if os_type == 'Windows':
        cfg.OUTPUT_DIR = r'D:\projects\output_finetune'
        cfg.PRETRAINED_ROOT = r'D:\tools\pretrained_weights'
    elif os_type == 'Linux':
        cfg.OUTPUT_DIR = '/rinoshinme/projects/content_supervision/output_finetune'
        cfg.PRETRAINED_ROOT = '/rinoshinme/weights'
    else:
        raise ValueError("OS type not supported")

    return cfg


def test_options():
    cfg = basic_options()
    cfg.PHASE = 'test'

    cfg.CKPT_PATH = r'D:\code\slim_finetune\checkpoints\ResNetV1_101\20210514_031240\ckpt\model-10000'
    cfg.TEST_DATASET_PATH = './data/flower.txt'
    cfg.BATCH_SIZE = 4
    return cfg


def demo_options():
    cfg = basic_options()

    cfg.PHASE = 'demo'
    cfg.CKPT_PATH = r'D:\code\slim_finetune\checkpoints\ResNetV1_101\20210514_031240\ckpt\model-10000'
    return cfg


def protobuf_options():
    cfg = basic_options()
    cfg.PHASE = 'protobuf'

    cfg.INPUT_NODE_NAME = 'input/x_input'
    cfg.OUTPUT_NODE_NAME = 'probability/probability'
    cfg.DROPOUT_NODE_NAME = None
    # cfg.CKPT_PATH = r'D:\projects\ContentSupervision\code\ResNetV1_101\20191029_060119\model-12000'
    return cfg
