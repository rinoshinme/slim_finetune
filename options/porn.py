from easydict import EasyDict
import platform

os_type = platform.system()


def basic_options():
    cfg = EasyDict()

    # Dataset
    cfg.CLASS_NAMES = ['generalnormal', 'generalsexy', 'generalporn', 
                       'animationnormal', 'animationsexy', 'animationporn']
    cfg.NUM_CLASSES = 6

    # Model
    cfg.MODEL_NAME = 'ResNetV1_101'
    cfg.IMAGE_SIZE = 224
    cfg.IMAGE_CHANNELS = 3

    return cfg


def train_options():
    cfg = basic_options()

    return cfg


