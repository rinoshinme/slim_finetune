from easydict import EasyDict
import platform

os_type = platform.system()


def basic_options():
    cfg = EasyDict()
    cfg.CLASS_NAMES = ['normal', '']
    cfg.NUM_CLASSES = []

    cfg.MODEL_NAME = ''
    cfg.IMAGE_SIZE = 0
    cfg.IMAGE_CHANNELS = 3

    return cfg
