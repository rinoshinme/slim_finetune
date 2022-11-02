from easydict import EasyDict


def basic_options():
    cfg = EasyDict()

    # Dataset
    
    cfg.CLASS_NAMES = ['animationnormal', 'animationporn', 'generalporn', 
                       'generalsexy', 'generalnormal', 'malenormal', 'maleporn']
    cfg.NUM_CLASSES = len(cfg.CLASS_NAMES)

    # Model
    cfg.MODEL_NAME = 'ResNetV1_101'
    cfg.IMAGE_SIZE = 224
    cfg.IMAGE_CHANNELS = 3

    return cfg


def protobuf_options():
    cfg = basic_options()
    cfg.PHASE = 'protobuf'

    cfg.INPUT_NODE_NAME = 'input/x_input'
    # cfg.OUTPUT_NODE_NAME = 'resnet_v1_101_1/predictions/Softmax'
    cfg.OUTPUT_NODE_NAME = 'probability/probability'
    cfg.DROPOUT_NODE_NAME = None
    cfg.CKPT_PATH = None
    return cfg
