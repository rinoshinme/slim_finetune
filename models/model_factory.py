"""
Pretrained models from
https://github.com/tensorflow/models/tree/master/research/slim
"""
from models.inception_v3 import InceptionV3
from models.inception_v4 import InceptionV4
from models.vgg16 import Vgg16
from models.resnet_v1_50 import ResNetV1_50
from models.resnet_v1_101 import ResNetV1_101
from models.densenet_121 import DenseNet121
from models.mobilenet_v1 import MobileNetV1
from config import cfg


def get_model(model_name):
    if model_name == 'VGG16':
        model = Vgg16(num_classes=cfg.NUM_CLASSES, train_layers=cfg.TRAIN_LAYERS)
    elif model_name == 'InceptionV3':
        model = InceptionV3(num_classes=cfg.NUM_CLASSES, train_layers=cfg.TRAIN_LAYERS)
    elif model_name == 'InceptionV4':
        model = InceptionV4(num_classes=cfg.NUM_CLASSES, train_layers=cfg.TRAIN_LAYERS)
    elif model_name == 'ResNetV1_50':
        model = ResNetV1_50(num_classes=cfg.NUM_CLASSES, train_layers=cfg.TRAIN_LAYERS)
    elif model_name == 'ResNetV1_101':
        model = ResNetV1_101(num_classes=cfg.NUM_CLASSES, train_layers=cfg.TRAIN_LAYERS)
    elif model_name == 'DenseNet121':
        model = DenseNet121(num_classes=cfg.NUM_CLASSES, train_layers=cfg.TRAIN_LAYERS)
    elif model_name == 'MobileNetV1':
        model = MobileNetV1(num_classes=cfg.NUM_CLASSES, train_layers=cfg.TRAIN_LAYERS)
    else:
        raise ValueError('model name not supported')

    return model
