"""
Pretrained models from
https://github.com/tensorflow/models/tree/master/research/slim
"""
from models.inception_v3 import InceptionV3
from models.inception_v4 import InceptionV4
from models.vgg16 import Vgg16
from models.resnet_v1_50 import ResNetV1_50
from models.resnet_v1_101 import ResNetV1_101
from models.resnet_v2_152 import ResNetV2_152
from models.densenet_121 import DenseNet121
from models.mobilenet_v1 import MobileNetV1
from models.efficientnet import EfficientNet


def get_model(model_name, num_classes, train_layers=None):
    if train_layers is None:
        train_layers = 'DEFAULT'
    if model_name == 'VGG16':
        model = Vgg16(num_classes=num_classes, train_layers=train_layers)
    elif model_name == 'InceptionV3':
        model = InceptionV3(num_classes=num_classes, train_layers=train_layers)
    elif model_name == 'InceptionV4':
        model = InceptionV4(num_classes=num_classes, train_layers=train_layers)
    elif model_name == 'ResNetV1_50':
        model = ResNetV1_50(num_classes=num_classes, train_layers=train_layers)
    elif model_name == 'ResNetV1_101':
        model = ResNetV1_101(num_classes=num_classes, train_layers=train_layers)
    elif model_name == 'ResNetV2_152':
        model = ResNetV2_152(num_classes=num_classes, train_layers=train_layers)
    elif model_name == 'DenseNet121':
        model = DenseNet121(num_classes=num_classes, train_layers=train_layers)
    elif model_name == 'MobileNetV1':
        model = MobileNetV1(num_classes=num_classes, train_layers=train_layers)
    elif model_name.startswith('efficientnet'):
        model = EfficientNet(model_name, num_classes=num_classes, train_layers=train_layers)
    else:
        raise ValueError('model name not supported')

    return model

