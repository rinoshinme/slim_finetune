"""
Pretrained models from
https://github.com/tensorflow/models/tree/master/research/slim
"""

from tfvortex.models.inception_v3 import InceptionV3
from tfvortex.models.inception_v4 import InceptionV4
from tfvortex.models.vgg16 import Vgg16
from tfvortex.models.resnet_v1_50 import ResNetV1_50
from tfvortex.models.resnet_v1_101 import ResNetV1_101
from tfvortex.models.resnet_v1_152 import ResNetV1_152
from tfvortex.models.resnet_v2_152 import ResNetV2_152
from tfvortex.models.densenet_121 import DenseNet121
from tfvortex.models.mobilenet_v1 import MobileNetV1
# from tfvortex.models.efficientnet import EfficientNet


def get_model(options):
    model_name = options.MODEL_NAME

    if model_name == 'VGG16':
        model = Vgg16(options)
    elif model_name == 'InceptionV3':
        model = InceptionV3(options)
    elif model_name == 'InceptionV4':
        model = InceptionV4(options)
    elif model_name == 'ResNetV1_50':
        model = ResNetV1_50(options)
    elif model_name == 'ResNetV1_101':
        model = ResNetV1_101(options)
    elif model_name == 'ResNetV1_152':
        model = ResNetV1_152(options)
    elif model_name == 'ResNetV2_152':
        model = ResNetV2_152(options)
    elif model_name == 'DenseNet121':
        model = DenseNet121(options)
    elif model_name == 'MobileNetV1':
        model = MobileNetV1(options)
    # elif model_name.startswith('efficientnet'):
    #     model = EfficientNet(options)
    else:
        raise ValueError('model name not supported')

    return model
