import easydict
import platform
os_type = platform.system()

__C = easydict.EasyDict()
cfg = __C


# __C.CLASS_NAMES = ['normal', 'army', 'weapon', 'fire', 'bloody', 'terror', 'terrorflag']
__C.CLASS_NAMES = ['normal', 'medium', 'bloody']
__C.NUM_CLASSES = len(__C.CLASS_NAMES)

__C.MODEL_NAME = 'ResNetV1_101'
__C.IMAGE_SIZE = 224
__C.IMAGE_CHANNELS = 3

__C.INPUT_NODE_NAME = 'input/x_input'
__C.OUTPUT_NODE_NAME = 'resnet_v1_101/predictions/Softmax'
__C.DROPOUT_NODE_NAME = None

__C.BATCH_SIZE = 1
if os_type == 'Windows' or os_type == 'Darwin':
    __C.DATASET_PATH = './dataset.txt'
    __C.CHECKPOINT_PATH = 'checkpoints/ckpt/model-12000'
elif os_type == 'Linux':
    __C.DATASET_PATH = ''
    __C.CHECKPOINT_PATH = ''
