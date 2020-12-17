import easydict

__C = easydict.EasyDict()
cfg = __C

# Categories
__C.class_names = [
    'normal', 
    'bloody', 'corpse', 
    'riot', 'bomb',
    'army_police', 'army_terror', 'army_other', 
    'sign_police', 'sign_terror', 
    'weapon_large', 'weapon_small', 'weapon_knife'
]
__C.num_classes = len(__C.class_names)

# Model parameters
__C.MODEL_NAME = 'ResNetV1_101'
__C.IMAGE_SIZE = 224
__C.IMAGE_CHANNELS = 3
__C.TRAIN_LAYERS = 'DEFAULT'
