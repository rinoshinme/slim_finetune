cfg_bloody = dict({
    'model_name': 'ResNetV1_101',
    'class_names': ['normal', 'medium', 'bloody'],
    'ng_indices': [0, 0.5, 1],
    'num_classes': 3,
    'image_size': 224,
    # 3 category model v1
    # 'ckpt_path': r'E:\Training\output_finetune\ResNetV1_101\20191018_051256\ckpt\model-6000',
    # 3 category model v2
    'ckpt_path': r'F:\Training\output_finetune\ResNetV1_101\20191018_102037\ckpt\model-12000'
})

cfg_violence = dict({
    'model_name': 'ResNetV1_101',
    'class_names': ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest'],
    'ng_indices': [0, 1, 1, 1, 0, 1, 0, 1, 1],
    'num_classes': 9,
    'image_size': 224,
    # 9 category model
    'ckpt_path': r'D:\incoming\ResNetV1_101_20190719_132123\ckpt\model-25000'
})

cfg_baokong7 = dict({
    'model_name': 'ResNetV1_101',
    'class_names': ['normal', 'army', 'weapon', 'fire', 'bloody', 'terrorism', 'terrorflag'],
    'ng_indices': [0, 0, 0, 1, 1, 1, 1],
    'num_classes': 7,
    'image_size': 224,
    # 7 category model
    'ckpt_path': r'F:\Training\output_finetune\ResNetV1_101\20191029_060119\ckpt\model-12000'
})

# Lost?
cfg_baokong7_v2 = dict({
    'model_name': 'ResNetV1_101',
    'class_names': ['normal', 'army', 'weapon', 'fire', 'bloody', 'mild_bloody', 'terrorism'],
    'ng_indices': [0, 0, 0, 1, 1, 0.5, 1],
    'num_classes': 7,
    'image_size': 224,
    # 7 category model
    'ckpt_path': r'F:\Training\output_finetune\ResNetV1_101\20191106_072716\ckpt\model-18000'
})
