import easydict

__C = easydict.EasyDict()
cfg = __C


__C.class_names = [
    'normal_person',
    'normal_nonperson',
    'normal_anime_person',
    'normal_anime_noperson',
    'general_porn',
    'general_sexy',
    'anime_porn',
    'anime_sexy',
    
]
__C.num_classes = len(__C.class_names)

