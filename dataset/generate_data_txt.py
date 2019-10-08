"""
1. copy valid images into train/val/test folder
2. generate train.txt/val.txt/test.txt
3. generate image augmentation for training data
"""
import os
import random

# SRC_LABELS = ['normal', 'riot', 'crash', 'fire',
#               'army', 'terrorism', 'weapon', 'bloody',
#                'protest']

SRC_LABELS = ['hat_on',
              'hat_off',
              'other']

# DST_LABELS = ['normal', 'riot', 'crash', 'fire',
#               'army', 'terrorism', 'weapon', 'bloody',
#               'protest']

DST_LABELS = SRC_LABELS

LABEL_MAP = {key: val for (key, val) in zip(SRC_LABELS, DST_LABELS)}

VAL_RATIO = 0.15
TEST_RATIO = 0.15


def generate_text(folder, class_names, dst_text):
    fp = open(dst_text, 'wt')
    for idx, name in enumerate(class_names):
        subfolder = os.path.join(folder, name)
        files = os.listdir(subfolder)
        # if len(files) > max_files_per_category and max_files_per_category > 0:
        #     files = files[:max_files_per_category]
        for f in files:
            filepath = os.path.join(subfolder, f)
            fp.write('%s\t%s\n' % (filepath, idx))
    fp.close()


def generate_text_shuffle(folder, class_names, dst_text):
    dataset = []
    for idx, name in enumerate(class_names):
        subfolder = os.path.join(folder, name)
        files = os.listdir(subfolder)
        for f in files:
            filepath = os.path.join(subfolder, f)
            dataset.append((filepath, idx))

    # shuffle dataset
    random.shuffle(dataset)
    with open(dst_text, 'w') as fp:
        for d in dataset:
            fp.write('%s\t%s\n' % (d[0], d[1]))


if __name__ == '__main__':
    # src_folder = r'D:\data\baokong3'
    # dst_folder = r'E:\DATASET2019\baokong13_20190731'

    dst_folder = '/factory/data/safetyhat_20190919'
    # split_images(src_folder, dst_folder, LABEL_MAP)

    # generate text
    train_dir = os.path.join(dst_folder, 'train')
    val_dir = os.path.join(dst_folder, 'val')
    test_dir = os.path.join(dst_folder, 'test')
    train_text = train_dir + '.txt'
    val_text = val_dir + '.txt'
    test_text = test_dir + '.txt'
    class_names = DST_LABELS
    generate_text(train_dir, class_names, train_text)
    generate_text(val_dir, class_names, val_text)
    generate_text(test_dir, class_names, test_text)
