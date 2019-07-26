"""
1. copy valid images into train/val/test folder
2. generate train.txt/val.txt/test.txt
3. generate image augmentation for training data
"""
import os
import cv2
import numpy as np
import random

SRC_LABELS = ['normal', 'riot', 'crash', 'fire',
              'army', 'terrorism', 'weapon', 'bloody',
              'protest']
DST_LABELS = ['normal', 'riot', 'crash', 'fire',
              'army', 'terrorism', 'weapon', 'bloody',
              'protest']

LABEL_MAP = {key: val for (key, val) in zip(SRC_LABELS, DST_LABELS)}

VAL_RATIO = 0.15
TEST_RATIO = 0.15


# Make sure that src dir contains only jpg, jpeg, png or bmp files
# remove duplicate images before generating dataset
def split_images(src_dir, dst_dir, name_map):
    for key, value in name_map.items():
        print(key)
        sub_folder = os.path.join(src_dir, key)
        files = os.listdir(sub_folder)
        print(len(files))
        # split valid_files into train/val/test
        random.shuffle(files)
        num_images = len(files)
        num_val = int(VAL_RATIO * num_images)
        num_test = int(TEST_RATIO * num_images)
        num_train = num_images - num_val - num_test
        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]

        # move files into respective folders
        train_folder = os.path.join(dst_dir, 'train', value)
        val_folder = os.path.join(dst_dir, 'val', value)
        test_folder = os.path.join(dst_dir, 'test', value)
        for folder in (train_folder, val_folder, test_folder):
            if not os.path.exists(folder):
                os.makedirs(folder)

        idx = 0
        for f in train_files:
            src_path = os.path.join(sub_folder, f)
            dst_path = os.path.join(train_folder, '%s_%06d.jpg' % (value, idx))
            idx += 1
            # img = cv2.imread(src_path)
            img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), -1)
            if img is not None:
                cv2.imwrite(dst_path, img)
        for f in val_files:
            src_path = os.path.join(sub_folder, f)
            dst_path = os.path.join(val_folder, '%s_%06d.jpg' % (value, idx))
            # print(src_path)
            # print(dst_path)
            idx += 1
            # shutil.copyfile(src_path, dst_path)
            # img = cv2.imread(src_path)
            img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), -1)
            if img is not None:
                cv2.imwrite(dst_path, img)
        for f in test_files:
            src_path = os.path.join(sub_folder, f)
            dst_path = os.path.join(test_folder, '%s_%06d.jpg' % (value, idx))
            idx += 1
            # shutil.copyfile(src_path, dst_path)
            # img = cv2.imread(src_path)
            img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), -1)
            if img is not None:
                cv2.imwrite(dst_path, img)


def generate_text(folder, class_names, dst_text, max_files_per_category=5000):
    fp = open(dst_text, 'wt')
    for idx, name in enumerate(class_names):
        subfolder = os.path.join(folder, name)
        files = os.listdir(subfolder)
        if max_files_per_category > 0:
            files = files[:max_files_per_category]
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
    src_folder = r'D:\data\baokong2'
    dst_folder = r'F:\DATASET2019\baokong09_20190717'
    # split_images(src_folder, dst_folder, LABEL_MAP)

    # generate text
    train_dir = os.path.join(dst_folder, 'train')
    val_dir = os.path.join(dst_folder, 'val')
    test_dir = os.path.join(dst_folder, 'test')
    train_text = train_dir + '.txt'
    val_text = val_dir + '.txt'
    test_text = test_dir + '.txt'
    # class_names = ['normal', 'army', 'fire', 'terrorflag']
    class_names = DST_LABELS
    generate_text(train_dir, class_names, train_text, max_files_per_category=5000)
    generate_text(val_dir, class_names, val_text, max_files_per_category=1500)
    generate_text(test_dir, class_names, test_text, max_files_per_category=1500)
