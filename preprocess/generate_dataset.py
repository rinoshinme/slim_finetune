"""
Generate dataset for training.

images are arranged in the following order
ROOT_DIR
    train
        class1
            img1.jpg
            img2.jpg
            ...
        class2
        ...
    val
        class1
        class2
        ...
    test
        class1
        class2
        ...
"""

import cv2
import os
import random
from utils.image import cvsafe_imread


def generate_dataset(root_dir, target_dir, src_names, dst_names, val_ratio=0.1, test_ratio=0.15):
    target_nums = {}
    for srcname, dstname in zip(src_names, dst_names):
        print(srcname)
        src_folder = os.path.join(root_dir, srcname)
        files = os.listdir(src_folder)
        random.shuffle(files)
        num_images = len(files)
        print(num_images)
        num_val = int(num_images * val_ratio)
        num_test = int(num_images * test_ratio)
        num_train = num_images - num_val - num_test

        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]
        train_folder = os.path.join(target_dir, 'train', dstname)
        val_folder = os.path.join(target_dir, 'val', dstname)
        test_folder = os.path.join(target_dir, 'test', dstname)
        for folder in (train_folder, val_folder, test_folder):
            if not os.path.exists(folder):
                os.makedirs(folder)

        if dstname not in target_nums.keys():
            target_nums[dstname] = 0

        idx = target_nums[dstname]
        for f in train_files:
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(train_folder, '%s_%06d.jpg' % (dstname, idx))
            idx += 1
            img = cvsafe_imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)

        for f in val_files:
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(val_folder, '%s_%06d.jpg' % (dstname, idx))
            idx += 1
            img = cvsafe_imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)

        for f in test_files:
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(test_folder, '%s_%06d.jpg' % (dstname, idx))
            idx += 1
            img = cvsafe_imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)

        # save idx for possible future reuse.
        target_nums[dstname] = idx


def generate_text(root_dir, class_names, phase, shuffle=True):
    img_dir = os.path.join(root_dir, phase)

    dataset = []
    for idx, name in enumerate(class_names):
        subfolder = os.path.join(img_dir, name)
        files = os.listdir(subfolder)
        for f in files:
            filepath = os.path.join(subfolder, f)
            dataset.append((filepath, idx))

    # shuffle dataset and save text
    if shuffle:
        random.shuffle(dataset)

    target_text = os.path.join(root_dir, '%s.txt' % phase)
    with open(target_text, 'w') as fp:
        for d in dataset:
            fp.write('%s\t%s\n' % (d[0], d[1]))


def generate_dataset_text(root_dir, class_names):
    """
    generate shuffled text dataset
    """
    generate_text(root_dir, class_names, 'train', shuffle=True)
    generate_text(root_dir, class_names, 'val', shuffle=True)
    generate_text(root_dir, class_names, 'test', shuffle=False)


if __name__ == '__main__':
    srcfolder = r'D:\data\baokong21cn'
    dstfolder = r'E:\Training\DATASET2019\bloody3_20191018'

    # may map multiple categories to single target group
    srcnames = ['正常', '正常2', '正常3', '正常21cn', '正常动漫', '轻微', '血腥', '血腥动漫']
    dstnames = ['normal', 'normal', 'normal', 'normal', 'normal', 'medium', 'bloody', 'bloody']

    generate_dataset(srcfolder, dstfolder, srcnames, dstnames, val_ratio=0.1, test_ratio=0.15)

    class_labels = ['normal', 'medium', 'bloody']
    generate_dataset_text(dstfolder, class_labels)
