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
import os
import cv2
import random
from utils.image import cvsafe_imread


def split_images(root_dir, target_dir, src_names, dst_names, val_ratio, test_ratio):
    target_nums = dict()
    for srcname, dstname in zip(src_names, dst_names):
        print(srcname)
        src_folder = os.path.join(root_dir, srcname)
        files = os.listdir(src_folder)
        random.shuffle(files)
        num_images = len(files)
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


if __name__ == '__main__':
    srcfolder = r'D:\data\baokong'
    dstfolder = r'E:\DATASET2019\baokong09_20190814'

    # srcnames = ['normal', 'army', 'bloody', 'crash', 'fire', 'identity',
    #             'normal_artificial', 'normal_crowd', 'normal_document',
    #             'protest', 'riot', 'terrorism', 'weapon']
    # dstnames = ['normal', 'army', 'bloody', 'crash', 'fire', 'identity',
    #             'normal_artificial', 'normal_crowd', 'normal_document',
    #             'protest', 'riot', 'terrorism', 'weapon']

    # map multiple categories to normal
    srcnames = ['14. normal', '9. army', '2. bloody', '7. crash', '5. fire', '6. identity',
                '13. normal_artificial', '11. normal_multiple_person', '12. normal_document',
                '4. protest', '3. riot', '1. terrorism', '8. weapon', '10. normal_person123']
    dstnames = ['normal', 'army', 'bloody', 'crash', 'fire', 'normal',
                'normal', 'normal', 'normal',
                'protest', 'riot', 'terrorism', 'weapon', 'normal']

    split_images(srcfolder, dstfolder, srcnames, dstnames, val_ratio=0.15, test_ratio=0.15)
