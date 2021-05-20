"""
Generate dataset, the 2nd kind.
images of same category are included in same folder, no separate for train/val/test.
the text files are generated with randomization on the fly.

For mixup data augmentation.
"""

import os
# import shutil
import cv2
from utils.image import cvsafe_imread


def generate_dataset_v2(root_dir, target_dir, src_names, dst_names):
    target_nums = {}
    for srcname, dstname in zip(src_names, dst_names):
        print(srcname)
        src_folder = os.path.join(root_dir, srcname)
        dst_folder = os.path.join(target_dir, dstname)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        file_names = os.listdir(src_folder)

        if dstname not in target_nums.keys():
            target_nums[dstname] = 0
        idx = target_nums[dstname]

        for name in file_names:
            src_path = os.path.join(src_folder, name)
            dst_path = os.path.join(dst_folder, '%s_%06d.jpg' % (dstname, idx))
            idx += 1
            # shutil.copy(src_path, dst_path)
            img = cvsafe_imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)

        target_nums[dstname] = idx


def generate_dataset_text_v2(dataset_dir):
    path_list = {}


if __name__ == '__main__':
    pass
