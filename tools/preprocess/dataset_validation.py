"""
Validate that every image in the dataset is readable from opencv
"""
import os
import shutil
import cv2
from utils.image import cvsafe_imread, is_image_file


def validate(src_dir, dst_dir):
    """
    Move valid files
    """
    for root, dirs, files in os.walk(src_dir):
        print(root)
        for name in files:
            if not is_image_file(name):
                continue
            fpath = os.path.join(root, name)
            img = cvsafe_imread(fpath)
            if img is None:
                continue
            # move file into target directory
            new_root = root.replace(src_dir, dst_dir, 1)
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            new_fpath = os.path.join(new_root, name)
            shutil.move(fpath, new_fpath)


def validate2(src_dir, dst_dir):
    """
    Move invalid files
    """
    for root, dirs, files in os.walk(src_dir):
        for name in files:
            if not is_image_file(name):
                continue
            fpath = os.path.join(root, name)
            print(fpath)
            img = cvsafe_imread(fpath)
            if img is None:
                # move file into target directory
                new_root = root.replace(src_dir, dst_dir, 1)
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                new_fpath = os.path.join(new_root, name)
                shutil.move(fpath, new_fpath)


def validate_dataset(src_dir, dst_dir, class_names):
    """
    make sure every image is a valid jpg image,
    and reading with opencv (tensorflow) do not produce error
    """
    for c in class_names:
        src_path = os.path.join(src_dir, c)
        dst_path = os.path.join(dst_dir, c)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        files = os.listdir(src_path)
        for f in files:
            print('processing %s' % f)
            base_name = f.split('.')[0]
            src_file = os.path.join(src_path, f)
            img = cv2.imread(src_file)
            if img is None:
                continue
            dst_file = os.path.join(dst_path, '%s.jpg' % base_name)
            cv2.imwrite(dst_file, img)


def validate_all(src_folder, dst_folder):
    for root, dirs, files in os.walk(src_folder):
        target_root = root.replace(src_folder, dst_folder)
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        for f in files:
            source_path = os.path.join(root, f)
            img = cv2.imread(source_path)
            if img is None:
                continue
            target_path = os.path.join(target_root, f)
            # cv2.imwrite(target_path, img)
            shutil.move(source_path, target_path)


if __name__ == '__main__':
    dsrc = r'D:\data\baokong'
    ddst = r'D:\data\tempfile\invalid'
    # validate(dsrc, ddst)
    validate2(dsrc, ddst)
