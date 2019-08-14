"""
Validate that every image in the dataset is readable from opencv
"""
import os
import shutil
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


if __name__ == '__main__':
    dsrc = r'D:\data\baokong'
    ddst = r'D:\data\tempfile\invalid'
    # validate(dsrc, ddst)
    validate2(dsrc, ddst)



