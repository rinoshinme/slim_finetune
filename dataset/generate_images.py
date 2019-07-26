"""
Generated a clean dataset
one folder for each cls
"""
import cv2
import os
import numpy as np

SRC_LABELS = ['正常', '暴乱打架', '车祸废墟', '放火', '警察治安', '恐怖分子', '武器', '血腥尸体', '游行集会']
DST_LABELS = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']


# Only jpg and png files are supported.
def convert_name_to_jpg(fname):
    fname = fname.lower()
    if fname.endswith('.jpg'):
        return fname
    elif fname.endswith('.jpeg'):
        return fname[:-4] + '.jpg'
    elif fname.endswith('.png'):
        return fname[:-3] + '.jpg'
    return None


def generate_images(src_folder, dst_folder):
    assert len(SRC_LABELS) == len(DST_LABELS)
    for i in range(len(SRC_LABELS)):
        print(SRC_LABELS[i])
        src_sub_folder = os.path.join(src_folder, SRC_LABELS[i])
        if not os.path.exists(src_sub_folder):
            continue
        dst_sub_folder = os.path.join(dst_folder, DST_LABELS[i])
        if not os.path.exists(dst_sub_folder):
            os.makedirs(dst_sub_folder)
        files = os.listdir(src_sub_folder)
        for f in files:
            file_path = os.path.join(src_sub_folder, f)
            print(file_path)
            jpg_name = convert_name_to_jpg(f)
            if jpg_name is None:
                continue
            dst_file_path = os.path.join(dst_sub_folder, jpg_name)
            if os.path.exists(dst_file_path):
                continue

            # filepath contain Chinese character cannot be opened
            # img = cv2.imread(file_path)
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            if img is None:
                continue

            print('write to {}'.format(dst_file_path))
            cv2.imwrite(dst_file_path, img)


if __name__ == '__main__':
    srcfolder = r'D:\data\baokong'
    dstfolder = r'D:\data\DATASET2019\baokong10'
    generate_images(srcfolder, dstfolder)
