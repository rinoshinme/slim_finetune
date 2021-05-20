"""
generate dataset augmentation by image blending
also called Mixup.
"""

import cv2
import numpy as np
import os
import random


def image_blend(src1, src2, dst, alpha):
    img1 = cv2.imdecode(np.fromfile(src1, dtype=np.uint8), -1)
    img2 = cv2.imdecode(np.fromfile(src2, dtype=np.uint8), -1)
    if img1 is None:
        return
    if img2 is None:
        return
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if img1.shape[2] == 4:
        img1 = img1[:, :, :3]
    if img2.shape[2] == 4:
        img2 = img2[:, :, :3]
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    height = max(h1, h2)
    width = max(w1, w2)
    # print(height, width)
    img1 = cv2.copyMakeBorder(img1, 0, height - h1, 0, width - w1, cv2.BORDER_CONSTANT, 255)
    img2 = cv2.copyMakeBorder(img2, 0, height - h2, 0, width - w2, cv2.BORDER_CONSTANT, 255)

    # print(img1.shape)
    # print(img2.shape)

    dst_img = cv2.addWeighted(img1, alpha, img2, 1.0-alpha, 0)
    cv2.imwrite(dst, dst_img)


def augment_folder(folder, num_aug, name=None):
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files]
    num_files = len(files)
    for i in range(num_aug):
        f1 = files[random.randint(0, num_files - 1)]
        f2 = files[random.randint(0, num_files - 1)]
        if name is not None:
            dst_name = os.path.join(folder, 'aug_%s_%06d.jpg' % (name, i))
        else:
            dst_name = os.path.join(folder, 'aug_%06d.jpg' % i)
        alpha = random.uniform(0.2, 0.8)
        image_blend(f1, f2, dst_name, alpha)


if __name__ == '__main__':
    folder = r'D:\data\DATASET2019\baokong12_20190701\train'

    # 1500 samples for each category
    num_aug = {
        'army': 0,
        'bloody': 0,
        'crash': 500,
        'falungong': 1000,
        'fire': 0,
        'normal': 0,
        'privacy': 500,
        'protest': 400,
        'riot': 500,
        'terrorflag': 800,
        'terrorism': 0,
        'weapon': 300
    }

    for name in num_aug.keys():
        print('processing', name)
        subfolder = os.path.join(folder, name)
        augment_folder(subfolder, num_aug[name], name)
