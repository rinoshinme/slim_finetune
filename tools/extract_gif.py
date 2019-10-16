"""
Extract image frames from gif files.
"""

import cv2
import os


def read_gif(gif_file, write_folder, name_prefix):
    capture = cv2.VideoCapture(gif_file)
    cnt = 1
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        save_name = os.path.join(write_folder, '%s_%06d.jpg' % (name_prefix, cnt))
        cv2.imwrite(save_name, frame)
        cnt += 1


if __name__ == '__main__':
    gif_folder = r'D:\data\21cn_baokong\bad_format'
    fnames = os.listdir(gif_folder)
    for name in fnames:
        gif_path = os.path.join(gif_folder, name)
        prefix = name.split('.')[0]
        read_gif(gif_path, gif_folder, prefix)
