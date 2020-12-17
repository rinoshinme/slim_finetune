"""
video related functionalities
"""
import cv2
import os
import sys


def video2frames(video_path, frames_folder, name_template='{:06d}.jpg', step=1):
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step != 0:
            continue
        save_path = os.path.join(frames_folder, name_template.format(count))
        cv2.imwrite(save_path, frame)
        count += 1
    return True


def video2framesffmpeg(video_path, frames_folder):
    """
    USE FFMPEG.
    """
    pass


def frames2video(frames_folder, video_path):
    pass


def gif2frames(gif_path, frames_folder, step=1):
    pass


