"""
Image data augmentation using opencv
"""
import cv2
import random


def rotate(image, angle):
    # rotate image with angle (degree)
    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    return cv2.warpAffine(image, mat, (w, h))


def random_rotate(image, low=-15, high=15):
    angle = random.uniform(low, high)
    return rotate(image, angle)


def channel_mutate(image, perm012):
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 1:
        return image
    assert image.shape[2] == 3
    return image[:, :, perm012]


def random_channel_mutate(image):
    channels = [0, 1, 2]
    random.shuffle(channels)
    return channel_mutate(image, channels)
