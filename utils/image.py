import cv2
import numpy as np


IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']


def is_image_file(image_path):
    if any([image_path.lower().endswith(ext) for ext in IMAGE_EXTS]):
        return True
    else:
        return False


def cvsafe_imread(image_path):
    """
    protect reading from non-ascii file paths
    """
    return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
