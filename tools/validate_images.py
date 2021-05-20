import os
import cv2
import shutil



def is_valid(filepath):
    if filepath.lower().split('.')[-1] not in ['jpg', 'jpeg', 'png']:
        return False
    
    try:
        image = cv2.imread(filepath)
        if image is None:
            return False
        if len(image.shape) != 3:
            return False
        if image.shape[2] != 3:
            return False
    except:
        return False

    return True


def run(image_root):
    cnt = 0
    for root, dirs, files in os.walk(image_root):
        for f in files:
            cnt += 1
            if cnt % 1000 == 0:
                print('{} files processed'.format(cnt))
            filepath = os.path.join(root, f)
            if not is_valid(filepath):
                newpath = os.path.join('/invalid', f)
                shutil.move(filepath, newpath)


if __name__ == '__main__':
    run('/rinoshinme/data/violence/sorted')
