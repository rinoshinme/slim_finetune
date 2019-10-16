import cv2
import os


def is_image_file(filename):
    fname = filename.lower()
    exts = ['jpg', 'jpeg', 'bmp', 'png']
    if any([fname.endswith(e) for e in exts]):
        return True
    else:
        return False


def copy_valid_files(src_folder, dst_folder):
    fnames = os.listdir(src_folder)
    for name in fnames:
        print(name)
        if not is_image_file(name):
            continue

        image_path = os.path.join(src_folder, name)
        img = cv2.imread(image_path)

        if img is None:
            continue

        # move file
        dst_path = os.path.join(dst_folder, name)
        if not os.path.exists(dst_path):
            # shutil.copy(image_path, dst_path)
            cv2.imwrite(dst_path, img)
        del img


if __name__ == '__main__':
    src_folder = r'D:\data\baokong\normal'
    dst_folder = r'D:\data\baokong21cn\normal'
    copy_valid_files(src_folder, dst_folder)
