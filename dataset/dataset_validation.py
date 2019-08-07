import os
import cv2
import shutil


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
            dst_file = os.path.join(dst_path, base_name + '.jpg')
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
    src_path = r'D:\data\data_scrawler\temp'
    dst_path = r'D:\data\data_scrawler\temp1'
    # class_labels = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    # validate_dataset(src_path, dst_path, class_labels)
    validate_all(src_path, dst_path)
