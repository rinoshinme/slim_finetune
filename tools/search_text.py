"""
search text within folder
"""
import os

FILE_EXTS = ['.py', '.sh', '.conf']


def check_text(file, text):
    with open(file, 'rt', encoding='utf-8') as fp:
        data = fp.read()
        if text in data:
            return True
        else:
            return False


def search_text(root_dir, target_text):
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            if any([name.endswith(ext) for ext in FILE_EXTS]):
                file_path = os.path.join(root, name)
                # print(file_path)
                if check_text(file_path, target_text):
                    print('--------result:', file_path)


if __name__ == '__main__':
    directory = r'D:\code\object detection\balancap.SSD-Tensorflow'
    # text = 'saved_model.pb'
    text = 'no_annotation_label'
    search_text(directory, text)
