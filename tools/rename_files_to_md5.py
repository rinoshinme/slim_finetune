import hashlib
import os
import shutil

JUNK_DIR = r'D:\data\baokong21cn\junk'


def get_md5_value(filename):
    m = hashlib.md5()
    with open(filename, 'rb') as f:
        data = f.read()
    m.update(data)
    return m.hexdigest()


def rename_file(file_dir, file_name):
    file_path = os.path.join(file_dir, file_name)
    md5_val = get_md5_value(file_path)
    file_name_fields = file_name.split('.')
    if len(file_name_fields) == 1:
        new_name = md5_val
    else:
        ext = file_name_fields[-1]
        new_name = '%s.%s' % (md5_val, ext)
    new_path = os.path.join(file_dir, new_name)
    if not os.path.exists(new_path):
        shutil.move(file_path, new_path)
    else:
        new_path = os.path.join(JUNK_DIR, new_name)
        shutil.move(file_path, new_path)


if __name__ == '__main__':
    image_dir = r'D:\data\baokong21cn\正常3'
    fnames = os.listdir(image_dir)
    for name in fnames:
        rename_file(image_dir, name)
