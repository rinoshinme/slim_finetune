import os
import random


def generate_data(data_root, text_path, class_names, shuffle=True):
    items = []
    for idx, name in enumerate(class_names):
        subfolder = os.path.join(data_root, name)
        filenames = os.listdir(subfolder)
        filepaths = [os.path.join(subfolder, f) for f in filenames]
        for p in filepaths:
            items.append((p, idx))
    
    if shuffle:
        random.shuffle(items)
    
    with open(text_path, 'w') as f:
        for item in items:
            f.write('{}\t{}\n'.format(item[0], item[1]))


if __name__ == '__main__':
    class_names = ['normal', 'bloody', 'fire', 'army', 'terrorism', 'terrorflag', 'weapon']
    data_root = '/rinoshinme/data/classification/baokong07_20191029/'
    generate_data(os.path.join(data_root, 'train'), './data/train.txt', class_names, shuffle=True)
    generate_data(os.path.join(data_root, 'test'), './data/test.txt', class_names, shuffle=True)
