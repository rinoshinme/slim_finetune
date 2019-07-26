import os
import random
import platform

CLASS_NAMES = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']
os_type = platform.system()
if os_type == 'Windows':
    SRC_DIR = r'D:\data\DATASET2019\baokong10'
elif os_type == 'Linux':
    SRC_DIR = '/home/deploy/rinoshinme/data/violence_data'
else:
    raise ValueError('OS type not supported')


def generate_dataset(root_folder, class_names, shuffle=False):
    """
    generate list of (path, cls_index) pairs from a sorted folder
    """
    dataset = []
    for i, name in enumerate(class_names):
        sub_folder = os.path.join(root_folder, name)
        if not os.path.exists(sub_folder):
            continue
        files = os.listdir(sub_folder)
        for f in files:
            path = os.path.join(sub_folder, f)
            dataset.append((path, i))
    if shuffle:
        random.shuffle(dataset)
    return dataset


def split_dataset(dataset, val_ratio=0.1, test_ratio=0.2):
    """
    split (path, cls_idx) dataset into train, val, test datasets
    """
    assert (val_ratio > 0.0) and (val_ratio < 1.0)
    num_samples = len(dataset)
    num_val = int(num_samples * val_ratio)
    num_test = int(num_samples * test_ratio)
    num_train = num_samples - num_val - num_test
    # 0 for train, 1 for val, 2 for test
    flags = [0 for _ in range(num_train)] + [1 for _ in range(num_val)] + [2 for _ in range(num_test)]
    random.shuffle(flags)
    train_dataset = [dataset[k] for k in range(num_samples) if flags[k] == 0]
    val_dataset = [dataset[k] for k in range(num_samples) if flags[k] == 1]
    test_dataset = [dataset[k] for k in range(num_samples) if flags[k] == 2]
    return train_dataset, val_dataset, test_dataset


def save_dataset(dataset, filepath, sep='\t'):
    """
    save (path, cls_idx) dataset into folder
    each line is a sample separated with sep
    """
    with open(filepath, 'w') as f:
        for item in dataset:
            f.write('%s%s%d\n' % (item[0], sep, item[1]))


def generate_server_test_txt(server_txt, host_txt, data_root):
    """
    transfer dataset txt from server to host.
    Paths on server have different root directory, so only base name
        is extracted and mapped to host root directory.
    """
    labels = []
    host_paths = []
    with open(server_txt, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            path = parts[0]
            label = parts[1]
            fields = path.split('/')
            host_path = os.path.join(data_root, fields[-2], fields[-1])
            labels.append(label)
            host_paths.append(host_path)

    with open(host_txt, 'w') as f:
        for i in range(len(labels)):
            f.write('%s %s\n' % (host_paths[i], labels[i]))


def split_softlabel_dataset(csv_file):
    # read dataset
    pass


def run_generate_dataset():
    """
    RUN: generate 3 text files of dataset
    """

    dataset = generate_dataset(SRC_DIR, CLASS_NAMES, False)
    train_set, val_set, test_set = split_dataset(dataset, val_ratio=0.1, test_ratio=0.2)

    save_dataset(train_set, os.path.join(SRC_DIR, 'train.txt'))
    save_dataset(val_set, os.path.join(SRC_DIR, 'val.txt'))
    save_dataset(test_set, os.path.join(SRC_DIR, 'test.txt'))


def run_generate_server_txt():
    """
    RUN: transform server text file into host text file
    """
    data_root = r'D:\data\violence_data\violence_data'
    server_txt = os.path.join(data_root, 'test_server.txt')
    host_txt = os.path.join(data_root, 'test_host.txt')
    generate_server_test_txt(server_txt, host_txt, data_root)


if __name__ == '__main__':
    run_generate_dataset()
    # run_generate_server_txt()
