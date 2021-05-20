import os
import random
import copy


def data_aug(filenames, size):
    if len(filenames) > size:
        return
    result = []
    fnames = copy.copy(filenames)
    for i in range(size):
        if i % size == 0:
            random.shuffle(fnames)
        result.append(i % size)
    return result


def generate_data(data_root, target_folder, class_names, shuffle=True):
    train_items = []
    val_items = []
    for idx, name in enumerate(class_names):
        subfolder = os.path.join(data_root, name)
        filenames = os.listdir(subfolder)
        filepaths = [os.path.join(subfolder, f) for f in filenames]
        if shuffle:
            random.shuffle(filepaths)
        nfiles = len(filepaths)
        nval = int(nfiles * 0.15)
        trainpaths = filepaths[nval:]
        train_paths = data_aug(trainpaths, 10000)
        valpaths = filepaths[:nval]
        valpaths = data_aug(valpaths, 15000)
        for p in trainpaths:
            train_items.append((p, idx))
        for p in valpaths:
            val_items.append((p, idx))

    random.shuffle(train_items)
    random.shuffle(val_items)

    with open(os.path.join(target_folder, 'train.txt'), 'w') as f:
        for item in train_items:
            f.write('{}\t{}\n'.format(item[0], item[1]))
    with open(os.path.join(target_folder, 'test.txt'), 'w') as f:
        for item in val_items:
            f.write('{}\t{}\n'.format(item[0], item[1]))


if __name__ == '__main__':
    class_names = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']
    data_root = '/rinoshinme/data/violence/sorted'

    generate_data(data_root, '../data/', class_names, shuffle=True)
