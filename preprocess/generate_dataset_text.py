import os
import random


def generate_text(folder, classnames, target_text, shuffle=False):
    dataset = []
    for idx, name in enumerate(classnames):
        subfolder = os.path.join(folder, name)
        for fname in os.listdir(subfolder):
            filepath = os.path.join(subfolder, fname)
            dataset.append((filepath, idx))

    if shuffle:
        random.shuffle(dataset)

    # save to text file
    with open(target_text, 'w') as fp:
        for path, idx in dataset:
            fp.write('%s\t%s\n' % (path, idx))


if __name__ == '__main__':
    root_dir = r'E:\DATASET2019\baokong09_20190717'
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')

    # class_names = ['normal', 'army', 'bloody', 'crash', 'fire', 'identity',
    #                'normal_artificial', 'normal_crowd', 'normal_document',
    #                'protest', 'riot', 'terrorism', 'weapon']
    class_names = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']

    for d in [train_dir, val_dir, test_dir]:
        generate_text(d, class_names, d + '.txt', shuffle=False)
