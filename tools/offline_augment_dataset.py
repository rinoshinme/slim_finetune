import os
import random
import cv2
import utils.augment_cv as augment_cv


def dataset_augmentation(root_dir, class_names, aug_ratio=1.0):
    for name in class_names:
        # print('processing {}'.format(name))
        subpath = os.path.join(root_dir, name)
        files = os.listdir(subpath)
        num_augs = int(aug_ratio * len(files))
        aug_files = random.sample(files, num_augs)
        for idx, f in enumerate(aug_files):
            print('processing {}: {} / {}'.format(name, idx, num_augs))
            img = cv2.imread(os.path.join(subpath, f))

            # do some augmentation
            img = augment_cv.random_rotate(img)
            img = augment_cv.random_channel_mutate(img)

            save_file = os.path.join(subpath, 'aug_%06d.jpg' % idx)
            cv2.imwrite(save_file, img)


if __name__ == '__main__':
    # only training images are augmented
    train_dir = r'E:\DATASET2019\baokong13_20190731\train'
    classnames = ['normal', 'army', 'bloody', 'crash', 'fire', 'identity',
                  'normal_artificial', 'normal_crowd', 'normal_document',
                  'protest', 'riot', 'terrorism', 'weapon']

    dataset_augmentation(train_dir, classnames, aug_ratio=1.0)
