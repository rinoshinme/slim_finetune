import numpy as np
import os
import scipy.misc
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework import dtypes
import tensorflow as tf
from collections import defaultdict
import random


class ImageDataGenerator(object):
    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=300, img_size=224, model_name=None):
        self.num_classes = num_classes
        print('num_classes = ', self.num_classes)
        self.image_size = img_size
        self.model_name = model_name
        self.IMAGENET_MEAN = tf.constant([121.55213, 113.84197, 99.5037], dtype=tf.float32)

        # self.img_paths is a list of strings
        # self.labels is a list of [probs/one-hot vectors]
        self.img_paths, self.labels = self._read_txt_file(txt_file)

        if mode == 'training':
            # each category has at least target_size sample lines.
            self.img_paths, self.labels = self._augment_list(self.img_paths, self.labels, target_size=10000)

        self.data_size = len(self.img_paths)

        if shuffle:
            self.img_paths, self.labels = self._shuffle_lists(self.img_paths, self.labels)

        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.float32)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # parse image
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=20)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=20)
        else:
            raise ValueError('Invalid mode {}'.format(mode))

        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # make new dataset with batches of images
        data = data.batch(batch_size)
        data = data.repeat()
        self.iterator = data.make_one_shot_iterator()
    
    def __len__(self):
        return self.data_size

    def _read_txt_file(self, txt_file):
        """
        labels are represented in one-hot format
        """
        img_paths = []
        labels = []
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                items = line.split('\t')
                # do not use interpolation-augmented images
                if os.path.basename(items[0]).startswith('aug'):
                    continue
                img_paths.append(items[0])
                label = int(items[1])
                label = self.one_hot_label(label)
                labels.append(label)
        return img_paths, labels

    def _augment_list(self, img_paths, labels, target_size=6000):
        samples = defaultdict(list)

        for path, label in zip(img_paths, labels):
            idx = self._get_label_from_onehot(label)
            samples[idx].append(path)

        img_paths_result = []
        labels_result = []
        for label in samples.keys():
            # print(label)
            size = len(samples[label])
            if size < target_size:
                more_samples = self._get_random_samples(samples[label], target_size - size)
                img_paths_result.extend(samples[label])
                img_paths_result.extend(more_samples)
                label_samples = [label for _ in range(len(samples[label]) + len(more_samples))]
                labels_result.extend(label_samples)
            else:
                img_paths_result.extend(samples[label])
                label_samples = [label for _ in range(len(samples[label]))]
                labels_result.extend(label_samples)

        # turn labels_result into one_hot
        labels_result = [self.one_hot_label(v) for v in labels_result]
        return img_paths_result, labels_result

    @staticmethod
    def _get_random_samples(samples, size):
        more_samples = []
        data_size = len(samples)
        for i in range(size):
            idx = random.randint(0, data_size - 1)
            more_samples.append(samples[idx])
        return more_samples

    @staticmethod
    def _read_csv_file(csv_file):
        img_paths = []
        labels = []
        with open(csv_file, 'r') as f:
            for line in f.readlines():
                items = line.split(',')
                img_paths.append(items[0])
                probs = items[1:]
                probs = [float(s) for s in probs]
                labels.append(probs)
        return img_paths, labels

    @staticmethod
    def _get_label_from_onehot(one_hot):
        return one_hot.index(1)

    @staticmethod
    def _shuffle_lists(img_list, label_list):
        assert len(img_list) == len(label_list)
        data_size = len(img_list)
        permutation = np.random.permutation(data_size)
        rimg_list = []
        rlabel_list = []
        for i in permutation:
            rimg_list.append(img_list[i])
            rlabel_list.append(label_list[i])
        return rimg_list, rlabel_list

    def one_hot_label(self, label):
        oh = [0 for _ in range(self.num_classes)]
        oh[label] = 1
        return oh

    def _parse_function_train(self, filename, label):
        # one_hot = tf.one_hot(label, self.num_classes)
        one_hot = label
        img_string = tf.read_file(filename)
        # decode image as RGB format, be careful when using opencv to run interence
        img_decode = tf.image.decode_jpeg(img_string, channels=3)

        # do some image augmentation
        img_enhance = self._data_augment(img_decode, model_type=self.image_size)

        img = tf.image.resize_images(img_enhance, [self.image_size, self.image_size])
        if isinstance(self.model_name, str) and self.model_name.startswith('efficientnet'):
            # efficientnet has its own normalization function.
            img_centered = img
        else:
            img_centered = tf.subtract(img, self.IMAGENET_MEAN)
        return img_centered, one_hot

    def _parse_function_inference(self, filename, label):
        one_hot = label
        img_string = tf.read_file(filename)
        # decode image as RGB format, be careful when using opencv to run interence
        img_decode = tf.image.decode_jpeg(img_string, channels=3)
        img_resize = tf.image.resize_images(img_decode, [self.image_size, self.image_size])
        img_centered = tf.subtract(img_resize, self.IMAGENET_MEAN)
        return img_centered, one_hot

    @staticmethod
    def _data_augment(img, model_type=224):
        # img = tf.image.random_jpeg_quality(img, 50, 90)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=64./255)
        img = tf.image.random_contrast(img, lower=0.4, upper=1.5)

        # random rotation
        # img = tf.py_func(self.random_rotate_func, [img], tf.uint8)

        # add random crop
        if model_type == 224:
            img = tf.image.resize_images(img, [256, 256])
            img = tf.image.random_crop(img, [224, 224, 3])
        elif model_type == 299:
            # no cropping for inception model
            pass
        else:
            raise ValueError("model type not supported")
        return img

    @staticmethod
    def random_rotate_func(image, low=-30.0, high=30.0):
        angle = np.random.uniform(low=low, high=high)
        return scipy.misc.imrotate(image, angle, interp='bicubic')


if __name__ == '__main__':
    # test convert_to_tensor
    vec = [[0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0]]
    t = convert_to_tensor(vec, tf.float32)
    print(t.shape)
