import numpy as np
import tensorflow as tf
import os
import cv2
import shutil
import scipy.special
import time
from tensorflow.python.framework import graph_util

from tfvortex.models.model_factory import get_model
from tfvortex.dataset.data_generator import ImageDataGenerator


class Tester(object):
    def __init__(self, options):
        self.options = options
        self.model_name = self.options.MODEL_NAME
        self.class_names = self.options.CLASS_NAMES
        self.num_classes = self.options.NUM_CLASSES
        self.image_size = self.options.IMAGE_SIZE
        self.image_mean = np.array([121.55213, 113.84197, 99.5037])

        self.dataset, self.next_batch = self.load_data()
        self.session = tf.Session()
        self.model = self.load_model(self.session)
        

    def load_data(self):
        print('loading data')
        with tf.device('/cpu:0'):
            test_dataset = ImageDataGenerator(txt_file=self.options.TEST_DATASET_PATH,
                                              mode='inference',
                                              batch_size=self.options.BATCH_SIZE,
                                              num_classes=self.num_classes,
                                              shuffle=True,
                                              img_size=self.image_size)
            test_next_batch = test_dataset.iterator.get_next()
        print('data loaded')
        return test_dataset, test_next_batch

    def load_model(self, session):
        model = get_model(self.options)
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(session, self.options.CKPT_PATH)
        return model

    def test(self, save_path=None):
        acc_list = []
        niter = len(self.dataset) // self.options.BATCH_SIZE
        time_start = time.time()
        for i in range(niter):
            print('processing batch %d' % i)
            x_batch_test, y_batch_test = self.session.run(self.next_batch)

            if self.model.keep_prob is not None:
                feed_dict = {self.model.x_input: x_batch_test,
                             self.model.y_input: y_batch_test,
                             self.model.keep_prob: 1.0}
            else:
                feed_dict = {self.model.x_input: x_batch_test,
                             self.model.y_input: y_batch_test}

            accuracy = self.session.run(self.model.accuracy, feed_dict=feed_dict)
            print('accuracy = ', accuracy)

            acc_list.append(accuracy)
        time_end = time.time()
        time_per_sample = (time_end - time_start) * 1.0 / (niter * self.options.BATCH_SIZE)
        print('samples_per_sec = {}'.format(1.0 / time_per_sample))

        print("accuracy on test dataSet: {}".format(np.mean(acc_list)))
