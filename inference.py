import numpy as np
import tensorflow as tf
import os
import cv2
import shutil
import scipy.special
import time
from tensorflow.python.framework import graph_util
from models.model_factory import get_model
from dataset.data_generator import ImageDataGenerator
from config import cfg
# cfg = cfg.TEST

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Tester(object):
    def __init__(self, load_data=True, load_model=True):
        self.model_name = cfg.MODEL_NAME
        self.class_names = cfg.CLASS_NAMES
        self.num_classes = cfg.NUM_CLASSES
        self.image_size = cfg.IMAGE_SIZE

        self.image_mean = np.array([121.55213, 113.84197, 99.5037])

        # test configurations
        if load_data:
            self.batch_size = cfg.TEST.BATCH_SIZE
            self.test_data_path = cfg.TEST.TEST_DATASET_PATH
            # load data
            self.test_set, self.next_batch = self.load_dataset()

            self.num_test = cfg.TEST.NUM_TEST
            if self.num_test == 0:
                self.num_test = self.test_set.data_size

        if load_model:
            self.input_node_name = cfg.TEST.INPUT_NODE_NAME
            self.output_node_name = cfg.TEST.OUTPUT_NODE_NAME
            self.dropout_node_name = cfg.TEST.DROPOUT_NODE_NAME

            # load model
            self.session = tf.Session()
            ckpt_path = cfg.TEST.CHECKPOINT_PATH
            self.model = self.load_model(ckpt_path)

    def load_dataset(self):
        print('loading data')
        with tf.device('/cpu:0'):
            test_dataset = ImageDataGenerator(txt_file=self.test_data_path,
                                              mode='inference',
                                              batch_size=self.batch_size,
                                              num_classes=self.num_classes,
                                              shuffle=True,
                                              img_size=self.image_size)
            test_next_batch = test_dataset.iterator.get_next()
        print('data loaded')
        return test_dataset, test_next_batch

    def load_model(self, ckpt_path):
        model = get_model(self.model_name)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        print(ckpt_path)
        saver.restore(self.session, ckpt_path)
        return model

    def test(self, num=None):
        num_batchs_one_validation = int(self.num_test / self.batch_size)
        acc_list = []
        if num is None:
            num = num_batchs_one_validation

        time_start = time.time()
        for i in range(num):
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
        time_per_sample = (time_end - time_start) * 1.0 / (num * self.batch_size)
        print('samples_per_sec = {}'.format(1.0 / time_per_sample))

        print("accuracy on test dataSet: {}".format(np.mean(acc_list)))

    def test_2class(self, csv_file, pos_labels, neg_labels):
        # pos_labels and neg_labels are integer indices, not label names
        img_paths, labels = self._read_csv_file(csv_file)
        num_images = len(img_paths)
        image_mean = np.array([121.55213, 113.84197, 99.5037])

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for idx, path in enumerate(img_paths):
            print('testing {}/{}'.format(idx, num_images))
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # do image preprocessing
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img - image_mean
            img = np.expand_dims(img, axis=0)

            if self.model.keep_prob is not None:
                feed_dict = {self.model.x_input: img,
                             self.model.keep_prob: 1.0}
            else:
                feed_dict = {self.model.x_input: img}

            logits = self.session.run(self.model.logits_val, feed_dict=feed_dict)
            logits = np.squeeze(logits, axis=0)

            # type 1 evaluation using max prob label
            result = np.argmax(logits)
            gt = np.argmax(labels[idx])

            if result in pos_labels and gt in pos_labels:
                true_positive += 1
            elif result in pos_labels and gt in neg_labels:
                false_positive += 1
            elif result in neg_labels and gt in pos_labels:
                false_negative += 1
            elif result in neg_labels and gt in neg_labels:
                true_negative += 1

            # type 2 evaluation using sum prob
            result_pos = np.sum([logits[k] for k in pos_labels])
            gt_pos = np.sum([labels[idx][k] for k in pos_labels])

            if result_pos > 0.5 and gt_pos > 0.5:
                true_positive += 1
            elif result_pos > 0.5 and gt_pos <= 0.5:
                false_positive += 1
            elif result_pos <= 0.5 and gt_pos > 0.5:
                false_positive += 1
            else:
                false_negative += 1

        print('tp = ', true_positive)
        print('fp = ', false_positive)
        print('tn = ', true_negative)
        print('fn = ', false_negative)

    def test_folder(self, folder_path, result_path):
        """
        Test all images in the folder and save results into separate folders
        """
        image_paths = os.listdir(folder_path)
        num_images = len(image_paths)

        # image_mean = np.array([121.55213, 113.84197, 99.5037])
        for idx, path in enumerate(image_paths):
            print('testing {}/{}'.format(idx, num_images))
            img_path = os.path.join(folder_path, path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # do image preprocessing
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img - self.image_mean
            img = np.expand_dims(img, axis=0)

            if self.model.keep_prob is not None:
                feed_dict = {self.model.x_input: img,
                             self.model.keep_prob: 1.0}
            else:
                feed_dict = {self.model.x_input: img}

            logits = self.session.run(self.model.logits_val, feed_dict=feed_dict)
            logits = np.squeeze(logits, axis=0)
            label = np.argmax(logits)

            result_name = self.class_names[label]
            result_sub_path = os.path.join(result_path, result_name)
            if not os.path.exists(result_sub_path):
                os.makedirs(result_sub_path)

            target_path = os.path.join(result_sub_path, path)

            # copy into result folder
            # print(img_path)
            # print(target_path)
            shutil.copyfile(img_path, target_path)

    def test_folder_iterative(self, folder_path, result_path):
        """
        Test all images in the folder and save results into separate folders
        """
        # image_paths = os.listdir(folder_path)
        # num_images = len(image_paths)

        # image_mean = np.array([121.55213, 113.84197, 99.5037])
        for root, files, dirs in os.walk(folder_path):
            for filename in files:
                img_path = os.path.join(root, filename)

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # do image preprocessing
                img = cv2.resize(img, (self.image_size, self.image_size))
                img = img - self.image_mean
                img = np.expand_dims(img, axis=0)

                if self.model.keep_prob is not None:
                    feed_dict = {self.model.x_input: img,
                                 self.model.keep_prob: 1.0}
                else:
                    feed_dict = {self.model.x_input: img}

                logits = self.session.run(self.model.logits_val, feed_dict=feed_dict)
                logits = np.squeeze(logits, axis=0)
                label = np.argmax(logits)

                result_name = self.class_names[label]
                result_sub_path = os.path.join(result_path, result_name)
                if not os.path.exists(result_sub_path):
                    os.makedirs(result_sub_path)

                target_path = os.path.join(result_sub_path, path)

                # copy into result folder
                # print(img_path)
                # print(target_path)
                shutil.copyfile(img_path, target_path)

    def save_pb(self, pb_path):
        graph = tf.get_default_graph()
        # self.print_all_nodes(graph, r'D:\graph_resnet.txt')

        # inputs = graph.get_tensor_by_name(self.input_node_name + ':0')
        # outputs = graph.get_tensor_by_name(self.output_node_name + ':0')
        # if self.dropout_node_name is not None:
        #     dropout = graph.get_tensor_by_name(self.dropout_node_name + ':0')
        # else:
        #     dropout = None

        graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            self.session,
            graph_def,
            [self.output_node_name]
        )
        with tf.gfile.GFile(pb_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

    @staticmethod
    def print_all_nodes(graph, fname=None):
        # check nodes of the graph
        if fname is None:
            for node in graph.as_graph_def().node:
                print(node.name)
        else:
            with open(fname, 'w') as f:
                for node in graph.as_graph_def().node:
                    f.write('%s\n' % node.name)

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

    def test_jpeg_file(self, image_file):
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # do image preprocessing
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img - self.image_mean
        img = np.expand_dims(img, axis=0)

        if self.model.keep_prob is not None:
            feed_dict = {self.model.x_input: img,
                         self.model.keep_prob: 1.0}
        else:
            feed_dict = {self.model.x_input: img}

        logits = self.session.run(self.model.logits_val, feed_dict=feed_dict)
        logits = np.squeeze(logits, axis=0)
        # print(logits)
        s = scipy.special.softmax(logits)
        # print(s)
        return s

    def test_gif_file(self, gif_file):
        vid = cv2.VideoCapture(gif_file)
        if not vid.isOpened():
            return []

        logits_list = []
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            img = cv2.resize(frame, (self.image_size, self.image_size))
            img = img - self.image_mean
            img = np.expand_dims(img, axis=0)

            if self.model.keep_prob is not None:
                feed_dict = {self.model.x_input: img,
                             self.model.keep_prob: 1.0}
            else:
                feed_dict = {self.model.x_input: img}

            logits = self.session.run(self.model.logits_val, feed_dict=feed_dict)
            logits = np.squeeze(logits, axis=0)
            s = scipy.special.softmax(logits)
            logits_list.append(s)
        return logits_list

    def test_21cn_invalid_38(self, folder):
        fnames = os.listdir(folder)
        for name in fnames:
            print(name)
            path = os.path.join(folder, name)
            if path.endswith('.gif'):
                res = self.test_gif_file(path)
                for a in res:
                    self.print_array(a)
            else:
                res = self.test_jpeg_file(path)
                self.print_array(res)

    @staticmethod
    def print_array(array):
        n = array.shape[0]
        buf = ','.join(str(array[i]) for i in range(n))
        print(buf)


if __name__ == '__main__':
    # 测试训练好的模型，测试参数见config.py
    # when testing, make sure that the configuration of model matches
    # the configuration used in the training
    # - activation_fn
    # - sgd/adam
    tester = Tester(load_data=False)
    # tester.test(num=100)
    # tester.save_pb(r'D:/resnetv1_50_violence_clf.pb')
    # invalid_21cn = r'D:\data\tests\ng38'
    # tester.test_21cn_invalid_38(invalid_21cn)
    folder = r'D:\data\baokong\baokong'
    tester.test_folder(folder, './result.txt')
