import numpy as np
import tensorflow as tf
import os
import cv2
import shutil
import random
from tensorflow.python.framework import graph_util
from models.model_factory import get_model
# from config import cfg
from utils.image import is_image_file

IMAGE_EXTS = ['.jpg', '.png', '.jpeg', ]


class MetricTester(object):
    def __init__(self, config):
        self.model_name = config['model_name']  # cfg.MODEL_NAME
        self.class_names = config['class_names']  # cfg.CLASS_NAMES
        self.num_classes = config['num_classes']  # cfg.NUM_CLASSES

        self.name2idx = {self.class_names[idx]: idx for idx in range(self.num_classes)}
        self.idx2name = {idx: self.class_names[idx] for idx in range(self.num_classes)}

        self.image_size = config['image_size']  # cfg.IMAGE_SIZE
        self.ng_indices = config['ng_indices']

        self.session = tf.Session()
        self.model = self.load_model(config['ckpt_path'])

    def load_model(self, ckpt_path):
        model = get_model(self.model_name, self.num_classes)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(self.session, ckpt_path)
        return model

    def test_folder(self, folder_path):
        """
        Test all images in the folder and save result probability into text file
        TODO: Contain some bugs, accuracy is lower than testing using tensorflow dataset.
        """
        image_paths = os.listdir(folder_path)
        # image_paths = [f for f in image_paths if f.endswith('.jpg')]
        # image_paths = image_paths[:20]
        num_images = len(image_paths)
        image_mean = np.array([121.55213, 113.84197, 99.5037])

        paths = []
        probs = []
        print(self.model.logits_val.get_shape().as_list())
        softmax_op = tf.nn.softmax(self.model.logits_val)
        for idx, path in enumerate(image_paths):
            print('testing {}/{}'.format(idx, num_images))
            img_path = os.path.join(folder_path, path)
            img = cv2.imread(img_path)
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

            # logits = self.session.run(self.model.logits_val, feed_dict=feed_dict)
            # logits = np.squeeze(logits, axis=0)
            # label = np.argmax(logits)

            softmax_results = self.session.run(softmax_op, feed_dict=feed_dict)
            softmax_results = np.squeeze(softmax_results, axis=0)

            prob_values = [float(softmax_results[i]) for i in range(self.num_classes)]

            paths.append(img_path)
            probs.append(prob_values)

            del img
            del feed_dict

        txt_path = folder_path + '.txt'
        self.write_probs(paths, probs, txt_path)

    @staticmethod
    def read_probs(txt_file):
        paths = []
        probs = []
        with open(txt_file, 'r') as fp:
            for line in fp.readlines():
                parts = line.split('\t')
                values = parts[1]
                prob_values = [float(v) for v in values.split(',')]
                paths.append(parts[0])
                probs.append(prob_values)
        return paths, probs

    def write_probs(self, paths, probs, txt_file):
        with open(txt_file, 'w') as fp:
            for path, prob in zip(paths, probs):
                prob_values = [str(prob[i]) for i in range(self.num_classes)]
                prob_text = ','.join(prob_values)
                fp.write('%s\t%s\n' % (path, prob_text))

    def calculate_acc(self, root_folder):
        accs = []
        total_positive = 0
        total_num = 0
        for cls_idx, name in enumerate(self.class_names):
            num_positive = 0
            txt_file = os.path.join(root_folder, name + '.txt')
            if not os.path.exists(txt_file):
                continue
            paths, probs = self.read_probs(txt_file)
            num_category = len(paths)
            for prob in probs:
                max_idx = np.argmax(prob)
                if max_idx == cls_idx:
                    num_positive += 1
            total_positive += num_positive
            total_num += num_category
            accs.append(num_positive * 1.0 / num_category)
        total_acc = total_positive * 1.0 / total_num
        return accs, total_acc

    def calculate_acc_binary(self, root_folder, positive_labels, threshold=0.5):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        positive_idx = [self.name2idx[name] for name in positive_labels]
        # negative_idx = [self.name2idx[name] for name in negative_labels]
        for cls_idx, name in enumerate(self.class_names):
            txt_path = os.path.join(root_folder, name + '.txt')
            paths, probs = self.read_probs(txt_path)
            for prob in probs:
                sum_positive = sum(prob[i] for i in positive_idx)
                # sum_negative = sum(prob[i] for i in negative_idx)
                if name in positive_labels and sum_positive >= threshold:
                    true_positive += 1
                elif name in positive_labels and sum_positive < threshold:
                    false_negative += 1
                elif name not in positive_labels and sum_positive >= threshold:
                    false_positive += 1
                else:
                    true_negative += 1
        return true_positive, false_positive, true_negative, false_negative

    def check_result(self, result_text, target_folder, thresh=None):
        paths, probs = self.read_probs(result_text)
        for idx, path in enumerate(paths):
            print(path)
            max_prob = np.max(probs[idx])
            if thresh is not None and max_prob < thresh:
                continue
            result_label = np.argmax(probs[idx])
            result_name = self.class_names[result_label]
            result_folder = os.path.join(target_folder, result_name)
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            base_name = os.path.basename(path)
            result_path = os.path.join(result_folder, base_name)
            shutil.copyfile(path, result_path)

    @staticmethod
    def read_paths(text_path):
        if not os.path.exists(text_path):
            return []
        file_paths = []
        with open(text_path, 'r') as fp:
            for line in fp.readlines():
                fields = line.split('\t')
                file_paths.append(fields[0])
        return file_paths

    def test_folder_all(self, folder, text_path):
        """
        test on all image files in folder and save results
        """
        print('start testing...')
        image_mean = np.array([121.55213, 113.84197, 99.5037])

        # ignore files that are already tested and saved in the text file.
        filepaths = self.read_paths(text_path)
        # cnt = len(filepaths)
        cnt = 0
        # print(self.model.logits_val.get_shape().as_list())
        softmax_op = tf.nn.softmax(self.model.logits_val)
        f = open(text_path, 'a+')
        for root, dirs, files in os.walk(folder):
            for name in files:
                cnt += 1
                img_path = os.path.join(root, name)
                if not is_image_file(img_path):
                    continue
                # skip detected images
                # print('testing: %06d - %s' % (cnt, img_path))
                if not any([name.lower().endswith(ext) for ext in IMAGE_EXTS]):
                    continue
                try:
                    if img_path in filepaths:
                        continue
                    # img = cv2.imread(img_path)
                    buf = np.fromfile(img_path, dtype=np.uint8)
                    if buf is not None:
                        img = cv2.imdecode(buf, -1)
                    else:
                        continue
                    # ignore bad images
                    if img is None:
                        print('image is None')
                        continue
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    if img.shape[2] != 3:
                        print('image not 3 channel')
                        continue

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

                    # logits = self.session.run(self.model.logits_val, feed_dict=feed_dict)
                    # logits = np.squeeze(logits, axis=0)
                    # label = np.argmax(logits)
                    softmax_results = self.session.run(softmax_op, feed_dict=feed_dict)
                    softmax_results = np.squeeze(softmax_results, axis=0)

                    probs = [float(softmax_results[i]) for i in range(self.num_classes)]
                    # if self.num_classes == 9:
                    #     violence_index = sum([probs[k] for k in [1, 2, 3, 5, 7, 8]])
                    #     # violence_index = probs[1] + probs[2]
                    # elif self.num_classes == 3:
                    #     violence_index = probs[1] * 0.5 + probs[2]
                    # elif self.num_classes == 7:
                    #     violence_index = sum([probs[i] * self.ng_indices[i] for i in range(self.num_classes)])
                    # else:
                    #     violence_index = 0
                    violence_index = sum([probs[i] * self.ng_indices[i] for i in range(self.num_classes)])
                    print('testing %d - %s: %f' % (cnt, img_path, violence_index))

                    prob_values = [str(probs[i]) for i in range(self.num_classes)]
                    prob_text = ','.join(prob_values)
                    f.write('%s\t%s\n' % (img_path, prob_text))
                    f.flush()

                    del buf
                    del img
                    del feed_dict
                except RuntimeError as e:
                    print('RuntimeError: ', e)
                    continue
                except Exception as e:
                    print('Exception: ', e)
                    continue
        f.close()

    def move_files(self, text_file, target_folder, threshold=None, move_indices=None):
        with open(text_file, 'r') as fp:
            for line in fp.readlines():
                fields = line.split('\t')
                path = fields[0]
                # ignore files that are already moved.
                if not os.path.exists(path):
                    continue
                base_name = os.path.basename(path)
                scores = [float(v) for v in fields[1].split(',')]
                max_index = np.argmax(scores)

                if move_indices is not None and max_index not in move_indices:
                    continue

                max_score = np.max(scores)

                # ignore low-confidence results
                if threshold is not None and max_score < threshold:
                    continue
                label = self.class_names[max_index]
                sub_target_folder = os.path.join(target_folder, label)
                if not os.path.exists(sub_target_folder):
                    os.makedirs(sub_target_folder)
                target_path = os.path.join(sub_target_folder, base_name)
                # tmp = random.randint(1, 10000000)
                # target_path = os.path.join(target_folder, label, '%08d_%s' % (tmp, base_name))

                # prevent file over-writing
                while os.path.exists(target_path):
                    tmp = random.randint(1, 10000000)
                    target_path = os.path.join(target_folder, label, '%08d_%s' % (tmp, base_name))
                shutil.move(path, target_path)

    def save_pb(self, pb_path):
        graph = tf.get_default_graph()
        # self.print_all_nodes(graph, r'D:\graph_resnet.txt')

        # inputs = graph.get_tensor_by_name(self.input_node_name + ':0')
        # outputs = graph.get_tensor_by_name(self.output_node_name + ':0')
        # if self.dropout_node_name is not None:
        #     dropout = graph.get_tensor_by_name(self.dropout_node_name + ':0')
        # else:
        #     dropout = None

        output_node_name = 'resnet_v1_101_1/predictions/Softmax'
        graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            self.session,
            graph_def,
            [output_node_name]
        )
        with tf.gfile.GFile(pb_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    cfg_bloody = dict({
        'model_name': 'ResNetV1_101',
        'class_names': ['normal', 'medium', 'bloody'],
        'ng_indices': [0, 0.5, 1],
        'num_classes': 3,
        'image_size': 224,
        # 3 category model v1
        # 'ckpt_path': r'E:\Training\output_finetune\ResNetV1_101\20191018_051256\ckpt\model-6000',
        # 3 category model v2
        'ckpt_path': r'F:\Training\output_finetune\ResNetV1_101\20191018_102037\ckpt\model-12000'
    })

    cfg_violence = dict({
        'model_name': 'ResNetV1_101',
        'class_names': ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest'],
        'ng_indices': [0, 1, 1, 1, 0, 1, 0, 1, 1],
        'num_classes': 9,
        'image_size': 224,
        # 9 category model
        'ckpt_path': r'D:\incoming\ResNetV1_101_20190719_132123\ckpt\model-25000'
    })

    cfg_baokong7 = dict({
        'model_name': 'ResNetV1_101',
        'class_names': ['normal', 'army', 'weapon', 'fire', 'bloody', 'terrorism', 'terrorflag'],
        'ng_indices': [0, 0, 0, 1, 1, 1, 1],
        'num_classes': 7,
        'image_size': 224,
        # 7 category model
        'ckpt_path': r'F:\Training\output_finetune\ResNetV1_101\20191029_060119\ckpt\model-12000'
    })

    # Lost?
    cfg_baokong7_v2 = dict({
        'model_name': 'ResNetV1_101',
        'class_names': ['normal', 'army', 'weapon', 'fire', 'bloody', 'mild_bloody', 'terrorism'],
        'ng_indices': [0, 0, 0, 1, 1, 0.5, 1],
        'num_classes': 7,
        'image_size': 224,
        # 7 category model
        'ckpt_path': r'F:\Training\output_finetune\ResNetV1_101\20191106_072716\ckpt\model-18000'
    })

    tester = MetricTester(cfg_violence)
    tester.save_pb(r'D:\baokong7.pb')

    test_folder = r'D:\data\baokong\baokong'
    test_text = r'D:\data\baokong\baokong.txt'
    split_path = r'E:\dataset_utils\splits9'

    task = 'run'  # ['run', 'split']

    if task == 'run':
        tester.test_folder_all(test_folder, test_text)
    elif task == 'split':
        thresholds = [99, 98, 97, 96, 95, 90, 70, 50, 10, 0]
        # thresholds = [99, 98, 95]
        for th in thresholds:
            sub_path = os.path.join(split_path,  '{}'.format(th))
            tester.move_files(test_text, sub_path, threshold=th/100.0)
