import tensorflow as tf
import time
from models.model_factory import get_model
from dataset.data_generator import ImageDataGenerator


class Tester(object):
    def __init__(self, cfg):
        self.class_names = cfg.CLASS_NAMES
        self.num_classes = cfg.NUM_CLASSES
        self.model_name = cfg.MODEL_NAME
        self.image_size = cfg.IMAGE_SIZE
        self.ckpt_path = cfg.CHECKPOINT_PATH
        self.dataset_path = cfg.DATASET_PATH
        self.batch_size = cfg.BATCH_SIZE

        self.dataset, self.next_batch = self.load_data()
        self.session, self.model = self.load_model()
        self.input_node_name = cfg.INPUT_NODE_NAME
        self.output_node_name = cfg.OUTPUT_NODE_NAME
        self.dropout_node_name = cfg.DROPOUT_NODE_NAME

    def run(self):
        nbatch = len(self.dataset) // self.batch_size
        for i in range(nbatch):
            print('processing batch {}'.format(i))
            batch_x, batch_y = self.session.run(self.next_batch)
            if self.model.keep_prob is not None:
                feed_dict = {self.model.x_input: batch_x,
                             self.model.y_input: batch_y,
                             self.model.keep_prob: 1.0}
            else:
                feed_dict = {self.model.x_input: batch_x,
                             self.model.y_input: batch_y}
            t1 = time.time()
            accuracy = self.session.run(self.model.accuracy, feed_dict=feed_dict)
            t2 = time.time()
            print('inference time: {:.06f}s'.format(t2 - t1))
            print('accuracy = ', accuracy)

    def load_model(self):
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        model = get_model(self.model_name, self.num_classes)
        # load model weights
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(session, self.ckpt_path)
        return session, model

    def load_data(self):
        print('loading data')
        with tf.device('/cpu:0'):
            dataset = ImageDataGenerator(self.dataset_path, 
                                         mode='inference', 
                                         batch_size=self.batch_size,
                                         num_classes=self.num_classes, 
                                         shuffle=False, 
                                         img_size=self.image_size)
            next_batch = dataset.iterator.get_next()
        print('done loading data')
        return dataset, next_batch
