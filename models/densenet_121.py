import tensorflow as tf
from nets import densenet
from models.model_utils import load_initial_weights
from tensorflow.contrib.slim import arg_scope
import os
from config import cfg

DEFAULT_TRAIN_LAYERS = ['logits']


class DenseNet121(object):
    def __init__(self, num_classes, train_layers=None, weights_path='DEFAULT'):
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = os.path.join(cfg.TRAIN.PRETRAINED_WEIGHT_PATH, 'tf-densenet121.ckpt')
        else:
            self.WEIGHTS_PATH = weights_path

        if train_layers == 'DEFAULT':
            self.train_layers = DEFAULT_TRAIN_LAYERS
        else:
            self.train_layers = train_layers

        with tf.variable_scope("input"):
            self.image_size = cfg.IMAGE_SIZE
            self.x_input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="x_input")
            self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.keep_prob = None

            # train
        with arg_scope(densenet.densenet_arg_scope()):
            self.logits, _ = densenet.densenet121(self.x_input,
                                                  num_classes=num_classes,
                                                  is_training=True,
                                                  reuse=tf.AUTO_REUSE)
            self.logits = tf.squeeze(self.logits, [1, 2])

            # validation
        with arg_scope(densenet.densenet_arg_scope()):
            self.logits_val, _ = densenet.densenet121(self.x_input,
                                                      num_classes=num_classes,
                                                      is_training=False,
                                                      reuse=tf.AUTO_REUSE)
            self.logits_val = tf.squeeze(self.logits_val, [1, 2])

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_input))
            self.loss_val = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_val, labels=self.y_input))

        with tf.name_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            var_list = [v for v in tf.trainable_variables() if
                        v.name.split('/')[-2] in self.train_layers or v.name.split('/')[-3] in self.train_layers]
            gradients = tf.gradients(self.loss, var_list)
            self.grads_and_vars = list(zip(gradients, var_list))
            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt_name = cfg.TRAIN.OPTIMIZER
            if opt_name == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif opt_name == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            else:
                raise ValueError('Optimizer not supported')

            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,
                                                          global_step=self.global_step)

        with tf.name_scope("probability"):
            self.probability = tf.nn.softmax(self.logits_val, name="probability")

        with tf.name_scope("prediction"):
            self.prediction = tf.argmax(self.logits_val, 1, name="prediction")

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y_input, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

        print(self.logits.shape.as_list())
        print(self.logits_val.shape.as_list())

    def load_initial_weights(self, session):
        load_initial_weights(session=session, weight_path=self.WEIGHTS_PATH, train_layers=self.train_layers)
