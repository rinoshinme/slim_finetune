import os
import sys
import tensorflow as tf
from tensorflow.contrib.slim import arg_scope

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(current_dir, '../'))
from nets import vgg
from tfvortex.utils.model import get_optimizer


class Vgg16(object):
    NAME = 'VGG16'
    DEFAULT_TRAIN_LAYERS = ['fc7', 'fc8']

    def __init__(self, options):
        num_classes = options.NUM_CLASSES

        with tf.variable_scope('input'):
            self.image_size = options.IMAGE_SIZE
            self.x_input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='x_input')
            self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        if options.PHASE == 'train':

            if train_layers == 'DEFAULT':
                self.train_layers = DEFAULT_TRAIN_LAYERS
            else:
                self.train_layers = train_layers

            with arg_scope(vgg.vgg_arg_scope(weight_decay=options.WEIGHT_DECAY)):
                self.logits, _ = vgg.vgg_16(self.x_input,
                                            num_classes=num_classes,
                                            is_training=True,
                                            reuse=tf.AUTO_REUSE,
                                            dropout_keep_prob=self.keep_prob)
            with arg_scope(vgg.vgg_arg_scope(weight_decay=0.0)):
                self.logits_val, _ = vgg.vgg_16(self.x_input,
                                                num_classes=num_classes,
                                                is_training=False,
                                                reuse=tf.AUTO_REUSE,
                                                dropout_keep_prob=self.keep_prob)

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

                optimizer = get_optimizer(options.OPTIMIZER, self.learning_rate)

                with tf.control_dependencies(update_ops):
                    self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,
                                                              global_step=self.global_step)
        else:
            with arg_scope(vgg.vgg_arg_scope(weight_decay=0.0)):
                self.logits_val, _ = vgg.vgg_16(self.x_input,
                                                num_classes=num_classes,
                                                is_training=False,
                                                reuse=tf.AUTO_REUSE,
                                                dropout_keep_prob=self.keep_prob)
        with tf.name_scope("probability"):
            self.probability = tf.nn.softmax(self.logits_val, name="probability")

        with tf.name_scope("prediction"):
            self.prediction = tf.argmax(self.logits_val, 1, name="prediction")

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y_input, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    def get_init_weights_path(self, weights_root):
        return os.path.join(weights_root, 'vgg_16.ckpt')
