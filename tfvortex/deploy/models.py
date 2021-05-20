"""
define models for tfserving deployment
"""
import os
import sys
import tensorflow as tf
from enum import Enum, unique
from tensorflow.contrib.slim import arg_scope


from tfvortex.utils.deploy import load_base64_tensor
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(current_dir, '../'))
from nets import resnet_v1


@unique
class InputType(Enum):
    TENSOR = 1
    BASE64_JPEG = 2


class ViolenceResNetV1L101(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def _build(self, weight_path, sess, input_type=InputType.BASE64_JPEG):
        self.input_tensor = None
        self.session = sess
        if input_type == InputType.TENSOR:
            self.input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
            self.input_tensor = self.input
        elif input_type == InputType.BASE64_JPEG:
            self.input = tf.placeholder(tf.string, shape=(None, ), name='input')
            self.input_tensor = load_base64_tensor(self.input)
        else:
            raise ValueError('invalid input type')

        # only load inference model
        with arg_scope(resnet_v1.resnet_arg_scope(activation_fn=tf.nn.relu,
                                                  weight_decay=0.0001)):
            self.logits_val, end_points = resnet_v1.resnet_v1_101(self.input_tensor,
                                                                  num_classes=self.num_classes,
                                                                  is_training=False,
                                                                  reuse=tf.AUTO_REUSE)
        # self.predictions = tf.nn.softmax(self.logits_val, name='Softmax')
        self.predictions = end_points['predictions']
        self.output = tf.identity(self.predictions, name='outputs')

        if weight_path is not None:
            self.load_trained_weights(weight_path)

    def load_trained_weights(self, weight_path):
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(self.session, weight_path)


class NsfwResNetV1L152(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, weight_path, sess, input_type=InputType.BASE64_JPEG):
        self.input_tensor = None
        self.session = sess
        if input_type == InputType.TENSOR:
            self.input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
            self.input_tensor = self.input
        elif input_type == InputType.BASE64_JPEG:
            self.input = tf.placeholder(tf.string, shape=(None, ), name='input')
            self.input_tensor = load_base64_tensor(self.input)
        else:
            raise ValueError('invalid input type')

        # only load inference model
        with arg_scope(resnet_v1.resnet_arg_scope(activation_fn=tf.nn.relu,
                                                  weight_decay=0.0001)):
            self.logits_val, end_points = resnet_v1.resnet_v1_152(self.input_tensor,
                                                         num_classes=self.num_classes,
                                                         is_training=False,
                                                         reuse=tf.AUTO_REUSE)
        # self.predictions = tf.nn.softmax(self.logits_val, name='Softmax')
        self.predictions = end_points['predictions']
        self.output = tf.identity(self.predictions, name='outputs')

        if weight_path is not None:
            self.load_trained_weights(weight_path)

    def load_trained_weights(self, weight_path):
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(self.session, weight_path)



