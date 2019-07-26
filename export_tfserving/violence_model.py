import tensorflow as tf
from enum import Enum, unique
from export_tfserving.image_utils import load_base64_tensor
from nets import resnet_v1
from tensorflow.contrib.slim import arg_scope
from models.model_utils import load_initial_weights


"""
# class_labels = [army, bloody, crash, fire, normal, protest, 
#                 riot, terrorflag, terrorism, weapon]
# violence_labels = [bloody, crash, fire, riot, terrorflag, terrorism]
# strategy for production:
1. prob(normal) > 0.95 => normal
2. sum(violence_labels) > 0.96 => violence
3. others: not sure
"""


@unique
class InputType(Enum):
    TENSOR = 1
    BASE64_JPEG = 2


class ViolenceModelResNetV1L101(object):
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
        # load_initial_weights(self.session, weight_path, train_layers=[])
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(self.session, weight_path)
