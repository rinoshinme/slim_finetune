import cv2
import numpy as np
import tensorflow as tf

from tfvortex.models.model_factory import get_model
from tfvortex.utils.image import is_image_file


class Demo(object):
    def __init__(self, options):
        self.options = options

        # parameters
        self.image_size = self.options.IMAGE_SIZE
        self.image_mean = np.array([121.55213, 113.84197, 99.5037])

        # load model
        self.model = get_model(self.options)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(self.session, self.options.CKPT_PATH)

        self.softmax_op = tf.nn.softmax(self.model.logits_val)
    
    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image - self.image_mean
        image = np.expand_dims(image, axis=0)
        return image

    def test(self, image_path):
        image = cv2.imread(image_path)
        inputs = self.preprocess(image)
        if self.model.keep_prob is not None:
            feed_dict = {
                self.model.x_input: inputs,
                self.model.keep_prob: 1.0,
            }
        else:
            feed_dict = {self.model.x_input: inputs}
        
        results = self.session.run(self.softmax_op, feed_dict=feed_dict)
        results = results[0]
        return results
