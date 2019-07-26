import tensorflow as tf
import cv2
import numpy as np
import os
from config import cfg


class PbTester(object):
    def __init__(self, pb_path):
        self.image_size = cfg.IMAGE_SIZE
        self.image_channels = cfg.IMAGE_CHANNELS
        self.class_names = cfg.CLASS_NAMES

        self.input_node_name = cfg.TEST.INPUT_NODE_NAME
        self.dropout_node_name = cfg.TEST.DROPOUT_NODE_NAME
        self.output_node_name = cfg.TEST.OUTPUT_NODE_NAME

        self.imagenet_mean = np.array([121.55213, 113.84197, 99.5037])

        # load pb
        self.inputs, self.outputs, self.dropout = self.load_pb(pb_path)

    def load_pb(self, pb_path):
        with tf.gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            if self.dropout_node_name is None:
                outputs, inputs = tf.import_graph_def(graph_def,
                                                      return_elements=[
                                                          self.output_node_name + ":0",
                                                          self.input_node_name + ":0"])
                dropout = None
            else:
                outputs, inputs, dropout = tf.import_graph_def(graph_def,
                                                               return_elements=[
                                                                   self.output_node_name + ":0",
                                                                   self.input_node_name + ":0",
                                                                   self.dropout_node_name + ':0'])
        return [inputs, outputs, dropout]

    def test_on_images(self, image_paths):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        num_images = len(image_paths)

        input_images = []

        for i in range(num_images):
            img = cv2.imread(image_paths[i])
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img - self.imagenet_mean
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = img[:, :, ::-1]
            input_images.append(img)
        input_array = np.stack(input_images, axis=0)
        print(input_array.shape)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.dropout is None:
                feed_dict = {self.inputs: input_array}
            else:
                feed_dict = {self.inputs: input_array, self.dropout: 1.0}
            output = sess.run(self.outputs, feed_dict=feed_dict)
        output = np.argmax(output, 1)
        return output


if __name__ == '__main__':
    pb_file = r'D:\classify.pb'
    pt = PbTester(pb_file)

    image_folder = r'D:\data\violence_data\violence_data_test\weapon'
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
    images = images[:20]

    result = pt.test_on_images(images)
    print(result)
