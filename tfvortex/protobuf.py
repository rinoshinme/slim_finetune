import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tfvortex.models.model_factory import get_model


class PbExporter(object):
    def __init__(self, options):
        self.options = options
        self.output_node_name = self.options.OUTPUT_NODE_NAME

        # load model
        self.model = get_model(self.options)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(self.session, self.options.CKPT_PATH)

    def export(self, pb_path):
        graph = tf.get_default_graph()
        # self.display_all_nodes(graph)

        graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            self.session,
            graph_def,
            [self.output_node_name])
        with tf.gfile.GFile(pb_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

    def display_all_nodes(self, graph):
        for node in graph.as_graph_def().node:
            print(node.name)


class PbTester(object):
    def __init__(self, pb_path, options):
        self.pb_path = pb_path
        self.input_node_name = options.INPUT_NODE_NAME
        self.output_node_name = options.OUTPUT_NODE_NAME
        self.dropout_node_name = options.DROPOUT_NODE_NAME
        self.image_size = options.IMAGE_SIZE
        self.imagenet_mean = np.array([121.55213, 113.84197, 99.5037])

        self.inputs, self.outputs, self.dropout = self.load_pb()

    def load_pb(self):
        with tf.gfile.GFile(self.pb_path, 'rb') as f:
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
        return inputs, outputs, dropout
    
    def infer(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img - self.imagenet_mean
        input_array = np.expand_dims(img, axis=0)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.dropout is None:
                feed_dict = {self.inputs: input_array}
            else:
                feed_dict = {self.inputs: input_array, self.dropout: 1.0}
            output = sess.run(self.outputs, feed_dict=feed_dict)
            # print(output.shape)
        return output
