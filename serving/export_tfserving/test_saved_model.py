import tensorflow as tf
import numpy as np
import cv2
import os
import random
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.signature_constants\
    import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS


class PbTester(object):
    def __init__(self, class_names, pb_path):
        self.image_size = 224
        self.image_channels = 3
        self.class_names = class_names

        self.input_node_name = 'input'
        self.dropout_node_name = None
        self.output_node_name = 'outputs'

        self.imagenet_mean = np.array([121.55213, 113.84197, 99.5037])

        # load pb
        # self.inputs, self.outputs, self.dropout = self.load_pb(pb_path)
        self.load_saved_model(pb_path)

    def load_saved_model(self, pb_path):
        self.session = tf.Session()
        meta_graph_def = tf.saved_model.loader.load(self.session, [SERVING], pb_path)
        signature = meta_graph_def.signature_def

        # # print nodes
        # print('tensors: BEGIN')
        # graph_def = self.session.graph.as_graph_def()
        # for tensor in graph_def.node:
        #     print(tensor.name)
        # print('tensors: END')

        inputs_name = signature[DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[PREDICT_INPUTS].name
        # outputs_name = signature[DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[PREDICT_OUTPUTS].name
        outputs_name = 'outputs:0'
        print('inputs_name: ', inputs_name)
        print('outputs_name: ', outputs_name)
        self.inputs = self.session.graph.get_tensor_by_name(inputs_name)
        self.outputs = self.session.graph.get_tensor_by_name(outputs_name)
        self.dropout = None

        # graph_def = self.session.graph.as_graph_def()
        # if self.dropout_node_name is None:
        #     self.outputs, self.inputs = tf.import_graph_def(graph_def, name='',
        #                                           return_elements=[
        #                                               self.output_node_name + ":0",
        #                                               self.input_node_name + ":0"])
        #     self.dropout = None
        # else:
        #     self.outputs, self.inputs, self.dropout = tf.import_graph_def(graph_def, name='',
        #                                                    return_elements=[
        #                                                        self.output_node_name + ":0",
        #                                                        self.input_node_name + ":0",
        #                                                        self.dropout_node_name + ':0'])

    def test_on_images(self, image_paths):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        num_images = len(image_paths)

        input_images = []

        for i in range(num_images):
            img = cv2.imread(image_paths[i])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img - self.imagenet_mean
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = img[:, :, ::-1]
            input_images.append(img)
        input_array = np.stack(input_images, axis=0)
        print(input_array.shape)
        # print(input_array)

        # self.session.run(tf.global_variables_initializer())
        if self.dropout is None:
            feed_dict = {self.inputs: input_array}
        else:
            feed_dict = {self.inputs: input_array, self.dropout: 1.0}
        output = self.session.run(self.outputs, feed_dict=feed_dict)
        # output = self.session.run(self.outputs)
        print(output)
        print(output.shape)
        output = np.argmax(output, 1)
        return output


if __name__ == '__main__':
    class_names = []
    pb_path = r'D:\projects\slim_finetune\export_tfserving\serve_models_tensor\1'
    tester = PbTester(class_names, pb_path)

    image_folder = r'D:\data\baokong2\bloody'
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]

    images = random.sample(images, 5)
    for img in images:
        print(img)

    result = tester.test_on_images(images)
    print(result)
