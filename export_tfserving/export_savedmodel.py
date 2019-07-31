import os
import cv2
import platform
import tensorflow as tf
import base64
import numpy as np

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils\
    import predict_signature_def

from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.signature_constants\
    import DEFAULT_SERVING_SIGNATURE_DEF_KEY

from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS

from export_tfserving.violence_model import ViolenceModelResNetV1L101, InputType
from export_tfserving.nsfw_model import NsfwResNetV1L152

"""Builds a SavedModel which can be used for deployment with
gcloud ml-engine, tensorflow-serving, ...
"""

os_type = platform.system()


def test_model(model, image_path, sess=None):
    if sess is not None:
        with open(image_path, 'rb') as f:
            data = f.read()
            data64 = base64.b64encode(data, b'-_')
            # data64 = data

        # sess is already initialized
        output = sess.run(model.output, feed_dict={model.input: [data64]})
        print(output)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("target", help="output directory")
    #
    # parser.add_argument("-i", "--input_type", required=True,
    #                     default=InputType.TENSOR.name.lower(),
    #                     help="Input type",
    #                     choices=[InputType.TENSOR.name.lower(),
    #                              InputType.BASE64_JPEG.name.lower()])
    #
    # parser.add_argument("-v", "--export_version",
    #                     help="export model version",
    #                     default="1")
    #
    # parser.add_argument("-m", "--model_weights", required=True,
    #                     help="Path to trained model weights file")
    #
    # args = parser.parse_args()

    # model = OpenNsfwModel()
    model = ViolenceModelResNetV1L101(num_classes=9)
    # model = NsfwResNetV1L152(num_classes=7)

    # parameters
    if os_type == 'Darwin':
        export_base_path = os.path.expanduser('~/Desktop/projects/tfserving/serve_models9_tensor')
        model_weights = '/Volumes/Elements/output_finetune/ResNetV1_101/20190719_132123/ckpt/model-25000'
        img_path = os.path.expanduser('~/Desktop/fire_002626.jpg')
    elif os_type == 'Windows':
        export_base_path = r'D:/projects/tfserving/violence/'
        model_weights = r'F:\output_finetune\ResNetV1_101\20190719_132123\ckpt\model-25000'
        # model_weights = r'D:\temp\model-150000'
        img_path = r'D:\data\baokong2\normal\27.jpg'
    else:
        raise ValueError('os_type not supported')
    export_version = '1'
    input_type = InputType.BASE64_JPEG
    # input_type = InputType.TENSOR

    export_path = os.path.join(export_base_path, export_version)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.build(weight_path=model_weights,
                    sess=sess,
                    input_type=input_type)

        # test_model(model, img_path, sess)

        builder = saved_model_builder.SavedModelBuilder(export_path)

        builder.add_meta_graph_and_variables(
            sess, [SERVING],
            signature_def_map={
                DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def(
                    inputs={PREDICT_INPUTS: model.input},
                    outputs={PREDICT_OUTPUTS: model.output}
                )
            }
        )

        builder.save()
