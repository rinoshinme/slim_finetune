"""
Builds a SavedModel which can be used for deployment with
gcloud ml-engine, tensorflow-serving, ...
"""
import os
import platform
import tensorflow as tf
import base64

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS

from .models import ViolenceResNetV1L101, NsfwResNetV1L152, InputType


def run():
    model = ViolenceResNetV1L101(num_classes=7)
    weight_path = r'D:\projects\ContentSupervision\code\ResNetV1_101\20191029_060119\model-12000'
    export_root = './data/saved_models'
    
    export_version = '1'
    input_type = InputType.BASE64_JPEG
    export_path = os.path.join(export_root, export_version)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model._build(weight_path=weight_path, sess=sess, input_type=input_type)
        
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
