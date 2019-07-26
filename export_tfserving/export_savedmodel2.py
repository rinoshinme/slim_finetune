import os

import tensorflow as tf
from export_tfserving.export_savedmodel import test_model

from export_tfserving.violence_model import ViolenceModelResNetV1L101, InputType

"""Builds a SavedModel which can be used for deployment with
gcloud ml-engine, tensorflow-serving, ...
"""

if __name__ == "__main__":
    # parameters
    export_base_path = r'D:/projects/slim_finetune/export_tfserving/violence_model9/'
    export_version = '1'
    export_path = os.path.join(export_base_path, export_version)
    input_type = InputType.BASE64_JPEG
    model_weights = r'F:\output_finetune\ResNetV1_101\20190719_132123\ckpt\model-25000'

    model = ViolenceModelResNetV1L101(num_classes=9)
    model.build(weight_path=None,
                input_type=input_type)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
    tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input': tensor_info_input},
        outputs={'output': tensor_info_output},
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load weights
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(sess, model_weights)

        test_model(model, r'D:/72.jpg')

        # builder.add_meta_graph_and_variables(
        #     sess, [SERVING],
        #     signature_def_map={
        #         DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
        #         'predict': prediction_signature
        #     },
        #     legacy_init_op=legacy_init_op)
        #
        # builder.save()
