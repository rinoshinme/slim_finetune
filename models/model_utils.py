from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
import cv2
import os
from nets import dataset_utils


def load_initial_weights(session, weight_path, train_layers):
    print('Loading parameters')

    # # output all graph nodes
    # graph = session.graph
    # for node in graph.as_graph_def().node:
    #     print(node.name)

    reader = pywrap_tensorflow.NewCheckpointReader(weight_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for op_name in var_to_shape_map:
        if op_name == 'global_step':
            continue

        op_name_list = op_name.split('/')
        union_list = [item for item in op_name_list if item in train_layers]
        if len(union_list) != 0:
            continue

        try:
            with tf.variable_scope('/'.join(op_name.split('/')[0:-1]), reuse=True):
                data = reader.get_tensor(op_name)
                var = tf.get_variable(op_name.split('/')[-1], trainable=False)
                # var = tf.get_variable(op_name.split('/')[-1], trainable=True)
                session.run(var.assign(data))
        except ValueError:
            tmp1 = list(op_name in str(item) for item in tf.global_variables())
            tmp2 = np.sum([int(item) for item in tmp1])
            if tmp2 == 0:
                print("Don't be loaded: {}, cause: {}".format(op_name, "new model no need this variable."))
            else:
                print("Don't be loaded: {}, cause: {}".format(op_name, ValueError))
    print('Loading parameters finished...')


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def download_ckpt(url):
    target_dir = os.path.join("./pre_trained_models/")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    dataset_utils.download_and_uncompress_tarball(url, target_dir)


def compute_mean(train_path="./data/train.txt", validation_path="./data/validation.txt"):
    count = 0
    imgs_mean = np.zeros(shape=(3,), dtype=np.float32)
    for filepath in [train_path, validation_path]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                img = cv2.cvtColor(cv2.imread(line.split(" ")[0]), cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)
                imgs_mean += np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
                count += 1
    image_means = imgs_mean / count
    print(image_means)


def get_optimizer(opt_name, learn_rate):
    if opt_name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    elif opt_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learn_rate)
    elif opt_name == 'moment':
        optimizer = tf.train.MomentumOptimizer(learn_rate, momentum=0.9)
    elif opt_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learn_rate)
    else:
        raise ValueError('Optimizer not supported')
    return optimizer


def top_softmax_loss(logits, y_input, alpha=1.0):
    loss_softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_input)
    softmax_res = tf.nn.softmax(logits)

    # make top prob as close to 1 as possible
    loss_top = 1.0 - tf.reduce_max(softmax_res, reduction_indices=[1])
    loss_top = tf.square(loss_top)
    return tf.reduce_mean(loss_softmax + alpha * loss_top, reduction_indices=[0])
