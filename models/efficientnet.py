import tensorflow as tf
import os
from config import cfg
import nets.efficientnet.efficientnet_builder as model_builder


class EfficientNet(object):
    def __init__(self, model_name, num_classes, train_layers=None, weights_path='DEFAULT'):
        self.model_name = model_name
        self.params = model_builder.efficientnet_params(self.model_name)
        self.image_size = self.params[2]
        self.num_classes = num_classes
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = os.path.join(cfg.TRAIN.PRETRAINED_WEIGHT_PATH, self.model_name, 'model.ckpt')
        else:
            self.WEIGHTS_PATH = weights_path

        self.train_layers = train_layers

        with tf.variable_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='inputs')
            self.normalized_inputs = self.normalize_features(self.inputs, model_builder.MEAN_RGB, model_builder.STDDEV_RGB)
            self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name='labels')
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.keep_prob = None

        additional_params = {'num_classes': self.num_classes}

        with tf.variable_scope('efficientnet', reuse=tf.AUTO_REUSE):
            self.logits, _ = model_builder.build_model(
                self.normalized_inputs,
                model_name=self.model_name,
                training=True,
                override_params=additional_params,
                model_dir='./models'
            )

            self.logits_val, _ = model_builder.build_model(
                self.normalized_inputs,
                model_name=self.model_name,
                training=False,
                override_params=additional_params,
                model_dir=None
            )

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels))
            self.loss_val = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_val, labels=self.labels))

        with tf.name_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            if self.train_layers is None:
                var_list = tf.trainable_variables()
                print('var_list = ', var_list)
            else:
                var_list = [v for v in tf.trainable_variables() if
                            v.name.split('/')[-2] in self.train_layers or
                            v.name.split('/')[-3] in self.train_layers]

            gradients = tf.gradients(self.loss, var_list)
            self.grads_and_vars = list(zip(gradients, var_list))

            # optimizer = get_optimizer(cfg.TRAIN.OPTIMIZER, self.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,
                                                          global_step=self.global_step)

        with tf.name_scope("probability"):
            self.probability = tf.nn.softmax(self.logits_val, name="probability")

        with tf.name_scope("prediction"):
            self.prediction = tf.argmax(self.logits_val, 1, name="prediction")

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    @staticmethod
    def normalize_features(features, mean_rgb, stddev_rgb):
        """Normalize the image given the means and stddevs."""
        stats_shape = (1, 1, 3)
        features -= tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
        features /= tf.constant(stddev_rgb, shape=stats_shape, dtype=features.dtype)
        return features

    def load_weights(self, session):
        session.run(tf.global_variables_initializer())
        # load_initial_weights(self.session, weight_path, train_layers=[])
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(session, self.WEIGHTS_PATH)
