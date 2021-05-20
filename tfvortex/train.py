import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

from tfvortex.models.model_factory import get_model
from tfvortex.utils.model import load_initial_weights
from tfvortex.utils.training import get_learning_rate
from tfvortex.dataset.data_generator import ImageDataGenerator


class Trainer(object):
    def __init__(self, options):
        self.options = options
        self.image_size = self.options.IMAGE_SIZE
        self.model_name = self.options.MODEL_NAME
        self.num_classes = self.options.NUM_CLASSES

        # load data
        self.train_iterator, self.train_next_batch, self.val_iterator, self.val_next_batch = self.load_data()

        # load model
        self.model = get_model(self.options)
    
    def load_data(self):
        print('loading data')
        with tf.device('/cpu:0'):
            train_iterator = ImageDataGenerator(txt_file=self.options.TRAIN_DATASET_PATH, 
                                                mode='training',
                                                batch_size=self.options.BATCH_SIZE,
                                                num_classes=self.options.NUM_CLASSES,
                                                shuffle=True,
                                                img_size=self.options.IMAGE_SIZE)
            val_iterator = ImageDataGenerator(txt_file=self.options.VAL_DATASET_PATH,
                                              mode='inference',
                                              batch_size=self.options.BATCH_SIZE,
                                              num_classes=self.options.NUM_CLASSES,
                                              shuffle=True,
                                              img_size=self.options.IMAGE_SIZE)
            print('train size: ', train_iterator.data_size)
            print('validation size: ', val_iterator.data_size)
            train_next_batch = train_iterator.iterator.get_next()
            val_next_batch = val_iterator.iterator.get_next()
        print('done loading data...')
        return train_iterator, train_next_batch, val_iterator, val_next_batch

    def train(self):
        with tf.Session() as sess:
            # timestamp = str(int(time.time()))
            dt = datetime.now()
            dt_str = dt.strftime('%Y%m%d_%H%M%S')
            out_dir = os.path.abspath(os.path.join(self.options.OUTPUT_DIR, self.model_name, dt_str))
            print('writing to {}\n'.format(out_dir))

            # define summary
            grad_summaries = []
            for g, v in self.model.grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            # grad_summaries_merged = tf.summary.merge(grad_summaries)
            loss_summary = tf.summary.scalar("loss", self.model.loss)
            acc_summary = tf.summary.scalar("accuracy", self.model.accuracy)

            # merge all the train summary
            # train_summary_merged = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_merged = tf.summary.merge([loss_summary, acc_summary])  # ignore grad summaries.
            train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), graph=sess.graph)
            # merge all the dev summary
            val_summary_merged = tf.summary.merge([loss_summary, acc_summary])
            val_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "val"), graph=sess.graph)

            # checkPoint saver
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "ckpt"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.options.MAX_CHECKPOINTS)

            sess.run(tf.global_variables_initializer())
            
            init_weights_path = self.model.get_init_weights_path(self.options.PRETRAINED_ROOT)
            load_initial_weights(sess, init_weights_path, self.model.train_layers)

            # start training
            current_lr = self.options.LEARNING_RATE
            print('current lr is {}'.format(current_lr))
            while True:
                current_step = tf.train.global_step(sess, self.model.global_step)

                current_lr = get_learning_rate(self.options, current_step)

                # train loop
                x_batch_train, y_batch_train = sess.run(self.train_next_batch)

                if self.model.keep_prob is not None:
                    feed_dict = {
                        self.model.x_input: x_batch_train,
                        self.model.y_input: y_batch_train,
                        self.model.keep_prob: self.options.KEEP_PROB,
                        self.model.learning_rate: current_lr
                    }
                else:
                    feed_dict = {
                        self.model.x_input: x_batch_train,
                        self.model.y_input: y_batch_train,
                        self.model.learning_rate: current_lr
                    }

                # the step here is (current_step + 1) because global_step incremented every time
                # train_op is evaluated.
                _, step, train_summaries, loss, accuracy = sess.run(
                    [self.model.train_op, self.model.global_step, train_summary_merged,
                     self.model.loss, self.model.accuracy], feed_dict=feed_dict)

                train_summary_writer.add_summary(train_summaries, step)
                time_str = datetime.now().isoformat()

                if current_step % self.options.DISPLAY_STEP == 0 and current_step > 0:
                    print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, current_step, loss, accuracy))

                # validation
                if current_step % self.options.EVALUATE_STEP == 0 and current_step > 0:
                    print("\nEvaluation:")
                    # num_batches in one validation
                    num_batchs_one_validation = int(self.options.VALIDATION_SIZE / self.options.BATCH_SIZE)
                    loss_list = []
                    acc_list = []

                    for i in range(num_batchs_one_validation):
                        x_batch_val, y_batch_val = sess.run(self.val_next_batch)

                        if self.model.keep_prob is not None:
                            feed_dict = {
                                self.model.x_input: x_batch_val,
                                self.model.y_input: y_batch_val,
                                self.model.keep_prob: 1
                            }
                        else:
                            feed_dict = {
                                self.model.x_input: x_batch_val,
                                self.model.y_input: y_batch_val,
                            }

                        step, dev_summaries, loss, accuracy = sess.run(
                            [self.model.global_step, val_summary_merged, self.model.loss_val, self.model.accuracy],
                            feed_dict=feed_dict)
                        loss_list.append(loss)
                        acc_list.append(accuracy)
                        val_summary_writer.add_summary(dev_summaries, step)
                    time_str = datetime.now().isoformat()
                    print(
                        "{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, current_step, np.mean(loss_list),
                                                                     np.mean(acc_list)))
                    print("\n")

                if current_step % self.options.CHECKPOINT_STEP == 0 and current_step > 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                # break conditon
                if current_step == self.options.MAX_STEP:
                    break
