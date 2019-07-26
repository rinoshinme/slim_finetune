import tensorflow as tf
from datetime import datetime
import os
import numpy as np
from models.model_factory import get_model
# from training_utils import get_learning_rate
from dataset.data_generator import ImageDataGenerator
from config import cfg
import sys

cfg_train = cfg.TRAIN
cfg_test = cfg.TEST

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class Trainer(object):
    def __init__(self):
        self.train_iterator, self.train_next_batch, self.val_iterator, self.val_next_batch = self.load_data()
        self.model, self.model_name = self.load_model()

    @staticmethod
    def load_data():
        print('loading data...')
        with tf.device('/cpu:0'):
            train_iterator = ImageDataGenerator(txt_file=cfg_train.TRAIN_DATASET_PATH,
                                                mode='training',
                                                batch_size=cfg_train.BATCH_SIZE,
                                                num_classes=cfg.NUM_CLASSES,
                                                shuffle=True,
                                                img_size=cfg.IMAGE_SIZE)
            val_iterator = ImageDataGenerator(txt_file=cfg_train.VAL_DATASET_PATH,
                                              mode='inference',
                                              batch_size=cfg_train.BATCH_SIZE,
                                              num_classes=cfg.NUM_CLASSES,
                                              shuffle=True,
                                              img_size=cfg.IMAGE_SIZE)
            print('train size: ', train_iterator.data_size)
            print('validation size: ', val_iterator.data_size)

            train_next_batch = train_iterator.iterator.get_next()
            val_next_batch = val_iterator.iterator.get_next()
        print('loading data finished...')
        return train_iterator, train_next_batch, val_iterator, val_next_batch

    @staticmethod
    def load_model():
        # initialize model
        model_name = cfg.MODEL_NAME
        model = get_model(model_name)
        return model, model_name

    def train(self):
        with tf.Session() as sess:
            # timestamp = str(int(time.time()))
            dt = datetime.now()
            dt_str = dt.strftime('%Y%m%d_%H%M%S')
            # out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', 'inceptionv3', timestamp))
            out_dir = os.path.abspath(os.path.join(cfg.OUTPUT_DIR, self.model_name, dt_str))
            print('writing to {}\n'.format(out_dir))

            # define summary
            grad_summaries = []
            for g, v in self.model.grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            loss_summary = tf.summary.scalar("loss", self.model.loss)
            acc_summary = tf.summary.scalar("accuracy", self.model.accuracy)

            # merge all the train summary
            train_summary_merged = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), graph=sess.graph)
            # merge all the dev summary
            val_summary_merged = tf.summary.merge([loss_summary, acc_summary])
            val_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "val"), graph=sess.graph)

            # checkPoint saver
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "ckpt"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg_train.MAX_CHECKPOINTS)

            sess.run(tf.global_variables_initializer())
            self.model.load_initial_weights(sess)

            # start training
            current_lr = cfg_train.LEARNING_RATE
            print('current lr is {}'.format(current_lr))
            while True:
                current_step = tf.train.global_step(sess, self.model.global_step)

                # decay learn rate when hit a step point
                if current_step in cfg_train.LEARNING_RATE_STEPVALUES:
                    current_lr = current_lr * cfg_train.LEARNING_RATE_DECAY
                    print('changed lr to {}'.format(current_lr))

                # current_lr = get_learning_rate(cfg_train, current_step)

                # train loop
                x_batch_train, y_batch_train = sess.run(self.train_next_batch)

                if self.model.keep_prob is not None:
                    feed_dict = {
                        self.model.x_input: x_batch_train,
                        self.model.y_input: y_batch_train,
                        self.model.keep_prob: cfg_train.KEEP_PROB,
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

                if current_step % cfg_train.DISPLAY_STEP == 0 and current_step > 0:
                    print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, current_step, loss, accuracy))

                # validation
                if current_step % cfg_train.EVALUATE_STEP == 0 and current_step > 0:
                    print("\nEvaluation:")
                    # num_batches in one validation
                    num_batchs_one_validation = int(cfg_train.VALIDATION_SIZE / cfg_train.BATCH_SIZE)
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

                if current_step % cfg_train.CHECKPOINT_STEP == 0 and current_step > 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                # break conditon
                if current_step == cfg_train.MAX_STEP:
                    break

    @staticmethod
    def write_training_options(cfg_train, file=None):
        # if file is not None:
        #     fp = open(file, 'w')
        # else:
        #     fp = sys.stdout
        # fd = fp.fileno()
        # os.write(fd, b'***')
        # os.write(fd, b'---')
        #
        # if file is not None:
        #     fp.close()

        stdout = sys.stdout
        if file is not None:
            sys.stdout = open(file, 'w')

        # do printing
        print('*' * 60)
        print('Learning rate: {}'.format(cfg_train.LEARNING_RATE))
        print('Batch size: {}'.format(cfg_train.BATCH_SIZE))
        print('Optimizer: {}'.format(cfg_train.OPTIMIZER))
        print('Weight decay: {}'.format(cfg_train.WEIGHT_DECAY))
        print('Activation: {}'.format(cfg_train.ACTIVATION_FN))
        print('*' * 60)

        # recover stdout to system default value
        if file is not None:
            sys.stdout = stdout


if __name__ == '__main__':
    print('pid = ', os.getpid())
    # train()
    trainer = Trainer()
    trainer.write_training_options(cfg_train)
    trainer.train()
