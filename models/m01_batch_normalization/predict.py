# -*- coding:utf-8 -*-

from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np
import os
import time

from mnist_cnn import Model


class Settings(object):
    def __init__(self):
        self.model_name = 'mnist_cnn'
        self.img_size = 28
        self.n_channel = 1
        self.n_class = 10
        self.drop_rate = 0.5
        self.learning_rate = 0.001
        self.decay_step = 2000
        self.decay_rate = 0.5
        self.training_steps = 10000
        self.batch_size = 100

        self.summary_path = 'summary/' + self.model_name + '/'
        self.ckpt_path = 'ckpt/' + self.model_name + '/'

        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)


def main():
    """模型训练。"""
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("../../data/MNIST_data", one_hot=False)
    print(mnist.test.labels.shape)
    print(mnist.train.labels.shape)

    my_setting = Settings()
    with tf.variable_scope(my_setting.model_name):
        model = Model(my_setting)

    # 模型要保存的变量
    var_list = tf.trainable_variables()
    if model.global_step not in var_list:
        var_list.append(model.global_step)
    # 添加 BN 层的均值和方差
    global_vars = tf.global_variables()
    bn_moving_vars = [v for v in global_vars if 'moving_mean' in v.name]
    bn_moving_vars += [v for v in global_vars if 'moving_variance' in v.name]
    var_list += bn_moving_vars
    # 创建Saver
    saver = tf.train.Saver(var_list=var_list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if not os.path.exists(my_setting.ckpt_path + 'checkpoint'):
            print("There is no checkpoit, please check out.")
            exit()
        saver.restore(sess, tf.train.latest_checkpoint(my_setting.ckpt_path))
        tic = time.time()
        n_batch = len(mnist.test.labels) // my_setting.batch_size
        predict_labels = list()
        true_labels = list()
        for step in range(n_batch):
            X_batch, y_batch = mnist.test.next_batch(my_setting.batch_size, shuffle=False)
            X_batch = X_batch.reshape([-1, 28, 28, 1])
            pred_label, test_loss, test_acc = sess.run([model.pred_lables, model.test_loss, model.test_acc],
                                           feed_dict={model.X_inputs: X_batch, model.y_inputs: y_batch})
            predict_labels.append(pred_label)
            true_labels.append(y_batch)
        predict_labels = np.hstack(predict_labels)
        true_labels = np.hstack(true_labels)
        acc = np.sum(predict_labels == true_labels) / len(true_labels)
        print("Test sample number = {}, acc = {:.4f}, pass {:.2f}s".format(len(true_labels), acc, time.time() - tic))


if __name__ == '__main__':
    main()
