# -*- coding:utf-8 -*- 

from __future__ import print_function, division, absolute_import

import tensorflow as tf
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
        self.training_steps = 10000   # 耗时 90s
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
        print("initializing variables.")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if os.path.exists(my_setting.ckpt_path + 'checkpoint'):
            print("restore checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(my_setting.ckpt_path))
        tic = time.time()
        for step in range(my_setting.training_steps):
            if 0 == step % 100:
                X_batch, y_batch = mnist.train.next_batch(my_setting.batch_size, shuffle=True)
                X_batch = X_batch.reshape([-1, 28, 28, 1])
                _, g_step, train_loss, train_acc = sess.run(
                    [model.train_op, model.global_step, model.train_loss, model.train_acc],
                    feed_dict={model.X_inputs: X_batch, model.y_inputs: y_batch})
                X_batch, y_batch = mnist.test.next_batch(my_setting.batch_size, shuffle=True)
                X_batch = X_batch.reshape([-1, 28, 28, 1])
                test_loss, test_acc = sess.run([model.test_loss, model.test_acc],
                                               feed_dict={model.X_inputs: X_batch, model.y_inputs: y_batch})
                print(
                    "Global_step={:.2f}, train_loss={:.2f}, train_acc={:.2f}; test_loss={:.2f}, test_acc={:.2f}; pass {:.2f}s".format(
                        g_step, train_loss, train_acc, test_loss, test_acc, time.time() - tic
                    ))
            else:
                X_batch, y_batch = mnist.train.next_batch(my_setting.batch_size, shuffle=True)
                X_batch = X_batch.reshape([-1, 28, 28, 1])
                sess.run([model.train_op], feed_dict={model.X_inputs: X_batch, model.y_inputs: y_batch})
            if 0 == (step + 1) % 1000:
                path = saver.save(sess, os.path.join(my_setting.ckpt_path, 'model.ckpt'),
                                  global_step=sess.run(model.global_step))
                print("Save model to {} ".format(path))


if __name__ == '__main__':
    main()
