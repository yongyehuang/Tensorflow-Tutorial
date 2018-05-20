# -*- coding:utf-8 -*-

"""网络结构定义。
关于 tf.layers.batch_normalization() 的理解参考： [tensorflow中batch normalization的用法](https://www.cnblogs.com/hrlnw/p/7227447.html)
"""

from __future__ import print_function, division, absolute_import

import tensorflow as tf


class Model(object):
    def __init__(self, settings):
        self.model_name = settings.model_name
        self.img_size = settings.img_size
        self.n_channel = settings.n_channel
        self.n_class = settings.n_class
        self.drop_rate = settings.drop_rate
        self.global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.learning_rate = tf.train.exponential_decay(settings.learning_rate,
                                                        self.global_step, settings.decay_step,
                                                        settings.decay_rate, staircase=True)

        self.conv_weight_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        self.conv_biases_initializer = tf.zeros_initializer()
        # 最后一个全连接层的初始化
        self.fc_weight_initializer = tf.truncated_normal_initializer(0.0, 0.005)
        self.fc_biases_initializer = tf.constant_initializer(0.1)

        # placeholders
        with tf.name_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.n_channel],
                                           name='X_inputs')
            self.y_inputs = tf.placeholder(tf.int64, [None], name='y_input')

        self.logits_train = self.inference(is_training=True, reuse=False)
        self.logits_test = self.inference(is_training=False, reuse=True)

        # 预测结果
        self.pred_lables = tf.argmax(self.logits_test, axis=1)
        self.pred_probas = tf.nn.softmax(self.logits_test)
        self.test_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(self.y_inputs, dtype=tf.int32), logits=self.logits_test))
        self.test_acc = tf.reduce_mean(tf.cast(tf.equal(self.pred_lables, self.y_inputs), tf.float32))

        # 训练结果
        self.train_lables = tf.argmax(self.logits_train, axis=1)
        self.train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(self.y_inputs, dtype=tf.int32), logits=self.logits_train))
        self.train_acc = tf.reduce_mean(tf.cast(tf.equal(self.train_lables, self.y_inputs), tf.float32))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        """
        **注意：** 下面一定要使用这样的方式来写。
        with tf.control_dependencies(update_ops) 这句话的意思是当运行下面的内容(train_op) 时，一定先执行 update_ops 的所有操作。
        这里的 update_ops 在这里主要是更新 BN 层的滑动平均值和滑动方差。

        除了 BN 层外，还有 center loss 中也采用这样的方式，在 center loss 中，update_ops 操作主要更新类中心向量。
        因为之前在 center loss 犯过没更新 center 的错误，所以印象非常深刻。
        """
        with tf.control_dependencies(update_ops):  # 这句话的意思是当运行下面的内容(train_op) 时，一定先执行 update_ops 的所有操作
            self.train_op = self.optimizer.minimize(self.train_loss, global_step=self.global_step)

    def inference(self, is_training, reuse=False):
        """带 BN 层的CNN """
        with tf.variable_scope('cnn', reuse=reuse):
            # 第一个卷积层 + BN  + max_pooling
            conv1 = tf.layers.conv2d(self.X_inputs, filters=32, kernel_size=5, strides=1, padding='same',
                                     kernel_initializer=self.conv_weight_initializer, name='conv1')
            bn1 = tf.layers.batch_normalization(conv1, training=is_training, name='bn1')
            bn1 = tf.nn.relu(bn1)  # 一般都是先经过 BN 层再加激活函数的
            pool1 = tf.layers.max_pooling2d(bn1, pool_size=2, strides=2, padding='same', name='pool1')

            # 第二个卷积层 + BN  + max_pooling
            conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=5, strides=1, padding='same',
                                     kernel_initializer=self.conv_weight_initializer, name='conv2')
            bn2 = tf.layers.batch_normalization(conv2, training=is_training, name='bn2')
            bn2 = tf.nn.relu(bn2)  # 一般都是先经过 BN 层再加激活函数的
            pool2 = tf.layers.max_pooling2d(bn2, pool_size=2, strides=2, padding='same', name='pool2')

            # 全连接,使用卷积来实现
            _, k_height, k_width, k_depth = pool2.get_shape().as_list()
            fc1 = tf.layers.conv2d(pool2, filters=1024, kernel_size=k_height, name='fc1')
            bn3 = tf.layers.batch_normalization(fc1, training=is_training, name='bn3')
            bn3 = tf.nn.relu(bn3)

            # dropout, 如果 is_training = False 就不会执行 dropout
            fc1_drop = tf.layers.dropout(bn3, rate=self.drop_rate, training=is_training)

            # 最后的输出层
            flatten_layer = tf.layers.flatten(fc1_drop)
            out = tf.layers.dense(flatten_layer, units=self.n_class)
        return out

    def inference2(self, is_training, reuse=False):
        """不带 BN 层的 CNN。"""
        with tf.variable_scope('cnn', reuse=reuse):
            # 第一个卷积层 + BN  + max_pooling
            conv1 = tf.layers.conv2d(self.X_inputs, filters=32, kernel_size=5, strides=1, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.conv_weight_initializer, name='conv1')
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='same', name='pool1')

            # 第二个卷积层 + BN  + max_pooling
            conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=5, strides=1, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.conv_weight_initializer, name='conv2')
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='same', name='pool2')

            # 全连接,使用卷积来实现
            _, k_height, k_width, k_depth = pool2.get_shape().as_list()
            fc1 = tf.layers.conv2d(pool2, filters=1024, kernel_size=k_height, activation=tf.nn.relu, name='fc1')

            # dropout, 如果 is_training = False 就不会执行 dropout
            fc1_drop = tf.layers.dropout(fc1, rate=self.drop_rate, training=is_training)

            # 最后的输出层
            flatten_layer = tf.layers.flatten(fc1_drop)
            out = tf.layers.dense(flatten_layer, units=self.n_class)
        return out
