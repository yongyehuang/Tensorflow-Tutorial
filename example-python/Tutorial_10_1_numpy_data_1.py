"""Use tf.data.Dataset to create dataset for numpy data.
With dataset.make_one_shot_iterator(), it is easy but very slow.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import warnings

warnings.filterwarnings('ignore')  # 不打印 warning
import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import numpy as np
import time
import sys

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

X = mnist.train.images
y = mnist.train.labels

batch_size = 128
# 创建 dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))
print('dataset is', dataset)

# 对数据进行操作
# def pre_func(X, y):
#     X = tf.multiply(X, 2)
#     return X, y
# dataset = dataset.map(pre_func)

dataset = dataset.shuffle(buffer_size=5000)  # 设置 buffle_size 越大，shuffle 越均匀
dataset = dataset.repeat().batch(batch_size)
print('after get batch', dataset)

# 生成迭代器
iterator = dataset.make_one_shot_iterator()
print(iterator)

# 迭代取值
time0 = time.time()
for count in range(100):  # 100batch  125 seconds
    X_batch, y_batch = sess.run(iterator.get_next())
    sys.stdout.write("\rloop {}, pass {:.2f}s".format(count, time.time() - time0))
    sys.stdout.flush()
    # print('count = {} : y = {}'.format(count, y_batch.reshape([-1])))

