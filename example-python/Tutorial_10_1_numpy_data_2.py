"""Use tf.data.Dataset to create dataset for numpy data.
With dataset.make_initializable_iterator(), it is more faster then dataset.make_one_shot_iterator().
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

# 使用 placeholder 来取代 array，并使用 initiable iterator， 在需要的时候再将 array 传进去
# 这样能够避免把大数组保存在图中
X_placeholder = tf.placeholder(dtype=X.dtype, shape=X.shape)
y_placeholder = tf.placeholder(dtype=y.dtype, shape=y.shape)

batch_size = 128
# 创建 dataset
dataset = tf.data.Dataset.from_tensor_slices((X_placeholder, y_placeholder))
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
iterator = dataset.make_initializable_iterator()
print(iterator)
sess.run(iterator.initializer, feed_dict={X_placeholder: X, y_placeholder: y})

# 迭代取值
time0 = time.time()
for count in range(100):  # 100batch  1.95 seconds
    X_batch, y_batch = sess.run(iterator.get_next())
    sys.stdout.write("\rloop {}, pass {:.2f}s".format(count, time.time() - time0))
    sys.stdout.flush()
    # print('count = {} : y = {}'.format(count, y_batch.reshape([-1])))

