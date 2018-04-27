"""tfrecord 写入数据.
将固定shape的矩阵写入 tfrecord 文件。这种形式的数据写入 tfrecord 是最简单的。
refer: http://blog.csdn.net/qq_16949707/article/details/53483493
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../../data/MNIST_data', one_hot=False)
X = mnist.train.images
y = mnist.train.labels

NUM_SHARDS = 64  # tfrecord 文件的数量，稍微大些对 shuffle 会好些
n_sample = len(X)
num_per_shard = n_sample // NUM_SHARDS  # 每个 tfrecord 的样本数量
# 在打包之前先手动打乱一次
new_idxs = np.random.permutation(n_sample)
X = X[new_idxs]
y = y[new_idxs]

tfrecord_dir = 'tfrecord/numpy/'
if not os.path.exists(tfrecord_dir):
    os.makedirs(tfrecord_dir)

time0 = time.time()
for shard_id in range(NUM_SHARDS):
    output_filename = '%d-of-%d.tfrecord' % (shard_id, NUM_SHARDS)
    output_path = os.path.join(tfrecord_dir, output_filename)
    with tf.python_io.TFRecordWriter(output_path) as writer:
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, n_sample)
        for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d, %g s' % (
                i + 1, n_sample, shard_id, time.time() - time0))
            sys.stdout.flush()

            X_sample = X[i].tolist()
            y_sample = y[i]
            # **3.定义数据类型，按照这里固定的形式写，有float_list(好像只有32位), int64_list, bytes_list.
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'X': tf.train.Feature(float_list=tf.train.FloatList(value=X_sample)),
                             'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y_sample]))}))
            # **4.序列化数据并写入文件中
            serialized = example.SerializeToString()
            writer.write(serialized)

print('Finished writing training data to tfrecord files.')
