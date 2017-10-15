# -*- coding:utf-8 -*- 

import tensorflow as tf
import numpy as np
from tqdm import tqdm

'''tfrecord 写入数据.
将固定shape的矩阵写入 tfrecord 文件。这种形式的数据写入 tfrecord 是最简单的。
refer: http://blog.csdn.net/qq_16949707/article/details/53483493
'''

# **1.创建文件，可以创建多个文件，在读取的时候只需要提供所有文件名列表就行了
writer1 = tf.python_io.TFRecordWriter('../data/test1.tfrecord')
writer2 = tf.python_io.TFRecordWriter('../data/test2.tfrecord')

"""
有一点需要注意的就是我们需要把矩阵转为数组形式才能写入
就是需要经过下面的 reshape 操作
在读取的时候再 reshape 回原始的 shape 就可以了
"""
X = np.arange(0, 100).reshape([50, -1]).astype(np.float32)
y = np.arange(50)

for i in tqdm(xrange(len(X))):  # **2.对于每个样本
    if i >= len(y) / 2:
        writer = writer2
    else:
        writer = writer1
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

print('Finished.')
writer1.close()
writer2.close()
