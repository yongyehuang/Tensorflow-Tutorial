# -*- coding:utf-8 -*- 

import tensorflow as tf
import numpy as np
from tqdm import tqdm

'''tfrecord 写入序列数据，每个样本的长度不固定。
和固定 shape 的数据处理方式类似，前者使用 tf.train.Example() 方式，而对于变长序列数据，需要使用 
tf.train.SequenceExample()。 在 tf.train.SequenceExample() 中，又包括了两部分：
context 来放置非序列化部分；
feature_lists 放置变长序列。

refer: 
https://github.com/tensorflow/magenta/blob/master/magenta/common/sequence_example_lib.py
https://github.com/dennybritz/tf-rnn
http://leix.me/2017/01/09/tensorflow-practical-guides/
https://github.com/siavash9000/im2txt_demo/blob/master/im2txt/im2txt/ops/inputs.py
'''

# **1.创建文件
writer1 = tf.python_io.TFRecordWriter('../../data/seq_test1.tfrecord')
writer2 = tf.python_io.TFRecordWriter('../../data/seq_test2.tfrecord')

# 非序列数据
labels = [1, 2, 3, 4, 5, 1, 2, 3, 4]
# 长度不固定的序列
frames = [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5],
          [1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]


writer = writer1
for i in tqdm(xrange(len(labels))):  # **2.对于每个样本
    if i == len(labels) / 2:
        writer = writer2
        print('\nThere are %d sample writen into writer1' % i)
    label = labels[i]
    frame = frames[i]
    # 非序列化
    label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    # 序列化
    frame_feature = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_])) for frame_ in frame
    ]

    seq_example = tf.train.SequenceExample(
        # context 来放置非序列化部分
        context=tf.train.Features(feature={
            "label": label_feature
        }),
        # feature_lists 放置变长序列
        feature_lists=tf.train.FeatureLists(feature_list={
            "frame": tf.train.FeatureList(feature=frame_feature),
        })
    )

    serialized = seq_example.SerializeToString()
    writer.write(serialized)  # **4.写入文件中

print('Finished.')
writer1.close()
writer2.close()
