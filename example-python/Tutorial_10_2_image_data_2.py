"""Use tf.data.Dataset to create dataset for image(png) data.
With TF Queue, shuffle data

refer: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
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
import sys
import os
import time


def get_file_path(data_path='../data/sketchy_000000000000/'):
    """解析文件夹，获取每个文件的路径和标签。"""
    img_paths = list()
    labels = list()
    class_dirs = sorted(os.listdir(data_path))
    dict_class2id = dict()
    for i in range(len(class_dirs)):
        label = i
        class_dir = class_dirs[i]
        dict_class2id[class_dir] = label
        class_path = os.path.join(data_path, class_dir)  # 每类的路径
        file_names = sorted(os.listdir(class_path))
        for file_name in file_names:
            file_path = os.path.join(class_path, file_name)
            img_paths.append(file_path)
            labels.append(label)
    return img_paths, labels


def get_batch(img_paths, labels, batch_size=128, height=256, width=256, channel=3):
    """根据 img_path 读入图片并做相应处理"""
    # 从硬盘上读取图片
    img_paths = np.asarray(img_paths)
    labels = np.asarray(labels)

    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([img_paths, labels], shuffle=True)
    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=channel)
    # Resize images to a common size
    image = tf.image.resize_images(image, [height, width])
    # Normalize
    image = image * 1.0 / 127.5 - 1.0
    # Create batches
    X_batch, y_batch = tf.train.batch([image, label], batch_size=batch_size,
                                      capacity=batch_size * 8,
                                      num_threads=4)
    return X_batch, y_batch


img_paths, labels = get_file_path()
X_batch, y_batch = get_batch(img_paths, labels)

sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

time0 = time.time()
for count in range(100):   # 11s for 100batch
    _X_batch, _y_batch = sess.run([X_batch, y_batch])
    sys.stdout.write("\rloop {}, pass {:.2f}s".format(count, time.time() - time0))
    sys.stdout.flush()
