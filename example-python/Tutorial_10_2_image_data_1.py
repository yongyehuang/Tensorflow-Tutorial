"""Use tf.data.Dataset to create dataset for image(png) data.
With one-shot
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


def parse_png(img_path, label, height=256, width=256, channel=3):
    """根据 img_path 读入图片并做相应处理"""
    # 从硬盘上读取图片
    img = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img, channels=channel)
    # resize
    img_resized = tf.image.resize_images(img_decoded, [height, width])
    # normalize
    img_norm = img_resized * 1.0 / 127.5 - 1.0
    return img_norm, label


img_paths, labels = get_file_path()
batch_size = 128
dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
dataset = dataset.map(parse_png)
print('parsing image', dataset)
dataset = dataset.shuffle(buffer_size=5000).repeat().batch(batch_size)
print('batch', dataset)

# 生成迭代器
iterator = dataset.make_one_shot_iterator()
print(iterator)

time0 = time.time()
for count in range(100):
    X_batch, y_batch = sess.run(iterator.get_next())
    sys.stdout.write("\rloop {}, pass {:.2f}s".format(count, time.time() - time0))
    sys.stdout.flush()
