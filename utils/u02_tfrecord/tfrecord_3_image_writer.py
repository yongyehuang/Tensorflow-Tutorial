# -*- coding:utf-8 -*- 

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import os
import time

'''tfrecord 写入数据.
将图片数据写入 tfrecord 文件。以 png格式数据集为例。

现在网上关于打包图片的例子非常多，实现方式各式各样，效率也相差非常多。
选择合适的方式能够有效地节省时间和硬盘空间。
有几点需要注意：
1.打包 tfrecord 的时候，千万不要使用 Image.open() 或者 matplotlib.image.imread() 等方式读取。
 1张小于10kb的png图片，前者（Image.open) 打开后，生成的对象100+kb, 后者直接生成 numpy 数组，大概是原图片的几百倍大小。
 所以应该直接使用 tf.gfile.FastGFile() 方式读入图片。
2.从 tfrecord 中取数据的时候，再用 tf.image.decode_png() 对图片进行解码。
3.不要随便使用 tf.image.resize_image_with_crop_or_pad 等函数，可以直接使用 tf.reshape()。前者速度极慢。
'''

# png 文件路径
IMG_DIR = '../../data/sketchy_000000000000/'
TFRECORD_DIR = 'tfrecord/sketchy_image/'
NUM_SHARDS = 64  # tfrecord 文件的数量，稍微大些对 shuffle 会好些


def get_file_path(data_path='../../data/sketchy_000000000000/'):
    """解析文件夹，获取每个文件的路径和标签。"""
    img_paths = list()
    labels = list()
    # 必须保证测试集 和 训练集中类别数相同，如果不同的话应该使用 dict_class2id 来保证类别对应正确。
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
    img_paths = np.asarray(img_paths)
    labels = np.asarray(labels)
    return img_paths, labels


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def convert_tfrecord_dataset(tfrecord_dir, n_shards, img_paths, labels, shuffle=True):
    """ convert samples to tfrecord dataset.
    Args:
        dataset_dir: 数据集的路径。
        tfrecord_dir: 保存 tfrecord 文件的路径。
        n_shards： tfrecord 文件个数
        img_paths: 图片的名字。
        labels：图片的标签。
    """
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)
    n_sample = len(img_paths)
    num_per_shard = n_sample // n_shards  # 每个 tfrecord 的样本数量

    # 在打包之前先手动打乱一次
    if shuffle:
        new_idxs = np.random.permutation(n_sample)
        img_paths = img_paths[new_idxs]
        labels = labels[new_idxs]

    time0 = time.time()
    for shard_id in range(n_shards):
        output_filename = '%d-of-%d.tfrecord' % (shard_id, n_shards)
        output_path = os.path.join(tfrecord_dir, output_filename)
        with tf.python_io.TFRecordWriter(output_path) as writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, n_sample)
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d, %g s' % (
                    i + 1, n_sample, shard_id, time.time() - time0))
                sys.stdout.flush()
                png_path = img_paths[i]
                label = labels[i]
                img = tf.gfile.FastGFile(png_path, 'rb').read()  # 读入图片
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': bytes_feature(img),
                            'label': int64_feature(label)
                        }))
                serialized = example.SerializeToString()
                writer.write(serialized)
    print('\nFinished writing data to tfrecord files.')


if __name__ == '__main__':
    img_paths, labels = get_file_path()
    convert_tfrecord_dataset(TFRECORD_DIR, NUM_SHARDS, img_paths, labels, shuffle=True)
