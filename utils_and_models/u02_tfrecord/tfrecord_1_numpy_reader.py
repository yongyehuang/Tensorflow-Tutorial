# -*- coding:utf-8 -*- 

import tensorflow as tf
import os
import sys
import time

'''read data
从 tfrecord 文件中读取数据，对应数据的格式为固定shape的数据。
'''

# **1.把所有的 tfrecord 文件名列表写入队列中
tfrecord_dir = 'tfrecord/numpy/'
tfrecord_files = os.listdir(tfrecord_dir)
tfrecord_files = list(map(lambda s: os.path.join(tfrecord_dir, s), tfrecord_files))

filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=None, shuffle=True)

# **2.创建一个读取器
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# **3.根据你写入的格式对应说明读取的格式
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'X': tf.FixedLenFeature([784], tf.float32),  # 注意如果不是标量，需要说明数组长度
                                       'y': tf.FixedLenFeature([], tf.int64)}     # 而标量就不用说明
                                   )
X_out = features['X']
y_out = features['y']

print(X_out)
print(y_out)
# **4.通过 tf.train.shuffle_batch 或者 tf.train.batch 函数读取数据
"""
在shuffle_batch 函数中，有几个参数的作用如下：
capacity: 队列的容量，容量越大的话，shuffle 得就更加均匀，但是占用内存也会更多
num_threads: 读取进程数，进程越多，读取速度相对会快些，根据个人配置决定
min_after_dequeue: 保证队列中最少的数据量。
   假设我们设定了队列的容量C，在我们取走部分数据m以后，队列中只剩下了 (C-m) 个数据。然后队列会不断补充数据进来，
   如果后勤供应（CPU性能,线程数量）补充速度慢的话，那么下一次取数据的时候，可能才补充了一点点，如果补充完后的数据个数少于
   min_after_dequeue 的话，不能取走数据，得继续等它补充超过 min_after_dequeue 个样本以后才让取走数据。
   这样做保证了队列中混着足够多的数据，从而才能保证 shuffle 取值更加随机。
   但是，min_after_dequeue 不能设置太大，否则补充时间很长，读取速度会很慢。
"""
X_batch, y_batch = tf.train.shuffle_batch([X_out, y_out], batch_size=128,
                                          capacity=2000, min_after_dequeue=100, num_threads=4)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# **5.启动队列进行数据读取
# 下面的 coord 是个线程协调器，把启动队列的时候加上线程协调器。
# 这样，在数据读取完毕以后，调用协调器把线程全部都关了。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# y_outputs = list()
# for i in range(5):
#     _X_batch, _y_batch = sess.run([X_batch, y_batch])
#     print('** batch %d' % i)
#     print('_X_batch:\n', _X_batch)
#     print('_y_batch:\n', _y_batch)
#     y_outputs.extend(_y_batch.tolist())
# print('All y_outputs: \n', y_outputs)

# 迭代取值
time0 = time.time()
for count in range(100):  # 100batch  125 seconds
    _X_batch, _y_batch = sess.run([X_batch, y_batch])
    sys.stdout.write("\rloop {}, pass {:.2f}s".format(count, time.time() - time0))
    sys.stdout.flush()

# **6.最后记得把队列关掉
coord.request_stop()
coord.join(threads)

