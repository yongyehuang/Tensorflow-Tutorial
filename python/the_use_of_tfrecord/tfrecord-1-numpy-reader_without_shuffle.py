# -*- coding:utf-8 -*- 

import tensorflow as tf

'''read data.
without shuffle, for validation or test data.
遍历一次数据集，针对验证集或者训练集，我们不需要使用 shuffle 的方式读取 batch，而是需要保证完整的遍历一次数据集。
和 tf.train.shuffle_batch 的读取方式类似，只是有几个地方需要注意：
1.把文件名字写入队列的时候，记得设 shuffle=False
2.同时设置 num_epochs 参数为 1. 这样当读取完一个epoch以后就会抛出 OutOfRangeError.
3.tf.train.batch() 函数的 allow_smaller_final_batch 参数一定要设置 True， 才不会漏掉最后不满 batch_size 的一小部分数据
4.在初始化的时候加上 tf.initialize_local_variables()， 否则会报错。
5.记得加上 try...except... 来捕获 OutOfRangeError
'''

# output file name string to a queue
filename_queue = tf.train.string_input_producer(['../data/test1.tfrecord', '../data/test2.tfrecord'], num_epochs=1,
                                                shuffle=False)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'X': tf.FixedLenFeature([2], tf.float32),
                                       'y': tf.FixedLenFeature([], tf.int64)}
                                   )
X_out = features['X']
y_out = features['y']

print(X_out)
print(y_out)
X_batch, y_batch = tf.train.batch([X_out, y_out],
                                  batch_size=6,
                                  capacity=20,
                                  num_threads=2,
                                  allow_smaller_final_batch=True)  # 保证遍历完整个数据集

# 由于设定了读取轮数（num_epochs）为1，所以就算使用 shuffle_batch, 也会保证只取一个epoch，不重复，也不会漏失
# X_batch, y_batch = tf.train.shuffle_batch([X_out, y_out],
#                                           batch_size=6,
#                                           capacity=20,
#                                           min_after_dequeue=10,
#                                           num_threads=2,
#                                           allow_smaller_final_batch=True)

sess = tf.Session()
# 注意下面这个，如果设了 num_epoch，需要初始化 local_variables
init = tf.group(tf.global_variables_initializer(),
                tf.initialize_local_variables())
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
y_outputs = list()
for i in xrange(25):
    try:
        _X_batch, _y_batch = sess.run([X_batch, y_batch])
        print('** batch %d' % i)
        print('_y_batch:', _y_batch)
        y_outputs.extend(_y_batch.tolist())
    except tf.errors.OutOfRangeError as e:
        print(e.message)
        break
print(y_outputs)
print(sorted(y_outputs))

coord.request_stop()
coord.join(threads)
