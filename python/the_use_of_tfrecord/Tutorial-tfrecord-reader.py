# -*- coding:utf-8 -*- 

import tensorflow as tf

'''read data'''

# output file name string to a queue
filename_queue = tf.train.string_input_producer(['../data/test1.tfrecord', '../data/test2.tfrecord'], num_epochs=None,
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
X_batch, y_batch = tf.train.shuffle_batch([X_out, y_out], batch_size=2,
                                          capacity=200, min_after_dequeue=100, num_threads=2)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
y_outputs = list()
for i in xrange(5):
    _X_batch, _y_batch = sess.run([X_batch, y_batch])
    print('** batch %d' % i)
    print('_X_batch:', _X_batch)
    print('_y_batch:', _y_batch)
    y_outputs.extend(_y_batch.tolist())
print(y_outputs)
print(sorted(y_outputs))

