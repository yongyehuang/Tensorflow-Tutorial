# -*- coding:utf-8 -*- 

import tensorflow as tf
import os
config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

"""TensorBoard 简单例子。
tf.summary.scalar('var_name', var)     # 记录标量的变化
tf.summary.histogram('vec_name', vec)  # 记录向量或者矩阵，tensor的数值分布变化。

merged = tf.summary.merge_all()        # 把所有的记录并把他们写到 log_dir 中
train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)  # 保存位置

运行完后，在命令行中输入 tensorboard --logdir=log_dir_path
"""

tf.app.flags.DEFINE_string('log_dir', 'summary/graph/', 'log saving path')
FLAGS = tf.app.flags.FLAGS
if os.path.exists(FLAGS.log_dir):
    os.rmdir(FLAGS.log_dir)
os.makedirs(FLAGS.log_dir)
print 'created log_dir path'


with tf.name_scope('add_example'):
    a = tf.Variable(tf.truncated_normal([100,1], mean=0.5, stddev=0.5), name='var_a')
    tf.summary.histogram('a_hist', a)
    b = tf.Variable(tf.truncated_normal([100,1], mean=-0.5, stddev=1.0), name='var_b')
    tf.summary.histogram('b_hist', b)
    increase_b = tf.assign(b, b + 0.05)
    c = tf.add(a, b)
    tf.summary.histogram('c_hist', c)
    c_mean = tf.reduce_mean(c)
    tf.summary.scalar('c_mean', c_mean)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(FLAGS.log_dir+'add_example', sess.graph)


def main(_):
    sess.run(tf.global_variables_initializer())
    for step in xrange(500):
        sess.run([merged, increase_b])  # 每步改变一次 b 的值
        summary = sess.run(merged)
        writer.add_summary(summary, step)
    writer.close()


if __name__ == '__main__':
    tf.app.run()
