# -*- coding:utf-8 -*-

import tensorflow as tf 

"""tf.app.flags 用来实现对命令行参数进行解析。
一般先使用 tf.app.flags.DEFINE_ ... 来定义参数；
然后通过 tf.app.flags.FLAGS 对象取各个参数。
"""

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'huang', 'str_name')
flags.DEFINE_integer('age', 99, 'people age')
flags.DEFINE_float('weight', 51.1, 'people weight')

def main(_):
    print 'FLAGS.name=', FLAGS.name 
    print 'FLAGS.age=', FLAGS.age
    print 'FLAGS.weight=', FLAGS.weight 

if __name__ == '__main__':
    tf.app.run()      # 会自动运行 main() 函数
