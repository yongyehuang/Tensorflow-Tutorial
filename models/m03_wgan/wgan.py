# -*- coding:utf-8 -*- 

from __future__ import print_function, division, absolute_import

import warnings

warnings.filterwarnings('ignore')  # 不打印 warning

import matplotlib

matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import time
import os
import sys


# Generator Network
# Input: Noise, Output: Image
def generator(z, reuse=False):
    """输入噪声，输出图片。
       G网络中使用ReLU作为激活函数，最后一层使用tanh.
       每一层都加 BN
    """
    with tf.variable_scope('Generator', reuse=reuse) as scope:
        # 按照论文的网络结构，第一层用一个全连接层
        x = tf.layers.dense(z, units=4 * 4 * 1024)
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=True, name='bn1'))
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # 然后把全链接层的输出 reshape 成 4D 的 tensor
        x = tf.reshape(x, shape=[-1, 4, 4, 1024])
        print('tr_conv4:', x)
        # Deconvolution, image shape: (batch, 8, 8, 512)
        x = tf.layers.conv2d_transpose(x, 512, 5, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=True, name='bn2'))
        print('tr_conv3:', x)
        # Deconvolution, image shape: (batch, 16, 16, 256)
        x = tf.layers.conv2d_transpose(x, 256, 5, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=True, name='bn3'))
        print('tr_conv2:', x)
        # Deconvolution, image shape: (batch, 32, 32, 128)
        x = tf.layers.conv2d_transpose(x, 128, 5, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=True, name='bn4'))
        print('tr_conv1:', x)
        # Deconvolution, image shape: (batch, 64, 64, 3)
        x = tf.layers.conv2d_transpose(x, 3, 5, strides=2, padding='same')
        x = tf.nn.tanh(tf.layers.batch_normalization(x, training=True, name='bn5'))
        print('output_image:', x)
        return x


def discriminator(x, reuse=False):
    """判别器。
    D网络中使用LeakyReLU作为激活函数。
    """
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        print('input_image:', x)
        x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=True, name='bn1'))
        print('conv1:', x)
        x = tf.layers.conv2d(x, 256, 5, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=True, name='bn2'))
        print('conv2:', x)
        x = tf.layers.conv2d(x, 512, 5, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=True, name='bn3'))
        print('conv3:', x)
        x = tf.layers.conv2d(x, 1024, 5, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=True, name='bn4'))
        print('conv4:', x)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 1)
        print('output:', x)
    return x


def get_file_path(data_path='../../data/anime/'):
    """解析文件夹，获取每个文件的路径和标签(在这个例子中所有的真实图片的标签都是1)。"""
    files = os.listdir(data_path)
    img_paths = [file for file in files if file.endswith('.jpg')]
    img_paths = list(map(lambda s: os.path.join(data_path, s), img_paths))
    labels = [1] * len(img_paths)
    return img_paths, labels


def get_batch(img_paths, labels, batch_size=50, height=64, width=64, channel=3):
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


def generate_img(sess, seed=3, show=False, save_path='generated_img'):
    """利用随机噪声生成图片， 并保存图片。"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    step = sess.run(global_step)
    n_row, n_col = 5, 10
    f, a = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    np.random.seed(seed)  # 每次都用相同的随机数便于对比结果
    z_batch = np.random.uniform(-1., 1., size=[n_row * n_col, noise_dim])
    fake_imgs = sess.run(gen_image, feed_dict={noise_input: z_batch})
    for i in range(n_row):
        # Noise input.
        for j in range(n_col):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = (fake_imgs[n_col * i + j] + 1) / 2
            a[i][j].imshow(img)
            a[i][j].axis('off')
    f.savefig(os.path.join(save_path, '{}.png'.format(step)), dpi=100)
    if show:
        f.show()


# 构建网络
# Network Params
image_dim = 64  # 头像的尺度
n_channel = 3
noise_dim = 100  # 输入噪声 z 的维度

noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_img_input = tf.placeholder(tf.float32, shape=[None, image_dim, image_dim, n_channel])

# 构造网络：生成器
gen_image = generator(noise_input)

# 判别器有两种输入，一种真实图像，一种生成器生成的假图像
disc_real = discriminator(real_img_input)  # 真图像输出的结果
disc_fake = discriminator(gen_image, reuse=True)

# 损失函数 (与DCGAN第1个改变)
disc_loss = tf.reduce_mean(disc_fake - disc_real)
gen_loss = tf.reduce_mean(-disc_fake)

# 两个优化器
# Build Optimizers (第2个改变)
global_step = tf.Variable(0, trainable=False, name='Global_Step')
optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=5e-5)
optimizer_disc = tf.train.RMSPropOptimizer(learning_rate=5e-5)

# G 和 D 的参数
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# 对于 D 的参数进行裁剪  （第3个改变，注意每次更新完 D 后都要执行裁剪）
clipped_disc = [tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)) for v in disc_vars]

# Create training operations
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):  # 这句话的意思是当运行下面的内容(train_op) 时，一定先执行 update_ops 的所有操作
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars, global_step=global_step)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# 模型要保存的变量
# var_list = tf.global_variables() + tf.local_variables()  # 这样保存所有变量不会出错，但是很多没必要的变量也保存了。414M 每个 ckpt
var_list = tf.trainable_variables()
if global_step not in var_list:
    var_list.append(global_step)
# 添加 BN 层的均值和方差
global_vars = tf.global_variables()
bn_moving_vars = [v for v in global_vars if 'moving_mean' in v.name]
bn_moving_vars += [v for v in global_vars if 'moving_variance' in v.name]
var_list += bn_moving_vars
# 创建Saver

saver = tf.train.Saver(var_list=var_list)

# 模型保存路径
ckpt_path = 'ckpt/'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

# Training Params
num_steps = 500000
batch_size = 64
d_iters = 5  # 更新 d_iters 次 D, 更新 1 次 G

# 构建数据读取函数
img_paths, labels = get_file_path()
print("n_sample={}".format(len(img_paths)))
get_X_batch, get_y_batch = get_batch(img_paths, labels, batch_size=batch_size)

# Start training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print("initializing variables.")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if os.path.exists(ckpt_path + 'checkpoint'):
        print("restore checkpoint.")
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

    # 启动数据读取队列
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    def get_batch():
        # 准备输入数据：真实图片
        X_batch, _ = sess.run([get_X_batch, get_y_batch])  # 本例中不管 _y_batch 的label
        # 准备噪声输入
        z_batch = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        # Training 这里的输入不需要 target
        feed_dict = {real_img_input: X_batch, noise_input: z_batch}
        return feed_dict


    try:
        tic = time.time()
        for i in range(1, num_steps + 1):
            # update 5 次 D，update 1 次 G
            dl = 0.0
            for _ in range(d_iters):
                _, _, dl = sess.run([train_disc, clipped_disc, disc_loss], feed_dict=get_batch())
            _, gl = sess.run([train_gen, gen_loss], feed_dict=get_batch())

            if i % 5 == 0:
                print(
                    'Step {}: Generator Loss: {:.2f}, Discriminator Loss: {:.2f}. Time passed {:.2f}s'.format(i, gl, dl,
                                                                                                              time.time() - tic))
                if i % 200 == 0:  # 每训练 1000 step，生成一次图片
                    generate_img(sess)
                    path = saver.save(sess, os.path.join(ckpt_path, 'model.ckpt'), global_step=sess.run(global_step))
                    print("Save model to {} ".format(path))

        generate_img(sess)  # 最后一次生成图像
    except Exception as e:
        print(e)
    finally:
        coord.request_stop()
        coord.join(threads)
