# TensorFlow 实战
下面的每个例子都是相互独立的，每个文件夹下面的代码都是可以单独运行的，不依赖于其他文件夹。

## m01_batch_normalization: Batch Normalization 的使用。
参考：[tensorflow中batch normalization的用法](https://www.cnblogs.com/hrlnw/p/7227447.html)

## m02_dcgan: 使用 DCGAN 生成二次元头像
参考：
- [原论文:Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1511.06434)
- [GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059)
- [代码：carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [代码：aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dcgan.py)

这里的 notebook 和 .py 文件的内容是一样的。本例子和下面的 GAN 模型用的数据集也是用了[GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059) 的二次元头像，感觉这里例子比较有意思。如果想使用其他数据集的话，只需要把数据集换一下就行了。

下载链接: https://pan.baidu.com/s/1HBJpfkIFaGh0s2nfNXJsrA 密码: x39r

下载后把所有的图片解压到一个文件夹中，比如本例中是： `data_path = '../../data/anime/'`

运行： `python dcgan.py `

## m03_wgan: 使用 WGAN 生成二次元头像
这里的生成器和判别器我只实现了 DCGAN，没有实现 MLP. 如果想实现的话可以参考下面的两个例子。
参考：
- [原论文:Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
- [代码：jiamings/wgan](https://github.com/jiamings/wgan)
- [代码：Zardinality/WGAN-tensorflow](https://github.com/Zardinality/WGAN-tensorflow)

原版的 wgan： `python wgan.py `

改进的 wgan-gp: `python wgan_gp.py`


## m04_pix2pix: image-to-image GAN


