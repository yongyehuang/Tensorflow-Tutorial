# Tensorflow-Tutorial

2018-04 更新说明

时间过去一年，TensorFlow 已经从 1.0 版本更新到了 1.8 版本，而且最近更新的非常频繁。最烦的就是每次更新很多 API 都改了，一些老版本的代码就跑不通了。因为本项目关注的人越来越多了，所以自己也感觉到非常有必要更新并更正一些之前的错误，否则误人子弟就不好了。这里不少内容可以直接在官方的教程中找到，官方文档也在不断完善中，我也是把里边的例子跑一下，加深理解而已，更多的还是要自己在具体任务中去搭模型，训模型才能很好地掌握。

这一次更新主要内容如下：

- 使用较新版本的 tf1.7.0
- 所有的代码改成 python3.5
- 重新整理了基础用例
- 添加实战例子

因为工作和学习比较忙，所以这些内容也没办法一下子完成。和之前的版本不同，之前我是作为一个入门菜鸟一遍学一边做笔记。虽然现在依然还是理解得不够，但是比之前掌握的知识应该多了不少，希望能够整理成一个更好的教程。

之前的代码我放在了另外一个分支上： https://github.com/yongyehuang/Tensorflow-Tutorial/tree/1.2.1

如果有什么问题或者建议，欢迎开issue或者邮件与我联系：yongye@bupt.edu.cn


## 运行环境
- python 3.5
- tensorflow 1.7.0 (gpu version)


## 文件结构
```
|- Tensorflow-Tutorial
|　　|- example-notebook　　　　　# 入门教程 notebook 版
|　　|- example-python　　　　　　# 入门教程 .py 版
|　　|- utils_and_models　　　　　# 一些工具函数和一些实战的例子
|　　|- data　　　　　　　        # 数据
|　　|- doc　　　　　　　　　　   # 相关文档
```

## Notes
### T_01.TensorFlow 的基本用法
- [notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_01%20Basic%20Usage.ipynb)

介绍 TensorFlow 的变量、常量和基本操作，最后介绍了一个非常简单的回归拟合例子。

### T_02.实现一个两层的全连接网络对 MNIST 进行分类
- [notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_02%20A%20simple%20feedforward%20network%20for%20MNIST.ipynb)

### T_03.TensorFlow 变量命名管理机制
- [notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_03%20The%20usage%20of%20%20name_scope%20and%20variable_scope.ipynb)

介绍  tf.Variable() 和 tf.get_variable() 创建变量的区别；介绍如何使用 tf.name_scope() 和 tf.variable_scope() 管理命名空间。


### T_04.实现一个两层的卷积神经网络（CNN）对 MNIST 进行分类
- [notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_04_1%20Convolutional%20network%20for%20MNIST(1).ipynb)

构建一个非常简单的 CNN 网络，同时输出中间各个核的可视化来理解 CNN 的原理。
<center>
<img src="https://raw.githubusercontent.com/yongyehuang/Tensorflow-Tutorial/1.7.0/figs/conv_mnist.png" width="60%" height="60%">
第一层卷积核可视化
</center>

### T_05.实现多层的 LSTM 和 GRU 网络对 MNIST 进行分类
- [LSTM-notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_05_1%20An%20understandable%20example%20to%20implement%20Multi-LSTM%20for%20MNIST.ipynb)
- [GRU-notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_05_2%20An%20understandable%20example%20to%20implement%20Multi-GRU%20for%20MNIST.ipynb)
- [Bi-GRU-notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_05_3%20Bi-GRU%20for%20MNIST.ipynb)

<center>
<img src="https://raw.githubusercontent.com/yongyehuang/Tensorflow-Tutorial/1.7.0/figs/lstm_8.png" width="60%" height="60%">
字符 8
</center>

<center>
<img src="https://raw.githubusercontent.com/yongyehuang/Tensorflow-Tutorial/1.7.0/figs/lstm_mnist.png" width="60%" height="60%">
lstm 对字符 8 的识别过程
</center>

### T_06.tensorboard 的简单用法
- [notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_06%20A%20very%20simple%20example%20for%20tensorboard.ipynb)

<center>
<img src="https://raw.githubusercontent.com/yongyehuang/Tensorflow-Tutorial/1.7.0/figs/graph2.png" width="30%" height="30%">
简单的 tensorboard 可视化
</center>

### T_07.使用 tf.train.Saver() 来保存模型
- [notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_07%20How%20to%20save%20the%20model.ipynb)

### T_08.【迁移学习】往一个已经保存好的 模型添加新的变量
- [notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_08%20%20%5Btransfer%20learning%5D%20Add%20new%20variables%20to%20graph%20and%20save%20the%20new%20model.ipynb)


### T_09.使用 tfrecord 打包不定长的序列数据
- [notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_09%20%5Btfrecord%5D%20use%20tfrecord%20to%20store%20sequences%20of%20different%20length.ipynb)
- [reader-code](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/utils_and_models/u02_tfrecord/tfrecord_2_seqence_reader.py)
- [writer-code](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/utils_and_models/u02_tfrecord/tfrecord_2_seqence_writer.py)


### T_10.使用 tf.data.Dataset 和 tfrecord 给 numpy 数据构建数据集
 - [dataset-notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_10%20%5BDataset%5D%20numpy%20data.ipynb)
 - [tfrecord-reader-code](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/utils_and_models/u02_tfrecord/tfrecord_1_numpy_reader.py)
 - [tfrecord-writer-code](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/utils_and_models/u02_tfrecord/tfrecord_1_numpy_writer.py)


下面是对 MNIST 数据训练集 55000 个样本 读取的一个速度比较，统一 `batch_size=128`，主要比较 `one-shot` 和 `initializable` 两种迭代方式：

|iter_mode|buffer_size|100 batch(s)|
|:----:|:---:|:---:|
|one-shot|2000|125|
|one-shot|5000|149|
|initializable|2000|0.7|
|initializable|5000|0.7|

可以看到，使用 `initializable` 方式的速度明显要快很多。因为使用 `one-shot` 方式会把整个矩阵放在图中，计算非常非常慢。


### T_11.使用 tf.data.Dataset 和 tfrecord 给 图片数据 构建数据集
- [dataset-notebook](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/example-notebook/Tutorial_11%20%5BDataset%5D%20image%20data.ipynb)
- [tfrecord-writer-code](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/utils_and_models/u02_tfrecord/tfrecord_3_image_writer.py)
- [tfrecord-reader-code](https://github.com/yongyehuang/Tensorflow-Tutorial/blob/1.7.0/utils_and_models/u02_tfrecord/tfrecord_3_image_reader.py)

对于 png 数据的读取,我尝试了 3 组不同的方式: one-shot 方式, tf 的队列方式(queue), tfrecord 方式. 同样是在机械硬盘上操作, 结果是 tfrecord 方式明显要快一些。（batch_size=128,图片大小为256*256,机械硬盘)

|iter_mode|buffer_size|100 batch(s)|
|:----:|:---:|:---:|
|one-shot|2000|75|
|one-shot|5000|86|
|tf.queue|2000|11|
|tf.queue|5000|11|
|tfrecord|2000|5.3|
|tfrecord|5000|5.3|

如果是在 SSD 上面的话,tf 的队列方式应该也是比较快的.打包成 tfrecord 格式只是减少了小文件的读取，其实现也是使用队列的。