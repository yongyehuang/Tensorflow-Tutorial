### 使用 tfrecord 打包数据
这里主要针对的是 tf1.2 及以前的版本，tf1.3和tf1.4 提供了新的 data API。等这个月把手头的实验做完更新后再尝试新的接口。

#### 数据维度相同
比如固定大小的矩阵，数组，图片等，比较简单，可以使用 tf.train.Example() 进行打包。参考：
- tfrecord-1-numpy-writer.py
- tfrecord-1-numpy-reader.py
- tfrecord-1-numpy-reader_without_shuffle.py

#### 数据维度不同(变长序列)
比如文本数据，每个序列的长度都是不固定的。可以使用 tf.train.SequenceExample() 进行打包。参考：
- tfrecord-2-seqence-writer.py
- tfrecord-2-seqence-reader.py

[sequence_example_lib.py](https://github.com/tensorflow/magenta/blob/master/magenta/common/sequence_example_lib.py)是官方提供一个例子，但是在官方文档中并没有提到这个例子，我也是费了好大力气才找到这个例子。tf1.3增加了 Dataset API，但是在 tf1.2 中的文档中，却没有一点说明，巨坑。

在 tf.train.Example() 中，通过 tf.train.Features() 把数据写入 tfrecord 文件。这些数据必须是具有相同的长度的**数组**，如果我们需要写入的是矩阵的话，我一般都是先reshape成 1D 的数组再进行写入。在读取数据的时候可以使用 tf.train.shuffle_batch() 直接读取数据。在读取数据的时候一定要明确指定数组的维度。

在 tf.train.SequenceExample()，包括 context 和 feature_lists 两部分。context 和 tf.train.Example() 作用类似，用于保存维度固定的数据。feature_lists 用于保存任意长度的序列数据。具体的写法参照 Tutorial-tfrecord-seqence-writer.py

在数据读取的时候，不能简单的通过 tf.train.shuffle_batch() 进行 batch 读取，因为我们没法给定数据的维度。解决方法是使用 tf.train.batch() 来进行读取，而且 **dynamic_pad 参数必须为 True，自动把batch中的数据填充0至一样的长度。** (所以在定义词典的时候，一定要把 '<PAD>' 的 id 对应为 0.)由于 batch() 函数不能进行 shuffle，所以我们只能借助 tf.RandomShuffleQueue() 函数维护一个队列来实现 shuffle。


#### 图片数据
- tfrecord-3-image-writer.py
- tfrecord-3-image-reader.py
