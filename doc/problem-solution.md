## 1.为什么 relu 激活函数会有失活现象？
> [一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了。](http://blog.csdn.net/cyh_24/article/details/50593400)
> [Must Know Tips/Tricks in Deep Neural Networks (by Xiu-Shen Wei)](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)
> [深度学习系列（8）：激活函数](https://plushunter.github.io/2017/05/12/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97%EF%BC%888%EF%BC%89%EF%BC%9A%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/)
A: 在某次反向传播中神经元 C 传过一个很大的梯度，和 C 相连的所有权重系数 w 都变成了很大的负数。假设前一层输出也是 relu 激活函数，那么前一层的输出(C 的输入)一定是非负数。那么神经元 C 一定会处于0梯度状态，没有任何数据(输入)能够拯救他。也就是失活了。

那么，**我觉得，在初始化的时候，如果本层(L)使用 relu 激活函数，那么 biases 最好使用较小的正数初始化，如0.01，0.005等。这样可以在一定程度上使得输出偏向于正数，从而避免梯度消失。** 但是也有一点好处，由于部分神经元失活了，这和做了 dropout 一样，能够有助于防止过拟合。

**除了上面的失活现象以外，relu 激活函数还有偏移现象：输出均值恒大于零。** 失活现象和偏移现象会共同影响网络的收敛性。



## 2.[code]AlexNet(等 CNN model) 调参过程中要注意什么？
> 在 github 上面有很多不同版本的 AlexNet，网络结构基本都是一样的，在官方的 tensorflow slim 框架中也提供了 alexnet-v2 版本。在我的任务中，发现 slim 版本的 alexnet 效果要比其他版本的好很多(2~4个百分点)。明明结构一样，为什么差别这么大。

A：虽然我觉得 slim 框架很不友好，但是里边的参数应该都是经过测试得到的较好的结果。很多小的细节都会影响最后的结果，具体要注意的有下面这些点：
- 初始化方式。在我的任务中，初始化方式影响很大。slim 中的每一层的初始化方式和 caffe 的版本中基本一致。包括 conv 层， fc 层的 weights, biases。我就是最后一个 fc 层改了一下初始化，loss 嗖嗖地就降了。
- 优化器。在原版中用的是 RMSProp 优化器，但坑的是用的不是默认参数，最好设置成跟 slim 一致。
- 数据预处理。我用 caffe 算出了整个数据集的均值，然后把原始图片[0.~255.]减去均值，效果还不错。
- lrn。在 slim 的 alexnet-v2 版本中去掉了局部响应层。结果没有变差，速度快了大概一倍吧。



## 3.[code]tensorflow 训练模型，报无法分配 memory，但是程序继续运行，这时运行的结果有问题吗？
>**ran out of memory trying to allocate 3.27GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.** 这时候，使用 watch nvidia-smi 看显卡使用情况，就会发现使用率非常低。

A：这种情况下就应该把batch_size减小了！！！虽然程序还在运行，但是你会发现，结果根本不对，loss 根本就没降。这不是因为你的模型不好，学习率不对。把 batch_size 调小看看结果再下定论。



## 4.什么情况下用 sigmoid 会比 relu 好呢？
> sigmoid 函数有陷入饱和区（造成梯度消失）,非零中心和计算效率低的缺点。那么有什么情况下用 sigmoid 会好呢？

A: 如果只是作为一个普通隐含层的激活函数的话，还是用 relu 就好了。但是如果你后继需要对某层特征进行量化处理的话，sigmoid 就起作用了。relu 层的输出范围太大，不适合做量化。这时候就体现出 0 和 1 的优越性了。同样，使用 tanh 层也是可以的。



## 5.sigmoid 非 zero-center 会有什么影响，relu 有这样的影响吗？怎么理解因此导致梯度下降权重更新时出现 z 字型的下降？
> [Sigmoid函数的输出不是零中心的。这个性质并不是我们想要的,因为在神经网络后面层中神经元得到的数据不是零中心的。这一情况将影响梯度下降的运作，因为如果输入神经元的数据总是正数（比如在 f=wx+b 中每个元素都是 x>0)，那么关于 w 的梯度在反向传播中，将会要么全部是正数，或者全部是负数（比如f=-wx+b）.这将会导致梯度下降权重更新时出现z字型的下降（如下图所示）。](https://plushunter.github.io/2017/05/12/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97%EF%BC%888%EF%BC%89%EF%BC%9A%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/)




## 6.在 relu 层前面的 BN 层中，为什么不用对 scale 做处理
> [mnist_4.1_batchnorm_five_layers_relu.py](https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.1_batchnorm_five_layers_relu.py)

A：没想明白. 



## 7.center loss 训练有什么需要注意的？



## 8.在问题 1 中，relu 激活函数的偏移现象就是数据的输入分布（zero-center）和输出分布变了（no-zero-center）.在 BN 中也提到过，这种数据分布的变化为什么会影响网络的收敛性？
> [对于神经网络的各层输出，由于它们经过了层内操作作用，其分布显然与各层对应的输入信号分布不同，而且差异会随着网络深度增大而增大，可是它们所能“指示”的样本标记（label）仍然是不变的，这便符合了covariate shift的定义。](https://www.zhihu.com/question/38102762) 这样源空间 -> 目标空间 的变化怎么影响网络的收敛？



## 9.在BN中，为什么通过将activation规范为均值和方差一致的手段使得原本会减小的activation的scale变大？
[在BN中，是通过将activation规范为均值和方差一致的手段使得原本会减小的activation的scale变大。可以说是一种更有效的local response normalization方法（见4.2.1节）。](https://www.zhihu.com/question/38102762) 在 BN 中，通过mini-batch来规范化某些层/所有层的输入，从而可以固定每层输入信号的均值与方差。对于 mini-batch，首先减去均值，方差归一；接着进行 scale and shift 操作，也就是用学习到的全局均值和方差替换掉了 mini-batch 的均值和方差。 然后送入激活函数如（sigmoid）中，**这时候并没有把sigmoid的输入约束到 0 附近呀，为什么就能够避免梯度消失？**

A：首先，对于 sigmoid 和 tanh 来说，当输入的绝对值很大时就会陷入饱和区。这个激活函数的输入一般指的是 z(=wx)。
z 很大 -> 陷入饱和区
z 很大 -> wx 这两个向量的点积很大 -> 如果 w 或者 x 的元素值很大，较容易使得 wx 很大？ -> w，x 的值很大
BN 避免了梯度消失 -> BN 处理使得 x 向 0 靠近了。但是没有呀？？









