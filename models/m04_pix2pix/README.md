# pix2pix 

代码来自：[affinelayer/pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

pix2pix.py 是原代码。

原始的代码比较长，包括数据读取，模型结构，模型训练和测试都是放在一个文件中。

根据原始代码，整理成了四个文件：
- utils.py: 数据读取和数据处理函数。
- model.py: pix2pix GAN 模型代码。
- train.py: 模型训练的代码。
- test.py: 模型测试的代码。

运行：
- 训练: `sh train.sh`
- 测试：`sh test.sh`
 