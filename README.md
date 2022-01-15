# StyleGAN-by-Jittor

本项目为使用[计图（jittor）](https://github.com/Jittor/jittor) 实现的 StyleGAN，作为对GAN网络的探索和计图框架的学习。

关于计图平台的使用：
+ [Jittor: a novel deep learning framework with meta-operators and unified graph execution](https://cg.cs.tsinghua.edu.cn/jittor/papers/)
+ [Jittor 官方文档](https://cg.cs.tsinghua.edu.cn/jittor/)

关于StyleGAN：
+ [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
+ [style-based-gan-pytorch](https://github.com/rosinality/style-based-gan-pytorch)

配置环境：
1. 需使用 Ubuntu 16.04 及以后版本的操作系统，并且 python 版本 >= 3.
2. pip install -r requirements.txt即可。若计图的安装遇到问题，也可查阅其官方文档。

使用方式：
1. 预训练模型在checkpoint中，输出图像结果和渐进式训练效果视频都在result中。
2. 可从我的网盘获取实验所用字符数据集：https://cloud.tsinghua.edu.cn/f/c9aa9bc2146e491ea8ba/?dl=1 ，并解压至color_symbol_7k目录下。
3. 执行 python generate.py checkpoint/080000.model --size=128即可执行前馈测试以及隐向量插值，如需别的效果请自行更改generator.py中的参数
4. 自行训练则需要修改train.py中的默认路径，再执行python train.py。数据集需先通过progressive_data.py进行处理，也需要改其中的默认路径，将数据存入data文件夹下。
