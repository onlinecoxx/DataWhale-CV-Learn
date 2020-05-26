# Task03 - 字符识别模型

> 特点对于字符识别模型，为了区分字符之间的差异，这种模型往往又通用的方式叫做分类器模型，但字符模型常常是已图片的方式存在，在训练上存在多维度情况。

## 1. 机器学习的分类方法

### 近邻算法
### SVM 向量机
### 朴素判别线性分类器
### 主成分分析


## 2. 神经网络分类方法

### CNN卷积神经网络 

CNN是一种多层非全连接的神经网络，主要包括卷积层和采样层，LeNet-5是CNN的经典例子，用于手写体数字字符识别，其网络结构如下

![LeNet](Task03/LeNet-CNN模型.jpeg)

最经典的就是 MNIST 集合的训练方式

#### 卷积操作

二维卷积是一个相当简单的操作：从卷积核开始，这是一个小的权值矩阵。这个卷积核在 2 维输入数据上 `滑动` ，对当前输入的部分元素进行矩阵乘法，然后将结果汇为单个输出像素。

![卷积动态图](Task03/Cov-动态图.gif)

```python
import cv2
import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
```