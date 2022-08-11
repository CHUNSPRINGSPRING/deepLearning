import torch.nn as nn
from torch.nn.functional import relu


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # channel为3，输入的是3个通道r\g\b，卷积核数为16个，卷积核的大小为5*5，这是他的构建！！！！
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化核 2*2， 步距为 2
        self.conv2 = nn.Conv2d(16, 32, 5)  # 前一个卷积核数16，深度16
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # 全连接层，是一个一维的向量，所以要展开得到第一个参数
        self.fc2 = nn.Linear(120, 84)  # 第二个参数，全连接层后节点84，是根据LeCNN图片得到的
        self.fc3 = nn.Linear(84, 10)  # 输出 的数据集个数，10个

    # 正向传播
    def forward(self, x):  # x是数据 通道排列顺序
        x = relu(self.conv1(x))  # input(3,32,32) output(16 ,28,28) 16:16个卷积核，28表示矩阵大小 ，relu为激活函数，这是它的输入！！！！
        x = self.pool1(x)  # output(16 ,14,14) 池化层改变的是高度和宽度，缩小为原来的一半，池化层不用加激励函数
        x = relu(self.conv2(x))  # output(32 ,10,10)，（之后）
        x = self.pool2(x)  # output(32,5,5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)，view 的第一个参数 -1表示可以自主生成一个一维数组
        x = relu(self.fc1(x))  # output(120) 全连接层需要乘以激励函数
        x = relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


import torch

input1 = torch.rand([32, 3, 32, 32])  # batch,channel,height,width
model = LeNet()
print(model)
output = model(input1)
