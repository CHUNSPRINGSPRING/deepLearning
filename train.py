#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/7/21 18:56
# @Author: Spring
# @File  : train.py

import torch
import torchvision
import torch.nn as nn
from testTensor import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 对图片进行预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练集为50000张，下载到当前地址的data文件夹下，下载train类型的图片,transform预处理器
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

# windows num_workers 设置为0，将图片分批以36张为一组,shuffle:打乱
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True, num_workers=0)

# 10000张测试图片
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=False, num_workers=0)

# 将加载的图片转换为一个迭代器，通过next函数获得对应的图片和标签
test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

# 元祖不可改变值
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize 反标准化
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 高度 、 宽度、 通道，与tensor的不匹配，所以要交换
#     plt.show()
#
#
# # print labels
# print(' '.join(f'{classes[test_label[j]]:5s}' for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(test_image))

net = LeNet()
loss_function = nn.CrossEntropyLoss()  # 损失函数 ， 包含了softmax函数,对误差求导的函数是误差损失
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化器使用Adam优化器，lr：学习率 0.001 ，训练参数


for epoch in range(5):  # loop over the dataset multiple times，迭代5次

    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients，清除历史梯度
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)  # 正向输出
        loss = loss_function(outputs, labels)  # 损失函数
        loss.backward()  # 损失反向传播（误差反向传播）
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:    # print every 500 mini-batches
            with torch.no_grad():  # with 上下文管理 无梯度损失
                outputs = net(test_image)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]  # 最可能归于哪个类别,寻找最大值，输出的十个节点中
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)  # 比较与真实是否相同，正确率

                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f'%
                      (epoch + 1, step + 1, running_loss / 500, accuracy))  # running_loss训练过程中的累加误差
                running_loss = 0.0

print('Finished Training')


save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)