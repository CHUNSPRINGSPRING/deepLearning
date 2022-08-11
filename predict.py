#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/7/22 10:51
# @Author: Spring
# @File  : predict.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from testTensor import LeNet

# 如何转入到gpu当中呢，将net传输到gpu：net.to(device),将每一步的输入和目标发送到Gpu
# 如下：inputs, labels = data[0].to(device), data[1].to(device)注意，应该要在模型训练是！！！！
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

# compose对图片进行预处理，第一步缩放到32*32 ，转化为tensor，之后用标准化处理
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 元祖不可改变值
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = LeNet()

net.load_state_dict(torch.load('Lenet.pth'), strict=True)

im = Image.open('1.jpg')
im = transform(im)  # [c,h,w]
im = torch.unsqueeze(im, dim=0)  # 增加新的维度，增加batch，在维度dim为0处增加

with torch.no_grad():
    outputs = net(im)
    # predict = torch.max(outputs, dim=1)[1].data.numpy()  # 传入index
    predict = torch.softmax(outputs, dim =1)
# print(classes[int(predict)])
print(predict)