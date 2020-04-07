"""
eg.1

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

"定义两个二维数组"
sample = Variable(torch.ones(2,2))
a = torch.Tensor(2,2)
a[0,0]=0
a[0,1]=1
a[1,0]=2
a[1,1]=3
target = Variable(a)
print(sample,target)

"nn.L1Loss()损失函数"
criterion = nn.L1Loss()
loss = criterion(sample,target)
print(loss)

"nn.SmoothL1Loss()损失函数"
criterion = nn.SmoothL1Loss()
loss = criterion(sample,target)
print(loss)

"nn.MSELoss()损失函数"
criterion = nn.MSELoss()
loss = criterion(sample,target)
print(loss)

"nn.BCELoss()损失函数"
criterion = nn.BCELoss()
loss = criterion(sample,target)
print(loss)

知识点：
1. nn.L1Loss()【取预测值和真实值的绝对值误差的平均数】
2. nn.SmoothL1Loss()【也称作Huber Loss,误差在（-1，1）上是平方损失，其它情况是L1损失】
3. nn.MSELoss()【平方损失函数，预测值和真实值之间的平方和的平均数】
4. nn.BCELoss()【二分类用的交叉熵】
5. nn.CrossEntropyLoss()【交叉熵损失函数，用于图像识别，对输入参数有格式要求】
6. nn.NLLLoss()【负对数似然损失函数（Negative Log Likelihood）,用于图像识别】
7. nn.NLLLoss2d()【用于图片】



"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
"定义两个二维数组"
sample = Variable(torch.ones(2,2))
a = torch.Tensor(2,2)
a[0,0]=0
a[0,1]=1
a[1,0]=2
a[1,1]=3
target = Variable(a)
criterion = nn.BCELoss()
loss = criterion(sample,target)
print(loss)
