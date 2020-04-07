"""
eg.1
import torch
x = torch.Tensor(3,4)
print('x Tensor :',x)
print(b.data)

eg.2
import torch
from torch.autograd import Variable
x = Variable(torch.Tensor(2,2))
print('x Variable :',x)
print(x.data)
print(x.grad)

知识点：
1. pytorch中的数据都是封装成Tensor来引用的
2. 引用Variable包含导数grad,x.data是数据，是Tensor,x.grad是计算过程中动态变化的导数，初始值为None





"""


