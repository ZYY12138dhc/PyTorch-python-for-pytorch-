"""
#eg.1
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

print("conv2d sample")
a = torch.ones(4,4)
x = Variable(torch.Tensor(a))
x = x.view(1,1,4,4)
print("x variable:",x)
b = torch.ones(2,2)
b[0,0]=0.1
b[0,1]=0.2
b[1,0]=0.3
b[1,1]=0.4
weights = Variable(b)
weights = weights.view(1,1,2,2)
print("weights:",weights)
y = F.conv2d(x,weights,padding = 0)
print("y:",y)

#eg.2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
print("conv1d sample")
a = range(16)
x = Variable(torch.Tensor(a))
x = x.view(1,1,16)
print("x variable:",x)
b = torch.ones(3)
b[0]=0.1
b[1]=0.2
b[2]=0.3
weights = Variable(b)
weights = weights.view(1,1,3)
print("weights:weights")
y = F.conv1d(x,weights,padding=0)
print("y:",y)

#eg.3
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
print("conv1d sample")
a = range(16)
x = Variable(torch.Tensor(a))
x = x.view(1,2,8)
print("x variable:",x)
b = torch.ones(6)
b[0]=0.1
b[1]=0.2
b[2]=0.3
weights = Variable(b)
weights = weights.view(1,2,3)
print("weights:weights")
y = F.conv1d(x,weights,padding=0)
print("y:",y)



知识点：
卷积层
1.  Conv2d
    函数定义:torch.nn.functional.conv2d(input,weight,bias=None,stride=1,padding=0,dilation=  groups=  )
2.  Conv1d
    函数定义：torch.nn.functional.conv1d(input,weight,bias= None,stride=1,padding=0,dilation=  groups=  )
    

"""













 
