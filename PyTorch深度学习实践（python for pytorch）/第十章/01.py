"""
#eg.1
import torch
import torch.nn.functional as F
from torch.autograd import Variable
print("conv2d sample")
a = range(20)
x = Variable(torch.Tensor(a))
x = x.view(1,1,4,5)
print("x variable :",x)
y = F.max_pool2d(x,kernel_size=2,stride=2)
print("y:",y)

#eg.2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
print("conv2d sample")
a = range(20)
x = Variable(torch.Tensor(a))
x = x.view(1,1,4,5)
print("x variable :",x)
y = F.avg_pool2d(x,kernel_size=2,stride=2)
print("y:",y)

#eg.3
import torch
import torch.nn.functional as F
from torch.autograd import Variable
print("conv1d sample")
a = range(16)
x = Variable(torch.Tensor(a))
x = x.view(1,1,16)
print("x variable:",x)
y = F.max_pool1d(x,kernel_size=2,stride=2)
print("y:",y)

#eg.4
import torch
import torch.nn.functional as F
from torch.autograd import Variable
print("conv1d sample")
a = range(16)
x = Variable(torch.Tensor(a))
x = x.view(1,2,8)
print("x variable:",x)
y = F.max_pool1d(x,kernel_size=2,stride=2)
print("y:",y)

知识点：
1.  池化层
2.  max_pool2d
    用法：torch.nn.functional.max_pool2d(input,lernel_size,stride=None,padding=0,ceil_mode=  count_include_pad=  )
3.  avg_pool2d
4.  max_pool1d


"""




