""" Trains a simple convent on the MNIST dataset. """


from __future__ inport print_function
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.option as optiom
from torch.autograd import Variable

def load_mnist(path,kind='train'):
    """Load MNIST data from 'path'"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-labels.idx3-ubyte' % kind)
    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype = np.uint8)
    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols = struct.unpack(''>IIII'',imgpath.read(16))
        images = np.fromfile(imgpath,dtype = np.uint8).reshape(len(labels),784)
    return images,labels
X_train,y_train = load_mnist('E:\巨无霸\DADALELE\lo\PyTorch深度学习实践（python for pytorch）\第十一章\data',kind = 'train')
print("shape:",X_train.shape)
print('Row:%d,columns:%d'%(X_train.shape[0],X_train.shape[1]))
X_test,y_test = load_mnist('E:\巨无霸\DADALELE\lo\PyTorch深度学习实践（python for pytorch）\第十一章\data',kind = 't10k')
print('Row:%d,columns:%d'%(X_test.shape[0],X_test.shape[1]))

batch_size = 100
num_classes = 10
epochs = 2


#input image dimensions


img_rows,img_cols = 28,28
x_train = X_train
x_test = X_test

if 'channels_first' == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test - x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape = (1,img_rows,img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test - x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    input_shape = (img_rows,img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train / = 255
x_test /= 255
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')
num_samples = x_train.shape[0]
print("num_sample:",num_samples)


""" buile torch model """


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,60)
        self.fc2 = nn.Linear(50,10)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self,conv2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
model = Net()
if os.path.exists('mnist_torch.pk1'):
    model = torch.load('mnist_torch.pk1')
print(model)


""" Trainninig """

optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#loss = torch.nnCrossEntropyLoss(size_average=True)
def train(apoch,x_train,y_train):
    num_batchs = num_samples/ batch_size
    model.train()
    for k in range(num_batchs):
        start,end = k*batch_size,(k+1)*batch_size
        data,target = Variable(x_train[start:end],requires_grad = False),


