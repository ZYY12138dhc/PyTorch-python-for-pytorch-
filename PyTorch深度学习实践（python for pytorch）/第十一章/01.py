"""
#eg.1
from __future__ import print_function
                            
import os
import struct
import numpy as np
import cv2

def load_mnist(path,kind='train'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
    with open(labels_path,'rb')as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path,'rb')as imgpath:
        magic,num,rows,cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
    return images,labels
X_train,y_train = load_mnist('E:\巨无霸\DADALELE\lo\PyTorch深度学习实践（python for pytorch）\第十一章\data',kind = 'train')
print('Row:%d,columns:%d'%(X_train.shape[0],X_train.shape[1]))
X_test,y_test = load_mnist('E:\巨无霸\DADALELE\lo\PyTorch深度学习实践（python for pytorch）\第十一章\data',kind = 't10k')
print('Row:%d,columns:%d'%(X_test.shape[0],X_test.shape[1]))

image1 = X_train[1]
image1 = image1.astype('float32')
image1 = image1.reshape(28,28,1)
cv2.imwrite('1.jpg',image1)

img = cv2.imread("E:\time.jpg")

cv2.namedWindow('lo')
cv2.imshow('time',img)
cv2.waitKey(10000)

#eg.2



知识点：
1.  Mnist手写数字图像识别
2.  用numpy读取mnist.npz
    Mnist.npz是一个numpy数组文件
    
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

"""




import tensorflow
import numpy as np
class mnist_data(object):
    def load_npz(self,path):
        f = np.load(path)
        for i in f:
            print (i)
        x_train = f['trainInps']
        y_train = f['trainTargs']
        x_test = f['testInps']
        y_test = f['testTargs']
        f.close()
        return (x_train,y_train),(x_test,y_test)
a = mnist_data()
(x_train,y_train),(x_test,y_test) = a.load_npz('E:\巨无霸\DADALELE\lo\PyTorch深度学习实践（python for pytorch）\第十一章\data\mnist.npz')
print("train rows:%d,test rows:%d"% (x_train.shape[0],x_test.shape[0]))
print("x_train shape",x_train.shape)
print("y_train shape",y_train.shape)

