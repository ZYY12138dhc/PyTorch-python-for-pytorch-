from __future__ import print_function
import os
import struct
import numpy as np
import cv2

import cv2
#两个回调函数
def GaussianBlurSize(GaussianBlur_size):
    global KSIZE 
    KSIZE = GaussianBlur_size * 2 +3
    print(KSIZE, SIGMA)
    dst = cv2.GaussianBlur(scr, (KSIZE,KSIZE), SIGMA, KSIZE) 
    cv2.imshow(window_name,dst)





KSIZE = 3
SIGMA = 3
image = cv2.imread("C:/Users/15064/Desktop/lena.png")#地址最好是本地的地址，不要移动硬盘的地址
print("image shape:",image.shape)
dst = cv2.GaussianBlur(image, (KSIZE,KSIZE), SIGMA, KSIZE) 
cv2.imshow("img1",image)
cv2.imshow("img2",dst)
cv2.waitKey()
