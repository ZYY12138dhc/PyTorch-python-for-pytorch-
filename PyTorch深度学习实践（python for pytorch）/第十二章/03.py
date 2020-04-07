from __future__ import print_function
import os
import struct
import math
import numpy as np
import cv2

def rotate(
    img,  #image matrix
    angle #angle of rotation
    ):    
    height = img.shape[0]
    width = img.shape[1]    
    if angle%180 == 0:
        scale = 1
    elif angle%90 == 0:
        scale = float(max(height, width))/min(height, width)
    else:
        scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)
    #print 'scale %f\n' %scale        
    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))
    return rotateImg #rotated image


image = cv2.imread("C:/Users/15064/Desktop/lena.png")
dst = rotate(image,60)
cv2.imshow("img1",image)
cv2.imshow("img2",dst)
cv2.waitKey()



