# coding=utf-8
import cv2
import numpy as np
'''Harris算法角点特征提取'''
img = cv2.imread('C:/Users/15064/Desktop/contour.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# {标记点大小，敏感度（3~31,越小越敏感）}
# OpenCV函数cv2.cornerHarris() 有四个参数 其作用分别为 :
dst = cv2.cornerHarris(gray,2,23,0.04)
img[dst>0.01 * dst.max()] = [0,0,255]
cv2.imshow('corners',img)
cv2.waitKey()
cv2.destroyAllWindows()
