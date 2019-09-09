import re
import numpy as np
import cv2
import os

img=cv2.imread('/Users/tingjiejameszhang/Desktop/1.jpg')
cv2.imshow('ori',img)
img=cv2.imread('/Users/tingjiejameszhang/Desktop/1.jpg',0)
cv2.imshow('gray',img)
img=cv2.bilateralFilter(img,10,30,30)
cv2.imshow('filtered',img)
img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('adaptiveThreshold',img)
cv2.waitKey(0)
