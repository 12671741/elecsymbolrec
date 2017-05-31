import numpy as np
import cv2
import camprofile

kernel = camprofile.kernal
width=camprofile.width
height=camprofile.height

def imgproc(frame):
    out=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #out = cv2.medianBlur(out,5)
    out=cv2.bilateralFilter(out,4,30,30)
    out=cv2.adaptiveThreshold(out,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    #out=cv2.dilate(out, kernel,iterations=1)
    #out=cv2.erode(out, kernel,iterations=5)

    out, contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out=cv2.rectangle(out, (width/2-camprofile.cropw-1,height/2-camprofile.croph-2), (width/2+camprofile.cropw+1,height/2+camprofile.croph+2),(0,255,0))
    #out=cv2.drawContours(frame, cont,-1, (0,255,0), 1)
    return out
