import cv2
import numpy as np

for b in xrange(1000):
    fname='data/face'+str(b)+'.jpg'
    img=cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test",img)
    cv2.imwrite(fname,img)
    key=cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
