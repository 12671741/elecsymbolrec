import numpy as np
from scipy.signal import convolve2d

ker0 = np.load('weights/ker0.npy')
W0 = np.load('weights/W0.npy')
W1 = np.load('weights/W1.npy')
W2 = np.load('weights/W2.npy')

def SoftPlus(x, deriv=False):  #using the solftplus
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x));

def bipola_logi(x, deriv=False):  #using the solftplus
    if(deriv==True):
        return 0.5*(1-x*x)
    return (1-np.exp(-x))/(1+np.exp(-x));

def max_pool(x):
    h,w = x.shape
    x = x.reshape(h/2,2,w/2,2).swapaxes(1,2).reshape(h/2,w/2,4)
    return np.amax(x,axis=2)

def classify(frame):
    frame=frame.astype(float)/128-1
    covxl0=np.empty((4,14,14))
    for r in range(4):
        covxl0[r] = SoftPlus(max_pool(convolve2d(frame,ker0[r],'valid')))#32 X 32
    l0 = np.reshape(covxl0,(1,-1))
    l1 = bipola_logi(np.dot(l0, W0))
    l2 = bipola_logi(np.dot(l1, W1))
    l3 = bipola_logi(np.dot(l2, W2))
    return l3
