import re
import numpy as np
import time
import cv2
import tflearn
import os
from scipy.signal import convolve2d
import remax
import matplotlib.pyplot as plt

def SoftPlus(x, deriv=False):  #using the solftplus
    if(deriv==True):
        return (x-np.square(x))
    return 1/(1+np.exp(-x));

def bipola_logi(x, deriv=False):  #using the solftplus
    if(deriv==True):
        return 0.5*(1-np.square(x))
    return (1-np.exp(-x))/(1+np.exp(-x));

def max_pool(x):
    n,h,w = x.shape
    x = x.reshape(n,h/2,2,w/2,2).swapaxes(2,3).reshape(n,h/2,w/2,4)
    return np.amax(x,axis=3)

batch_size = 1
updateData = 1
randomweight=1

dlen=10
neul1=785 #1024
neul2=64
neul3=24
learningrate=0.001


if updateData:
    print("creating data from database")
    f=open('../data/d.txt','r')
    f=f.read()
    f=f.split('\n')
    f=f[0:len(f)-1]
    totallen=len(f)
    total_batch = int(totallen/batch_size)
    batch_ys=np.empty((total_batch,batch_size,dlen))
    batch_xs=np.empty((total_batch,batch_size,32,32))
    ys=0
    for j in xrange(total_batch):
        for i in xrange(batch_size):
            fname='../data/face'+str(ys)+'.jpg'
            ys=ys+1
            img=cv2.imread(fname,0)
            batch_xs[j][i]=img.astype(float)/128-1
            batch_ys[j][i]=eval(f[i+j*batch_size].replace(" ",","))

    batch_xs,batch_ys=tflearn.data_utils.shuffle(batch_xs,batch_ys)
    np.savez_compressed("data/X",X=batch_xs)
    np.save("data/Y",batch_ys)
else:
    print "loading saved data"
    z=np.load("data/X.npz")
    namelist = z.zip.namelist()
    z.zip.extract(namelist[0])
    batch_xs = np.load(namelist[0], mmap_mode='r+')
    os.remove(namelist[0])
    batch_ys=np.load("data/Y.npy")
    total_batch=len(batch_xs)
    totallen = total_batch*batch_size

if randomweight:
    print("randomise weights")
    np.random.seed(2)
    ker0  = 2*np.random.random((4,5,5))-1
    W0 = 2*np.random.random((neul1,neul2)) - 1
    W1 = 2*np.random.random((neul2,neul3)) - 1
    W2 = 2*np.random.random((neul3,dlen)) - 1
    cycleerrors=[]
    cycleerrorsval=[]
else:
    cycleerrors = np.load('weights/cycleerrors.npy')
    cycleerrorsval = np.load('weights/cycleerrorsval.npy')
    cycleerrors=cycleerrors.tolist()
    cycleerrorsval=cycleerrorsval.tolist()
    ker0 = np.load('weights/ker0.npy')
    W0 = np.load('weights/W0.npy')
    W1 = np.load('weights/W1.npy')
    W2 = np.load('weights/W2.npy')
np.set_printoptions(precision=3)
training_portion=0.8

print 'there are ',totallen,' datas,'
print 'start training'

covx0_size=32-len(ker0[0])+1
covxl0=np.zeros((4,covx0_size,covx0_size),dtype=float)
covx_l0=np.zeros((4,covx0_size/2,covx0_size/2),dtype=float)
ker0_delta2=np.zeros((4,covx0_size,covx0_size),dtype=float)
kerchange  = np.zeros((4,len(ker0[0]),len(ker0[0])),dtype=float)
t=time.clock()
aug=np.array(-1).reshape(1,-1)
for j in range(1000):
    trainacu=0
    testacu=0
    errors=0
    errorsval=0
    for i in range(total_batch):
        for r in range(4):
            covxl0[r] = convolve2d(batch_xs[i][0],ker0[r],'valid')#32 X 32
            covx_l0[r]=SoftPlus(max_pool(covxl0)[r])
        l0 = np.hstack((np.reshape(covx_l0,(1,-1)),aug))
        l1 = bipola_logi(np.dot(l0, W0))#shape (1, 32)
        l2 = bipola_logi(np.dot(l1, W1))#shape (1, 16)
        l3 = bipola_logi(np.dot(l2, W2))#shape (1, 10)
        l3_error = batch_ys[i] - l3#shape (1, 10)
        if i > int(total_batch*training_portion):
            errorsval+=np.sum(l3_error*l3_error)/2

        if i < int(total_batch*training_portion):#back propagation
             # Back propagation of errors using the chain rule.
            l3_delta = l3_error*bipola_logi(l3, deriv=True)#shape (1, 10)

            l2_error = l3_delta.dot(W2.T)#(1, 16)
            l2_delta = l2_error*bipola_logi(l2, deriv=True)#(1, 16)

            l1_error = l2_delta.dot(W1.T)#(1, 32)
            l1_delta = l1_error * bipola_logi(l1,deriv=True)#(1, 32)

            #covlayer
            ker0_delta = l1_delta.dot(W0.T)*SoftPlus(l0,deriv=True)
            ker0_delta=ker0_delta[0,0:784].reshape(4,-1)
            for r in range(4):
                ker0_delta2[r][::2][:,::2]=ker0_delta[r].reshape((covx0_size/2,covx0_size/2))[r]
                kerchange[r]=convolve2d(batch_xs[i][0],ker0_delta2[r],'valid')*learningrate
            ker0 += kerchange
            W2 += l2.T.dot(l3_delta)*learningrate
            W1 += l1.T.dot(l2_delta)*learningrate
            W0 += l0.T.dot(l1_delta)*learningrate
            #print("Error: " + str(np.mean(np.abs(l2_error))))
            trainacu+=np.sum(np.argmax(l3,axis=1)==np.argmax(batch_ys[i],axis=1))
            #print ker
            #print("ticks: ", trainacu," i: ",i)
            errors+=np.sum(l3_error*l3_error)/2
        else:
            testacu+=np.sum(np.argmax(l3,axis=1)==np.argmax(batch_ys[i],axis=1))
    cycleerrors.append(errors)
    cycleerrorsval.append(errorsval)
    print("accu: ", trainacu.astype(float)/(totallen*training_portion),",iteration: ",j,"  velaccu = ", testacu.astype(float)/(totallen*(1-training_portion)))
    #print("Error: " + str(np.mean(np.abs(l3_error))),",iteration: ",j,"  velerror = ", acu.astype(float)/ys)
    print time.clock()-t,'s'
    t=time.clock()
    if j%5==4:
        print ker0
        np.save('weights/cycleerrors', cycleerrors)
        np.save('weights/cycleerrorsval', cycleerrorsval    )
        np.save('weights/ker0', ker0)
        np.save('weights/W0', W0)
        np.save('weights/W1', W1)
        np.save('weights/W2', W2)
        print "weight saved"
        #batch_xs,batch_ys=tflearn.data_utils.shuffle(batch_xs,batch_ys)
