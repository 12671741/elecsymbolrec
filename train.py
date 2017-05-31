import re
import numpy as np
import time
import cv2
import tflearn
import os

def nonlin(x, deriv=False):  #using the softmax
    if(deriv==True):
        return (x*(1-x))
        #return 0.5*(1-x)*(1-x)
    return 1/(1+np.exp(-x));
    #return (1-np.exp(-x))/(1+np.exp(-x))
batch_size = 10
updateData=1
randomweight=1

dlen=10
neul1=1025
neul2=64
neul3=32

learningrate=0.01
if updateData:
    print("creating data from database")
    f=open('../data/d.txt','r')
    f=f.read()
    f=f.split('\n')
    f=f[0:len(f)-1]
    totallen=len(f)
    total_batch = int(totallen/batch_size)
    batch_ys=np.empty((total_batch,batch_size,dlen))
    batch_xs=np.empty((total_batch,batch_size,neul1))
    ys=0
    for j in xrange(total_batch):
        for i in xrange(batch_size):
            fname='../data/face'+str(ys)+'.jpg'
            img=cv2.imread(fname,0)
            img=np.reshape(img,(-1))
            img=img.astype(float)
            batch_xs[j][i]=np.hstack((img,255))/255
            batch_ys[j][i]=eval(f[i+j*batch_size].replace(" ",","))
            ys=ys+1

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
    W0 = 2*np.random.random((neul1,neul2)) - 1
    W1 = 2*np.random.random((neul2,neul3)) - 1
    W2 = 2*np.random.random((neul3,10)) - 1
else:
    W0 = np.load('weights/W0.npy')
    W1 = np.load('weights/W1.npy')
    W2 = np.load('weights/W2.npy')

testing_portion=0.95
for j in range(1000):
    trainacu=0
    testacu=0
    for i in range(total_batch):
        if i < int(total_batch*testing_portion):
            l1 = nonlin(np.dot(batch_xs[i], W0))
            l2 = nonlin(np.dot(l1, W1))
            l3 = nonlin(np.dot(l2, W2))

             # Back propagation of errors using the chain rule.
            l3_error = batch_ys[i] - l3
            l3_delta = l3_error*nonlin(l3, deriv=True)#(batch_ys - l2)*(l2*(1-l2))

            l2_error = l3_delta.dot(W2.T)#error signal
            l2_delta = l2_error*nonlin(l2, deriv=True)#(batch_ys - l2)*(l2*(1-l2))

            l1_error = l2_delta.dot(W1.T)
            l1_delta = l1_error * nonlin(l1,deriv=True)
            #update weights (no learning rate term)
            W2 += l2.T.dot(l3_delta)*learningrate
            W1 += l1.T.dot(l2_delta)*learningrate
            W0 += batch_xs[i].T.dot(l1_delta)*learningrate
            #print("Error: " + str(np.mean(np.abs(l2_error))))
            trainacu+=np.sum(np.argmax(l3,axis=1)==np.argmax(batch_ys[i],axis=1))
        else:
            l1 = nonlin(np.dot(batch_xs[i], W0))
            l2 = nonlin(np.dot(l1, W1))
            l3 = nonlin(np.dot(l2, W2))
            testacu+=np.sum(np.argmax(l3,axis=1)==np.argmax(batch_ys[i],axis=1))
    print("training accuracy: ", trainacu.astype(float)/(totallen*testing_portion),",iteration: ",j,"  validation accuracy = ", testacu.astype(float)/(totallen*(1-testing_portion)))
    #print("Error: " + str(np.mean(np.abs(l3_error))),",iteration: ",j,"  velerror = ", acu.astype(float)/ys)
    if j%20==19:
        np.save('weights/W0', W0)
        np.save('weights/W1', W1)
        np.save('weights/W2', W2)
        print "weight saved"
        #batch_xs,batch_ys=tflearn.data_utils.shuffle(batch_xs,batch_ys)
print("Output trained W")
print W0
print W1
print W2

np.save('weights/W0', W0)
np.save('weights/W1', W1)
np.save('weights/W2', W2)






#im=X[0:18].sum(axis=0)
#show_im(im)
