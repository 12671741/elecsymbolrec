import re
import numpy as np
import time
import cv2
import tflearn
import os

def Sigmoid(x, deriv=False):  #using the solftplus
    if(deriv==True):
        return (x-np.square(x))
    return 1/(1+np.exp(-x))

def bipola_logi(x, deriv=False):  #defining the bipola_logi
    if(deriv==True):
        return 0.5*(1-np.square(x))
    return 2/(1+np.exp(-x))-1

def SoftPlus(x, deriv=False):
    if(deriv==True):
        return 1/(1+np.exp(x))
    return np.log(1+np.exp(x))
def ReLU(x, deriv=False):
    if(deriv==True):
        return x>0
    return np.maximum(x, 0, x)

batch_size = 50
updateData=1
randomweight=1

Nonputs=26
Ninputs=1025
Nhidden_neural=512

#Adaptive Moment Estimation(adam) optimizer parameters
m0=0
v0=0
m1=0
v1=0
b1=0.9
b2=0.999
eps=1e-8

learningrate=0.001
if updateData:
    print("creating data from database")
    f=open('../data26/d.txt','r')
    f=f.read()
    f=f.split('\n')
    f=f[0:len(f)-1]
    totallen=len(f)
    total_batch = int(totallen/batch_size)
    batch_ys=np.empty((total_batch,batch_size,Nonputs))
    batch_xs=np.empty((total_batch,batch_size,Ninputs))
    ys=0
    for j in xrange(total_batch):
        for i in xrange(batch_size):
            fname='../data26/img'+str(ys)+'.jpg'
            img=cv2.imread(fname,0)
            img=np.reshape(img,(-1))
            img=img.astype(np.float32)
            batch_xs[j][i]=1-np.hstack((img,0))/255
            batch_ys[j][i]=eval(f[i+j*batch_size].replace(" ",","))
            ys=ys+1
    batch_ys=batch_ys
    batch_ys=np.reshape(batch_ys,(total_batch*batch_size,1,Nonputs))
    batch_xs=np.reshape(batch_xs,(total_batch*batch_size,1,Ninputs))
    batch_xs,batch_ys=tflearn.data_utils.shuffle(batch_xs,batch_ys)
    batch_ys=np.reshape(batch_ys,(total_batch,batch_size,Nonputs))
    batch_xs=np.reshape(batch_xs,(total_batch,batch_size,Ninputs))
    np.savez_compressed("data26/X",X=batch_xs)
    np.save("data26/Y",batch_ys)
else:
    print "loading saved data"
    z=np.load("data26/X.npz")
    namelist = z.zip.namelist()
    z.zip.extract(namelist[0])
    batch_xs = np.load(namelist[0], mmap_mode='r+')
    os.remove(namelist[0])
    batch_ys=np.load("data26/Y.npy")
    total_batch=len(batch_xs)
    totallen = total_batch*batch_size
if randomweight:
    print("randomise weights")
    np.random.seed(2)
    cycleerrors=[]
    cycleerrorsval=[]
    W0 = 2*np.random.random((Ninputs,Nhidden_neural)) - 1
    W1 = 2*np.random.random((Nhidden_neural+1,Nonputs)) - 1

else:
    cycleerrors = np.load('weights/cycleerrors.npy')
    cycleerrorsval = np.load('weights/cycleerrorsval.npy')
    cycleerrors=cycleerrors.tolist()
    cycleerrorsval=cycleerrorsval.tolist()
    W0 = np.load('weights/W0.npy')
    W1 = np.load('weights/W1.npy')
training_portion=0.6
valing_portion=0.2
aug=np.ones((batch_size,1))
for j in range(60):
    trainacu=0
    valacu=0
    errors=0
    errorsval=0
    for i in xrange(int(total_batch*(training_portion+valing_portion))):
        a1 = np.dot(batch_xs[i], W0)
        l1 = ReLU(a1)
        l1 = np.hstack((l1,aug))
        a2 = np.dot(l1, W1)
        l2 = SoftPlus(a2)
        l2_error = batch_ys[i] - l2#loss
        #l2_error1=(batch_ys[i]/l2-(1-batch_ys[i])/(1-l2))/(2*total_batch*training_portion)#crossentropy
        #print batch_ys[i]/l2
        #print l2_error1
        # Back propagation of errors using the chain rule.
        if i < int(total_batch*training_portion):
            l2_delta = l2_error*SoftPlus(a2, deriv=True)#(batch_ys - l2)*(l2*(1-l2))

            l1_error = l2_delta.dot(W1[0:Nhidden_neural].T)
            l1_delta = l1_error * ReLU(a1,deriv=True)
            #calculate momentum and adaGrad
            dW1=-l1.T.dot(l2_delta)
            m1=b1*m1+(1-b1)*dW1
            v1=b2*v1+(1-b2)*dW1**2
            dW0=-batch_xs[i].T.dot(l1_delta)
            m0=b1*m0+(1-b1)*dW0
            v0=b2*v0+(1-b2)*dW0**2
            #update weights
            W1 -= m1/(np.sqrt(v1)+eps)*learningrate
            W0 -= m0/(np.sqrt(v0)+eps)*learningrate

            #print("Error: " + str(np.mean(np.abs(l2_error))))
            #print np.sum(np.argmax(l2,axis=1)==np.argmax(batch_ys[i],axis=1))
            trainacu+=np.sum(np.argmax(l2,axis=1)==np.argmax(batch_ys[i],axis=1))
            errors+=np.sum(l2_error**2)
        else:
            errorsval+=np.sum(l2_error**2)
            valacu+=np.sum(np.argmax(l2,axis=1)==np.argmax(batch_ys[i],axis=1))
    print("training accuracy: ", trainacu.astype(float)/(totallen*training_portion),",iteration: ",j,"  validation accuracy = ", valacu.astype(float)/(totallen*valing_portion))
    cycleerrors.append((errors)/(totallen*training_portion))
    cycleerrorsval.append((errorsval)/(totallen*valing_portion))
    #print("Error: " + str(np.mean(np.abs(l3_error))),",iteration: ",j,"  velerror = ", acu.astype(float)/ys)
    if j%20==19:
        np.save('weights/cycleerrors', cycleerrors)
        np.save('weights/cycleerrorsval', cycleerrorsval)
        np.save('weights/W0', W0)
        np.save('weights/W1', W1)
        print "weight saved"

np.save('weights/cycleerrors', cycleerrors)
np.save('weights/cycleerrorsval', cycleerrorsval)
np.save('weights/W0', W0)
np.save('weights/W1', W1)
print "weight saved"
print "testing"
testacu=0
for i in range(int(total_batch*(training_portion+valing_portion)),total_batch):
    l1 = ReLU(np.dot(batch_xs[i], W0))
    l1 = np.hstack((l1,aug))#augmenting
    l2 = SoftPlus(np.dot(l1, W1))
    testacu+=np.sum(np.argmax(l2,axis=1)==np.argmax(batch_ys[i],axis=1))
testacu=testacu.astype(float)/(totallen*(1-training_portion-valing_portion))
print("testing accuracy: ",testacu)
#im=X[0:18].sum(axis=0)
#show_im(im)
