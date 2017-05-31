import numpy as np
import time
import remax

a=np.random.random((30,30))
ker = np.load('weights/ker.npy')
t=time.clock()
for i in xrange(1000):
	covx_1=remax.cresize0(a)
print time.clock()-t
