import numpy as np
cimport numpy as np
import time
cdef inline np.ndarray[np.float64_t, ndim=2] cmaxpool(np.ndarray[np.float64_t, ndim=2] mat):
    lenmat=len(mat)
    mat_1=np.empty((lenmat/2,lenmat/2))
    t=time.clock()
    for n in xrange(lenmat/2):
            for m in xrange(lenmat/2):
                mat_1[n][m]=max([mat[n*2][m*2],mat[n*2+1][m*2],mat[n*2][m*2+1],mat[n*2+1][m*2+1]])
    return mat_1

cdef inline np.ndarray[np.float64_t, ndim=2] cresize0(np.ndarray[np.float64_t, ndim=2] mat):
    lenmat=len(mat)
    mat_2=np.zeros((lenmat*2,lenmat*2))
    for n in xrange(lenmat):
        for m in xrange(lenmat):
            mat_2[n*2][m*2]=mat[n][m]

    return mat_2

cdef inline np.ndarray[np.float64_t, ndim=2] max_pool(np.ndarray[np.float64_t, ndim=2] x):
    """Return maximum in groups of 2x2 for a N,h,w image"""
    N,h,w = x.shape
    x = x.reshape(N,h/2,2,w/2,2).swapaxes(2,3).reshape(N,h/2,w/2,4)
    return np.amax(x,axis=3)

def maxpool(mat):
    return max_pool(mat)

def resize0(mat):
    return cresize0(mat)
