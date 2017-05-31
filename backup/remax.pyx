import numpy as np
cimport numpy as np
cdef inline np.ndarray[np.float64_t, ndim=2] cmaxpool(np.ndarray[np.float64_t, ndim=2] mat):
    mat_1=np.empty((len(mat)/2,len(mat)/2))
    for n in xrange(len(mat)/2):
            for m in xrange(len(mat)/2):
                mat_1[n][m]=np.max([mat[n*2][m*2],mat[n*2+1][m*2],mat[n*2][m*2+1],mat[n*2+1][m*2+1]])
    return mat_1

cdef inline np.ndarray[np.float64_t, ndim=2] cresize0(np.ndarray[np.float64_t, ndim=2] mat):
    mat_2=np.empty((len(mat)*2,len(mat)*2))
    for n in xrange(len(mat)):
        for m in xrange(len(mat)):
            mat_2[n*2][m*2]=mat[n][m]
            mat_2[n*2+1][m*2]=mat[n][m]
            mat_2[n*2][m*2+1]=mat[n][m]
            mat_2[n*2+1][m*2+1]=mat[n][m]
    return mat_2

def maxpool(mat):
    return cmaxpool(mat)

def resize0(mat):
    return cresize0(mat)
