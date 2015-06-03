
# NOTE:
# THIS FUNCTION IS FASTER THAN np.einsum FOR SMALLER MATRICES AND SLOWER THAN np.einum
# FOR LARGER MATRICES. 
 
import cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def doublecontract(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B):
    cdef int i,j
    cdef int a1 = A.shape[0]
    cdef int b1 = B.shape[0]
    cdef int a2 = A.shape[1]
    cdef int b2 = B.shape[1]
    if a1!=b1 or a2!=b2:
        raise ValueError('incompatible dimensions for double contraction')
    
    cdef DTYPE_t dc = 0
    for i in range(0,a1):
        for j in range(0,b2):
            dc += A[i,j]*B[i,j]
    return dc
