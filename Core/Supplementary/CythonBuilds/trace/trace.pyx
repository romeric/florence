
# WORKS ONLY FOR FLOATING POINT ARRAYS
import cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def trace(np.ndarray[DTYPE_t, ndim=2] A):
    cdef int i
    cdef DTYPE_t trA = 0
    cdef int n = A.shape[0]
    
    for i in range(0,n):
       trA += A[i,i] 
    return trA