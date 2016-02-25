cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.profile(False)
@cython.wraparound(False)
def whereLT(np.ndarray[DTYPE_t, ndim=2] Arr, DTYPE_t i):
    x=(); y=()
    cdef int j,k
    cdef int rows = Arr.shape[0]
    cdef int cols = Arr.shape[1]
    
    for j in range(0,rows):
        for k in range(0,cols):
            if Arr[j,k] < i:
                x = x+(j,)
                y = y+(k,)
    return x,y
    # return np.asarray(x),np.asarray(y)