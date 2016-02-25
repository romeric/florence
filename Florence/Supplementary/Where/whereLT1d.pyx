cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.profile(False)
@cython.wraparound(False)
def whereLT1d(np.ndarray[DTYPE_t, ndim=1] Arr, DTYPE_t i):
    x=()
    cdef int j
    cdef int rows = Arr.shape[0]
    
    for j in range(0,rows):
        if Arr[j] < i:
            x = x+(j,)
    return x