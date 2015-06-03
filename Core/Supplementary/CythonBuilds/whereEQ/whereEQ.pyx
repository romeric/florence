import numpy as np
cimport numpy as np
import cython

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t

@cython.boundscheck(False)
@cython.profile(False)
@cython.wraparound(False)
def whereEQ(np.ndarray[DTYPE_t, ndim=2] elements, DTYPE_t i):
    x=(); y=()
    cdef int j,k
    cdef int rows = elements.shape[0]
    cdef int cols = elements.shape[1]
    
    for j in range(0,rows):
        for k in range(0,cols):
            if elements[j,k]==i:
                x = x+(j,)
                y = y+(k,)
    return x,y