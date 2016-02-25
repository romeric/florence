import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, profile

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t



cdef get_pair(DTYPE_t *elements, long i, int rows, int cols):
    
    cdef int j,k
    cdef int counter = 0
    cdef int counter1 = 0
    
    cdef int a = 0
    cdef int b = 0

    for j in range(0,rows):
        for k in range(0,cols):
            if elements[counter] == i:
                if counter1 == 0:
                    a = j
                    b = k
                    return (a,b)
                
            counter += 1
            
                    
    return (a,b)
    
    
    

@boundscheck(False)
@wraparound(False)
def whereEQ1HitCount(np.ndarray[DTYPE_t, ndim=2] elements, DTYPE_t i):

    cdef long rows = elements.shape[0]
    cdef long cols = elements.shape[1]
    
    return get_pair(&elements[0,0],i,rows,cols)