
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, profile

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t


cdef get_pairs(DTYPE_t *elements, DTYPE_t i, long rows, long cols):
    
    cdef int j,k
    cdef int counter = 0
    cdef int counter1 = 0
    
    cdef int a = 0
    cdef int b = 0
    cdef int c = 0
    cdef int d = 0

    for j in range(0,rows):
        for k in range(0,cols):
            if elements[counter] == i:
                if counter1 == 0:
                    a = j
                    b = k
                    counter1 += 1
                else:
                    c = j
                    d = k
                    return (a,c),(b,d)
                    
            counter += 1
            
                    
    return (a,c),(b,d)
    
    
    

@boundscheck(False)
@wraparound(False)
def whereEQ_2HitCounts(np.ndarray[DTYPE_t, ndim=2] elements, DTYPE_t i):

    cdef int rows = elements.shape[0]
    cdef int cols = elements.shape[1]
    cdef int counter = rows*cols
    
    return get_pairs(&elements[0,0],i,rows,cols)