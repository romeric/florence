import numpy as np
cimport numpy as np 
import cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def makezero(np.ndarray[DTYPE_t, ndim=2] A, DTYPE_t tol=1.0e-14):
	cdef int i,j
	cdef int a1 = A.shape[0]
	cdef int a2 = A.shape[1]
	for i in range(a1):
		for j in range(a2):
			if np.abs(A[i,j]) < tol:
				A[i,j] = 0

	return A 


