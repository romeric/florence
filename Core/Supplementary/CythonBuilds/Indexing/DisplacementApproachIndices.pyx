import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

cdef inline void DisplacementApproachIndices_Constitutive(double *B,double* SpatialGradient, int nvar, int rows, int cols):
	cdef int i
	for i in range(rows):
		B[i*cols*nvar] = SpatialGradient[i]
		B[i*cols*nvar+1] = SpatialGradient[i+rows]
		
		B[i*cols*nvar+(cols+2)] = SpatialGradient[i]
		B[i*cols*nvar+(cols+2)+1] = SpatialGradient[i+rows]



cdef inline void DisplacementApproachIndices_Geometric(double *B,double* SpatialGradient, int nvar, int rows, int cols):
	cdef int i
	for i in range(rows):
		B[i*cols*nvar] = SpatialGradient[i]
		B[i*cols*nvar+1] = SpatialGradient[i+rows]
		
		B[i*cols*nvar+(cols+2)] = SpatialGradient[i]
		B[i*cols*nvar+(cols+2)+1] = SpatialGradient[i+rows]


@boundscheck(False)
@wraparound(False)
def GeometricIndices(np.ndarray[double,ndim=2] B, np.ndarray[double,ndim=2] SpatialGradient, int nvar):
	cdef int rows = SpatialGradient.shape[1]
	cdef int cols = B.shape[1]
	DisplacementApproachIndices_Geometric(&B[0,0],&SpatialGradient[0,0],nvar,rows,cols)