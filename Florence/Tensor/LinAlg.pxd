import numpy as np
cimport numpy as np

#----------------------------------------------------------------------------------------------------
cpdef inverse(np.ndarray[double,ndim=2] A)
cdef inv2(double *A)
cdef inv3(double *A)

#----------------------------------------------------------------------------------------------------
cpdef inverse_transpose(np.ndarray[double,ndim=2] A)
cdef invT2(double *A)
cdef invT3(double *A)

#----------------------------------------------------------------------------------------------------
# cpdef determinant(double [:,:] A)
cpdef determinant(np.ndarray[double,ndim=2] A)
cdef double det2(const double *A)
cdef double det3(const double *A)

# DO NOT USE THESE - NO PAY OFF AS NUMPY DOT IS VERY GOOD EVEN FOR 2X2 ARRAYS
#----------------------------------------------------------------------------------------------------
cpdef dgemm(np.ndarray[double,ndim=2] A, np.ndarray[double,ndim=2] B)
cdef dgemm2(const double *A, const double *B)
cdef dgemm3(const double *A, const double *B)
#----------------------------------------------------------------------------------------------------
cpdef daxpy(np.ndarray[double,ndim=2] A, np.ndarray[double,ndim=1] b)
cdef daxpy2(const double *A, const double *b)
cdef daxpy3(const double *A, const double *b)