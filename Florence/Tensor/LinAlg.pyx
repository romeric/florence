from cython cimport boundscheck, profile, wraparound
from warnings import warn

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

# A SET OF LINEAR ALGEBRA FUNCTIONS ON 2X2 AND 3X3 MATRICES IMPLEMENTED USING HAND UNROLLING
# IN ORDER TO AVOID/STAY AWAY FROM NUMPY'S OVERHEAD FOR SMALL MATRICES. TO BE USED ONLY AT CYTHON LEVEL
# ALTHOUGH COULD BE USED AT PYTHON LEVEL, BUT THE RETURN MATRICES ARE LISTS ARE LISTS OF LISTS NOT NUMPY
# ARRAYS

# COMPILE IUSING
# cython --cplus LinAlg.pyx
# g++ -shared -fPIC _Numeric.cpp LinAlg.cpp -o LinAlg.so -I/usr/include/python2.7 -lpython2.7 \
# -O3 -pthread -Wall -fwrapv -fno-strict-aliasing -ffast-math -funroll-loops

Integer = np.int32
ctypedef np.int32_t Integer_t


#----------------------------------------------------------------------------------------------------
cdef extern from "_LinAlg.cpp":
    void reverse_cuthill_mckee(Integer_t *ind, Integer_t *ptr, Integer_t num_rows, Integer_t *order)
    # vector[long] reverse_cuthill_mckee(long *ind, long *ptr, long num_rows, long *order) # ALSO AVAILABLE
    # vector[Integer_t] reverse_cuthill_mckee(Integer_t *ind, Integer_t *ptr, Integer_t num_rows)

# REVERSE Cuthill-McKee PERMUTATION FOR SPARSE MATRICES
def symrcm(A):
    """Reverse Cuthill-McKee algorithm for sparse csr_matrix and csc_matrix.
        The return value of symrcm(A) is a permutation vector (1D array) such
        that A(r,r) has its non-zero elements closer to the diagonal. Applying
        this permutation to matrices can be beneficial for the efficiency of the
        linear solvers.
    """

    cdef np.ndarray[Integer_t] ind = A.indices
    cdef np.ndarray[Integer_t] ptr = A.indptr
    cdef Integer_t num_rows = A.shape[0]
    cdef np.ndarray[Integer_t] order = np.zeros(num_rows,dtype=Integer) 

    # cdef vector[int] order = reverse_cuthill_mckee(&ind[0], &ptr[0], num_rows, order)
    reverse_cuthill_mckee(&ind[0], &ptr[0], num_rows, &order[0])
    return order


#---------------------------------------------------------------------------------------------------------------








#----------------------------------------------------------------------------------------------------
@profile(False)
@boundscheck(False)
@wraparound(False)
cpdef inverse(np.ndarray[double,ndim=2] A):
    cdef int ndim = A.shape[0]
    if ndim == 2:
        return inv2(&A[0,0])
    elif ndim == 3:
        return inv3(&A[0,0])
    else:
        warn("inverse of matrices > (3x3) falls back to numpy.linalg. Use that instead")
        return np.linalg.inv(A)
    
cdef inline inv2(double *A):
    cdef double A1_1 = A[0]
    cdef double A1_2 = A[1]
    cdef double A2_1 = A[2]
    cdef double A2_2 = A[3]

    cdef double invA[2][2] 
    
    invA[0][0] = A2_2/(A1_1*A2_2 - A1_2*A2_1)
    invA[0][1] = -A1_2/(A1_1*A2_2 - A1_2*A2_1)
    invA[1][0] = -A2_1/(A1_1*A2_2 - A1_2*A2_1)
    invA[1][1] = A1_1/(A1_1*A2_2 - A1_2*A2_1)
    
    return invA


cdef inline inv3(double *A):
    cdef double A1_1 = A[0]
    cdef double A1_2 = A[1]
    cdef double A1_3 = A[2]
    cdef double A2_1 = A[3]
    cdef double A2_2 = A[4]
    cdef double A2_3 = A[5]
    cdef double A3_1 = A[6]
    cdef double A3_2 = A[7]
    cdef double A3_3 = A[8]
    
    cdef double invA[3][3]
    
    invA[0][0] = (A2_2*A3_3 - A2_3*A3_2)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[0][1] = -(A1_2*A3_3 - A1_3*A3_2)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[0][2] = (A1_2*A2_3 - A1_3*A2_2)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[1][0] = -(A2_1*A3_3 - A2_3*A3_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[1][1] = (A1_1*A3_3 - A1_3*A3_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[1][2] = -(A1_1*A2_3 - A1_3*A2_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[2][0] = (A2_1*A3_2 - A2_2*A3_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[2][1] = -(A1_1*A3_2 - A1_2*A3_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[2][2] = (A1_1*A2_2 - A1_2*A2_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
 
    return invA
#----------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
@profile(False)
@boundscheck(False)
@wraparound(False)
cpdef inverse_transpose(np.ndarray[double,ndim=2] A):
    cdef int ndim = A.shape[0]
    if ndim == 2:
        return invT2(&A[0,0])
    elif ndim == 3:
        return invT3(&A[0,0])
    else:
        warn("inverse transpose of matrices > (3x3) falls back to numpy.linalg. Use that instead")
        return np.linalg.inv(A).T
    
cdef inline invT2(double *A):
    cdef double A1_1 = A[0]
    cdef double A1_2 = A[1]
    cdef double A2_1 = A[2]
    cdef double A2_2 = A[3]

    cdef double invA[2][2] 
    
    invA[0][0] = A2_2/(A1_1*A2_2 - A1_2*A2_1)
    invA[1][0] = -A1_2/(A1_1*A2_2 - A1_2*A2_1)
    invA[0][1] = -A2_1/(A1_1*A2_2 - A1_2*A2_1)
    invA[1][1] = A1_1/(A1_1*A2_2 - A1_2*A2_1)
    
    return invA


cdef inline invT3(double *A):
    cdef double A1_1 = A[0]
    cdef double A1_2 = A[1]
    cdef double A1_3 = A[2]
    cdef double A2_1 = A[3]
    cdef double A2_2 = A[4]
    cdef double A2_3 = A[5]
    cdef double A3_1 = A[6]
    cdef double A3_2 = A[7]
    cdef double A3_3 = A[8]
    
    cdef double invA[3][3]
    
    invA[0][0] = (A2_2*A3_3 - A2_3*A3_2)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[0][1] = -(A2_1*A3_3 - A2_3*A3_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[0][2] = (A2_1*A3_2 - A2_2*A3_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[1][0] = -(A1_2*A3_3 - A1_3*A3_2)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[1][1] = (A1_1*A3_3 - A1_3*A3_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[1][2] = -(A1_1*A3_2 - A1_2*A3_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[2][0] = (A1_2*A2_3 - A1_3*A2_2)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[2][1] = -(A1_1*A2_3 - A1_3*A2_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
    invA[2][2] = (A1_1*A2_2 - A1_2*A2_1)/(A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1)
 
    return invA

#----------------------------------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cpdef determinant(np.ndarray[double,ndim=2] A):
    cdef int ndim = A.shape[0]
    if ndim == 2:
        return det2(&A[0,0])
    elif ndim == 3:
        return det3(&A[0,0])
    else:
        warn("determinant of matrices > (3x3) falls back to numpy.linalg. Use that instead")
        return np.linalg.det(A)


cdef inline double det2(const double *A):
    cdef double A1_1 = A[0]
    cdef double A1_2 = A[1]
    cdef double A2_1 = A[2]
    cdef double A2_2 = A[3]

    return A1_1*A2_2 - A1_2*A2_1


cdef inline double det3(const double *A):
    cdef double A1_1 = A[0]
    cdef double A1_2 = A[1]
    cdef double A1_3 = A[2]
    cdef double A2_1 = A[3]
    cdef double A2_2 = A[4]
    cdef double A2_3 = A[5]
    cdef double A3_1 = A[6]
    cdef double A3_2 = A[7]
    cdef double A3_3 = A[8]

    return A1_1*A2_2*A3_3 - A1_1*A2_3*A3_2 - A1_2*A2_1*A3_3 + A1_2*A2_3*A3_1 + A1_3*A2_1*A3_2 - A1_3*A2_2*A3_1












# DO NOT USE THESE - NO PAY OFF AS NUMPY DOT IS VERY GOOD EVEN FOR 2X2 ARRAYS
#----------------------------------------------------------------------------------------------------

cpdef dgemm(np.ndarray[double,ndim=2] A, np.ndarray[double,ndim=2] B):
    cdef:
        int ndim1 = A.shape[1]
        int ndim2 = B.shape[0]

    if ndim1 != ndim2:
        raise ValueError("matrices are not aligned")

    if ndim1 == 2:
        return dgemm2(&A[0,0], &B[0,0])
    elif ndim1 == 3:
        return dgemm3(&A[0,0], &B[0,0])
    else:
        warn("dgemm of matrices > (3x3) falls back to numpy.dot. Use that instead")
        return A.dot(B)


cdef inline dgemm2(const double *A, const double *B):

    cdef:
        double A1_1 = A[0]
        double A1_2 = A[1]
        double A2_1 = A[2]
        double A2_2 = A[3]

        double B1_1 = B[0]
        double B1_2 = B[1]
        double B2_1 = B[2]
        double B2_2 = B[3]

        double AB[2][2]
    
    AB[:] = [[ A1_1*B1_1 + A1_2*B2_1, A1_1*B1_2 + A1_2*B2_2], [ A2_1*B1_1 + A2_2*B2_1, A2_1*B1_2 + A2_2*B2_2]]
    return AB


cdef inline dgemm3(const double *A, const double *B):

    cdef:
        double A1_1 = A[0]
        double A1_2 = A[1]
        double A1_3 = A[2]
        double A2_1 = A[3]
        double A2_2 = A[4]
        double A2_3 = A[5]
        double A3_1 = A[6]
        double A3_2 = A[7]
        double A3_3 = A[8]

        double B1_1 = B[0]
        double B1_2 = B[1]
        double B1_3 = B[2]
        double B2_1 = B[3]
        double B2_2 = B[4]
        double B2_3 = B[5]
        double B3_1 = B[6]
        double B3_2 = B[7]
        double B3_3 = B[8]

        double AB[3][3]

    AB[:] = [[ A1_1*B1_1 + A1_2*B2_1 + A1_3*B3_1, A1_1*B1_2 + A1_2*B2_2 + A1_3*B3_2, A1_1*B1_3 + A1_2*B2_3 + A1_3*B3_3],
        [ A2_1*B1_1 + A2_2*B2_1 + A2_3*B3_1, A2_1*B1_2 + A2_2*B2_2 + A2_3*B3_2, A2_1*B1_3 + A2_2*B2_3 + A2_3*B3_3],
        [ A3_1*B1_1 + A3_2*B2_1 + A3_3*B3_1, A3_1*B1_2 + A3_2*B2_2 + A3_3*B3_2, A3_1*B1_3 + A3_2*B2_3 + A3_3*B3_3]]

    return AB
 

#---------------------------------------------------------------------------------------------------------------

cpdef daxpy(np.ndarray[double,ndim=2] A, np.ndarray[double,ndim=1] b):
    cdef:
        int ndim1 = A.shape[1]
        int ndim2 = b.shape[0]

    if ndim1 != ndim2:
        raise ValueError("matrices are not aligned")

    if ndim1 == 2:
        return daxpy2(&A[0,0], &b[0])
    elif ndim1 == 3:
        return daxpy3(&A[0,0], &b[0])
    else:
        warn("daxpy of matrices > (3x3) falls back to numpy.dot. Use that instead")
        return np.dot(A,b)


cdef daxpy2(const double *A, const double *b):

    cdef:
        double A1_1 = A[0]
        double A1_2 = A[1]
        double A2_1 = A[2]
        double A2_2 = A[3]

        double b1 = b[0]
        double b2 = b[1]

        double Ab[2]
    
    Ab[:] = [A1_1*b1 + A1_2*b2, A2_1*b1 + A2_2*b2]
    return Ab


cdef daxpy3(const double *A, const double *b):

    cdef:
        double A1_1 = A[0]
        double A1_2 = A[1]
        double A1_3 = A[2]
        double A2_1 = A[3]
        double A2_2 = A[4]
        double A2_3 = A[5]
        double A3_1 = A[6]
        double A3_2 = A[7]
        double A3_3 = A[8]

        double b1 = b[0]
        double b2 = b[1]
        double b3 = b[2]

        double Ab[3]
    
    Ab[:] = [ A1_1*b1 + A1_2*b2 + A1_3*b3, A2_1*b1 + A2_2*b2 + A2_3*b3, A3_1*b1 + A3_2*b2 + A3_3*b3]
    return Ab


