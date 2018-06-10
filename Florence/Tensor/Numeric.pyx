from cython import boundscheck, wraparound, cdivision
import numpy as np
cimport numpy as np

from libc.stdint cimport int64_t
from libc.math cimport fabs
from libcpp.vector cimport vector
from cpython cimport bool

__all__ = ['trace','doublecontract','makezero','makezero3d','issymetric','tovoigt', 'tovoigt3', 'cross2d',
'fillin','findfirst','findequal','findequal_approx','findless','findgreater']

# COMPILE USING
# cython --cplus Numeric.pyx
# g++ -std=c++11 -fPIC -shared -pthread -O3 -fwrapv -fno-strict-aliasing -Wall -finline-functions \
# -ffast-math -funroll-loops Numeric.cpp _Numeric.cpp -o Numeric.so -I/usr/include/python2.7 -lpython2.7 -lm


#---------------------------------------------------
# FUSED TYPES ESSENTIALLY INTRODUCE THE SAME OVERHEAD AS THAT OF NUMPY UFUNCS
# ctypedef fused Float:
#     np.float32_t
#     np.float64_t

# ctypedef fused Int:
#     np.int32_t
#     np.int64_t

# ctypedef fused Generic:
#     np.int32_t
#     np.uint32_t
#     np.int64_t
#     np.uint64_t
#     np.float32_t
#     np.float64_t
#---------------------------------------------------


Real = np.float64
ctypedef np.float64_t Real_t

Integer = np.int64
# ctypedef np.int64_t Integer_t
ctypedef int64_t Integer_t

cdef extern from "<algorithm>" namespace "std":
    void fill(Real_t *first, Real_t *last, Real_t num)
    Integer_t* find(Integer_t *first, Integer_t *last, Integer_t num)

cdef extern from "_Numeric.cpp":
    vector[Integer_t] FindEqual(const Integer_t *arr, Integer_t size, Integer_t num)
    vector[Integer_t] FindEqualApprox(const Real_t *arr, Integer_t size,Real_t num, Real_t tolerance)
    vector[Integer_t] FindLessThan(const Real_t *arr, Integer_t size, Real_t num)
    vector[Integer_t] FindGreaterThan(const Real_t *arr, Integer_t size, Real_t num)
    # vector[Generic] FindGreaterThan(const Generic *arr, Integer_t size, Generic num) # NOT ALLOWED



# cpdef ALSO INTRODUCES A BIT OF OVERHEAD - AVOID IT UNLESS NECESSARY

@boundscheck(False)
@wraparound(False)
def trace(np.ndarray[Real_t, ndim=2] A):
    cdef:
        int i
        Real_t trA = 0.0
        int n1 = A.shape[0]
        int n2 = A.shape[1]

    if n1 != n2:
        raise AssertionError("Trace of non-Hermitian (non-square) matrix requested")

    for i in range(0,n1):
       trA += A[i,i]
    return trA



@boundscheck(False)
@wraparound(False)
def doublecontract(np.ndarray[Real_t, ndim=2] A, np.ndarray[Real_t, ndim=2] B):
    """Double contraction of 2D Arrays"""
    # THIS FUNCTION IS FASTER THAN np.einsum FOR SMALLER MATRICES AND SLOWER THAN FOR LARGER MATRICES.
    cdef:
        int a1 = A.shape[0]
        int b1 = B.shape[0]
        int a2 = A.shape[1]
        int b2 = B.shape[1]

    if a1!=b1 or a2!=b2:
        raise ValueError('Incompatible dimensions for double contraction')

    return _doublecontract(&A[0,0],&B[0,0], a1*a2)

# CYTHON DOES NOT ACCEPT THE const KEYWORD FOR A & B HERE FOR SOME REASON
cdef inline Real_t _doublecontract(Real_t *A, Real_t *B, int size):

    cdef int i
    cdef Real_t sum_reducter = 0.
    for i in range(size):
        sum_reducter += A[i]*B[i]
    return sum_reducter


@boundscheck(False)
@wraparound(False)
cpdef void makezero(np.ndarray[Real_t, ndim=2] A, Real_t tol=1.0e-14):
    """Substitute the elements of an array which are close to zero with zero.
        This is an in-place operation and does not return anything"""

    cdef:
        int i,j
        int a1 = A.shape[0]
        int a2 = A.shape[1]

    for i in range(a1):
        for j in range(a2):
            if fabs(A[i,j]) < tol:
                A[i,j] = 0.

@boundscheck(False)
@wraparound(False)
cpdef void makezero3d(np.ndarray[Real_t, ndim=3] A, Real_t tol=1.0e-14):
    """Substitute the elements of an array which are close to zero with zero.
        This is an in-place operation and does not return anything"""

    cdef:
        int i,j,k
        int a1 = A.shape[0]
        int a2 = A.shape[1]
        int a3 = A.shape[2]

    for i in range(a1):
        for j in range(a2):
            for k in range(a3):
                if fabs(A[i,j,k]) < tol:
                    A[i,j,k] = 0.


@boundscheck(False)
@wraparound(False)
cpdef bool issymetric(np.ndarray[Real_t, ndim=2] A, Real_t tol=1.0e-12):
    """Checks if a Hermitian floating point matrix is symmetric within a tolerance"""
    cdef:
        int i,j
        int a1 = A.shape[0]
        int a2 = A.shape[1]
        bool issym = True

    if a1 != a2:
        raise ValueError("Symmetricity of a non-Hermitian (non-square) matrix requested")

    for i in range(a1):
        for j in range(a2):
            if fabs(A[i,j] - A[j,i]) > tol:
                issym = False
                break
        if issym == True:
            break

    return issym


@boundscheck(False)
@wraparound(False)
def tovoigt(np.ndarray[Real_t, ndim=4, mode='c'] C):
    """Convert a 4D array to its Voigt represenation"""
    cdef np.ndarray[Real_t, ndim=2,mode='c'] VoigtA
    cdef int n1dim = C.shape[0]
    # DISPATCH CALL TO APPROPRIATE FUNCTION
    if n1dim == 3:
        VoigtA = np.zeros((6,6),dtype=np.float64)
        _Voigt3(&C[0,0,0,0],&VoigtA[0,0])
    elif n1dim == 2:
        VoigtA = np.zeros((3,3),dtype=np.float64)
        _Voigt2(&C[0,0,0,0],&VoigtA[0,0])

    return VoigtA


cdef _Voigt3(const Real_t *C, Real_t *VoigtA):
    VoigtA[0] = C[0]
    VoigtA[1] = C[4]
    VoigtA[2] = C[8]
    VoigtA[3] = 0.5*(C[1]+C[3])
    VoigtA[4] = 0.5*(C[2]+C[6])
    VoigtA[5] = 0.5*(C[5]+C[7])
    VoigtA[6] = VoigtA[1]
    VoigtA[7] = C[40]
    VoigtA[8] = C[44]
    VoigtA[9] = 0.5*(C[37]+C[39])
    VoigtA[10] = 0.5*(C[38]+C[42])
    VoigtA[11] = 0.5*(C[41]+C[43])
    VoigtA[12] = VoigtA[2]
    VoigtA[13] = VoigtA[8]
    VoigtA[14] = C[80]
    VoigtA[15] = 0.5*(C[73]+C[75])
    VoigtA[16] = 0.5*(C[74]+C[78])
    VoigtA[17] = 0.5*(C[77]+C[79])
    VoigtA[18] = VoigtA[3]
    VoigtA[19] = VoigtA[9]
    VoigtA[20] = VoigtA[15]
    VoigtA[21] = 0.5*(C[10]+C[12])
    VoigtA[22] = 0.5*(C[11]+C[15])
    VoigtA[23] = 0.5*(C[14]+C[16])
    VoigtA[24] = VoigtA[4]
    VoigtA[25] = VoigtA[10]
    VoigtA[26] = VoigtA[16]
    VoigtA[27] = VoigtA[22]
    VoigtA[28] = 0.5*(C[20]+C[24])
    VoigtA[29] = 0.5*(C[23]+C[25])
    VoigtA[30] = VoigtA[5]
    VoigtA[31] = VoigtA[11]
    VoigtA[32] = VoigtA[17]
    VoigtA[33] = VoigtA[23]
    VoigtA[34] = VoigtA[29]
    VoigtA[35] = 0.5*(C[50]+C[52])


cdef _Voigt2(const Real_t *C, Real_t *VoigtA):
    VoigtA[0] = C[0]
    VoigtA[1] = C[3]
    VoigtA[2] = 0.5*(C[1]+C[2])
    VoigtA[3] = VoigtA[1]
    VoigtA[4] = C[15]
    VoigtA[5] = 0.5*(C[13]+C[14])
    VoigtA[6] = VoigtA[2]
    VoigtA[7] = VoigtA[5]
    VoigtA[8] = 0.5*(C[5]+C[6])




@boundscheck(False)
@wraparound(False)
def tovoigt3(np.ndarray[Real_t, ndim=3, mode='c'] e):
    """Convert a 3D array to its Voigt represenation"""
    cdef np.ndarray[Real_t, ndim=2,mode='c'] VoigtA
    cdef int n1dim = e.shape[0]
    # DISPATCH CALL TO APPROPRIATE FUNCTION
    if n1dim == 3:
        VoigtA = np.zeros((6,3),dtype=np.float64)
        _Voigt33(&e[0,0,0],&VoigtA[0,0])
    elif n1dim == 2:
        VoigtA = np.zeros((3,2),dtype=np.float64)
        _Voigt23(&e[0,0,0],&VoigtA[0,0])

    return VoigtA


cdef _Voigt33(const Real_t *e, Real_t *VoigtA):
    VoigtA[0]  = e[0]
    VoigtA[1]  = e[1]
    VoigtA[2]  = e[2]
    VoigtA[3]  = e[12]
    VoigtA[4]  = e[13]
    VoigtA[5]  = e[14]
    VoigtA[6]  = e[24]
    VoigtA[7]  = e[25]
    VoigtA[8]  = e[26]
    VoigtA[9]  = 0.5*(e[3]+e[9])
    VoigtA[10] = 0.5*(e[4]+e[10])
    VoigtA[11] = 0.5*(e[5]+e[11])
    VoigtA[12] = 0.5*(e[6]+e[18])
    VoigtA[13] = 0.5*(e[7]+e[19])
    VoigtA[14] = 0.5*(e[8]+e[20])
    VoigtA[15] = 0.5*(e[15]+e[21])
    VoigtA[16] = 0.5*(e[16]+e[22])
    VoigtA[17] = 0.5*(e[17]+e[23])


cdef _Voigt23(const Real_t *e, Real_t *VoigtA):
    VoigtA[0] = e[0]
    VoigtA[1] = e[1]
    VoigtA[2] = e[6]
    VoigtA[3] = e[7]
    VoigtA[4] = 0.5*(e[2]+e[4])
    VoigtA[5] = 0.5*(e[3]+e[5])


@boundscheck(False)
def cross2d(np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B, str dim="3d", toarray=False):
    """Cross product of second order tensors A_ij x B_ij defined in the sense of R. de Boer
        [Vektor- und Tensorrechnung fur Ingenieure] and J. Bonet [A computational framework
        for polyconvex large strain elasticity].

        input:
            A:                      [ndarray] ndarry of 3x3
            B:                      [ndarray] ndarry of 3x3
            dim:                    [str] either "2d" or "3d". To get a cross product in a 2D
                                    space you still need to supply the matrices in 3D space
                                    and specify the dim argument as 2d. The dim="2d" essentially
                                    assumes that A and B have their zeros in the third direction
                                    as in:
                                        A [and B] = np.array([
                                                            [A00, A01, 0],
                                                            [A10, A11, 0],
                                                            [  0,   0, 0],
                                                            ])

            toarray:                [bool] The default outupt of this unction is a python list
                                    and not a numpy array as one would expect. This is for
                                    performance reason, as numpy for 3x3 array would introduce
                                    quite a bit of overhead. Specify toarray to true if a numpy
                                    array is desired as an output
    """


    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Dimension of matrix should be 3x3")

    cdef:
        Real_t A00=A[0,0]
        Real_t A11=A[1,1]
        Real_t A22=A[2,2]
        Real_t A01=A[0,1]
        Real_t A02=A[0,2]
        Real_t A12=A[1,2]
        Real_t A10=A[1,0]
        Real_t A20=A[2,0]
        Real_t A21=A[2,1]

        Real_t B00=B[0,0]
        Real_t B11=B[1,1]
        Real_t B22=B[2,2]
        Real_t B01=B[0,1]
        Real_t B02=B[0,2]
        Real_t B12=B[1,2]
        Real_t B10=B[1,0]
        Real_t B20=B[2,0]
        Real_t B21=B[2,1]

    cdef double AB[3][3]

    if dim == "3d":
        AB[:] = [
            [ A11*B22 - A12*B21 - A21*B12 + A22*B11, A12*B20 - A10*B22 + A20*B12 - A22*B10, A10*B21 - A11*B20 - A20*B11 + A21*B10],
            [ A02*B21 - A01*B22 + A21*B02 - A22*B01, A00*B22 - A02*B20 - A20*B02 + A22*B00, A01*B20 - A00*B21 + A20*B01 - A21*B00],
            [ A01*B12 - A02*B11 - A11*B02 + A12*B01, A02*B10 - A00*B12 + A10*B02 - A12*B00, A00*B11 - A01*B10 - A10*B01 + A11*B00]
            ]

    elif dim == "2d":
        # AB[:] = [
        #     [ 0., 0., 0.],
        #     [ 0., 0., 0.],
        #     [ 0., 0., A00*B11 - A01*B10 - A10*B01 + A11*B00]
        #     ]
        AB[2][2] = A00*B11 - A01*B10 - A10*B01 + A11*B00

    if toarray:
        return np.array(AB,copy=False)
    return AB



@boundscheck(False)
@cdivision(True)
def findfirst(np.ndarray A, Integer_t num):
    """Find the first occurence of a value in an ndarray"""

    if A.dtype != Integer:
        raise TypeError("Expected an array of integers")

    cdef:
        Integer_t size = A.size
        Integer_t ndim = A.ndim
        Integer_t[::1] arr = A.ravel()
        Integer_t col = A.shape[1]

    cdef Integer_t *idx = find(&arr[0],<Integer_t*>(&arr[-1]+1),num)
    cdef Integer_t ret = idx - &arr[0]

    if ret == size:
        return ()

    if ndim == 1:
        return ret
    elif ndim == 2:
        return (ret // col, ret % col)
    else:
        raise ValueError("Dimension mismatch. Array should be either 1D or 2D")


@boundscheck(False)
def fillin(np.ndarray A, Real_t num):
    """Fill ndarray with a value. This is an in-place operation"""

    if A.dtype != Real:
        raise TypeError("Expected an array of doubles")

    cdef Real_t[::1] arr = A.ravel()
    fill(&arr[0],<Real_t*>(&arr[-1]+1),num)
    arr[-1] = num


@boundscheck(False)
@wraparound(False)
def findequal(np.ndarray A, Integer_t num):
    """An equivalent function to numpy.where(A==num). The difference here is that
        numpy.where results in a boolean array of size equal to the original
        array A, even if number of counts is none or a single element. This function
        allocates and does push_back on per count"""

    if A.dtype != Integer:
        raise TypeError("Type mismatch. Array and value must have the same type")

    cdef Integer_t[::1] arr = A.ravel()
    cdef vector[Integer_t] idx = FindEqual(&arr[0],A.size,num)

    if A.ndim == 1:
        return np.array(idx,copy=False)
    elif A.ndim == 2:
        npidx = np.array(idx,copy=False)
        return npidx // A.shape[1], npidx % A.shape[1]
    else:
        raise ValueError("Dimension mismatch. Array should be either 1D or 2D")


@boundscheck(False)
@wraparound(False)
def findequal_approx(np.ndarray A, Real_t num, Real_t tol=1.0e-14):
    """An equivalent function to numpy.where(A==num). The difference here is that
        numpy.where results in a boolean array of size equal to the original
        array A, even if number of counts is none or a single element. This function
        allocates and does push_back on per count"""

    if A.dtype != Real:
        raise TypeError("Type mismatch. Array and value must have the same type")

    cdef Real_t[::1] arr = A.ravel()
    cdef vector[Integer_t] idx = FindEqualApprox(&arr[0], A.size, num, tol)

    if A.ndim == 1:
        return np.array(idx,copy=False)
    elif A.ndim == 2:
        npidx = np.array(idx,copy=False)
        return npidx // A.shape[1], npidx % A.shape[1]
    else:
        raise ValueError("Dimension mismatch. Array should be either 1D or 2D")


@boundscheck(False)
@wraparound(False)
def findless(np.ndarray A, Real_t num):
    """An equivalent function to numpy.where(A<num). The difference here is that
        numpy.where results in a boolean array of size equal to the original
        array A, even if number of counts is none or a single element. This function
        allocates and performs push_back per hit count"""

    cdef Real_t[::1] arr = A.ravel()
    cdef vector[Integer_t] idx = FindLessThan(&arr[0],A.size,num)

    if A.ndim == 1:
        return np.array(idx,copy=False)
    elif A.ndim == 2:
        npidx = np.array(idx,copy=False)
        return npidx // A.shape[1], npidx % A.shape[1]
    else:
        raise ValueError("Dimension mismatch. Array should be either 1D or 2D")


@boundscheck(False)
@wraparound(False)
def findgreater(np.ndarray A, Real_t num):
    """An equivalent function to numpy.where(A>num). The difference here is that
        numpy.where results in a boolean array of size equal to the original
        array A, even if number of counts is none or a single element. This function
        allocates and does push_back on per count"""

    cdef Real_t[::1] arr = A.ravel()
    cdef vector[Integer_t] idx = FindGreaterThan(&arr[0],A.size,num)

    if A.ndim == 1:
        return np.array(idx,copy=False)
    elif A.ndim == 2:
        npidx = np.array(idx,copy=False)
        return npidx // A.shape[1], npidx % A.shape[1]
    else:
        raise ValueError("Dimension mismatch. Array should be either 1D or 2D")


