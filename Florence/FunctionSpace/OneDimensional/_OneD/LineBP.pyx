import numpy as np
cimport numpy as np

ctypedef double Real
from cython cimport boundscheck

cdef extern from "LineBP.h":
    void _LagrangeBP(int n, Real xi, Real *N, Real *dN) except +
    void _LagrangeGaussLobattoBP(int n, Real xi, Real *eps, Real *N, Real *dN) except +



@boundscheck(False)
def LagrangeBP_(int C, Real xi):

    cdef int n = C+2
    cdef np.ndarray[Real,ndim=1, mode='c']  N = np.zeros(n,dtype=np.float64)
    cdef np.ndarray[Real,ndim=1, mode='c'] dN = np.zeros(n,dtype=np.float64)

    _LagrangeBP(n, xi, &N[0], &dN[0])

    return N, dN


@boundscheck(False)
def LagrangeGaussLobattoBP_(int C, Real xi, np.ndarray[Real,ndim=1, mode='c'] eps):

    cdef int n = C+2
    cdef np.ndarray[Real,ndim=1, mode='c']  N  = np.zeros(n,dtype=np.float64)
    cdef np.ndarray[Real,ndim=1, mode='c'] dN  = np.zeros(n,dtype=np.float64)

    _LagrangeGaussLobattoBP(n, xi, &eps[0], &N[0], &dN[0])

    return N, dN