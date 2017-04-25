#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np

ctypedef double Real


cdef extern from "_GeometricStiffness_.h":
    inline void _GeometricStiffnessFiller_(Real *geometric_stiffness, 
        const Real *SpatialGradient, const Real *CauchyStressTensor, const Real *detJ, 
        const int ndim, const int nvar, const int nodeperelem, const int nguass) nogil

def GeometricStiffnessIntegrand(np.ndarray[Real, ndim=3, mode='c'] SpatialGradient,
    np.ndarray[Real, ndim=3, mode='c'] CauchyStressTensor,
    np.ndarray[Real, ndim=1] detJ, int nvar):

    cdef int ngauss = SpatialGradient.shape[0]
    cdef int nodeperelem = SpatialGradient.shape[1]
    cdef int ndim = SpatialGradient.shape[2]
    cdef int local_size = nvar*nodeperelem

    cdef np.ndarray[Real, ndim=2, mode='c'] geometric_stiffness = np.zeros((local_size,
        local_size),dtype=np.float64)

    _GeometricStiffnessFiller_(&geometric_stiffness[0,0], &SpatialGradient[0,0,0], 
        &CauchyStressTensor[0,0,0], &detJ[0], ndim, nvar, nodeperelem, ngauss)


    return geometric_stiffness