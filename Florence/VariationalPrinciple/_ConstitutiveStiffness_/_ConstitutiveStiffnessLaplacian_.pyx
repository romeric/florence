#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np

ctypedef double Real

cdef extern from "_ConstitutiveStiffnessLaplacian_.h":

    inline void _ConstitutiveStiffnessLaplacian_Filler_(Real *stiffness, Real *traction,
        const Real* SpatialGradient,
        const Real* ElectricDisplacementx,
        const Real* H_Voigt,
        const Real* detJ,
        int ngauss,
        int noderpelem,
        int ndim,
        int nvar,
        int H_VoigtSize,
        int requires_geometry_update) nogil


def __ConstitutiveStiffnessIntegrandLaplacian__(np.ndarray[Real, ndim=3, mode='c'] SpatialGradient,
    np.ndarray[Real, ndim=3, mode='c'] ElectricDisplacementx,
    np.ndarray[Real, ndim=3, mode='c'] H_Voigt,
    np.ndarray[Real, ndim=1] detJ,
    int nvar,
    int requires_geometry_update):

    cdef int ngauss = SpatialGradient.shape[0]
    cdef int nodeperelem = SpatialGradient.shape[1]
    cdef int ndim = SpatialGradient.shape[2]
    cdef int local_size = nvar*nodeperelem
    cdef int H_VoigtSize = H_Voigt.shape[2]

    cdef np.ndarray[Real, ndim=2, mode='c'] traction = np.zeros((local_size,1),dtype=np.float64)
    cdef np.ndarray[Real, ndim=2, mode='c'] stiffness = np.zeros((local_size,
        local_size),dtype=np.float64)

    _ConstitutiveStiffnessLaplacian_Filler_(&stiffness[0,0], &traction[0,0],
        &SpatialGradient[0,0,0],
        &ElectricDisplacementx[0,0,0],
        &H_Voigt[0,0,0], &detJ[0],
        ngauss,
        nodeperelem,
        ndim,
        nvar,
        H_VoigtSize,
        requires_geometry_update)

    return stiffness, traction