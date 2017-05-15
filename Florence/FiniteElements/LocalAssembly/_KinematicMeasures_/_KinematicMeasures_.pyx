#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np


ctypedef double Real


cdef extern from "_KinematicMeasures_.h":
    void KinematicMeasures(Real *SpatialGradient_, Real *F_, Real *detJ, const Real *Jm_,
        const Real *AllGauss_, const Real *LagrangeElemCoords_, const Real *EulerElemCoords_,
        int ngauss, int ndim, int nodeperelem, int update) 


def _KinematicMeasures_(np.ndarray[Real, ndim=3, mode='c'] Jm,
    np.ndarray[Real, ndim=1] AllGauss,
    np.ndarray[Real, ndim=2, mode='c'] LagrangeElemCoords, 
    np.ndarray[Real, ndim=2, mode='c'] EulerElemCoords, requires_geometry_update):
    
    cdef int ndim = Jm.shape[0]
    cdef int nodeperelem = Jm.shape[1]
    cdef int ngauss = Jm.shape[2]
    cdef int update = int(requires_geometry_update)

    cdef np.ndarray[Real, ndim=3, mode='c'] F = np.zeros((ngauss,ndim,ndim),dtype=np.float64)
    cdef np.ndarray[Real, ndim=3, mode='c'] SpatialGradient = np.zeros((ngauss,nodeperelem,ndim),dtype=np.float64)
    cdef np.ndarray[Real, ndim=1] detJ = np.zeros(ngauss,dtype=np.float64)

    KinematicMeasures(&SpatialGradient[0,0,0], &F[0,0,0], &detJ[0], &Jm[0,0,0], &AllGauss[0],
        &LagrangeElemCoords[0,0], &EulerElemCoords[0,0], ngauss, ndim, nodeperelem, update)

    return SpatialGradient, F, detJ