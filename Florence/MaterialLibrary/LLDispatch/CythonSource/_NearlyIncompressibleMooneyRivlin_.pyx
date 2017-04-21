#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np

ctypedef double Real


cdef extern from "_NearlyIncompressibleMooneyRivlin_.h" nogil:
    cdef cppclass _NearlyIncompressibleMooneyRivlin_[Real]:
        _NearlyIncompressibleMooneyRivlin_() except +
        _NearlyIncompressibleMooneyRivlin_(Real alpha, Real beta, Real kappa) except +
        void SetParameters(Real alpha, Real beta, Real kappa) except +
        void KineticMeasures(Real *Snp, Real* Hnp, int ndim, int ngauss, const Real *Fnp) except +



def KineticMeasures(material, np.ndarray[Real, ndim=3, mode='c'] F):
    
    cdef int ndim = F.shape[2]
    cdef int ngauss = F.shape[0]
    cdef np.ndarray[Real, ndim=3, mode='c'] stress, hessian

    stress = np.zeros((ngauss,ndim,ndim),dtype=np.float64)
    if ndim==3:
        hessian = np.zeros((ngauss,6,6),dtype=np.float64)
    elif ndim==2:
        hessian = np.zeros((ngauss,3,3),dtype=np.float64)

    cdef _NearlyIncompressibleMooneyRivlin_[Real] mat_obj = _NearlyIncompressibleMooneyRivlin_()
    mat_obj.SetParameters(material.alpha,material.beta,material.kappa)
    mat_obj.KineticMeasures(&stress[0,0,0], &hessian[0,0,0], ndim, ngauss, &F[0,0,0])

    return stress, hessian