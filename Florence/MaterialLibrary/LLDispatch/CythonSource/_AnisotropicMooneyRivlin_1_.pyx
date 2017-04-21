#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np

ctypedef double Real


cdef extern from "_AnisotropicMooneyRivlin_1_.h" nogil:
    cdef cppclass _AnisotropicMooneyRivlin_1_[Real]:
        _AnisotropicMooneyRivlin_1_() except +
        _AnisotropicMooneyRivlin_1_(Real mu1, Real mu2, Real mu3, Real lamb) except +
        void SetParameters(Real mu1, Real mu2, Real mu3, Real lamb) except +
        void KineticMeasures(Real *Snp, Real* Hnp, int ndim, int ngauss, const Real *Fnp, const Real *Nnp) except +



def KineticMeasures(material, np.ndarray[Real, ndim=3, mode='c'] F,
    np.ndarray[Real, ndim=2] N):
    
    cdef int ndim = F.shape[2]
    cdef int ngauss = F.shape[0]
    cdef np.ndarray[Real, ndim=3, mode='c'] D, stress, hessian

    stress = np.zeros((ngauss,ndim,ndim),dtype=np.float64)
    if ndim==3:
        hessian = np.zeros((ngauss,6,6),dtype=np.float64)
    elif ndim==2:
        hessian = np.zeros((ngauss,3,3),dtype=np.float64)

    cdef _AnisotropicMooneyRivlin_1_[Real] mat_obj = _AnisotropicMooneyRivlin_1_()
    mat_obj.SetParameters(material.mu1,material.mu2,material.mu3,material.lamb)
    mat_obj.KineticMeasures(&stress[0,0,0], &hessian[0,0,0], ndim, ngauss, &F[0,0,0], &N[0,0])

    return stress, hessian