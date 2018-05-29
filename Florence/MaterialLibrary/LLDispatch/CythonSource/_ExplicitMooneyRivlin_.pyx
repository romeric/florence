#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np

ctypedef double Real


cdef extern from "_ExplicitMooneyRivlin_.h" nogil:
    cdef cppclass _ExplicitMooneyRivlin_[Real]:
        _ExplicitMooneyRivlin_() except +
        _ExplicitMooneyRivlin_(Real mu1, Real mu2, Real lamb) except +
        void SetParameters(Real mu1, Real mu2, Real lamb) except +
        void KineticMeasures(Real *Snp, int ndim, int ngauss, const Real *Fnp) except +



def KineticMeasures(material, np.ndarray[Real, ndim=3, mode='c'] F):

    cdef int ndim = F.shape[2]
    cdef int ngauss = F.shape[0]
    cdef np.ndarray[Real, ndim=3, mode='c'] stress

    stress = np.zeros((ngauss,ndim,ndim),dtype=np.float64)

    cdef _ExplicitMooneyRivlin_[Real] mat_obj = _ExplicitMooneyRivlin_()
    mat_obj.SetParameters(material.mu1,material.mu2,material.lamb)
    mat_obj.KineticMeasures(&stress[0,0,0], ndim, ngauss, &F[0,0,0])

    return stress