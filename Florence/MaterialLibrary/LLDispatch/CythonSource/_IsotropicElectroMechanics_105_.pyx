#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np

ctypedef double Real


cdef extern from "_IsotropicElectroMechanics_105_.h" nogil:
    cdef cppclass _IsotropicElectroMechanics_105_[Real]:
        _IsotropicElectroMechanics_105_() except +
        _IsotropicElectroMechanics_105_(Real mu1, Real mu2, Real lamb, Real eps_1, Real eps_2) except +
        void SetParameters(Real mu1, Real mu2, Real lamb, Real eps_1, Real eps_2) except +
        void KineticMeasures(Real *Dnp, Real *Snp, Real* Hnp, int ndim, int ngauss, const Real *Fnp, const Real *Enp) except +



def KineticMeasures(material, np.ndarray[Real, ndim=3, mode='c'] F, np.ndarray[Real, ndim=2] E):
    
    cdef int ndim = F.shape[2]
    cdef int ngauss = F.shape[0]
    cdef np.ndarray[Real, ndim=3, mode='c'] D, stress, hessian

    D = np.zeros((ngauss,ndim,1),dtype=np.float64)
    stress = np.zeros((ngauss,ndim,ndim),dtype=np.float64)

    if ndim==3:
        hessian = np.zeros((ngauss,9,9),dtype=np.float64)
    elif ndim==2:
        hessian = np.zeros((ngauss,5,5),dtype=np.float64)

    cdef _IsotropicElectroMechanics_105_[Real] mat_obj = _IsotropicElectroMechanics_105_()
    mat_obj.SetParameters(material.mu1,material.mu2,material.lamb,material.eps_1,material.eps_2)
    mat_obj.KineticMeasures(&D[0,0,0], &stress[0,0,0], &hessian[0,0,0], ndim, ngauss, &F[0,0,0], &E[0,0])

    return D, stress, hessian