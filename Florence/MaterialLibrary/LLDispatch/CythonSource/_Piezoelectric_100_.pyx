#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np

ctypedef double Real


cdef extern from "_Piezoelectric_100_.h" nogil:
    cdef cppclass _Piezoelectric_100_[Real]:
        _Piezoelectric_100_() except +
        _Piezoelectric_100_(Real mu1, Real mu2, Real mu3, Real lamb, Real eps_1, Real eps_2, Real eps_3) except +
        void SetParameters(Real mu1, Real mu2, Real mu3, Real lamb, Real eps_1, Real eps_2, Real eps_3) except +
        void KineticMeasures(Real *Dnp, Real *Snp, Real* Hnp, int ndim, int ngauss, const Real *Fnp, const Real *Enp, const Real *Nnp) except +



def KineticMeasures(material, np.ndarray[Real, ndim=3, mode='c'] F, np.ndarray[Real, ndim=2] E,
    np.ndarray[Real, ndim=2] N):
    
    cdef int ndim = F.shape[2]
    cdef int ngauss = F.shape[0]
    cdef np.ndarray[Real, ndim=3, mode='c'] D, stress, hessian

    D = np.zeros((ngauss,ndim,1),dtype=np.float64)
    stress = np.zeros((ngauss,ndim,ndim),dtype=np.float64)
    if ndim==3:
        hessian = np.zeros((ngauss,9,9),dtype=np.float64)
    elif ndim==2:
        hessian = np.zeros((ngauss,5,5),dtype=np.float64)

    cdef _Piezoelectric_100_[Real] mat_obj = _Piezoelectric_100_()
    mat_obj.SetParameters(material.mu1,material.mu2,material.mu3,material.lamb,material.eps_1,material.eps_2,material.eps_3)
    mat_obj.KineticMeasures(&D[0,0,0], &stress[0,0,0], &hessian[0,0,0], ndim, ngauss, &F[0,0,0], &E[0,0], &N[0,0])

    return D, stress, hessian









# cdef class Piezoelectric_100:

#     # CREATE A POINTER TO CPP BASE CLASS
#     cdef _Piezoelectric_100_[Real] *baseptr
#     cdef public dict material

#     def __cinit__(self, material):

#         # CREATE A NEW CPP OBJECT BY CALLING ITS CONSTRUCTOR
#         self.baseptr = new _Piezoelectric_100_()
#         # CHECK IF THE OBJECT WAS CREATED
#         if self.baseptr is NULL:
#             raise MemoryError("Could not create an instance of {} material" % type(self).__name__)

#         self.baseptr.SetParameters(material.mu1,material.mu2,material.mu3,material.lamb,material.eps_1,material.eps_2,material.eps_3)
#         self.material = material.__dict__

#     def KineticMeasures(self,np.ndarray[Real, ndim=3, mode='c'] F, np.ndarray[Real, ndim=2] E,
#         np.ndarray[Real, ndim=2] N):
        
#         cdef int ndim = F.shape[2]
#         cdef int ngauss = F.shape[0]
#         cdef np.ndarray[Real, ndim=3, mode='c'] D, stress, hessian

#         if ndim==3:
#             D = np.zeros((ngauss,3,1),dtype=np.float64)
#             stress = np.zeros((ngauss,3,3),dtype=np.float64)
#             hessian = np.zeros((ngauss,9,9),dtype=np.float64)
#         elif ndim==2:
#             D = np.zeros((ngauss,2,1),dtype=np.float64)
#             stress = np.zeros((ngauss,2,2),dtype=np.float64)
#             hessian = np.zeros((ngauss,5,5),dtype=np.float64)

#         self.baseptr.KineticMeasures(&D[0,0,0], &stress[0,0,0], &hessian[0,0,0], ndim, ngauss, &F[0,0,0], &E[0,0], &N[0,0])

#         return D, stress, hessian


#     def __dealloc__(self):
#         if self.baseptr != NULL:
#             del self.baseptr
