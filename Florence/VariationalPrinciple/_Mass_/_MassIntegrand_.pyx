#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t

ctypedef double Real
ctypedef int64_t Integer
ctypedef uint64_t UInteger


cdef extern from "_MassIntegrand_.h":

    void _MassIntegrand_Filler_(Real *mass,
        const Real* bases,
        const Real* detJ,
        int ngauss,
        int noderpelem,
        int ndim,
        int nvar,
        Real rho) nogil

    void _ConstantMassIntegrand_Filler_(Real *mass,
        const Real* constant_mass_integrand,
        const Real* detJ,
        int ngauss,
        int local_capacity) nogil

    void _ExplicitConstantMassIntegrand_(
        const UInteger* elements,
        const Real* points,
        const Real* Jm,
        const Real* AllGauss,
        const Real* constant_mass_integrand,
        Real *mass,
        Integer nelem,
        Integer ndim,
        Integer nvar,
        Integer ngauss,
        Integer nodeperelem,
        Integer local_capacity) nogil





def __MassIntegrand__(Real rho,
    np.ndarray[Real, ndim=2, mode='c'] bases,
    np.ndarray[Real, ndim=1] detJ,
    int ndim, int nvar):

    cdef int ngauss = detJ.shape[0]
    cdef int nodeperelem = bases.shape[0]
    cdef int local_size = nvar*nodeperelem

    cdef np.ndarray[Real, ndim=2, mode='c'] mass = np.zeros((local_size,
        local_size),dtype=np.float64)

    _MassIntegrand_Filler_(&mass[0,0],
        &bases[0,0],
        &detJ[0],
        ngauss,
        nodeperelem,
        ndim,
        nvar,
        rho)

    return mass



def __ConstantMassIntegrand__(np.ndarray[Real, ndim=3, mode='c'] constant_mass_integrand,
    np.ndarray[Real, ndim=1] detJ):

    cdef int ngauss = detJ.shape[0]
    cdef int local_size = constant_mass_integrand.shape[1]
    cdef int local_capacity = local_size*local_size

    cdef np.ndarray[Real, ndim=2, mode='c'] mass = np.zeros((local_size,
        local_size),dtype=np.float64)

    _ConstantMassIntegrand_Filler_(&mass[0,0],
        &constant_mass_integrand[0,0,0],
        &detJ[0],
        ngauss,
        local_capacity)

    return mass




def __ExplicitConstantMassIntegrand__(
    mesh,
    function_space,
    nvar,
    np.ndarray[Real, ndim=3, mode='c'] constant_mass_integrand):

    cdef Integer ndim                                   = mesh.points.shape[1]
    cdef Integer ngauss                                 = function_space.AllGauss.shape[0]
    cdef Integer nelem                                  = mesh.nelem
    cdef Integer nodeperelem                            = mesh.elements.shape[1]
    cdef Integer local_size                                 = constant_mass_integrand.shape[1]
    cdef Integer local_capacity                             = local_size*local_size
    cdef np.ndarray[UInteger,ndim=2, mode='c'] elements = mesh.elements
    cdef np.ndarray[Real,ndim=2, mode='c'] points       = mesh.points
    cdef np.ndarray[Real,ndim=3, mode='c'] Jm           = function_space.Jm
    cdef np.ndarray[Real,ndim=1, mode='c'] AllGauss     = function_space.AllGauss.flatten()

    cdef np.ndarray[Real, ndim=2, mode='c'] mass        = np.zeros((mesh.points.shape[0]*nvar,1),dtype=np.float64)

    _ExplicitConstantMassIntegrand_(    &elements[0,0],
                                        &points[0,0],
                                        &Jm[0,0,0],
                                        &AllGauss[0],
                                        &constant_mass_integrand[0,0,0],
                                        &mass[0,0],
                                        nelem,
                                        ndim,
                                        nvar,
                                        ngauss,
                                        nodeperelem,
                                        local_capacity)

    return mass