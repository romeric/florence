#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t

ctypedef int64_t Integer
ctypedef uint64_t UInteger
ctypedef double Real


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

    void _GenericConstantMassIntegrand_(
        const UInteger* elements,
        const Real* points,
        const Real* Jm,
        const Real* AllGauss,
        const Real* constant_mass_integrand,
        Integer nelem,
        Integer ndim,
        Integer nvar,
        Integer ngauss,
        Integer nodeperelem,
        Integer local_capacity,
        Integer mass_type,
        const Integer* local_rows_mass,
        const Integer* local_cols_mass,
        int *I_mass,
        int *J_mass,
        Real *V_mass,
        Real *mass,
        int recompute_sparsity_pattern,
        int squeeze_sparsity_pattern,
        const int *data_local_indices,
        const int *data_global_indices,
        const UInteger *sorted_elements,
        const Integer *sorter
        ) nogil

    void _SymmetricConstantMassIntegrand_(
        const UInteger* elements,
        const Real* points,
        const Real* Jm,
        const Real* AllGauss,
        const Real* constant_mass_integrand,
        Integer nelem,
        Integer ndim,
        Integer nvar,
        Integer ngauss,
        Integer nodeperelem,
        Integer local_capacity,
        Integer mass_type,
        const Integer* local_rows_mass,
        const Integer* local_cols_mass,
        int *I_mass,
        int *J_mass,
        Real *V_mass,
        Real *mass,
        int recompute_sparsity_pattern,
        int squeeze_sparsity_pattern,
        const int *data_local_indices,
        const int *data_global_indices,
        const UInteger *sorted_elements,
        const Integer *sorter
        ) nogil

    void _SymmetricNonZeroConstantMassIntegrand_(
        const UInteger* elements,
        const Real* points,
        const Real* Jm,
        const Real* AllGauss,
        const Real* constant_mass_integrand,
        Integer nelem,
        Integer ndim,
        Integer nvar,
        Integer ngauss,
        Integer nodeperelem,
        Integer local_capacity,
        Integer mass_type,
        const Integer* local_rows_mass,
        const Integer* local_cols_mass,
        int *I_mass,
        int *J_mass,
        Real *V_mass,
        Real *mass,
        int recompute_sparsity_pattern,
        int squeeze_sparsity_pattern,
        const int *data_local_indices,
        const int *data_global_indices,
        const UInteger *sorted_elements,
        const Integer *sorter
        ) nogil


    void _SymmetricNonZeroOneElementConstantMassIntegrand_(
        const UInteger* elements,
        const Real* points,
        const Real* Jm,
        const Real* AllGauss,
        const Real* constant_mass_integrand,
        Integer nelem,
        Integer ndim,
        Integer nvar,
        Integer ngauss,
        Integer nodeperelem,
        Integer local_capacity,
        Integer mass_type,
        const Integer* local_rows_mass,
        const Integer* local_cols_mass,
        int *I_mass,
        int *J_mass,
        Real *V_mass,
        Real *mass,
        int recompute_sparsity_pattern,
        int squeeze_sparsity_pattern,
        const int *data_local_indices,
        const int *data_global_indices,
        const UInteger *sorted_elements,
        const Integer *sorter
        ) nogil





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




def __TotalConstantMassIntegrand__(
    mesh,
    function_space,
    formulation,
    mass_type="lumped",
    recompute_sparsity_pattern=True,
    squeeze_sparsity_pattern=False,
    np.ndarray[int,ndim=1,mode='c'] indices=np.zeros(1,np.int32),
    np.ndarray[int,ndim=1,mode='c'] indptr=np.zeros(1,np.int32),
    np.ndarray[int,ndim=1,mode='c'] data_global_indices=np.zeros(1,np.int32),
    np.ndarray[int,ndim=1,mode='c'] data_local_indices=np.zeros(1,np.int32)
    ):

    cdef np.ndarray[Real, ndim=3, mode='c'] constant_mass_integrand = formulation.constant_mass_integrand
    cdef Integer ndim                                   = mesh.InferSpatialDimension()
    cdef Integer nvar                                   = formulation.nvar
    cdef Integer ngauss                                 = function_space.AllGauss.shape[0]
    cdef Integer nelem                                  = mesh.nelem
    cdef Integer nodeperelem                            = mesh.elements.shape[1]
    cdef Integer local_size                             = constant_mass_integrand.shape[1]
    cdef Integer local_capacity                         = local_size*local_size
    cdef np.ndarray[UInteger,ndim=2, mode='c'] elements = mesh.elements
    cdef np.ndarray[Real,ndim=2, mode='c'] points       = mesh.points
    cdef np.ndarray[Real,ndim=3, mode='c'] Jm           = function_space.Jm
    cdef np.ndarray[Real,ndim=1, mode='c'] AllGauss     = function_space.AllGauss.flatten()
    cdef Integer c_mass_type                            = 1 if mass_type == "consistent" else 0

    cdef np.ndarray[Integer,ndim=1,mode='c'] local_rows_mass        = formulation.local_rows_mass
    cdef np.ndarray[Integer,ndim=1,mode='c'] local_cols_mass        = formulation.local_columns_mass

    cdef np.ndarray[Integer,ndim=2, mode='c'] sorter                = np.zeros((1,1),np.int64)
    cdef np.ndarray[UInteger,ndim=2, mode='c'] sorted_elements      = np.zeros((1,1),np.uint64)

    # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
    cdef np.ndarray[int,ndim=1,mode='c'] I_mass         = np.zeros(1,np.int32)
    cdef np.ndarray[int,ndim=1,mode='c'] J_mass         = np.zeros(1,np.int32)
    cdef np.ndarray[Real,ndim=1,mode='c'] V_mass        = np.zeros(1,np.float64)

    if c_mass_type == 1:
        if recompute_sparsity_pattern:
            I_mass          = np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
            J_mass          = np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
            V_mass          = np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)
        else:
            I_mass = indptr
            J_mass = indices
            V_mass = np.zeros(indices.shape[0],dtype=np.float64)
            if squeeze_sparsity_pattern:
                sorter = mesh.element_sorter
                sorted_elements = mesh.sorted_elements

    # LUMPED MASS TYPE
    cdef np.ndarray[Real, ndim=2, mode='c'] mass        = np.zeros((1,1),dtype=np.float64)
    if c_mass_type == 0:
        mass                                            = np.zeros((nvar*mesh.points.shape[0],1),dtype=np.float64)


    structured_grid = False
    if ndim == 3:
        sizes = mesh.Volumes()
    elif ndim == 2:
        sizes = mesh.Areas()
    elif ndim == 1:
        sizes = mesh.Lengths()
    if np.allclose(sizes,sizes[0]):
        structured_grid = True

    if structured_grid:
        _SymmetricNonZeroOneElementConstantMassIntegrand_(
                                                    &elements[0,0],
                                                    &points[0,0],
                                                    &Jm[0,0,0],
                                                    &AllGauss[0],
                                                    &constant_mass_integrand[0,0,0],
                                                    nelem,
                                                    ndim,
                                                    nvar,
                                                    ngauss,
                                                    nodeperelem,
                                                    local_capacity,
                                                    c_mass_type,
                                                    &local_rows_mass[0],
                                                    &local_cols_mass[0],
                                                    &I_mass[0],
                                                    &J_mass[0],
                                                    &V_mass[0],
                                                    &mass[0,0],
                                                    recompute_sparsity_pattern,
                                                    squeeze_sparsity_pattern,
                                                    &data_local_indices[0],
                                                    &data_global_indices[0],
                                                    &sorted_elements[0,0],
                                                    &sorter[0,0]
                                                    )

        if recompute_sparsity_pattern:
            return mass, I_mass, J_mass, V_mass
        else:
            return mass, V_mass


    if mesh.element_type == "quad" or mesh.element_type == "hex":
        _SymmetricNonZeroConstantMassIntegrand_(    &elements[0,0],
                                                    &points[0,0],
                                                    &Jm[0,0,0],
                                                    &AllGauss[0],
                                                    &constant_mass_integrand[0,0,0],
                                                    nelem,
                                                    ndim,
                                                    nvar,
                                                    ngauss,
                                                    nodeperelem,
                                                    local_capacity,
                                                    c_mass_type,
                                                    &local_rows_mass[0],
                                                    &local_cols_mass[0],
                                                    &I_mass[0],
                                                    &J_mass[0],
                                                    &V_mass[0],
                                                    &mass[0,0],
                                                    recompute_sparsity_pattern,
                                                    squeeze_sparsity_pattern,
                                                    &data_local_indices[0],
                                                    &data_global_indices[0],
                                                    &sorted_elements[0,0],
                                                    &sorter[0,0]
                                                    )
    else:
        _SymmetricConstantMassIntegrand_(           &elements[0,0],
                                                    &points[0,0],
                                                    &Jm[0,0,0],
                                                    &AllGauss[0],
                                                    &constant_mass_integrand[0,0,0],
                                                    nelem,
                                                    ndim,
                                                    nvar,
                                                    ngauss,
                                                    nodeperelem,
                                                    local_capacity,
                                                    c_mass_type,
                                                    &local_rows_mass[0],
                                                    &local_cols_mass[0],
                                                    &I_mass[0],
                                                    &J_mass[0],
                                                    &V_mass[0],
                                                    &mass[0,0],
                                                    recompute_sparsity_pattern,
                                                    squeeze_sparsity_pattern,
                                                    &data_local_indices[0],
                                                    &data_global_indices[0],
                                                    &sorted_elements[0,0],
                                                    &sorter[0,0]
                                                    )

    if recompute_sparsity_pattern:
        return mass, I_mass, J_mass, V_mass
    else:
        return mass, V_mass