import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t

ctypedef int64_t Integer
ctypedef uint64_t UInteger
ctypedef double Real


cdef extern from "_LowLevelAssemblyDF_.h" nogil:
    void _GlobalAssemblyDF_(const Real *points,
                            const UInteger* elements,
                            const Real* Eulerx,
                            const Real* Eulerp,
                            const Real* bases,
                            const Real* Jm,
                            const Real* AllGauss,
                            Integer ndim,
                            Integer nvar,
                            Integer ngauss,
                            Integer nelem,
                            Integer nodeperelem,
                            Integer nnode,
                            Integer H_VoigtSize,
                            Integer requires_geometry_update,
                            Integer* local_rows_stiff,
                            Integer* local_cols_stiff,
                            int *I_stiff,
                            int *J_stiff,
                            Real *V_stiff,
                            Real *T,
                            int recompute_sparsity_pattern,
                            int squeeze_sparsity_pattern,
                            const int *data_local_indices,
                            const int *data_global_indices,
                            const UInteger *sorted_elements,
                            const Integer *sorter,
                            Real rho,
                            Real mu,
                            Real mu1,
                            Real mu2,
                            Real mu3,
                            Real mue,
                            Real lamb,
                            Real eps_1,
                            Real eps_2,
                            Real eps_3,
                            Real eps_e,
                            const Real *anisotropic_orientations
                            )


def _LowLevelAssemblyDF_(fem_solver, function_space, formulation, mesh, material, Real[:,::1] Eulerx, Real[::1] Eulerp):

    # GET VARIABLES FOR DISPATCHING TO C
    cdef Integer ndim                       = formulation.ndim
    cdef Integer nvar                       = formulation.nvar
    cdef Integer ngauss                     = function_space.AllGauss.shape[0]
    cdef Integer nelem                      = mesh.nelem
    cdef Integer nodeperelem                = mesh.elements.shape[1]
    cdef Integer nnode                      = mesh.points.shape[0]
    cdef Integer H_VoigtSize                = material.H_VoigtSize

    cdef np.ndarray[UInteger,ndim=2, mode='c'] elements = mesh.elements
    cdef np.ndarray[Real,ndim=2, mode='c'] points       = mesh.points
    cdef np.ndarray[Real,ndim=2, mode='c'] bases        = function_space.Bases
    cdef np.ndarray[Real,ndim=3, mode='c'] Jm           = function_space.Jm
    cdef np.ndarray[Real,ndim=1, mode='c'] AllGauss     = function_space.AllGauss.flatten()

    cdef Integer requires_geometry_update               = fem_solver.requires_geometry_update

    cdef np.ndarray[Integer,ndim=1,mode='c'] local_rows_stiffness   = formulation.local_rows
    cdef np.ndarray[Integer,ndim=1,mode='c'] local_cols_stiffness   = formulation.local_columns

    cdef np.ndarray[Integer,ndim=1,mode='c'] local_rows_mass        = formulation.local_rows_mass
    cdef np.ndarray[Integer,ndim=1,mode='c'] local_cols_mass        = formulation.local_columns_mass

    cdef np.ndarray[int,ndim=1,mode='c'] I_stiff        = np.zeros(1,np.int32)
    cdef np.ndarray[int,ndim=1,mode='c'] J_stiff        = np.zeros(1,np.int32)
    cdef np.ndarray[Real,ndim=1,mode='c'] V_stiff       = np.zeros(1,np.float64)

    cdef np.ndarray[Integer,ndim=2, mode='c'] sorter                    = np.zeros((1,1),np.int64)
    cdef np.ndarray[UInteger,ndim=2, mode='c'] sorted_elements          = np.zeros((1,1),np.uint64)
    cdef np.ndarray[int,ndim=1,mode='c'] data_global_indices            = np.zeros(1,np.int32)
    cdef np.ndarray[int,ndim=1,mode='c'] data_local_indices             = np.zeros(1,np.int32)
    cdef int squeeze_sparsity_pattern                                   = fem_solver.squeeze_sparsity_pattern
    cdef int recompute_sparsity_pattern                                 = fem_solver.recompute_sparsity_pattern

    if fem_solver.recompute_sparsity_pattern:
        I_stiff        = np.zeros(int((nvar*nodeperelem)**2*nelem),np.int32)
        J_stiff        = np.zeros(int((nvar*nodeperelem)**2*nelem),np.int32)
        V_stiff        = np.zeros(int((nvar*nodeperelem)**2*nelem),np.float64)
    else:
        I_stiff = fem_solver.indptr
        J_stiff = fem_solver.indices
        V_stiff = np.zeros(fem_solver.indices.shape[0],dtype=np.float64)
        data_global_indices = fem_solver.data_global_indices
        data_local_indices = fem_solver.data_local_indices
        if fem_solver.squeeze_sparsity_pattern:
            sorter = mesh.element_sorter
            sorted_elements = mesh.sorted_elements

    cdef np.ndarray[Real,ndim=1,mode='c'] T = np.zeros(mesh.points.shape[0]*nvar,np.float64)

    cdef np.ndarray[Real,ndim=2,mode='c'] anisotropic_orientations = np.zeros((1,1),np.float64)
    if material.is_transversely_isotropic:
        anisotropic_orientations = material.anisotropic_orientations

    cdef Real mu=0.,mu1=0.,mu2=0.,mu3=0.,mue=0.,lamb=0.,eps_1=0.,eps_2=0., eps_3=0., eps_e=0.

    mu1, mu2, lamb = material.mu1, material.mu2, material.lamb

    cdef Real rho = material.rho

    _GlobalAssemblyDF_(     &points[0,0],
                            &elements[0,0],
                            &Eulerx[0,0],
                            &Eulerp[0],
                            &bases[0,0],
                            &Jm[0,0,0],
                            &AllGauss[0],
                            ndim,
                            nvar,
                            ngauss,
                            nelem,
                            nodeperelem,
                            nnode,
                            H_VoigtSize,
                            requires_geometry_update,
                            &local_rows_stiffness[0],
                            &local_cols_stiffness[0],
                            &I_stiff[0],
                            &J_stiff[0],
                            &V_stiff[0],
                            &T[0],
                            recompute_sparsity_pattern,
                            squeeze_sparsity_pattern,
                            &data_local_indices[0],
                            &data_global_indices[0],
                            &sorted_elements[0,0],
                            &sorter[0,0],
                            rho,
                            mu,
                            mu1,
                            mu2,
                            mu3,
                            mue,
                            lamb,
                            eps_1,
                            eps_2,
                            eps_3,
                            eps_e,
                            &anisotropic_orientations[0,0]
                            )


    if fem_solver.recompute_sparsity_pattern:
        return I_stiff, J_stiff, V_stiff, T
    else:
        return V_stiff, T
