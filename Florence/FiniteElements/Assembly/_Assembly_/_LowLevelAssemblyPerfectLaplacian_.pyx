import numpy as np
cimport numpy as np

ctypedef long long Integer
ctypedef unsigned long long UInteger
ctypedef double Real


cdef extern from "_LowLevelAssemblyPerfectLaplacian_.h" nogil:
    void _GlobalAssemblyPerfectLaplacian_(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer ndim,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        int *I_stiff,
                        int *J_stiff,
                        Real *V_stiff,
                        Real *e_tensor,
                        Integer is_hessian_symmetric
                        ) nogil


def _LowLevelAssemblyPerfectLaplacian_(fem_solver, function_space, formulation, mesh, material, Real[:,::1] Eulerx, Real[::1] Eulerp):

    # GET VARIABLES FOR DISPATCHING TO C
    cdef Integer ndim                       = formulation.ndim
    cdef Integer nvar                       = formulation.nvar
    cdef Integer ngauss                     = function_space.AllGauss.shape[0]
    cdef Integer nelem                      = mesh.nelem
    cdef Integer nodeperelem                = mesh.elements.shape[1]
    cdef Integer nnode                      = mesh.points.shape[0]

    cdef np.ndarray[UInteger,ndim=2, mode='c'] elements = mesh.elements
    cdef np.ndarray[Real,ndim=2, mode='c'] points       = mesh.points
    cdef np.ndarray[Real,ndim=2, mode='c'] bases        = function_space.Bases
    cdef np.ndarray[Real,ndim=3, mode='c'] Jm           = function_space.Jm
    cdef np.ndarray[Real,ndim=1, mode='c'] AllGauss     = function_space.AllGauss.flatten()

    cdef np.ndarray[int,ndim=1,mode='c'] I_stiff        = np.zeros(int((nvar*nodeperelem)**2*nelem),np.int32)
    cdef np.ndarray[int,ndim=1,mode='c'] J_stiff        = np.zeros(int((nvar*nodeperelem)**2*nelem),np.int32)
    cdef np.ndarray[Real,ndim=1,mode='c'] V_stiff       = np.zeros(int((nvar*nodeperelem)**2*nelem),np.float64)

    # NEGATIVE DEFINITE
    if material.e.shape[0] != ndim:
        raise ValueError("Permittivity tensor has to have a size of (ndim x ndim)")

    cdef np.ndarray[Real,ndim=2, mode='c'] e            = -material.e
    cdef Integer is_hessian_symmetric                   = np.allclose(material.e, material.e.T, atol=1e-8)

    _GlobalAssemblyPerfectLaplacian_(   &points[0,0],
                                        &elements[0,0],
                                        &Eulerp[0],
                                        &bases[0,0],
                                        &Jm[0,0,0],
                                        &AllGauss[0],
                                        ndim,
                                        ngauss,
                                        nelem,
                                        nodeperelem,
                                        nnode,
                                        &I_stiff[0],
                                        &J_stiff[0],
                                        &V_stiff[0],
                                        &e[0,0],
                                        is_hessian_symmetric
                                        )


    return I_stiff, J_stiff, V_stiff

