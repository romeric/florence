import numpy as np
cimport numpy as np

ctypedef long long Integer
ctypedef unsigned long long UInteger
ctypedef double Real


cdef extern from "_LowLevelAssemblyExplicit_DF_DPF_.h" nogil:
    void _GlobalAssemblyExplicit_DF_DPF_(const Real *points,
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
                        Real *T,
                        Integer is_dynamic,
                        Integer* local_rows_mass,
                        Integer* local_cols_mass,
                        int *I_mass,
                        int *J_mass,
                        Real *V_mass,
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
                        const Real *anisotropic_orientations,
                        int material_number,
                        int formulation_number
                        ) nogil


def _LowLevelAssemblyExplicit_DF_DPF_(function_space, formulation, mesh, material, Real[:,::1] Eulerx, Real[::1] Eulerp):

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

    cdef Integer requires_geometry_update               = True
    cdef Integer is_dynamic                             = False

    cdef np.ndarray[Integer,ndim=1,mode='c'] local_rows_mass        = formulation.local_rows_mass
    cdef np.ndarray[Integer,ndim=1,mode='c'] local_cols_mass        = formulation.local_columns_mass

    cdef np.ndarray[int,ndim=1,mode='c'] I_mass         = np.zeros(1,np.int32)
    cdef np.ndarray[int,ndim=1,mode='c'] J_mass         = np.zeros(1,np.int32)
    cdef np.ndarray[Real,ndim=1,mode='c'] V_mass        = np.zeros(1,np.float64)

    if is_dynamic:
        I_mass          = np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        J_mass          = np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        V_mass          = np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)


    cdef np.ndarray[Real,ndim=1,mode='c'] T = np.zeros(mesh.points.shape[0]*nvar,np.float64)

    cdef np.ndarray[Real,ndim=2,mode='c'] anisotropic_orientations = np.zeros((1,1),np.float64)
    if material.is_transversely_isotropic:
        anisotropic_orientations = material.anisotropic_orientations

    cdef Real mu=0.,mu1=0.,mu2=0.,mu3=0.,mue=0.,lamb=0.,eps_1=0.,eps_2=0., eps_3=0., eps_e=0.

    cdef int material_number
    if material.mtype == "ExplicitMooneyRivlin":
        mu1, mu2, lamb = material.mu1, material.mu2, material.lamb
        material_number = 0
    elif material.mtype == "NeoHookean":
        mu, lamb = material.mu, material.lamb
        material_number = 1
    elif material.mtype == "MooneyRivlin":
        mu1, mu2, lamb = material.mu1, material.mu2, material.lamb
        material_number = 2
    elif material.mtype == "NearlyIncompressibleMooneyRivlin":
        mu1, mu2, mu3 = material.alpha, material.beta, material.kappa
        material_number = 3
    elif material.mtype == "IsotropicElectroMechanics_101":
        mu, lamb, eps_1 = material.mu, material.lamb, material.eps_1
        material_number = 4
    elif material.mtype == "IsotropicElectroMechanics_105":
        mu1, mu2, lamb, eps_1, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_1, material.eps_2
        material_number = 5
    elif material.mtype == "IsotropicElectroMechanics_106":
        mu1, mu2, lamb, eps_1, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_1, material.eps_2
        material_number = 6
    elif material.mtype == "IsotropicElectroMechanics_107":
        mu1, mu2, mue, lamb, eps_1, eps_2, eps_e = material.mu1, material.mu2, material.mue, material.lamb, \
            material.eps_1, material.eps_2, material.eps_e
        material_number = 7
    elif material.mtype == "IsotropicElectroMechanics_108":
        mu1, mu2, lamb, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_2
        material_number = 8
    elif material.mtype == "ExplicitIsotropicElectroMechanics_108":
        mu1, mu2, lamb, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_2
        material_number = 9
    elif material.mtype == "LinearElastic" or material.mtype == "IncrementalLinearElastic":
        mu, lamb = material.mu, material.lamb
        material_number = 10
    else:
        raise NotImplementedError("Low level assembly for material {} not available for explicit analysis."
            " Consider 'optimise=False' for now".format(material.mtype))

    cdef Real rho = material.rho

    cdef int formulation_number
    if formulation.fields == "mechanics":
        formulation_number = 0
    elif formulation.fields == "electro_mechanics":
        formulation_number = 1

    _GlobalAssemblyExplicit_DF_DPF_(    &points[0,0],
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
                                        &T[0],
                                        is_dynamic,
                                        &local_rows_mass[0],
                                        &local_cols_mass[0],
                                        &I_mass[0],
                                        &J_mass[0],
                                        &V_mass[0],
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
                                        &anisotropic_orientations[0,0],
                                        material_number,
                                        formulation_number
                                        )


    return I_mass, J_mass, V_mass, T



