from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

try:
    from ._LowLevelAssemblyDF__LinearElastic_ import _LowLevelAssemblyDF__LinearElastic_
    from ._LowLevelAssemblyDF__LinearElastic_ import _LowLevelAssemblyDF__LinearElastic_ as _LowLevelAssemblyDF__IncrementalLinearElastic_
    from ._LowLevelAssemblyDF__NeoHookean_ import _LowLevelAssemblyDF__NeoHookean_
    from ._LowLevelAssemblyDF__MooneyRivlin_ import _LowLevelAssemblyDF__MooneyRivlin_
    from ._LowLevelAssemblyDF__AnisotropicMooneyRivlin_1_ import _LowLevelAssemblyDF__AnisotropicMooneyRivlin_1_
    from ._LowLevelAssemblyDF__NearlyIncompressibleMooneyRivlin_ import _LowLevelAssemblyDF__NearlyIncompressibleMooneyRivlin_
    from ._LowLevelAssemblyDPF__IsotropicElectroMechanics_0_ import _LowLevelAssemblyDPF__IsotropicElectroMechanics_0_
    from ._LowLevelAssemblyDPF__IsotropicElectroMechanics_3_ import _LowLevelAssemblyDPF__IsotropicElectroMechanics_3_
    from ._LowLevelAssemblyDPF__SteinmannModel_ import _LowLevelAssemblyDPF__SteinmannModel_
    from ._LowLevelAssemblyDPF__IsotropicElectroMechanics_101_ import _LowLevelAssemblyDPF__IsotropicElectroMechanics_101_
    from ._LowLevelAssemblyDPF__IsotropicElectroMechanics_105_ import _LowLevelAssemblyDPF__IsotropicElectroMechanics_105_
    from ._LowLevelAssemblyDPF__IsotropicElectroMechanics_106_ import _LowLevelAssemblyDPF__IsotropicElectroMechanics_106_
    from ._LowLevelAssemblyDPF__IsotropicElectroMechanics_107_ import _LowLevelAssemblyDPF__IsotropicElectroMechanics_107_
    from ._LowLevelAssemblyDPF__IsotropicElectroMechanics_108_ import _LowLevelAssemblyDPF__IsotropicElectroMechanics_108_
    from ._LowLevelAssemblyDPF__Piezoelectric_100_ import _LowLevelAssemblyDPF__Piezoelectric_100_
    has_low_level_dispatcher = True
except ImportError:
    has_low_level_dispatcher = False
    warn("Cannot use low level dispatchers for Assembly")


try:
    from ._LowLevelAssemblyExplicit_DF_DPF_ import _LowLevelAssemblyExplicit_DF_DPF_
    has_low_level_dispatcher = True
except ImportError:
    has_low_level_dispatcher = False
    warn("Cannot use low level dispatchers for Assembly")

try:
    from ._LowLevelAssemblyPerfectLaplacian_ import _LowLevelAssemblyPerfectLaplacian_
    has_low_level_dispatcher = True
except ImportError:
    has_low_level_dispatcher = False
    warn("Cannot use low level dispatchers for Assembly")


__all__ = ['_LowLevelAssembly_', '_LowLevelAssemblyExplicit_', '_LowLevelAssemblyLaplacian_']


def _LowLevelAssembly_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    prefix = "_LowLevelAssemblyDF__"
    if formulation.fields == "electro_mechanics":
        prefix = "_LowLevelAssemblyDPF__"

    assembly_func = prefix + type(material).__name__ + "_"
    # CHECK IF LOW LEVEL ASSEMBLY EXISTS FOR MATERIAL
    ll_exists = False
    for key in globals().keys():
        if assembly_func == key:
            ll_exists = True
            break
    if ll_exists is False:
        raise NotImplementedError("Turning optimise option on for {} material is not supported yet. Consider 'optimise=False' for now".format(type(material).__name__))


    # MAKE MESH DATA CONTIGUOUS
    mesh.ChangeType()

    # CALL LOW LEVEL ASSEMBLER
    I_stiffness, J_stiffness, V_stiffness, I_mass, J_mass, V_mass, \
        T = eval(assembly_func)(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)

    nvar = formulation.nvar
    stiffness = csr_matrix((V_stiffness,(I_stiffness,J_stiffness)),
        shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64)

    F, mass = [], []

    if fem_solver.analysis_type != "static" and fem_solver.is_mass_computed is False:
        mass = csr_matrix((V_mass,(I_mass,J_mass)),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64)

    return stiffness, T, F, mass



def _LowLevelAssembly_Par_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    prefix = "_LowLevelAssemblyDF__"
    if formulation.fields == "electro_mechanics":
        prefix = "_LowLevelAssemblyDPF__"
        
    assembly_func = prefix + type(material).__name__ + "_"
    # CHECK IF LOW LEVEL ASSEMBLY EXISTS FOR MATERIAL
    ll_exists = False
    for key in globals().keys():
        if assembly_func == key:
            ll_exists = True
            break
    if ll_exists is False:
        raise NotImplementedError("Turning optimise option on for {} material is not supported yet. Consider 'optimise=False' for now".format(type(material).__name__))

    # CALL LOW LEVEL ASSEMBLER
    return eval(assembly_func)(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)



def _LowLevelAssemblyExplicit_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    # MAKE MESH DATA CONTIGUOUS
    mesh.ChangeType()

    # CALL LOW LEVEL ASSEMBLER
    I_mass, J_mass, V_mass, \
        T = _LowLevelAssemblyExplicit_DF_DPF_(function_space, formulation, mesh, material, Eulerx, Eulerp)

    F, mass = [], []

    if fem_solver.analysis_type != "static" and fem_solver.is_mass_computed is False:
        nvar = formulation.nvar
        mass = csr_matrix((V_mass,(I_mass,J_mass)),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64)

    return T, F, mass


def _LowLevelAssemblyExplicit_Par_(function_space, formulation, mesh, material, Eulerx, Eulerp):
    # CALL LOW LEVEL ASSEMBLER
    return _LowLevelAssemblyExplicit_DF_DPF_(function_space, formulation, mesh, material, Eulerx, Eulerp)[-1]



def _LowLevelAssemblyLaplacian_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    mesh.GetNumberOfNodes()
    # MAKE MESH DATA CONTIGUOUS
    mesh.ChangeType()

    # CALL LOW LEVEL ASSEMBLER
    I, J, V = _LowLevelAssemblyPerfectLaplacian_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
    stiffness = csr_matrix((V,(I,J)), shape=((mesh.nnode,mesh.nnode)),dtype=np.float64)

    return stiffness, np.zeros(mesh.nnode,np.float64)



