from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

try:
    from ._LowLevelAssemblyDF__NeoHookean_2_ import _LowLevelAssemblyDF__NeoHookean_2_
    from ._LowLevelAssemblyDF__MooneyRivlin_0_ import _LowLevelAssemblyDF__MooneyRivlin_0_
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
except IOError:
    has_low_level_dispatcher = False
    warn("Cannot use low level dispatchers for Assembly")


__all__ = ['_LowLevelAssembly_']


def _LowLevelAssembly_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    prefix = "_LowLevelAssemblyDF__"
    if formulation.fields == "electro_mechanics":
        prefix = "_LowLevelAssemblyDPF__"

    assembly_func = prefix + type(material).__name__ + "_"

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



