from warnings import warn

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

    return eval(assembly_func)(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)



