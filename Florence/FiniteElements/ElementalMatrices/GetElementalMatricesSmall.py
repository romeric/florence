import imp, os, sys
from .ElementalStiffness import *
from .ElementalMass import *
from Florence.FiniteElements.ApplyNeumannBoundaryConditions import *
from Florence.FiniteElements.StaticCondensation import StaticCondensation

def FindIndices(A):
    return np.repeat(np.arange(0,A.shape[0]),A.shape[0],axis=0), np.tile(np.arange(0,A.shape[0]),A.shape[0]), A.ravel()

def GetElementalMatricesSmall(elem, function_space, formulation, material, mesh, fem_solver, Eulerx,TotalPot):
    # ALLOCATE
    Domain = function_space

    massel=[]; f = []
    # GET THE FIELDS AT THE ELEMENT LEVEL
    LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
    EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
    if formulation.fields == 'electro_mechanics':
        ElectricPotentialElem = TotalPot[mesh.elements[elem,:],:]
    else:
        ElectricPotentialElem = []

    # COMPUTE THE STIFFNESS MATRIX
    if fem_solver.vectorise:
        stiffnessel, t = Stiffness(function_space, formulation, material, fem_solver,
            LagrangeElemCoords, EulerElemCoords, ElectricPotentialElem,elem)
    else:
        stiffnessel, t = Stiffness_NonVectorised(MainData,function_space, formulation, material, fem_solver,
            LagrangeElemCoords,EulerElemCoords,ElectricPotentialElem,elem)


    I_mass_elem = []; J_mass_elem = []; V_mass_elem = []
    if fem_solver.analysis_type != 'static':
        # COMPUTE THE MASS MATRIX
        massel = Mass(MainData,LagrangeElemCoords,EulerElemCoords,elem)

    if fem_solver.has_moving_boundary:
        # COMPUTE FORCE VECTOR
        f = ApplyNeumannBoundaryConditions3D(MainData, nmesh, elem, LagrangeElemCoords)

    # STATIC CONDENSATION
    # if C>0:
        # stiffnessel,f = StaticCondensation(stiffnessel,f,C,nvar)
        # massel,f = StaticCondensation(stiffnessel,f,C,nvar)

    I_stiff_elem, J_stiff_elem, V_stiff_elem = FindIndices(stiffnessel)
    if fem_solver.analysis_type != 'static':
        I_mass_elem, J_mass_elem, V_mass_elem = FindIndices(massel)

    return I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, I_mass_elem, J_mass_elem, V_mass_elem
