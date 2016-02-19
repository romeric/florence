import imp, os, sys
from Core.FiniteElements.ApplyNeumannBoundaryConditions import *
from ElementalStiffness import *
from ElementalMass import *
from Core.FiniteElements.StaticCondensation import StaticCondensation

def FindIndices(A):
    return np.repeat(np.arange(0,A.shape[0]),A.shape[0],axis=0), np.tile(np.arange(0,A.shape[0]),A.shape[0]), A.ravel()

def GetElementalMatricesSmall(elem,MainData,elements,points,Eulerx,TotalPot):
    # ALLOCATE
    Domain = MainData.Domain

    massel=[]; f = []
    # GET THE FIELDS AT THE ELEMENT LEVEL
    LagrangeElemCoords = points[elements[elem,:],:]
    EulerElemCoords = Eulerx[elements[elem,:],:]
    if MainData.Fields == 'ElectroMechanics':
        ElectricPotentialElem = TotalPot[elements[elem,:],:]
    else:
        ElectricPotentialElem = []

    # COMPUTE THE STIFFNESS MATRIX
    if MainData.__VECTORISATION__ is True:
        stiffnessel, t = Stiffness(MainData,LagrangeElemCoords,EulerElemCoords,ElectricPotentialElem,elem)
    else:
        stiffnessel, t = Stiffness_NonVectorised(MainData,LagrangeElemCoords,EulerElemCoords,ElectricPotentialElem,elem)

    I_mass_elem = []; J_mass_elem = []; V_mass_elem = []
    if MainData.Analysis != 'Static':
        # COMPUTE THE MASS MATRIX
        massel = Mass(MainData,LagrangeElemCoords,EulerElemCoords,elem)

    if MainData.AssemblyParameters.ExternalLoadNature == 'Nonlinear':
        # COMPUTE FORCE VECTOR
        f = ApplyNeumannBoundaryConditions3D(MainData, nmesh, elem, LagrangeElemCoords)

    # STATIC CONDENSATION
    # if C>0:
        # stiffnessel,f = StaticCondensation(stiffnessel,f,C,nvar)
        # massel,f = StaticCondensation(stiffnessel,f,C,nvar)

    I_stiff_elem, J_stiff_elem, V_stiff_elem = FindIndices(stiffnessel)
    if MainData.Analysis != 'Static':
        I_mass_elem, J_mass_elem, V_mass_elem = FindIndices(massel)

    return I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, I_mass_elem, J_mass_elem, V_mass_elem
