# import imp, os, sys
from .ElementalStiffness import *
from .ElementalMass import *
from Florence.FiniteElements.ApplyNeumannBoundaryConditions import *
from Florence.FiniteElements.SparseAssembly import SparseAssembly_Step_1

# pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
# St = imp.load_source('FiniteElements',pwd+'/FiniteElements/StaticCondensation.py')



def DistributedMatrices(elem,MainData,mesh,material,Eulerx,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem):
    
    massel=[]; f = []  
    # GET THE FIELDS AT THE ELEMENT LEVEL
    LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
    EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
    if MainData.Fields == 'ElectroMechanics':
        ElectricPotentialElem = TotalPot[elements[elem,:],:]
    else:
        ElectricPotentialElem = []

    nodeperelem = mesh.elements.shape[1]
    
    from Florence import FunctionSpace, QuadratureRule

    norder = 2*MainData.C
    if norder == 0:
        norder = 1
    QuadratureOpt=3

    quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder, mesh_type=mesh.element_type)
    function_space = FunctionSpace(mesh, quadrature, p=MainData.C+1)
    MainData.Domain, MainData.Boundary = function_space, function_space.Boundary

    norder_post = (MainData.C+1)+(MainData.C+1)
    post_quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder_post, mesh_type=mesh.element_type)
    function_space = FunctionSpace(mesh, post_quadrature, p=MainData.C+1)
    MainData.PostDomain, MainData.PostBoundary = function_space, function_space.Boundary

    # MainData.Domain, MainData.Boundary, MainData.Quadrature = GetBasesAtInegrationPoints(MainData.C,
    #     norder,QuadratureOpt,mesh.element_type)
    # norder_post = (MainData.C+1)+(MainData.C+1)
    # MainData.PostDomain, MainData.PostBoundary, MainData.PostQuadrature = GetBasesAtInegrationPoints(MainData.C,
    #     norder_post,QuadratureOpt,mesh.element_type)

    stiffnessel, t = Stiffness(MainData,material,LagrangeElemCoords,EulerElemCoords,ElectricPotentialElem,elem)

    # FROM THE LOCAL I & J VECTORS GET GLOBAL I & J VECTORS
    full_current_row_stiff, full_current_column_stiff = SparseAssembly_Step_1(I_stiff_elem,
        J_stiff_elem,MainData.nvar,nodeperelem,elem,mesh.elements)

    return full_current_row_stiff, full_current_column_stiff, stiffnessel.ravel(), t, f, [],[],[]





def FindIndices(A):
    return np.repeat(np.arange(0,A.shape[0]),A.shape[0],axis=0), np.tile(np.arange(0,A.shape[0]),A.shape[0]), A.ravel()


def GetElementalMatrices(elem,MainData,elements,points,nodeperelem,Eulerx,
    Eulerp,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem):
    # ALLOCATE
    Domain = MainData.Domain

    massel=[]; f = []  
    # GET THE FIELDS AT THE ELEMENT LEVEL
    LagrangeElemCoords = points[elements[elem,:],:]
    EulerElemCoords = Eulerx[elements[elem,:],:]
    if MainData.Fields == 'ElectroMechanics':
        ElectricPotentialElem = Eulerp[elements[elem,:],:]
    else:
        ElectricPotentialElem = []

    # COMPUTE THE STIFFNESS MATRIX
    if MainData.__VECTORISATION__ is True:
        stiffnessel, t = Stiffness(MainData,LagrangeElemCoords,EulerElemCoords,ElectricPotentialElem,elem)
    else:
        stiffnessel, t = Stiffness_NonVectorised(MainData,LagrangeElemCoords,EulerElemCoords,ElectricPotentialElem,elem)

    # FROM THE LOCAL I & J VECTORS GET GLOBAL I & J VECTORS
    full_current_row_stiff, full_current_column_stiff = SparseAssembly_Step_1(I_stiff_elem,
        J_stiff_elem,MainData.nvar,nodeperelem,elem,elements)

    # FOR TIME-DEPENDENT PROBLEMS   
    full_current_row_mass=[]; full_current_column_mass =[]; V_mass_elem=[]
    if MainData.Analysis != 'Static':
        # COMPUTE THE MASS MATRIX
        massel = Mass(MainData,LagrangeElemCoords,EulerElemCoords,elem)
        # FROM THE LOCAL I & J VECTORS GET GLOBAL I & J VECTORS
        full_current_row_mass, full_current_column_mass = SparseAssembly_Step_1(I_mass_elem,
            J_mass_elem,MainData.nvar,nodeperelem,elem,elements)
        # RAVEL MASS MATRIX
        V_mass_elem = massel.ravel()

    if MainData.AssemblyParameters.ExternalLoadNature == 'Nonlinear':
        # COMPUTE FORCE VECTOR
        f = ApplyNeumannBoundaryConditions3D(MainData, nmesh, elem, LagrangeElemCoords)
    
    # STATIC CONDENSATION
        # if C>0:
            # stiffnessel,f = St.StaticCondensation(stiffnessel,f,C,nvar)
            # massel,f = St.StaticCondensation(stiffnessel,f,C,nvar)



    return full_current_row_stiff, full_current_column_stiff, stiffnessel.ravel(), 
    t, f, full_current_row_mass, full_current_column_mass, V_mass_elem
