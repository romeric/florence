# import imp, os, sys
from .ElementalStiffness import *
from .ElementalMass import *
from Florence.FiniteElements.ApplyNeumannBoundaryConditions import *
from Florence.FiniteElements.SparseAssembly import SparseAssembly_Step_1

# pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
# St = imp.load_source('FiniteElements',pwd+'/FiniteElements/StaticCondensation.py')

def FindIndices(A):
	# NEW FASTER APPROACH - NO TEMPORARY
	return np.repeat(np.arange(0,A.shape[0]),A.shape[0],axis=0), np.tile(np.arange(0,A.shape[0]),A.shape[0]), A.ravel()


def GetElementalMatrices(elem,MainData,elements,points,nodeperelem,Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem):
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

	# FROM THE LOCAL I & J VECTORS GET GLOBAL I & J VECTORS
	full_current_row_stiff, full_current_column_stiff = SparseAssembly_Step_1(I_stiff_elem,J_stiff_elem,MainData.nvar,nodeperelem,elem,elements)

	# FOR TIME-DEPENDENT PROBLEMS	
	full_current_row_mass=[]; full_current_column_mass =[]; V_mass_elem=[]
	if MainData.Analysis != 'Static':
		# COMPUTE THE MASS MATRIX
		massel = Mass(MainData,LagrangeElemCoords,EulerElemCoords,elem)
		# FROM THE LOCAL I & J VECTORS GET GLOBAL I & J VECTORS
		full_current_row_mass, full_current_column_mass = SparseAssembly_Step_1(I_mass_elem,J_mass_elem,MainData.nvar,nodeperelem,elem,elements)
		# RAVEL MASS MATRIX
		V_mass_elem = massel.ravel()

	if MainData.AssemblyParameters.ExternalLoadNature == 'Nonlinear':
		# COMPUTE FORCE VECTOR
		f = ApplyNeumannBoundaryConditions3D(MainData, nmesh, elem, LagrangeElemCoords)
	
	# STATIC CONDENSATION
		# if C>0:
			# stiffnessel,f = St.StaticCondensation(stiffnessel,f,C,nvar)
			# massel,f = St.StaticCondensation(stiffnessel,f,C,nvar)



	return full_current_row_stiff, full_current_column_stiff, stiffnessel.ravel(), t, f, full_current_row_mass, full_current_column_mass, V_mass_elem
