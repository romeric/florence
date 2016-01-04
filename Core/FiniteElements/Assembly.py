import numpy as np 
import scipy as sp
import numpy.linalg as la
from scipy.sparse import coo_matrix, csc_matrix 

from ElementalMatrices.GetElementalMatrices import *
from SparseAssembly import SparseAssembly_Step_2

from ElementalMatrices.GetElementalMatricesSmall import *
from SparseAssemblySmall import SparseAssemblySmall
import pyximport; pyximport.install()
from SparseAssemblyNative import SparseAssemblyNative


# PARALLEL PROCESSING ROUTINES
# from multiprocessing import Pool
import multiprocessing as MP
import Core.ParallelProcessing.parmap as parmap


#-------------- MAIN ASSEMBLY ROUTINE --------------------#
#---------------------------------------------------------#

def Assembly(MainData,mesh,Eulerx,TotalPot):
	return AssemblyLarge(MainData,mesh,Eulerx,TotalPot) if mesh.nelem > 1e09 else AssemblySmall(MainData,mesh,Eulerx,TotalPot)


#-------------- ASSEMBLY ROUTINE FOR RELATIVELY LARGER MATRICES (NELEM > 100000)------------------------#
#-------------------------------------------------------------------------------------------------------#

def AssemblyLarge(MainData,nmesh,Eulerx,TotalPot):

	# GET MESH DETAILS
	C = MainData.C
	nvar = MainData.nvar
	ndim = MainData.ndim

	# nelem = nmesh.elements.shape[0]
	nelem = nmesh.nelem
	nodeperelem = nmesh.elements.shape[1]

	# ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX
	I_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int64)
	J_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int64)
	V_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.float64)

	# THE I & J VECTORS OF LOCAL STIFFNESS MATRIX DO NOT CHANGE, HENCE COMPUTE THEM ONCE
	I_stiff_elem = np.repeat(np.arange(0,nvar*nodeperelem),nvar*nodeperelem,axis=0)
	J_stiff_elem = np.tile(np.arange(0,nvar*nodeperelem),nvar*nodeperelem)

	I_mass=[];J_mass=[];V_mass=[]; I_mass_elem = []; J_mass_elem = []
	if MainData.Analysis !='Static':
		# ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX
		I_mass=np.zeros((nvar*nodeperelem)**2*nmesh.elements.shape[0],dtype=np.int64)
		J_mass=np.zeros((nvar*nodeperelem)**2*nmesh.elements.shape[0],dtype=np.int64)
		V_mass=np.zeros((nvar*nodeperelem)**2*nmesh.elements.shape[0],dtype=np.float64)

		# THE I & J VECTORS OF LOCAL MASS MATRIX DO NOT CHANGE, HENCE COMPUTE THEM ONCE
		I_mass_elem = np.repeat(np.arange(0,nvar*nodeperelem),nvar*nodeperelem,axis=0)
		J_mass_elem = np.tile(np.arange(0,nvar*nodeperelem),nvar*nodeperelem)


	# ALLOCATE RHS VECTORS
	F = np.zeros((nmesh.points.shape[0]*nvar,1)); T =  np.zeros((nmesh.points.shape[0]*nvar,1)) 
	# ASSIGN OTHER NECESSARY MATRICES
	full_current_row_stiff = []; full_current_column_stiff = []; coeff_stiff = [] 
	full_current_row_mass = []; full_current_column_mass = []; coeff_mass = []
	mass = []

	if MainData.Parallel:
		# COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
		ParallelTuple = parmap.map(GetElementalMatrices,np.arange(0,nelem),MainData,nmesh.elements,nmesh.points,nodeperelem,
			Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem,pool=MP.Pool(processes=MainData.nCPU))

	for elem in range(nelem):

		if MainData.Parallel:
			# UNPACK PARALLEL TUPLE VALUES
			full_current_row_stiff = ParallelTuple[elem][0]; full_current_column_stiff = ParallelTuple[elem][1]
			coeff_stiff = ParallelTuple[elem][2]; t = ParallelTuple[elem][3]; f = ParallelTuple[elem][4]
			full_current_row_mass = ParallelTuple[elem][5]; full_current_column_mass = ParallelTuple[elem][6]; coeff_mass = ParallelTuple[elem][6]

		else:
			# COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
			full_current_row_stiff, full_current_column_stiff, coeff_stiff, t, f, \
			full_current_row_mass, full_current_column_mass, coeff_mass = GetElementalMatrices(elem,
				MainData,nmesh.elements,nmesh.points,nodeperelem,Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem)

		# SPARSE ASSEMBLY - STIFFNESS MATRIX
		# I_stiffness, J_stiffness, V_stiffness = SparseAssembly_Step_2(I_stiffness,J_stiffness,V_stiffness,
			# full_current_row_stiff,full_current_column_stiff,coeff_stiff,
		# 	nvar,nodeperelem,elem)
		I_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row_stiff
		J_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column_stiff
		V_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff_stiff

		if MainData.Analysis != 'Static':
			# SPARSE ASSEMBLY - MASS MATRIX
			I_mass, J_mass, V_mass = SparseAssembly_Step_2(I_mass,J_mass,V_mass,full_current_row_mass,full_current_column_mass,coeff_mass,
				nvar,nodeperelem,elem)

		if MainData.AssemblyParameters.ExternalLoadNature == 'Nonlinear':
			# RHS ASSEMBLY
			for iter in range(0,nvar):
				F[nmesh.elements[elem,:]*nvar+iter,0]+=f[iter::nvar]
		# INTERNAL TRACTION FORCE ASSEMBLY
		for iter in range(0,nvar):
				T[nmesh.elements[elem,:]*nvar+iter,0]+=t[iter::nvar,0]

	# CALL BUILT-IN SPARSE ASSEMBLER 
	stiffness = coo_matrix((V_stiffness,(I_stiffness,J_stiffness)),shape=((nvar*nmesh.points.shape[0],nvar*nmesh.points.shape[0]))).tocsc()

	if MainData.Analysis != 'Static':
		# CALL BUILT-IN SPARSE ASSEMBLER
		mass = coo_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*nmesh.points.shape[0],nvar*nmesh.points.shape[0]))).tocsc()

	# GET STORAGE/MEMORY DETAILS
	MainData.spmat = stiffness.data.nbytes/1024./1024.
	MainData.ijv = (I_stiffness.nbytes + J_stiffness.nbytes + V_stiffness.nbytes)/1024./1024.

	return stiffness, T, F, mass




#-------------- ASSEMBLY ROUTINE FOR RELATIVELY SMALL SIZE MATRICES (NELEM < 100000)--------------------#
#-------------------------------------------------------------------------------------------------------#

def AssemblySmall(MainData,mesh,Eulerx,TotalPot):

	# GET MESH DETAILS
	C = MainData.C
	nvar = MainData.nvar
	ndim = MainData.ndim
	nelem = mesh.nelem
	nodeperelem = mesh.elements.shape[1]

	# ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX
	I_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int64)
	J_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int64)
	V_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.float64)

	I_mass=[];J_mass=[];V_mass=[]
	if MainData.Analysis !='Static':
		# ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX
		I_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.int64)
		J_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.int64)
		V_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.float64)

	F = np.zeros((mesh.points.shape[0]*nvar,1)); T =  np.zeros((mesh.points.shape[0]*nvar,1))  
	mass = []


	if MainData.Parallel:
		# COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
		ParallelTuple = parmap.map(GetElementalMatricesSmall,np.arange(0,nelem),MainData,mesh.elements,mesh.points,Eulerx,TotalPot,
			pool=MP.Pool(processes=MainData.numCPU))

	for elem in range(nelem):

		if MainData.Parallel:
			# UNPACK PARALLEL TUPLE VALUES
			I_stiff_elem = ParallelTuple[elem][0]; J_stiff_elem = ParallelTuple[elem][1]; V_stiff_elem = ParallelTuple[elem][2]
			t = ParallelTuple[elem][3]; f = ParallelTuple[elem][4]
			I_mass_elem = ParallelTuple[elem][5]; J_mass_elem = ParallelTuple[elem][6]; V_mass_elem = ParallelTuple[elem][6]

		else:
			# COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
			I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, I_mass_elem, J_mass_elem, V_mass_elem = GetElementalMatricesSmall(elem,MainData,
				mesh.elements,mesh.points,Eulerx,TotalPot)

		# SPARSE ASSEMBLY - STIFFNESS MATRIX
		SparseAssemblyNative(I_stiff_elem,J_stiff_elem,V_stiff_elem,I_stiffness,J_stiffness,V_stiffness,
			elem,nvar,nodeperelem,mesh.elements)

		# SparseAssemblySmall(I_stiff_elem,J_stiff_elem,V_stiff_elem,
		# 	I_stiffness,J_stiffness,V_stiffness,elem,nvar,nodeperelem,mesh.elements)

		
		if MainData.Analysis != 'Static':
			# SPARSE ASSEMBLY - MASS MATRIX
			I_mass, J_mass, V_mass = SparseAssemblySmall(I_mass_elem,J_mass_elem,V_mass_elem,
				I_mass,J_mass,V_mass,elem,nvar,nodeperelem,mesh.elements)

		if MainData.AssemblyParameters.ExternalLoadNature == 'Nonlinear':
			# RHS ASSEMBLY
			for iter in range(0,nvar):
				F[mesh.elements[elem,:]*nvar+iter,0]+=f[iter:f.shape[0]:nvar]
		# INTERNAL TRACTION FORCE ASSEMBLY
		for iterator in range(0,nvar):
				T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]

	# CALL BUILT-IN SPARSE ASSEMBLER 
	stiffness = coo_matrix((V_stiffness,(I_stiffness,J_stiffness)),shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0]))).tocsc()

	if MainData.Analysis != 'Static':
		# CALL BUILT-IN SPARSE ASSEMBLER
		mass = coo_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0]))).tocsc()

	# GET STORAGE/MEMORY DETAILS
	MainData.spmat = stiffness.data.nbytes/1024./1024.
	MainData.ijv = (I_stiffness.nbytes + J_stiffness.nbytes + V_stiffness.nbytes)/1024./1024.


	return stiffness, T, F, mass









#------------------------------- RHS ASSEMBLY -------------------------------#
#----------------------------------------------------------------------------#

# PRE-ASSEMBLY - ASSEMBLY FOR RHS ONLY
def AssemblyForces(MainData,mesh,nmesh,Quadrature,Domain,BoundaryData,Boundary):

	# H = MainData.MaterialArgs.H
	# Get mesh details
	C = MainData.C
	nvar = MainData.nvar
	ndim = MainData.ndim

	nmesh.points = np.array(nmesh.points)
	nmesh.elements = np.array(nmesh.elements)
	nelem = nmesh.elements.shape[0]
	nodeperelem = nmesh.elements.shape[1]


	F = np.zeros((nmesh.points.shape[0]*nvar,1)) 
	f = []

	for elem in range(0,nelem):
		LagrangeElemCoords = np.zeros((nodeperelem,ndim))
		for i in range(0,nodeperelem):
			LagrangeElemCoords[i,:] = nmesh.points[nmesh.elements[elem,i],:]

		
		if ndim==2:
			# Compute Force vector
			f = np.zeros(k.shape[0])
		elif ndim==3:
			# Compute Force vector
			f = ApplyNeumannBoundaryConditions3D(MainData, nmesh, BoundaryData, Domain, Boundary, Quadrature.weights, elem, LagrangeElemCoords)


		# Static Condensation
		# if C>0:
		# 	k,f = St.StaticCondensation(k,f,C,nvar)

		# RHS Assembly
		for iter in range(0,nvar):
			F[nmesh.elements[elem]*nvar+iter,0]+=f[iter:f.shape[0]:nvar]


	return F



# PRE-ASSEMBLY - ASSEMBLY FOR RHS ONLY (CHEAP)
def AssemblyForces_Cheap(MainData,mesh,nmesh,Quadrature,Domain,BoundaryData,Boundary,DynStep=0):

	# THIS IS A DIRICHLET TYPE STRATEGY FOR APPLYING NEUMANN BOUNDARY CONDITIONS - MEANING GLOBALY
	# Get mesh details
	C = MainData.C
	nvar = MainData.nvar
	ndim = MainData.ndim

	points = np.array(nmesh.points)
	elements = np.array(nmesh.elements)
	edges = np.array(nmesh.edges)
	faces = np.array(nmesh.faces)
	nelem = elements.shape[0]
	nodeperelem = elements.shape[1]

	F = np.zeros((points.shape[0]*nvar,1)) 
	f = []

	columns_out = []; AppliedNeumann = []; unique_edge_nodes = []

	if BoundaryData().NeuArgs.Applied_at == 'node':
		if MainData.ndim==3:
			unique_edge_nodes = np.unique(faces)
		else:
			unique_edge_nodes = np.unique(edges)
		# Further optimisation of this function and also a necessity
		# We need to know the number of nodes on the face to distribute the force
		BoundaryData().NeuArgs.no_nodes = np.zeros(len(BoundaryData().NeuArgs.cond))
		pp =  points[unique_edge_nodes,:]
		for i in range(0,BoundaryData().NeuArgs.cond.shape[0]):
			x =  np.where(pp[:,BoundaryData().NeuArgs.cond[i,0]]==BoundaryData().NeuArgs.cond[i,1])[0]
			# Number of nodes on which this Neumann is applied, is 
			BoundaryData().NeuArgs.no_nodes[i] = x.shape[0]

		for inode in range(0,unique_edge_nodes.shape[0]):
			coord_node = points[unique_edge_nodes[inode]]
			BoundaryData().NeuArgs.node = coord_node
			Neumann = BoundaryData().NeumannCriterion(BoundaryData().NeuArgs,MainData.Analysis,DynStep)

			if type(Neumann) is list:
				pass
			else:
				for i in range(0,nvar):
					if type(Neumann[i]) is list:
						pass
					else:
						columns_out = np.append(columns_out,nvar*inode+i)
						AppliedNeumann = np.append(AppliedNeumann,Neumann[i])

	columns_out = columns_out.astype(int)
	# In this case columns out is analogous to the rows of F where Neumann data appears
	for i in range(0,columns_out.shape[0]):
		if AppliedNeumann[i]!=0.0:
			F[columns_out[i],0] = AppliedNeumann[i]


	return F
