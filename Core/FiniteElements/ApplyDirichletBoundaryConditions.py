import numpy as np 
from DirichletBoundaryDataFromCAD import IGAKitWrapper, PostMeshWrapper
from time import time

# ROUTINE FOR APPLYING DIRICHLET BOUNDARY CONDITIONS
def ApplyDirichletBoundaryConditions(stiffness,F,mesh,MainData):

	#######################################################
	nvar = MainData.nvar
	ndim = MainData.ndim

	columns_out = []; AppliedDirichlet = []

	#----------------------------------------------------------------------------------------------------#
	#-------------------------------------- NURBS BASED SOLUTION ----------------------------------------#
	#----------------------------------------------------------------------------------------------------#
	if MainData.BoundaryData.Type == 'nurbs':

		tCAD = time()

		# GET DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY FROM CAD
		if MainData.BoundaryData.RequiresCAD:
			# CALL POSTMESH WRAPPER
			nodesDBC, Dirichlet = PostMeshWrapper(MainData,mesh)
		else:
			# CALL IGAKIT WRAPPER
			nodesDBC, Dirichlet = IGAKitWrapper(MainData,mesh)

		print 'Finished identifying Dirichlet boundary conditions from CAD geometry. Time taken ', time()-tCAD, 'seconds'


		# from Core.Supplementary.Tensors import makezero
		# Dirichlet = makezero(Dirichlet)
		# for i in range(Dirichlet.shape[0]):
		# 	for j in range(Dirichlet.shape[1]):
		# 		if abs(Dirichlet[i,j])<1e-02:
		# 			Dirichlet[i,j]= 0

		

		nOfDBCnodes = nodesDBC.shape[0]
		for inode in range(nOfDBCnodes):
			for i in range(nvar):
				columns_out = np.append(columns_out,nvar*nodesDBC[inode]+i)
				AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[inode,i])

		# MainData.nodesDBC = nodesDBC # REMOVE THIS
		# print Dirichlet
		# print nodesDBC.shape
		# print AppliedDirichlet
		# print nodesDBC[8]
		# print Dirichlet.shape
		# print Dirichlet[8,:]
		# print AppliedDirichlet.shape
		# from scipy.io import savemat
		# print nodesDBC.shape, mesh.points.shape

		# print nodesDBC.shape, Dirichlet.shape
		# for i in range(nodesDBC.shape[0]):
		# 	x,y = np.where(mesh.edges==nodesDBC[i])
		# 	# print x
		# 	if x.shape[0]!=0:
		# 		# print x,y
		# 		Dirichlet[mesh.edges[x,y],:]=0
		# 		# print Dirichlet[x,y]
		# 	# pass




		# import sys; sys.exit() 


		############################
		# To Rogelio
		# print mesh.points
		# print AppliedDirichlet.shape, mesh.points.shape
		# Dict = {'points':mesh.points,'element':mesh.elements,'displacements':AppliedDirichlet,'displacement_dof':columns_out}
		# from scipy.io import savemat
		# savemat('/home/roman/Desktop/wing_p2',Dict)

		# print mesh.edges.shape, AppliedDirichlet.shape, Dirichlet.shape
		# import sys; sys.exit(0)

		############################

	#----------------------------------------------------------------------------------------------------#
	#------------------------------------- NON-NURBS BASED SOLUTION -------------------------------------#
	#----------------------------------------------------------------------------------------------------#

	elif MainData.BoundaryData.Type == 'straight' or MainData.BoundaryData.Type == 'mixed':
		# IF DIRICHLET BOUNDARY CONDITIONS ARE APPLIED DIRECTLY AT NODES
		if MainData.BoundaryData().DirichArgs.Applied_at == 'node':
			# GET UNIQUE NODES AT THE BOUNDARY
			unique_edge_nodes = []
			if ndim==2:
				unique_edge_nodes = np.unique(mesh.edges)
			elif ndim==3:
				unique_edge_nodes = np.unique(mesh.faces)
			# ACTIVATE THIS FOR DEBUGGING ELECTROMECHANICAL PROBLEMS
			# unique_edge_nodes = np.unique(mesh.elements) 


			MainData.BoundaryData().DirichArgs.points = mesh.points
			MainData.BoundaryData().DirichArgs.edges = mesh.edges
			for inode in range(0,unique_edge_nodes.shape[0]):
				coord_node = mesh.points[unique_edge_nodes[inode]]
				MainData.BoundaryData().DirichArgs.node = coord_node
				MainData.BoundaryData().DirichArgs.inode = unique_edge_nodes[inode]

				Dirichlet = MainData.BoundaryData().DirichletCriterion(MainData.BoundaryData().DirichArgs)

				# COMMENTED RECENTLY IN FAVOR OF WHAT APPEARS BELOW
				# if type(Dirichlet) is None:
				# 	pass
				# else:
				# 	for i in range(nvar):
				# 		# if type(Dirichlet[i]) is list:
				# 		if Dirichlet[i] is None:
				# 			pass
				# 		else:
				# 			# columns_out = np.append(columns_out,nvar*inode+i) # THIS IS INVALID
				# 			# ACTIVATE THIS FOR DEBUGGING ELECTROMECHANICAL PROBLEMS
				# 			columns_out = np.append(columns_out,nvar*unique_edge_nodes[inode]+i)
				# 			AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[i])

				if type(Dirichlet) is not None:
					for i in range(nvar):
						if Dirichlet[i] is not None:
							# columns_out = np.append(columns_out,nvar*inode+i) # THIS IS INVALID
							# ACTIVATE THIS FOR DEBUGGING ELECTROMECHANICAL PROBLEMS
							columns_out = np.append(columns_out,nvar*unique_edge_nodes[inode]+i)
							AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[i])


	# GENERAL PROCEDURE - GET REDUCED MATRICES FOR FINAL SOLUTION
	columns_out = columns_out.astype(np.int64)
	columns_in = np.delete(np.arange(0,nvar*mesh.points.shape[0]),columns_out)

	for i in range(0,columns_out.shape[0]):
		# F = F - AppliedDirichlet[i]*(stiffness[:,columns_out[i]])
		F = F - AppliedDirichlet[i]*stiffness.getcol(columns_out[i])
	
	stiffness_b = []; F_b1 = [] 
	return stiffness_b, F_b1, F, columns_in, columns_out, AppliedDirichlet





def GetReducedMatrices(stiffness,F,columns_in,mass=0,Analysis=0):

	F_b = F[columns_in,0]
	stiffness_b = stiffness[columns_in,:][:,columns_in]

	if Analysis != 'Static':
		mass = mass[columns_in,:][:,columns_in]

	return stiffness_b, F_b, F, mass
