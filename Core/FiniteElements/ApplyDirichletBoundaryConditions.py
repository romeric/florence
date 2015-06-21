import numpy as np 

# ROUTINE FOR APPLYING DIRICHLET BOUNDARY CONDITIONS
def ApplyDirichletBoundaryConditions(stiffness,F,nmesh,MainData):

	#######################################################
	nvar = MainData.nvar
	ndim = MainData.ndim

	columns_out = []; AppliedDirichlet = []

	#----------------------------------------------------------------------------------------------------#
	#-------------------------------------- NURBS BASED SOLUTION ----------------------------------------#
	#----------------------------------------------------------------------------------------------------#
	if MainData.BoundaryData.Type == 'nurbs':
		from Core.Supplementary.nurbs.nurbs import Nurbs
		# GET THE NURBS CURVE FROM PROBLEMDATA
		nurbs = MainData.BoundaryData().NURBSParameterisation()
		# IDENTIFIY DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY
		nodesDBC, Dirichlet = Nurbs(nmesh,nurbs,MainData.BoundaryData,MainData.C)
		# print Dirichlet
		# print nodesDBC

		nOfDBCnodes = nodesDBC.shape[0]
		for inode in xrange(nOfDBCnodes):
			for i in xrange(nvar):
				# print inode
				columns_out = np.append(columns_out,nvar*nodesDBC[inode]+i)
				# AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[nodesDBC[inode],i])
				# columns_out = np.append(columns_out,nvar*inode+i)
				# print i,inode,
				AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[inode,i])

		# print columns_out.astype(int)
		# print
		# print AppliedDirichlet.shape, columns_out.shape
		# print AppliedDirichlet, Dirichlet
		# print AppliedDirichlet
		# print np.hstack((nmesh.points[nodesDBC,:],Dirichlet))
		# print np.linalg.norm(nmesh.points[nodesDBC,:],axis=1)
		# print np.linalg.norm(nmesh.points[nodesDBC,:]+Dirichlet,axis=1)
		# print nmesh.points[nodesDBC,:]
		# print nodesDBC
		# print nmesh.points[nodesDBC,:].shape, Dirichlet.shape
		# print nodesDBC.shape
		# for i in range(nmesh.points.shape[0]):
			# if np.allclose(np.sqrt(nmesh.points[i,0]**2+nmesh.points[i,1]**2),1):# or np.allclose(np.sqrt(nmesh.points[i,0]**2+nmesh.points[i,1]**2),5):
				# print i
		# print nodesDBC
		# print np.max(nmesh.elements), nmesh.points.shape[0]
		# import sys; sys.exit(0)
	#----------------------------------------------------------------------------------------------------#
	#------------------------------------- NON-NURBS BASED SOLUTION -------------------------------------#
	#----------------------------------------------------------------------------------------------------#

	elif MainData.BoundaryData.Type == 'straight' or MainData.BoundaryData.Type == 'mixed':
		# IF DIRICHLET BOUNDARY CONDITIONS ARE APPLIED DIRECTLY AT NODES
		if MainData.BoundaryData().DirichArgs.Applied_at == 'node':
			# GET UNIQUE NODES AT THE BOUNDARY
			unique_edge_nodes = []
			if ndim==2:
				unique_edge_nodes = np.unique(nmesh.edges)
			elif ndim==3:
				unique_edge_nodes = np.unique(nmesh.faces)
			# ACTIVATE THIS FOR DEBUGGING ELECTROMECHANICAL PROBLEMS
			# unique_edge_nodes = np.unique(nmesh.elements) 


			MainData.BoundaryData().DirichArgs.points = nmesh.points
			MainData.BoundaryData().DirichArgs.edges = nmesh.edges
			for inode in range(0,unique_edge_nodes.shape[0]):
				coord_node = nmesh.points[unique_edge_nodes[inode]]
				MainData.BoundaryData().DirichArgs.node = coord_node
				MainData.BoundaryData().DirichArgs.inode = unique_edge_nodes[inode]

				Dirichlet = MainData.BoundaryData().DirichletCriterion(MainData.BoundaryData().DirichArgs)

				if type(Dirichlet) is None:
					pass
				else:
					for i in range(nvar):
						# if type(Dirichlet[i]) is list:
						if Dirichlet[i] is None:
							pass
						else:
							# columns_out = np.append(columns_out,nvar*inode+i) # THIS IS INVALID
							# ACTIVATE THIS FOR DEBUGGING ELECTROMECHANICAL PROBLEMS
							columns_out = np.append(columns_out,nvar*unique_edge_nodes[inode]+i)
							AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[i])



	# GENERAL PROCEDURE - GET REDUCED MATRICES FOR FINAL SOLUTION
	columns_out = columns_out.astype(np.int64)
	columns_in = np.delete(np.arange(0,nvar*nmesh.points.shape[0]),columns_out)
	# print columns_in.shape
	# print AppliedDirichlet#,'\n',columns_out
	# print AppliedDirichlet.shape

	for i in range(0,columns_out.shape[0]):
		# F = F - AppliedDirichlet[i]*(stiffness[:,columns_out[i]])
		F = F - AppliedDirichlet[i]*stiffness.getcol(columns_out[i])
	
	stiffness_b = []; F_b1 = [] 
	return stiffness_b, F_b1, F, columns_in, columns_out, AppliedDirichlet






def ApplyIncrementalDirichletBoundaryConditions(stiffness,F,columns_in,columns_out,AppliedDirichlet,Iter,Minimal,nmesh,mass=0,Analysis=0):

	# for i in range(0,columns_out.shape[0]):
	# 	if AppliedDirichlet[i]!=0.0:
	# 		F = F - AppliedDirichlet[i]*(stiffness[:,columns_out[i]])
	# if Iter == 0:
	# 	for i in range(0,columns_out.shape[0]):
	# 		if AppliedDirichlet[i]!=0.0:
	# 			F = F - AppliedDirichlet[i]*(stiffness[:,columns_out[i]])


	# F_b = np.delete(F,columns_out[:])
	F_b = F[columns_in,0]

	# Check F_b's size
	if F_b.shape[0]==1:
		F_b = F_b.reshape(F_b.shape[1],1)

	F_b1 = np.zeros(F_b.shape[0])
	for i in range(0,F_b1.shape[0]):
		F_b1[i]=F_b[i]

	stiffness_b = stiffness[columns_in,:][:,columns_in]

	if Analysis != 'Static':
		mass = mass[columns_in,:][:,columns_in]



	return stiffness_b, F_b1, F, mass



def ApplyLinearDirichletBoundaryConditions(stiffness,F,columns_in,columns_out,AppliedDirichlet,Analysis,mass):

	F_b = F[columns_in,0]

	# Check F_b's size
	if F_b.shape[0]==1:
		F_b = F_b.reshape(F_b.shape[1],1)

	F_b1 = np.zeros(F_b.shape[0])
	for i in range(0,F_b1.shape[0]):
		F_b1[i]=F_b[i]
	if F_b1.shape[0]==1:
		F_b1 = F_b 


	stiffness_b = stiffness[columns_in,:][:,columns_in]

	if Analysis != 'Static':
		mass = mass[columns_in,:][:,columns_in]



	return stiffness_b, F_b1, F, mass