import numpy as np 
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
		from_cad = True 
		# from_cad = False

		## x= MainData.BoundaryData().ProjectionCriteria(mesh)
		## print x.reshape(x.shape[0])
		## import sys; sys.exit(0)

		tCAD = time()
		if from_cad == False:
			from Core.Supplementary.nurbs.nurbs import Nurbs
			# GET THE NURBS CURVE FROM PROBLEMDATA
			nurbs = MainData.BoundaryData().NURBSParameterisation()
			# IDENTIFIY DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY
			nodesDBC, Dirichlet = Nurbs(mesh,nurbs,MainData.BoundaryData,MainData.C)
		else:
			# GET BOUNDARY FEKETE POINTS
			if MainData.ndim == 2:
				from Core.QuadratureRules import GaussLobattoQuadrature
				boundary_fekete = GaussLobattoQuadrature(MainData.C+2)[0]
				# IT IS IMPORTANT TO ENSURE THAT THE DATA IS C-CONITGUOUS
				boundary_fekete = boundary_fekete.copy(order="c")

				from Core import PostMeshCurvePy as PostMeshCurve 
				# print dir(PostMesh) 
				# import sys; sys.exit(0)
				curvilinear_mesh = PostMeshCurve(mesh.element_type,dimension=MainData.ndim)
				curvilinear_mesh.SetMeshElements(mesh.elements)
				curvilinear_mesh.SetMeshPoints(mesh.points)
				curvilinear_mesh.SetMeshEdges(mesh.edges)
				curvilinear_mesh.SetMeshFaces(np.zeros((1,4),dtype=np.uint64))
				curvilinear_mesh.SetScale(MainData.BoundaryData.scale)
				curvilinear_mesh.SetCondition(MainData.BoundaryData.condition)
				curvilinear_mesh.SetProjectionPrecision(1.0e-04)
				curvilinear_mesh.SetProjectionCriteria(MainData.BoundaryData().ProjectionCriteria(mesh))
				curvilinear_mesh.ScaleMesh()
				# curvilinear_mesh.InferInterpolationPolynomialDegree();
				curvilinear_mesh.SetFeketePoints(boundary_fekete)
				curvilinear_mesh.GetBoundaryPointsOrder()
				# READ THE GEOMETRY FROM THE IGES FILE
				curvilinear_mesh.ReadIGES(MainData.BoundaryData.IGES_File)
				# EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
				curvilinear_mesh.GetGeomVertices()
				curvilinear_mesh.GetGeomEdges()
				curvilinear_mesh.GetGeomFaces()
				curvilinear_mesh.GetGeomPointsOnCorrespondingEdges()
				# FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
				curvilinear_mesh.IdentifyCurvesContainingEdges()
				# PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
				curvilinear_mesh.ProjectMeshOnCurve()
				# FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
				curvilinear_mesh.RepairDualProjectedParameters()
				# PERFORM POINT INVERTION FOR THE INTERIOR POINTS
				curvilinear_mesh.MeshPointInversionCurve()
				# GET DIRICHLET DATA
				nodesDBC, Dirichlet = curvilinear_mesh.GetDirichletData() 
				# FIND UNIQUE VALUES OF DIRICHLET DATA
				posUnique = np.unique(nodesDBC,return_index=True)[1]
				nodesDBC, Dirichlet = nodesDBC[posUnique], Dirichlet[posUnique,:]
				# print Dirichlet
				# print nodesDBC
				# import sys; sys.exit(0)

			elif MainData.ndim == 3:
				from Core.QuadratureRules.FeketePointsTri import FeketePointsTri
				boundary_fekete = FeketePointsTri(MainData.C)

				from Core import PostMeshSurfacePy as PostMeshSurface 
				# print dir(PostMesh) 
				# import sys; sys.exit(0)
				curvilinear_mesh = PostMeshSurface(mesh.element_type,dimension=MainData.ndim)
				curvilinear_mesh.SetMeshElements(mesh.elements)
				curvilinear_mesh.SetMeshPoints(mesh.points)
				if mesh.edges.ndim == 2 and mesh.edges.shape[1]==0:
					mesh.edges = np.zeros((1,4),dtype=np.uint64)
				else:
					curvilinear_mesh.SetMeshEdges(mesh.edges)
				curvilinear_mesh.SetMeshFaces(mesh.faces)
				curvilinear_mesh.SetScale(MainData.BoundaryData.scale)
				curvilinear_mesh.SetCondition(MainData.BoundaryData.condition)
				curvilinear_mesh.SetProjectionPrecision(1.0e-04)
				curvilinear_mesh.SetProjectionCriteria(MainData.BoundaryData().ProjectionCriteria(mesh))
				curvilinear_mesh.ScaleMesh()
				curvilinear_mesh.SetFeketePoints(boundary_fekete)
				# curvilinear_mesh.GetBoundaryPointsOrder()
				# READ THE GEOMETRY FROM THE IGES FILE
				curvilinear_mesh.ReadIGES(MainData.BoundaryData.IGES_File)
				# EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
				curvilinear_mesh.GetGeomVertices()
				# curvilinear_mesh.GetGeomEdges()
				curvilinear_mesh.GetGeomFaces()
				curvilinear_mesh.GetGeomPointsOnCorrespondingFaces()
				# FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
				curvilinear_mesh.IdentifySurfacesContainingFaces()
				# PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
				curvilinear_mesh.ProjectMeshOnSurface()
				# FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
				# curvilinear_mesh.RepairDualProjectedParameters()
				# PERFORM POINT INVERTION FOR THE INTERIOR POINTS
				curvilinear_mesh.MeshPointInversionSurface()
				# GET DIRICHLET DATA
				nodesDBC, Dirichlet = curvilinear_mesh.GetDirichletData() 
				# FIND UNIQUE VALUES OF DIRICHLET DATA
				posUnique = np.unique(nodesDBC,return_index=True)[1]
				nodesDBC, Dirichlet = nodesDBC[posUnique], Dirichlet[posUnique,:]

				# from Core.Supplementary.Tensors import makezero
				# print makezero(Dirichlet)
				# print nodesDBC
				# import sys; sys.exit(0)


		print 'Finished identifying Dirichlet boundary conditions from CAD geometry. Time taken ', time()-tCAD, 'seconds'

		nOfDBCnodes = nodesDBC.shape[0]
		for inode in range(nOfDBCnodes):
			for i in range(nvar):
				columns_out = np.append(columns_out,nvar*nodesDBC[inode]+i)
				AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[inode,i])

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
	columns_in = np.delete(np.arange(0,nvar*mesh.points.shape[0]),columns_out)
	# print columns_in.shape, columns_out.shape
	# print AppliedDirichlet#,'\n',columns_out
	# print AppliedDirichlet.shape
	# import sys; sys.exit(0)

	for i in range(0,columns_out.shape[0]):
		# F = F - AppliedDirichlet[i]*(stiffness[:,columns_out[i]])
		F = F - AppliedDirichlet[i]*stiffness.getcol(columns_out[i])
	
	stiffness_b = []; F_b1 = [] 
	return stiffness_b, F_b1, F, columns_in, columns_out, AppliedDirichlet






def ApplyIncrementalDirichletBoundaryConditions(stiffness,F,columns_in,columns_out,AppliedDirichlet,Iter,Minimal,mesh,mass=0,Analysis=0):

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
