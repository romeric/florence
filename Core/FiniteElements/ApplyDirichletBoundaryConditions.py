import numpy as np 
from time import time

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
		from_cad = True 
		# from_cad = False

		# MainData.BoundaryData().ProjectionCriteria(nmesh)
		# import sys; sys.exit(0)

		tCAD = time()
		if from_cad == False:
			from Core.Supplementary.nurbs.nurbs import Nurbs
			# GET THE NURBS CURVE FROM PROBLEMDATA
			nurbs = MainData.BoundaryData().NURBSParameterisation()
			# IDENTIFIY DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY
			nodesDBC, Dirichlet = Nurbs(nmesh,nurbs,MainData.BoundaryData,MainData.C)
		else:
			# GET BOUNDARY FEKETE POINTS
			if MainData.ndim == 2:
				from Core.QuadratureRules import GaussLobattoQuadrature
				boundary_fekete = GaussLobattoQuadrature(MainData.C+2)[0]
				# IT IS IMPORTANT TO ENSURE THAT THE DATA IS C-CONITGUOUS
				boundary_fekete = boundary_fekete.copy(order="c")

			elif MainData.ndim == 3:
				from Core.QuadratureRules.FeketePointsTri import FeketePointsTri
				boundary_fekete = FeketePointsTri(MainData.C)

			# import Core.Supplementary.cpp_src.occ_backend.OCCPluginPy as dd 
			# print dir(dd)
			from Core.Supplementary.cpp_src.occ_backend.OCCPluginPy import OCCPluginPy2 as OCCPlugin2
			# print dir(OCC_Interface)
			occ_interface = OCCPlugin2(nmesh.element_type,dimension=MainData.ndim)
			occ_interface.SetMeshElements(nmesh.elements)
			occ_interface.SetMeshPoints(nmesh.points)
			occ_interface.SetMeshEdges(nmesh.edges)
			occ_interface.SetMeshFaces(np.zeros((1,4),dtype=np.uint64))
			occ_interface.SetScale(MainData.BoundaryData.scale)
			occ_interface.SetCondition(MainData.BoundaryData.condition)
			occ_interface.ScaleMesh()
			# occ_interface.InferInterpolationPolynomialDegree();
			occ_interface.SetFeketePoints(boundary_fekete)
			occ_interface.GetBoundaryPointsOrder()
			# READ THE GEOMETRY FROM THE IGES FILE
			occ_interface.ReadIGES(MainData.BoundaryData.IGES_File)
			# EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
			occ_interface.GetGeomVertices()
			occ_interface.GetGeomEdges()
			occ_interface.GetGeomFaces()
			occ_interface.GetGeomPointsOnCorrespondingEdges()
			# FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
			occ_interface.IdentifyCurvesContainingEdges()
			# PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
			occ_interface.ProjectMeshOnCurve("Bisection")
			# FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
			occ_interface.RepairDualProjectedParameters()
			# PERFORM POINT INVERTION FOR THE INTERIOR POINTS
			occ_interface.MeshPointInversionCurve()

			# import sys; sys.exit(0)


			from Core.Supplementary.cpp_src.occ_backend.OCCPluginPy import OCCPluginPy as OCCPlugin
			# print dir(OCC_Interface)
			OCC_Interface = OCCPlugin(dimension=MainData.ndim)
			OCC_Interface.SetMesh(nmesh.points,nmesh.elements,nmesh.edges,np.zeros((1,4),dtype=np.uint64))
			OCC_Interface.SetCADGeometry(MainData.BoundaryData.IGES_File)
			OCC_Interface.SetScale(MainData.BoundaryData.scale)
			OCC_Interface.SetCondition(MainData.BoundaryData.condition)
			OCC_Interface.SetBoundaryFeketePoints(boundary_fekete)
			OCC_Interface.SetProjectionMethod("Bisection")
			nodesDBC, Dirichlet = OCC_Interface.ComputeDirichletBoundaryConditions()
			# print Dirichlet
			# import sys; sys.exit(0)
			# from Core.Supplementary.cpp_src.occ_backend.OCCPluginPy import __ComputeDirichletBoundaryConditions__
			# nodesDBC, Dirichlet = __ComputeDirichletBoundaryConditions__(MainData.BoundaryData.IGES_File, scale,
				# nmesh.points,nmesh.elements,nmesh.edges,np.zeros((1,4),dtype=np.int64),condition,boundary_fekete)
			posUnique = np.unique(nodesDBC,return_index=True)[1]
			nodesDBC, Dirichlet = nodesDBC[posUnique], Dirichlet[posUnique,:];
			# print type(nodesDBC[0])
			# print nodesDBC
			# print nodesDBC.flags
			# print nodesDBC[0]
			# print Dirichlet
			# print nmesh.points
			# import sys; sys.exit(0)
		print 'Finished identifying Dirichlet boundary conditions from CAD geometry. Time taken ', time()-tCAD, 'seconds'

		nOfDBCnodes = nodesDBC.shape[0]
		for inode in xrange(nOfDBCnodes):
			for i in xrange(nvar):
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
