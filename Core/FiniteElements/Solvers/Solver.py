import os, sys
from time import time
import numpy as np



from Core.FiniteElements.Assembly import *
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *
from Core.FiniteElements.PostProcess import *
# from Core.FiniteElements.StaticCondensationGlobal import *
from Core.FiniteElements.InitiateNonlinearAnalysisData import *
# from Core.FiniteElements.Solvers.DynamicSolver import *
from Core.FiniteElements.Solvers.StaticSolver import *

from Core.FiniteElements.PostProcess import * 

def MainSolver(MainData,mesh):

	# INITIATE DATA FOR NON-LINEAR ANALYSIS
	NodalForces, Residual = InitiateNonlinearAnalysisData(MainData,mesh)
	# SET NON-LINEAR PARAMETERS
	Tolerance = MainData.AssemblyParameters.NRTolerance
	LoadIncrement = MainData.AssemblyParameters.LoadIncrements
	ResidualNorm = { 'Increment_'+str(Increment) : [] for Increment in range(0,LoadIncrement) }
	
	# ALLOCATE FOR GEOMETRY AND SOLUTION FIELDS
	Eulerx = np.copy(mesh.points)
	TotalDisp = np.zeros((mesh.points.shape[0],MainData.nvar,LoadIncrement),dtype=np.float64)

	# PRE-ASSEMBLY
	print 'Assembling the system and acquiring neccessary information for the analysis...'
	tAssembly=time()
	# RHS
	# F = AssemblyForces(MainData,mesh,MainData.Quadrature,Domain,MainData.MaterialArgs,BoundaryData,Boundary)
	# F = AssemblyForces_Cheap(MainData,mesh,Quadrature,Domain,MainData.MaterialArgs,BoundaryData,Boundary)
	F = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
	# LHS
	M = []
	if MainData.Analysis == 'Static':
		K,TractionForces,_,_ = Assembly(MainData,mesh,Eulerx,np.zeros((mesh.points.shape[0],1),dtype=np.float64))
	else:
		K,_,_,M = Assembly(MainData,mesh,Eulerx,np.zeros((mesh.points.shape[0],1),dtype=np.float64))

	# APPLY DIRICHLET BOUNDARY CONDITIONS TO GET: columns_in, columns_out, AppliedDirichlet
	_, _, F, columns_in, columns_out, AppliedDirichlet = ApplyDirichletBoundaryConditions(K,F,mesh,MainData)

	if MainData.AnalysisType=='Nonlinear':
		print 'Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'sec'
	else:
		print 'Finished the assembly stage. Time elapsed was', time()-tAssembly, 'sec'

	if MainData.Analysis != 'Static':
		TotalDisp = DynamicSolver(LoadIncrement,MainData,K,F,M,NodalForces,Residual,ResidualNorm,mesh,TotalDisp,
			Eulerx,columns_in,columns_out,AppliedDirichlet,Domain,Boundary,MainData.MaterialArgs)
	else:
		TotalDisp = StaticSolver(LoadIncrement,MainData,K,F,M,NodalForces,Residual,ResidualNorm,mesh,TotalDisp,
			Eulerx,columns_in,columns_out,AppliedDirichlet)



	# MainData_ = copy.deepcopy(MainData)	
	# LoadFactor = 1./LoadIncrement
	# AppliedDirichletInc = np.zeros(AppliedDirichlet.shape[0])
	# DeltaF = LoadFactor*F
	
	# for Increment in range(0,LoadIncrement):

	# 	# NodalForces += DeltaF
	# 	Residual -= DeltaF
	# 	AppliedDirichletInc += LoadFactor*AppliedDirichlet

	# 	K_b, F_b = ApplyLinearDirichletBoundaryConditions(K,Residual,columns_in,MainData_.Analysis,M)[:2]
	# 	sol = spsolve(K_b,-F_b)
	# 	dU = PostProcess().TotalComponentSol(MainData_,sol,columns_in,columns_out,AppliedDirichletInc,0,F.shape[0]) 
	# 	TotalDisp[:,:,Increment] += dU
	# 	vmesh = copy.deepcopy(mesh)
	# 	vmesh.points = mesh.points + TotalDisp[:,:MainData_.ndim,Increment]
	# 	# print np.linalg.norm(vmesh.points)
	# 	Eulerx = np.copy(vmesh.points) 
	# 	K = Assembly(MainData_,vmesh,Eulerx,np.zeros((mesh.points.shape[0],1),dtype=np.float64))[0]
	# 	# F = np.zeros((mesh.points.shape[0]*MainData_.nvar,1),dtype=np.float64)
	# 	# _, _, F, _, _, AppliedDirichlet = ApplyDirichletBoundaryConditions(K,F,vmesh,MainData_)
	# 	# print AppliedDirichlet - AppliedDirichletInc
	# 	# AppliedDirichlet -= AppliedDirichletInc 

	# 	# xpoints = np.zeros(vmesh.points.shape[0]*2)
	# 	# xpoints[::2] = TotalDisp[:,0,Increment]
	# 	# xpoints[1::2] = TotalDisp[:,1,Increment]
	# 	# ypoints = TotalDisp[:,:,Increment].flatten()
	# 	# AppliedDirichlet -= ypoints[columns_out]
	# 	# for i in range(0,columns_out.shape[0]):
	# 	# 	F = F - AppliedDirichlet[i]*K.getcol(columns_out[i])
	# 	# print AppliedDirichlet

	# 	AppliedDirichlet = ApplyDirichletBoundaryConditions(K,F,vmesh,MainData_)[-1]
	# 	# K2 = copy.deepcopy(K)
	# 	# K2[:,:] = 0
	# 	# F2 = np.zeros_like(F)
	# 	# print F2
	# 	# AppliedDirichlet = ApplyDirichletBoundaryConditions(K2,F2,vmesh,MainData_)[-1]

	# 	# from Core.FiniteElements.DirichletBoundaryDataFromCAD import IGAKitWrapper, PostMeshWrapper
	# 	# # GET DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY FROM CAD
	# 	# if MainData.BoundaryData.RequiresCAD:
	# 	# 	# CALL POSTMESH WRAPPER
	# 	# 	nodesDBC, Dirichlet = PostMeshWrapper(MainData,mesh)
	# 	# else:
	# 	# 	# CALL IGAKIT WRAPPER
	# 	# 	nodesDBC, Dirichlet = IGAKitWrapper(MainData,mesh)
	# 	# if Increment==0:
	# 	# 	# nodesDBC, Dirichlet = PostMeshWrapper(MainData,vmesh)
	# 	# 	MainData.Dirichlet = np.copy(Dirichlet)
	# 	# # else:
	# 	# 	# Dirichlet = MainData.Dirichlet - vmesh.points[nodesDBC[:,0],:]

	# 	# AppliedDirichlet = []
	# 	# nOfDBCnodes = nodesDBC.shape[0]
	# 	# for inode in range(nOfDBCnodes):
	# 	# 	for i in range(MainData.nvar):
	# 	# 		# pass
	# 	# 		AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[inode,i])

	# 	# print Dirichlet
	# 	# print AppliedDirichlet - ypoints[columns_out]
	# 	# AppliedDirichlet + ypoints[columns_out]
	# 	# print Dirichlet 
	# 	# print np.linalg.norm(vmesh.points[MainData.nodesDBC[:,0],:],axis=1)
	# 	# print np.linalg.norm(vmesh.points[413,:])
	# 	# print vmesh.points[23,:]





	MainData.NRConvergence = ResidualNorm


	return TotalDisp
