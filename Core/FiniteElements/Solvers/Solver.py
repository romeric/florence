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
	# PURE NEUMANN NODAL FORCE VECTOR
	# F = AssemblyForces(MainData,mesh,MainData.Quadrature,Domain,MainData.MaterialArgs,BoundaryData,Boundary)
	# F = AssemblyForces_Cheap(MainData,mesh,Quadrature,Domain,MainData.MaterialArgs,BoundaryData,Boundary)
	NeumannForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
	# FORCES RESULTING FROM DIRICHLET BOUNDARY CONDITIONS
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
		TotalDisp = StaticSolver(LoadIncrement,MainData,K,F,NeumannForces,M,NodalForces,Residual,ResidualNorm,mesh,TotalDisp,
			Eulerx,columns_in,columns_out,AppliedDirichlet)


	#------------------------------------------------------------------------------------------------------
	# LoadFactor = 1./LoadIncrement
	# F = np.zeros_like(F)
	# AppliedDirichletInc2 = np.zeros_like(AppliedDirichlet)
	# for Increment in range(LoadIncrement):
	# 	AppliedDirichletInc = LoadFactor*AppliedDirichlet
	# 	AppliedDirichletInc2 += LoadFactor*AppliedDirichlet
	# 	# print AppliedDirichletInc
	# 	for i in range(columns_out.shape[0]):
	# 		F = F - AppliedDirichletInc[i]*K.getcol(columns_out[i])
	# 	# Residual = -LoadFactor*F
	# 	Residual = -np.copy(F)
	# 	# print -Residual[161*2-1:161*2+2:2,:]
	# 	# print np.linalg.norm(Residual)
	# 	# print K[0,0]
	# 	K_b, F_b = GetReducedMatrices(K,Residual,columns_in,M,MainData.Analysis)[:2]
	# 	sol = spsolve(K_b,-F_b)
	# 	dU = PostProcess().TotalComponentSol(MainData,sol,columns_in,columns_out,AppliedDirichletInc,0,F.shape[0]) 
	# 	TotalDisp[:,:,Increment] = dU

	# 	print mesh.points[161,:]
	# 	if Increment < MainData.AssemblyParameters.LoadIncrements-1:

	# 		# vmesh = copy.deepcopy(mesh)
	# 		# vmesh.points = mesh.points + TotalDisp[:,:MainData.ndim,Increment]	
	# 		# Eulerx = np.copy(vmesh.points)		
	# 		# # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
	# 		# K, _ = Assembly(MainData,vmesh,Eulerx,np.zeros_like(vmesh.points))[:2]
	# 		# # Get Reduced
	# 		# F = np.zeros_like(F)

	# 		# mesh.points = mesh.points + TotalDisp[:,:MainData.ndim,Increment]
	# 		mesh.points += TotalDisp[:,:MainData.ndim,Increment]	
	# 		Eulerx = np.copy(mesh.points)		
	# 		# RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
	# 		K, _ = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[:2]
	# 		# Get Reduced
	# 		F = np.zeros_like(F)
		
	# 	# print (mesh.points+TotalDisp[:,:,Increment])[161,:]
	# 	print mesh.points[161,:]
	# 	PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp[:,:,:Increment+1],show_plot=False)
	# 	# PostProcess().MeshQualityMeasures(MainData,mesh,np.zeros_like(TotalDisp[:,:,:Increment+1]),show_plot=False)
	# 	# print AppliedDirichletInc
	# 	# print 

	# print (mesh.points+TotalDisp[:,:,-1])[159,:]
	# # print (mesh.points+TotalDisp[:,:,-1])[161,:]
	#---------------------------------------------------------------------------------------------


	# LINEAR SOLVER VERIFIED AND WORKING
	#------------------------------------------------------------------------------------------------------------
	# LoadFactor = 1./LoadIncrement
	# for Increment in range(LoadIncrement):
	# 	AppliedDirichletInc = LoadFactor*AppliedDirichlet
	# 	F = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
	# 	K = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[0]
	# 	for i in range(columns_out.shape[0]):
	# 		F = F - AppliedDirichletInc[i]*K.getcol(columns_out[i])
	# 	K_b, F_b = GetReducedMatrices(K,F,columns_in,M,MainData.Analysis)[:2]
	# 	sol = spsolve(K_b,F_b)
	# 	dU = PostProcess().TotalComponentSol(MainData,sol,columns_in,columns_out,AppliedDirichletInc,0,F.shape[0]) 
	# 	TotalDisp[:,:,Increment] = dU
	# 	mesh.points += TotalDisp[:,:MainData.ndim,Increment]	
	# 	Eulerx = np.copy(mesh.points)

	# 	PostProcess().MeshQualityMeasures(MainData,mesh,np.zeros_like(TotalDisp[:,:,:Increment+1]),show_plot=False)
	#------------------------------------------------------------------------------------------------------------

	# exit(0)
	# UPDATE THE FIELDS
	# TotalDisp[:,:,Increment] += dU


	MainData.NRConvergence = ResidualNorm


	return TotalDisp
