from time import time
from copy import deepcopy
import numpy as np
import numpy.linalg as la
# from scipy.sparse.linalg import spsolve, cg, cgs, bicg, bicgstab, gmres, lgmres, minres
from scipy.sparse.linalg import spsolve, bicgstab, onenormest, cg, spilu 
# from scipy.sparse.linalg import svds, eigsh, eigs, inv as spinv, onenormest
from pyamg import *
from pyamg.gallery import *



from Core.FiniteElements.Assembly import *
from Core.FiniteElements.PostProcess import * 
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *


def LinearSolver(MainData,mesh,TotalDisp,
			Eulerx,LoadIncrement,NeumannForces,
			ColumnsIn,ColumnsOut,AppliedDirichlet):

	jacobian_postprocess = PostProcess()

	smesh = deepcopy(mesh)

	LoadFactor = 1./LoadIncrement
	for Increment in range(LoadIncrement):
		# COMPUTE INCREMENTAL FORCES
		NodalForces = LoadFactor*NeumannForces
		AppliedDirichletInc = LoadFactor*AppliedDirichlet
		# DIRICHLET FORCES IS SET TO ZERO EVERY TIME
		DirichletForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
		Residual = DirichletForces + NodalForces
		# ASSEMBLE
		# K = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[0]
		# K, TractionForces = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[:2]
		vmesh = deepcopy(mesh)
		vmesh.points -= TotalDisp[:,:MainData.ndim,Increment-1]
		K, TractionForces = Assembly(MainData,vmesh,Eulerx,np.zeros_like(mesh.points))[:2]

		Residual[ColumnsIn] = TractionForces[ColumnsIn] - NodalForces[ColumnsIn]
		# print TractionForces
		# APPLY DIRICHLET BOUNDARY CONDITIONS & GET REDUCED MATRICES 
		K_b, F_b = ApplyDirichletGetReducedMatrices(K,Residual,ColumnsIn,ColumnsOut,AppliedDirichletInc,MainData.Analysis,[])[:2]

		# SOLVE THE SYSTEM
		sol = spsolve(K_b,F_b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)


		dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 
		# STORE TOTAL SOLUTION DATA
		TotalDisp[:,:,Increment] += dU
		# UPDATE MESH GEOMETRY
		mesh.points += TotalDisp[:,:MainData.ndim,Increment]	
		Eulerx = np.copy(mesh.points)

		# print Eulerx

		# xx = Eulerx.flatten() - smesh.points.flatten()
		# print np.dot(xx.T,xx)


		# COMPUTE SCALED JACBIAN FOR THE MESH
		# if Increment == LoadIncrement - 1:
		jacobian_postprocess.MeshQualityMeasures(MainData,mesh,np.zeros_like(TotalDisp[:,:,:Increment+1]),show_plot=False)


	return TotalDisp


# def LinearSolver(MainData,mesh,TotalDisp,
# 			Eulerx,LoadIncrement,NeumannForces,
# 			ColumnsIn,ColumnsOut,AppliedDirichlet):

# 	jacobian_postprocess = PostProcess()

# 	LoadFactor = 1./LoadIncrement
# 	for Increment in range(LoadIncrement):
# 		# COMPUTE INCREMENTAL FORCES
# 		NodalForces = LoadFactor*NeumannForces
# 		AppliedDirichletInc = LoadFactor*AppliedDirichlet
# 		# print AppliedDirichletInc
# 		# DIRICHLET FORCES IS SET TO ZERO EVERY TIME
# 		# DirichletForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
# 		# Residual = DirichletForces + NodalForces
# 		# Residual = DirichletForces
# 		Residual = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)

# 		# ASSEMBLE
# 		K, TractionForces = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[:2]

# 		# print TractionForces[:]
# 		# print K[:5,:5]
# 		# print np.linalg.norm(TractionForces)
# 		# K = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[0]
# 		# print TractionForces
# 		Residual[ColumnsIn] = TractionForces[ColumnsIn] - NodalForces[ColumnsIn]
# 		# APPLY DIRICHLET BOUNDARY CONDITIONS & GET REDUCED MATRICES 
# 		# print Residual[:5]
# 		# print Residual

# 		# TotalDisp[:,:,Increment] = PostProcess().TotalComponentSolcc(MainData,ColumnsIn,
# 		# 	ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 

		
# 		K_b, F_b = ApplyDirichletGetReducedMatrices(K,Residual,ColumnsIn,ColumnsOut,AppliedDirichletInc,MainData.Analysis,[])[:2]
# 		# print np.linalg.norm(F_b)
# 		# print F_b
		
# 		# SOLVE THE SYSTEM
# 		sol = spsolve(K_b,F_b)
# 		# print sol

# 		TotalDisp[:,:,Increment] += PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,
# 			ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 

# 		# print np.max(np.abs(AppliedDirichletInc)), np.max(np.abs(sol)), np.max(np.abs(AppliedDirichlet))
# 		# exit(0)
# 		# STORE TOTAL SOLUTION DATA
# 		# TotalDisp[:,:,Increment] = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,
# 		# 	ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 
# 		# UPDATE MESH GEOMETRY
# 		# mesh.points += TotalDisp[:,:MainData.ndim,Increment]	
# 		# Eulerx = np.copy(mesh.points)



# 		Eulerx = Eulerx + TotalDisp[:,:MainData.ndim,Increment]	
# 		# Eulerx = mesh.points + TotalDisp[:,:MainData.ndim,Increment]	
# 		# print np.linalg.norm(Eulerx[165,:] - mesh.points[165,:])
# 		# # print np.linalg.norm(TotalDisp[:,:MainData.ndim,Increment])
# 		# xx = Eulerx.flatten() - mesh.points.flatten()
# 		# # print Eulerx.flags
# 		# xx = xx[:,None]
# 		# print np.dot(xx.T,xx)
# 		# print mesh.points
# 		# print Eulerx[6,:] - mesh.points[6,:]
# 		# exit(0)
# 		# print np.concatenate((mesh.points,Eulerx),axis=1)
# 		# print Eulerx - mesh.points
# 		# Eulerx = np.copy(mesh.points)
# 		# print Eulerx
# 		# print TotalDisp[:,:,-1]

# 		# COMPUTE SCALED JACBIAN FOR THE MESH
# 		# if Increment == LoadIncrement - 1:
# 		jacobian_postprocess.MeshQualityMeasures(MainData,mesh,TotalDisp[:,:,:Increment+1],show_plot=False)
# 			# jacobian_postprocess.MeshQualityMeasures(MainData,mesh,np.zeros_like(TotalDisp[:,:,:Increment+1]),show_plot=False)

# 	# TotalDisp[:,:,-1] = Eulerx - mesh.points
# 	# PostProcess.HighOrderPatchPlot(MainData,mesh,TotalDisp)
# 	# import matplotlib.pyplot as plt
# 	# plt.show()

# 	# jacobian_postprocess.is_scaledjacobian_computed
# 	# MainData.isScaledJacobianComputed = True


# 	return TotalDisp





# def LinearSolver(MainData,Increment,K,DirichletForces,NeumannForces,
# 			NodalForces,Residual,mesh,TotalDisp,Eulerx,
# 			ColumnsIn,ColumnsOut,AppliedDirichlet,AppliedDirichletInc):

# 	# print Residual
# 	# GET REDUCED MATRICES
# 	K_b, F_b = GetReducedMatrices(K,Residual,ColumnsIn,MainData.Analysis,[])[:2]	
# 	# SOLVE THE SYSTEM
# 	t_solver=time()
# 	if MainData.solve.type == 'direct':
# 		# CHECK FOR THE CONDITION NUMBER OF THE SYSTEM
# 		if Increment==MainData.AssemblyParameters.LoadIncrements-1:
# 			# MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
# 			MainData.solve.condA = onenormest(K_b) # REMOVE THIS
# 		# CALL DIRECT SOLVER
# 		# sol = spsolve(K_b,-F_b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)

# 		# invest_K = spilu(K_b)
# 		# print dir(invest_K)
# 		# sol = invest_K.solve(-F_b)
# 		# exit(0)
# 		# X0 = np.dot(invest_K)
# 		# sol = cg(K_b,-F_b,tol=MainData.solve.tol,M=invest_K)[0]
# 		# ml = ruge_stuben_solver(K_b)
# 		# sol = ml.solve(-F_b, tol=1e-6)

# 		sol = spsolve(K_b,-F_b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True) 
# 		# print sol 
# 		# exit(0)
# 	else:
# 		# CALL ITERATIVE SOLVER
# 		sol = bicgstab(K_b,-F_b,tol=MainData.solve.tol)[0]
# 	print 'Finished solving the system. Time elapsed was', time()-t_solver
	
# 	# GET THE TOTAL SOLUTION AND ITS COMPONENTS SUCH AS UX, UY, UZ, PHI ETC
# 	dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 

# 	# UPDATE THE FIELDS
# 	TotalDisp[:,:,Increment] += dU

# 	# LINEARISED ELASTICITY WITH EITHER OR ALL OF GEOMETRY, STRESS OR HESSIAN UPDATE(S)
# 	if MainData.Prestress:
# 		# THE IF CONDITION IS TO AVOID ASSEMBLING FOR THE INCREMENT WHICH WE DON'T SOLVE FOR
# 		if Increment < MainData.AssemblyParameters.LoadIncrements-1:
# 			# UPDATE THE GEOMETRY
# 			Eulerx = mesh.points + TotalDisp[:,:MainData.ndim,Increment]			
# 			# RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
# 			K, TractionForces = Assembly(MainData,mesh,Eulerx,TotalDisp[:,MainData.nvar-1,Increment,None])[:2]
# 			# FIND THE RESIDUAL
# 			Residual[ColumnsIn] = TractionForces[ColumnsIn] - NodalForces[ColumnsIn]

# 			# COMPUTE SCALED JACOBIAN
# 			PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp[:,:,:Increment+1],show_plot=False)

# 	print 'Load increment', Increment, 'for incrementally linearised elastic problem'


# 	return TotalDisp