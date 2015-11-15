from time import time
import numpy as np
import numpy.linalg as la
# from scipy.sparse.linalg import spsolve, cg, cgs, bicg, bicgstab, gmres, lgmres, minres
from scipy.sparse.linalg import spsolve, bicgstab, onenormest 
# from scipy.sparse.linalg import svds, eigsh, eigs, inv as spinv, onenormest
from copy import deepcopy

from Core.FiniteElements.Assembly import *
from Core.FiniteElements.PostProcess import * 
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *



def LinearSolver(MainData,Increment,DirichletForces,NeumannForces,
			NodalForces,Residual,mesh,TotalDisp,Eulerx,
			ColumnsIn,ColumnsOut,AppliedDirichlet,AppliedDirichletInc):

	pass

	# if MainData.MaterialArgs.Type == 'IncrementalLinearElastic':
		# vmesh = deepcopy(mesh)
		# TotalDisp = IncrementalLinearElasticitySolver(MainData,Increment,DirichletForces,NeumannForces,
		# 	NodalForces,Residual,vmesh,TotalDisp,Eulerx,
		# 	ColumnsIn,ColumnsOut,AppliedDirichlet)


	# ASSEMBLE STIFFNESS MATRIX AND TRACTION FORCES
	# K,TractionForces = Assembly(MainData,mesh,Eulerx,TotalDisp[:,MainData.nvar-1,Increment,None])[:2]
	# APPLY DIRICHELT BOUNDARY CONDITIONS AND GET DIRICHLET RELATED FORCES
	# DirichletForces, ColumnsIn, ColumnsOut, AppliedDirichlet = ApplyDirichletBoundaryConditions(K,DirichletForces,mesh,MainData)
	# GET REDUCED MATRICES
	# K_b, F_b = GetReducedMatrices(K,Residual,ColumnsIn,ColumnsOut,AppliedDirichletInc,MainData.Analysis,[])[:2]
	# K_b, F_b = GetReducedMatrices(K,Residual,ColumnsIn,ColumnsOut,AppliedDirichletStep,MainData.Analysis,[])[:2]

	# print -Residual[:6]
	# print K[1:5,1:5]
	# print AppliedDirichletInc[16:22]
	# print TotalDisp[161,:]
	
	# # SOLVE THE SYSTEM
	# t_solver=time()
	# if MainData.solve.type == 'direct':
	# 	# CHECK FOR THE CONDITION NUMBER OF THE SYSTEM
	# 	if Increment==MainData.AssemblyParameters.LoadIncrements-1:
	# 		# MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
	# 		MainData.solve.condA = onenormest(K_b) # REMOVE THIS
	# 	# CALL DIRECT SOLVER
	# 	sol = spsolve(K_b,-F_b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
	# else:
	# 	# CALL ITERATIVE SOLVER
	# 	sol = bicgstab(K_b,-F_b,tol=MainData.solve.tol)[0]
	# print 'Finished solving the system. Time elapsed was', time()-t_solver
	
	# # GET THE TOTAL SOLUTION AND ITS COMPONENTS SUCH AS UX, UY, UZ, PHI ETC
	# # dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 
	# dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletStep,0,K.shape[0]) 

	# # UPDATE THE FIELDS
	# TotalDisp[:,:,Increment] += dU

	# Eulerx = mesh.points + TotalDisp[:,:MainData.ndim,Increment]
	# print AppliedDirichletInc

	# LINEARISED ELASTICITY WITH EITHER OR ALL OF GEOMETRY, STRESS OR HESSIAN UPDATE(S)
	# if MainData.Prestress:
		# THE IF CONDITION IS TO AVOID ASSEMBLING FOR THE INCREMENT WHICH WE DON'T SOLVE FOR
		# if Increment < MainData.AssemblyParameters.LoadIncrements-1:
		# UPDATE THE GEOMETRY
		# Eulerx = mesh.points + TotalDisp[:,:MainData.ndim,Increment]			
		# RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
		# K, TractionForces = Assembly(MainData,mesh,Eulerx,TotalDisp[:,MainData.nvar-1,Increment,None])[:2]
		# DirichletForces = np.zeros_like(DirichletForces)
		# AppliedDirichletInc = np.zeros_like(AppliedDirichletInc)
		# print K[1:5,1:5]
		# print DirichletForces[:6]
		# print TotalDisp[161,:]
		# FIND THE RESIDUAL
		# Residual[ColumnsIn] = TractionForces[ColumnsIn] - NodalForces[ColumnsIn]
		# for i in range(0,ColumnsOut.shape[0]):
			# DirichletForces = DirichletForces - AppliedDirichletStep[i]*K.getcol(ColumnsOut[i])
		# 	DirichletForces = DirichletForces - AppliedDirichletStep[i]*K.getcol(ColumnsOut[i])
		# AppliedDirichletInc = np.zeros_like(AppliedDirichletInc)

			# PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp[:,:,:Increment+1],show_plot=False)



			# AppliedDirichletInc = AppliedDirichlet/LoadIncrement
			# AppliedDirichletInc = Increment*AppliedDirichletInc/MainData.AssemblyParameters.LoadIncrements
			# AppliedDirichletStep
			# DirichletForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
			# K = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[0]
			# for i in range(ColumnsOut.shape[0]):
			# 	DirichletForces = DirichletForces - AppliedDirichletInc[i]*K.getcol(ColumnsOut[i])
			# K_b, F_b = GetReducedMatrices(K,DirichletForces,ColumnsIn,MainData.Analysis,[])[:2]
			# sol = spsolve(K_b,F_b)
			# dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletStep,0,K.shape[0]) 
			# TotalDisp[:,:,Increment] = dU
			# mesh.points += TotalDisp[:,:MainData.ndim,Increment]	
			# Eulerx = np.copy(mesh.points)

		# PostProcess().MeshQualityMeasures(MainData,mesh,np.zeros_like(TotalDisp[:,:,:Increment+1]),show_plot=False)

	# PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp[:,:,:Increment+1],show_plot=False)

	print 'Load increment', Increment, 'for incrementally linearised elastic problem'

	# if Increment ==0:
		# exit(0) 
	# print Increment

	# RETURNING K IS NECESSARY
	return TotalDisp, DirichletForces, AppliedDirichletInc, Residual, NodalForces