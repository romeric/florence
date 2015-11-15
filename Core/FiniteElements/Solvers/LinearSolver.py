from time import time
import numpy as np
import numpy.linalg as la
# from scipy.sparse.linalg import spsolve, cg, cgs, bicg, bicgstab, gmres, lgmres, minres
from scipy.sparse.linalg import spsolve, bicgstab 
from scipy.sparse.linalg import svds, eigsh, eigs, inv as spinv, onenormest
import copy

from Core.FiniteElements.Assembly import *
from Core.FiniteElements.PostProcess import * 
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *

def LinearSolver(Increment,MainData,K,F,NeumannForces,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
			columns_in,columns_out,AppliedDirichletInc):

	# GET THE REDUCED ELEMENTAL MATRICES 
	# K_b, F_b, _, _ = ApplyLinearDirichletBoundaryConditions(K,F,columns_in,columns_out,AppliedDirichletInc,MainData.Analysis,M)
	K_b, F_b = GetReducedMatrices(K,Residual,columns_in,M,MainData.Analysis)[:2]
	
	# SOLVE THE SYSTEM
	t_solver=time()
	if MainData.solve.type == 'direct':
		# CHECK FOR THE CONDITION NUMBER OF THE SYSTEM
		if Increment==MainData.AssemblyParameters.LoadIncrements-1:
			# MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
			MainData.solve.condA = onenormest(K_b) # REMOVE THIS
		# CALL DIRECT SOLVER
		sol = spsolve(K_b,-F_b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
	else:
		# CALL ITERATIVE SOLVER
		sol = bicgstab(K_b,-F_b,tol=MainData.solve.tol)[0]
	print 'Finished solving the system. Time elapsed was', time()-t_solver
	
	# GET THE TOTAL SOLUTION AND ITS COMPONENTS SUCH AS UX, UY, UZ, PHI ETC
	dU = PostProcess().TotalComponentSol(MainData,sol,columns_in,columns_out,AppliedDirichletInc,0,F.shape[0]) 

	# UPDATE THE FIELDS
	TotalDisp[:,:,Increment] += dU

	print AppliedDirichletInc

	# LINEARISED ELASTICITY WITH STRESS AND HESSIAN UPDATE
	if MainData.Prestress:
		# THE IF CONDITION IS TO AVOID ASSEMBLING FOR THE INCREMENT WHICH WE DON'T SOLVE FOR
		if Increment < MainData.AssemblyParameters.LoadIncrements-1:
			# UPDATE THE GEOMETRY
			Eulerx = nmesh.points + TotalDisp[:,:MainData.ndim,Increment]			
			# RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
			K, TractionForces = Assembly(MainData,nmesh,Eulerx,TotalDisp[:,MainData.nvar-1,Increment].reshape(TotalDisp.shape[0],1))[:2]
			# print np.linalg.norm(TractionForces)

			F = np.zeros_like(F)
			Residual = np.zeros_like(Residual)
			NodalForces = np.zeros_like(NodalForces)

			# vmesh = copy.deepcopy(nmesh)
			# vmesh.points = np.copy(Eulerx)
			# F = ApplyDirichletBoundaryConditions(K,F,vmesh,MainData)[2]
			

			# FIND THE RESIDUAL
			# Residual[columns_in] = TractionForces[columns_in] - NodalForces[columns_in]
			for i in range(0,columns_out.shape[0]):
				F = F - AppliedDirichletInc[i]*K.getcol(columns_out[i])
			AppliedDirichletInc = np.zeros_like(AppliedDirichletInc)

			PostProcess().MeshQualityMeasures(MainData,nmesh,TotalDisp[:,:,:Increment+1],show_plot=False)

	print 'Load increment', Increment, 'for incrementally linearised elastic problem'

	# RETURNING K IS NECESSARY
	return TotalDisp, K, F, AppliedDirichletInc, Residual, NodalForces