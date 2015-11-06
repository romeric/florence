from time import time
import numpy as np
import numpy.linalg as la
# from scipy.sparse.linalg import spsolve, cg, cgs, bicg, bicgstab, gmres, lgmres, minres
from scipy.sparse.linalg import spsolve, bicgstab 
from scipy.sparse.linalg import svds, eigsh, eigs, inv as spinv, onenormest
from scipy.io import savemat 

from Core.FiniteElements.Assembly import *
from Core.FiniteElements.PostProcess import * 
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *

def LinearSolver(Increment,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
			columns_in,columns_out,AppliedDirichletInc):

	# GET THE REDUCED ELEMENTAL MATRICES 
	# print Residual + F 
	# K_b, F_b, _, _ = ApplyLinearDirichletBoundaryConditions(K,F,columns_in,columns_out,AppliedDirichletInc,MainData.Analysis,M)
	# print np.linalg.norm(Residual)
	K_b, F_b= ApplyLinearDirichletBoundaryConditions(K,Residual,columns_in,columns_out,AppliedDirichletInc,MainData.Analysis,M)[:2]

	
	# SOLVE THE SYSTEM
	t_solver=time()
	if MainData.solve.type == 'direct':
		# CHECK FOR THE CONDITION NUMBER OF THE SYSTEM
		if Increment==MainData.AssemblyParameters.LoadIncrements-1:
			# MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
			# MainData.solve.condA = np.linalg.cond(K_b.toarray()) # REMOVE THIS
			# savemat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/xx.mat',{'Stiffness':K_b})
			# print onenormest(K_b)
			MainData.solve.condA = onenormest(K_b) # REMOVE THIS
		# sol = spsolve(K_b,-F_b)
		sol = spsolve(K_b,-F_b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
		# sol = spsolve(K_b,-F_b,use_umfpack=True)
		# sol = spsolve(K_b,-F_b,permc_spec='MMD_AT_PLUS_A')
		# sol = spsolve(K_b,-F_b)
	else:
		sol = bicgstab(K_b,-F_b,tol=MainData.solve.tol)[0]
	print 'Finished solving the system. Time elapsed was', time()-t_solver
	
	# GET THE TOTAL SOLUTION AND ITS COMPONENTS SUCH AS UX, UY, UZ, PHI ETC
	dU = PostProcess().TotalComponentSol(MainData,sol,columns_in,columns_out,AppliedDirichletInc,0,F.shape[0]) 

	# UPDATE THE FIELDS
	TotalDisp[:,:,Increment] += dU

	if MainData.Prestress:
		# UPDATE THE GEOMETRY
		Eulerx = nmesh.points + TotalDisp[:,:MainData.ndim,Increment]			
		# RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
		K, TractionForces = Assembly(MainData,nmesh,Eulerx,TotalDisp[:,MainData.nvar-1,Increment].reshape(TotalDisp.shape[0],1))[:2]
		# print np.linalg.norm(TractionForces)
		# FIND THE RESIDUAL
		Residual[columns_in] = TractionForces[columns_in] - NodalForces[columns_in]

	print 'Load increment', Increment, 'for incrementally linearised elastic problem'

	return TotalDisp