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



def LinearSolver(MainData,Increment,K,DirichletForces,NeumannForces,
			NodalForces,Residual,mesh,TotalDisp,Eulerx,
			ColumnsIn,ColumnsOut,AppliedDirichlet,AppliedDirichletInc):

	# GET REDUCED MATRICES
	K_b, F_b = GetReducedMatrices(K,Residual,ColumnsIn,MainData.Analysis,[])[:2]	
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
	dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 

	# UPDATE THE FIELDS
	TotalDisp[:,:,Increment] += dU

	# LINEARISED ELASTICITY WITH EITHER OR ALL OF GEOMETRY, STRESS OR HESSIAN UPDATE(S)
	if MainData.Prestress:
		# THE IF CONDITION IS TO AVOID ASSEMBLING FOR THE INCREMENT WHICH WE DON'T SOLVE FOR
		if Increment < MainData.AssemblyParameters.LoadIncrements-1:
			# UPDATE THE GEOMETRY
			Eulerx = mesh.points + TotalDisp[:,:MainData.ndim,Increment]			
			# RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
			K, TractionForces = Assembly(MainData,mesh,Eulerx,TotalDisp[:,MainData.nvar-1,Increment,None])[:2]
			# FIND THE RESIDUAL
			Residual[ColumnsIn] = TractionForces[ColumnsIn] - NodalForces[ColumnsIn]

			# COMPUTE SCALED JACOBIAN
			# PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp[:,:,:Increment+1],show_plot=False)

	print 'Load increment', Increment, 'for incrementally linearised elastic problem'


	return TotalDisp