import numpy as np
from scipy.sparse.linalg import spsolve, bicgstab, onenormest 

from Core.FiniteElements.Assembly import *
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *
from Core.FiniteElements.PostProcess import *


def IncrementalLinearElasticitySolver(MainData,mesh,TotalDisp,Eulerx,LoadIncrement,NeumannForces,
		ColumnsIn,ColumnsOut,AppliedDirichlet):
	"""An icremental linear elasticity solver, in which only the geometry is updated 
		and all the remaining quantities such as stresses and Hessians are evaluated once at 
		the origin. In this approach instead of solving the problem inside a non-linear routine,
		a somewhat explicit and more efficient way is adopted to avoid pre-assembly of the system
		of equations needed for non-linear analysis
	"""

	jacobian_postprocess = PostProcess()

	LoadFactor = 1./LoadIncrement
	for Increment in range(LoadIncrement):
		# COMPUTE INCREMENTAL FORCES
		NodalForces = LoadFactor*NeumannForces
		AppliedDirichletInc = LoadFactor*AppliedDirichlet
		# DIRICHLET FORCES IS SET TO ZERO EVERY TIME
		DirichletForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
		Residual = DirichletForces + NodalForces
		# ASSEMBLE
		K = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[0]
		# APPLY DIRICHLET BOUNDARY CONDITIONS & GET REDUCED MATRICES 
		K_b, F_b = ApplyDirichletGetReducedMatrices(K,Residual,ColumnsIn,ColumnsOut,AppliedDirichletInc,MainData.Analysis,[])[:2]

		# SOLVE THE SYSTEM
		t_solver=time()
		if MainData.solve.type == 'direct':
			# CHECK FOR THE CONDITION NUMBER OF THE SYSTEM
			if Increment==MainData.AssemblyParameters.LoadIncrements-1:
				# MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
				MainData.solve.condA = onenormest(K_b) # REMOVE THIS
			# CALL DIRECT SOLVER
			sol = spsolve(K_b,F_b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
		else:
			# CALL ITERATIVE SOLVER
			sol = bicgstab(K_b,F_b,tol=MainData.solve.tol)[0]
		print 'Finished solving the system. Time elapsed was', time()-t_solver

		dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 
		# STORE TOTAL SOLUTION DATA
		TotalDisp[:,:,Increment] += dU
		# UPDATE MESH GEOMETRY
		mesh.points += TotalDisp[:,:MainData.ndim,Increment]	
		Eulerx = np.copy(mesh.points)

		# COMPUTE SCALED JACBIAN FOR THE MESH
		jacobian_postprocess.MeshQualityMeasures(MainData,mesh,np.zeros_like(TotalDisp[:,:,:Increment+1]),show_plot=False)

	jacobian_postprocess.is_scaledjacobian_computed
	MainData.isScaledJacobianComputed = True


	return TotalDisp