from time import time
import numpy as np
from Core.FiniteElements.Solvers.NewtonRaphsonStatic import *
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *


def StaticSolver(MainData,LoadIncrement,K,DirichletForces,NeumannForces,
	NodalForces,Residual,ResidualNorm,mesh,TotalDisp,
	Eulerx,ColumnsIn,ColumnsOut,AppliedDirichlet):
	
	LoadFactor = 1./LoadIncrement
	AppliedDirichletInc = np.zeros(AppliedDirichlet.shape[0])
	
	for Increment in range(LoadIncrement):

		DeltaF = LoadFactor*NeumannForces
		NodalForces += DeltaF
		# RESIDUAL FORCES CONTAIN CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
		Residual -= (DeltaF + LoadFactor*DirichletForces)
		AppliedDirichletInc += LoadFactor*AppliedDirichlet

		# CALL THE LINEAR/NONLINEAR SOLVER
		if MainData.AnalysisType == 'Nonlinear':
			t_increment = time()

			# LET NORM OF THE FIRST RESIDUAL BE THE NORM WITH RESPECT TO WHICH WE
			# HAVE TO CHECK THE CONVERGENCE OF NEWTON RAPHSON. TYPICALLY THIS IS 
			# NORM OF NODAL FORCES
			if Increment==0:
				MainData.NormForces = np.linalg.norm(Residual[ColumnsOut])

			TotalDisp = NewtonRaphson(MainData,Increment,K,NodalForces,Residual,ResidualNorm,mesh,TotalDisp,Eulerx,
				ColumnsIn,ColumnsOut,AppliedDirichletInc)

			print '\nFinished Load increment', Increment, 'in', time()-t_increment, 'sec'
			try:
				print 'Norm of Residual is', np.abs(la.norm(Residual[ColumnsIn])/MainData.NormForces), '\n'
			except RuntimeWarning:
				print what

			# STORE THE INFORMATION IF NEWTON-RAPHSON FAILS
			if MainData.AssemblyParameters.FailedToConverge == True:
				MainData.solve.condA = np.NAN
				MainData.solve.scaledA = np.NAN
				MainData.solve.scaledAFF = np.NAN
				MainData.solve.scaledAHH = np.NAN
				break

	return TotalDisp