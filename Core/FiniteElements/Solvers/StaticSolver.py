from time import time
import numpy as np
from Core.FiniteElements.Solvers.NewtonRaphsonStatic import *
from Core.FiniteElements.Solvers.LinearSolver import *
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *


def StaticSolver(LoadIncrement,MainData,K,F,NeumannForces,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,
	Eulerx,columns_in,columns_out,AppliedDirichlet):

	LoadFactor = 1./LoadIncrement
	AppliedDirichletInc = np.zeros(AppliedDirichlet.shape[0])
	

	for Increment in range(0,LoadIncrement):

		DeltaF = LoadFactor*NeumannForces
		NodalForces += DeltaF
		# RESIDUAL FORCES CONTAIN CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
		# print np.linalg.norm(DeltaF)
		Residual -= (DeltaF + LoadFactor*F)
		AppliedDirichletInc += LoadFactor*AppliedDirichlet

		

		# CALL THE LINEAR/NONLINEAR SOLVER
		if MainData.AnalysisType == 'Nonlinear':
			t_increment = time()

			# LET NORM OF RESIDUAL BE THE NORM WITH RESPECT TO WHICH WE HAVE TO 
			# CHECK THE CONVERGENCE OF NEWTON RAPHSON. TYPICALLY THIS IS NORM OF
			# OF NODAL FORCES
			if Increment==0:
				MainData.NormForces = np.linalg.norm(Residual[columns_out])

			TotalDisp = NewtonRaphson(Increment,MainData,K,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
				columns_in,columns_out,AppliedDirichletInc)

			print '\nFinished Load increment', Increment, 'in', time()-t_increment, 'sec'
			print 'Norm of Residual is', np.abs(la.norm(Residual[columns_in])/MainData.NormForces), '\n'

			# STORE THE INFORMATION IF NEWTON-RAPHSON FAILS
			if MainData.AssemblyParameters.FailedToConverge == True:
				MainData.solve.condA = np.NAN
				MainData.solve.scaledA = np.NAN
				break

		elif MainData.AnalysisType == 'Linear':
			TotalDisp, K, F, AppliedDirichletInc, Residual, NodalForces = LinearSolver(Increment,MainData,K,F,NeumannForces,M,NodalForces,
				Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
				columns_in,columns_out,AppliedDirichletInc)

		# print nmesh.points[6,:]+TotalDisp[6,:,Increment]


	return TotalDisp