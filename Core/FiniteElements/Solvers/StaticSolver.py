from time import time
import numpy as np
from Core.FiniteElements.Solvers.NewtonRaphsonStatic import *
from Core.FiniteElements.Solvers.LinearSolver import *
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *


def StaticSolver(LoadIncrement,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,
	Eulerx,columns_in,columns_out,AppliedDirichlet):

	LoadFactor = 1./LoadIncrement
	AppliedDirichletInc = np.zeros(AppliedDirichlet.shape[0])
	
	for Increment in range(0,LoadIncrement):

		DeltaF = LoadFactor*F
		NodalForces += DeltaF
		Residual -= DeltaF
		AppliedDirichletInc += LoadFactor*AppliedDirichlet

		# CALL THE LINEAR/NONLINEAR SOLVER
		if MainData.AnalysisType == 'Nonlinear':
			t_increment = time()
			TotalDisp = NewtonRaphson(Increment,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
				columns_in,columns_out,AppliedDirichletInc)

			NormForces = la.norm(NodalForces[columns_in])
			if la.norm(NodalForces[columns_in]) < 1e-14:
				NormForces = 1e-14

			print '\nFinished Load increment', Increment, 'in', time()-t_increment, 'sec'
			print 'Norm of Residual is', np.abs(la.norm(Residual[columns_in])/NormForces), '\n'

			if MainData.AssemblyParameters.FailedToConverge == True:
				MainData.solve.condA = np.NAN
				MainData.solve.scaledA = np.NAN
				break

		elif MainData.AnalysisType == 'Linear':
			TotalDisp, K = LinearSolver(Increment,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
				columns_in,columns_out,AppliedDirichletInc)


	return TotalDisp