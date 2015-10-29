from time import time
import numpy as np
from Core.FiniteElements.Solvers.NewtonRaphsonStatic import *
from Core.FiniteElements.Solvers.LinearSolver import *
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *


def StaticSolver(LoadIncrement,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,
	Eulerx,columns_in,columns_out,AppliedDirichlet):

	LoadFactor = 1./LoadIncrement
	AppliedDirichletInc = np.zeros(AppliedDirichlet.shape[0])
	
	# MainData.xx = np.zeros(10)
	for Increment in range(0,LoadIncrement):

		# print Increment
		# if Increment==12:
		# 	np.savetxt('/home/roman/Desktop/step11.dat', 
		# 		TotalDisp[:,:,Increment-1],fmt='%10.9f',delimiter=',')

		DeltaF = LoadFactor*F
		NodalForces += DeltaF
		Residual -= DeltaF
		AppliedDirichletInc += LoadFactor*AppliedDirichlet

		# CALL THE ITERATIVE SOLVER
		if MainData.AnalysisType == 'Nonlinear':
			t_increment = time()
			TotalDisp = NewtonRaphson(Increment,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
				columns_in,columns_out,AppliedDirichletInc)

			NormForces = la.norm(NodalForces[columns_in])
			if la.norm(NodalForces[columns_in]) < 1e-14:
				NormForces = 1e-14

			print '\nFinished Load increment', Increment, 'in', time()-t_increment, 'sec'
			print 'Norm of Residual is', np.abs(la.norm(Residual[columns_in])/NormForces), '\n'
			# MainData.xx = np.append(MainData.xx,np.sum(MainData.xx)+Residual[2*107][0])
			# MainData.xx = np.append(MainData.xx,np.sum(MainData.xx)+NodalForces[2*107,:][0])
			# MainData.xx = np.append(MainData.xx,np.sum(MainData.xx)+NodalForces[824,:][0])


		elif MainData.AnalysisType == 'Linear':
			TotalDisp = LinearSolver(Increment,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
				columns_in,columns_out,AppliedDirichletInc)

	# print np.concatenate((TotalDisp[:,:,0],TotalDisp[:,:,1]),axis=1)
	# print Residual[2*107][0], np.sum(MainData.xx,axis=0)
	

		
		# sys.exit("STOPPED")
	return TotalDisp