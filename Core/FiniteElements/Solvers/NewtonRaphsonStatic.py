from time import time
import numpy as np
import numpy.linalg as la 
# from scipy.sparse.linalg import spsolve, cg, cgs, bicg, bicgstab, gmres, lgmres, minres
from scipy.sparse.linalg import spsolve, bicgstab 
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *
from Core.FiniteElements.StaticCondensationGlobal import *
from Core.FiniteElements.PostProcess import *
from Core.FiniteElements.Assembly import *


def NewtonRaphson(Increment,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,columns_in,columns_out,AppliedDirichletInc):

	Tolerance = MainData.AssemblyParameters.NRTolerance
	LoadIncrement = MainData.AssemblyParameters.LoadIncrements
	Iter = 0

	NormForces = la.norm(NodalForces[columns_in])
	if la.norm(NodalForces[columns_in]) < 1e-14:
		NormForces = 1e-14

	while np.abs(la.norm(Residual[columns_in])/NormForces) > Tolerance:
		# APPLY INCREMENTAL DIRICHLET BOUNDARY CONDITIONS
		K_b, F_b= ApplyIncrementalDirichletBoundaryConditions(K,Residual,columns_in,columns_out,AppliedDirichletInc,Iter,MainData.Minimal,nmesh,M,MainData.Analysis)[:2]

		# SOLVE THE SYSTEM
		if MainData.solve.type == 'direct':
			# CHECK FOR THE CONDITION NUMBER OF THE SYSTEM
			# MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
			sol = spsolve(K_b,-F_b)
		else:
			sol = bicgstab(K_b,-F_b,tol=MainData.solve.tol)[0]

		# GET THE TOTAL SOLUTION AND ITS COMPONENTS SUCH AS UX, UY, UZ, PHI ETC
		dU = PostProcess().TotalComponentSol(MainData,sol,columns_in,columns_out,AppliedDirichletInc,Iter,F.shape[0]) 

		# UPDATE THE FIELDS
		TotalDisp[:,:,Increment] += dU
		# UPDATE THE GEOMETRY
		Eulerx = nmesh.points + TotalDisp[:,:MainData.ndim,Increment]			
		# UPDATE & SAVE ITERATION NUMBER
		MainData.AssemblyParameters.IterationNumber +=1
		# RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
		K, TractionForces = Assembly(MainData,nmesh,Eulerx,TotalDisp[:,MainData.nvar-1,Increment].reshape(TotalDisp.shape[0],1))[:2]
		# print np.concatenate((Residual[columns_in],NodalForces[columns_in]),axis=1)
		# FIND THE RESIDUAL
		Residual[columns_in] = TractionForces[columns_in] - NodalForces[columns_in]
		# SAVE THE NORM 
		NormForces = la.norm(NodalForces[columns_in])
		ResidualNorm['Increment_'+str(Increment)] = np.append(ResidualNorm['Increment_'+str(Increment)],np.abs(la.norm(Residual[columns_in])/NormForces))
		
		print 'Iteration number', Iter, 'for load increment', Increment, 'with a residual of \t\t', np.abs(la.norm(Residual[columns_in])/NormForces)
		# sys.exit("STOPPED")
		# UPDATE ITERATION NUMBER
		Iter +=1 

		if Iter==MainData.AssemblyParameters.MaxIter:
			sys.exit("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

		# if Iter==1:
			# sys.exit("STOPPED")


	return TotalDisp