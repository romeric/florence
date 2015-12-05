from time import time
import numpy as np
import numpy.linalg as la 
# from scipy.sparse.linalg import spsolve, cg, cgs, bicg, bicgstab, gmres, lgmres, minres, lobpcg
from scipy.sparse.linalg import onenormest 
try:
	from scikits.umfpack import spsolve as umfpack_spsolve
	umfpack_solver = True
except ImportError:
	umfpack_solver = False

from SparseSolver import SparseSolver

from Core.FiniteElements.ApplyDirichletBoundaryConditions import *
from Core.FiniteElements.StaticCondensationGlobal import *
from Core.FiniteElements.PostProcess import *
from Core.FiniteElements.Assembly import *


def NewtonRaphson(MainData,Increment,K,NodalForces,Residual,
		ResidualNorm,mesh,TotalDisp,Eulerx,ColumnsIn,ColumnsOut,AppliedDirichletInc):

	Tolerance = MainData.AssemblyParameters.NRTolerance
	LoadIncrement = MainData.AssemblyParameters.LoadIncrements
	Iter = 0

	# NormForces = la.norm(NodalForces[ColumnsIn])
	NormForces = MainData.NormForces

	# AVOID DIVISION BY ZERO
	if la.norm(Residual[ColumnsIn]) < 1e-14:
		NormForces = 1e-14


	while np.abs(la.norm(Residual[ColumnsIn])/NormForces) > Tolerance:
		# APPLY INCREMENTAL DIRICHLET BOUNDARY CONDITIONS
		K_b, F_b = GetReducedMatrices(K,Residual,ColumnsIn,MainData.Analysis,[])[:2]

		# SOLVE THE SYSTEM

		# CHECK FOR THE CONDITION NUMBER OF THE SYSTEM
		if Increment==MainData.AssemblyParameters.LoadIncrements-1 and Iter>1:
			# MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
			MainData.solve.condA = onenormest(K_b) # REMOVE THIS

		# sol = SparseSolver(K_b,F_b,MainData.solve.type,sub_type='MUMPS')
		sol = SparseSolver(K_b,-F_b,MainData.solve.type,sub_type='UMFPACK')

		# GET THE TOTAL SOLUTION AND ITS COMPONENTS SUCH AS UX, UY, UZ, PHI ETC
		dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,Iter,K.shape[0]) 

		# UPDATE THE FIELDS
		TotalDisp[:,:,Increment] += dU
		# UPDATE THE GEOMETRY
		Eulerx = mesh.points + TotalDisp[:,:MainData.ndim,Increment]			
		# UPDATE & SAVE ITERATION NUMBER
		MainData.AssemblyParameters.IterationNumber +=1
		# RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
		K, TractionForces = Assembly(MainData,mesh,Eulerx,TotalDisp[:,MainData.nvar-1,Increment,None])[:2]
		# FIND THE RESIDUAL
		Residual[ColumnsIn] = TractionForces[ColumnsIn] - NodalForces[ColumnsIn]
		# SAVE THE NORM 
		NormForces = MainData.NormForces
		ResidualNorm['Increment_'+str(Increment)] = np.append(ResidualNorm['Increment_'+str(Increment)],\
			np.abs(la.norm(Residual[ColumnsIn])/NormForces))
		
		print 'Iteration number', Iter, 'for load increment', Increment, 'with a residual of \t\t', \
			np.abs(la.norm(Residual[ColumnsIn])/NormForces)			 

		# UPDATE ITERATION NUMBER
		Iter +=1 

		# if Iter==MainData.AssemblyParameters.MaxIter:
			# raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

		if Iter==MainData.AssemblyParameters.MaxIter or ResidualNorm['Increment_'+str(Increment)][-1] > 500:
			MainData.AssemblyParameters.FailedToConverge = True
			break
		if np.isnan(np.abs(la.norm(Residual[ColumnsIn])/NormForces)):
			MainData.AssemblyParameters.FailedToConverge = True
			break


	return TotalDisp





def NewtonRaphsonWithArcLength(Increment,MainData,K,F,M,NodalForces,Residual,
 	ResidualNorm,mesh,TotalDisp,Eulerx,ColumnsIn,ColumnsOut,AppliedDirichletInc):

	raise NotImplementedError('Arc length not implemented yet')