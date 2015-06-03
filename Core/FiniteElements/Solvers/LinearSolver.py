from time import time
import numpy as np
import numpy.linalg as la
from scipy.sparse.linalg import spsolve 

from Core.FiniteElements.PostProcess import * 
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *

def LinearSolver(Increment,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,Eulerx,
			columns_in,columns_out,AppliedDirichletInc):

	# GET THE REDUCED ELEMENTAL MATRICES 
	K_b, F_b, _, _ = ApplyLinearDirichletBoundaryConditions(K,F,columns_in,columns_out,AppliedDirichletInc,MainData.Analysis,M)

	# SOLVE THE SYSTEM
	sol = spsolve(K_b,F_b)

	# GET THE TOTAL SOLUTION AND ITS COMPONENTS SUCH AS UX, UY, UZ, PHI ETC
	dU = PostProcess().TotalComponentSol(MainData,sol,columns_in,columns_out,AppliedDirichletInc,0,F.shape[0]) 

	TotalDisp[:,:,Increment] += dU

	return TotalDisp