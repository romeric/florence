import os, sys
from time import time
import numpy as np



from Core.FiniteElements.Assembly import *
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *
from Core.FiniteElements.PostProcess import *
# from Core.FiniteElements.StaticCondensationGlobal import *
from Core.FiniteElements.InitiateNonlinearAnalysisData import *
# from Core.FiniteElements.Solvers.DynamicSolver import *
from Core.FiniteElements.Solvers.StaticSolver import *



def MainSolver(MainData,nmesh):

	# INITIATE DATA FOR NON-LINEAR ANALYSIS
	NodalForces, Residual = InitiateNonlinearAnalysisData(MainData,nmesh)
	# SET NON-LINEAR PARAMETERS
	Tolerance = MainData.AssemblyParameters.NRTolerance
	LoadIncrement = MainData.AssemblyParameters.LoadIncrements
	ResidualNorm = { 'Increment_'+str(Increment) : [] for Increment in range(0,LoadIncrement) }
	
	# ALLOCATE FOR GEOMETRY AND SOLUTION FIELDS
	Eulerx = np.copy(nmesh.points)
	TotalDisp = np.zeros((nmesh.points.shape[0],MainData.nvar,LoadIncrement),dtype=np.float64)

	# PRE-ASSEMBLY
	print 'Assembling the system and acquiring neccessary information for the analysis...'
	tAssembly=time()
	# RHS
	# F = AssemblyForces(MainData,nmesh,MainData.Quadrature,Domain,MainData.MaterialArgs,BoundaryData,Boundary)
	# F = AssemblyForces_Cheap(MainData,nmesh,Quadrature,Domain,MainData.MaterialArgs,BoundaryData,Boundary)
	F = np.zeros((nmesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
	# LHS
	M = []
	if MainData.Analysis == 'Static':
		K,TractionForces,_,_ = Assembly(MainData,nmesh,Eulerx,np.zeros((nmesh.points.shape[0],1),dtype=np.float64))
	else:
		K,_,_,M = Assembly(MainData,nmesh,Eulerx,np.zeros((nmesh.points.shape[0],1),dtype=np.float64))

	# APPLY DIRICHLET BOUNDARY CONDITIONS TO GET: columns_in, columns_out, AppliedDirichlet
	_, _, F, columns_in, columns_out, AppliedDirichlet = ApplyDirichletBoundaryConditions(K,F,nmesh,MainData)

	if MainData.AnalysisType=='Nonlinear':
		print 'Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'sec'
	else:
		print 'Finished the assembly stage. Time elapsed was', time()-tAssembly, 'sec'

	if MainData.Analysis != 'Static':
		TotalDisp = DynamicSolver(LoadIncrement,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,
			Eulerx,columns_in,columns_out,AppliedDirichlet,Domain,Boundary,MainData.MaterialArgs)
	else:
		TotalDisp = StaticSolver(LoadIncrement,MainData,K,F,M,NodalForces,Residual,ResidualNorm,nmesh,TotalDisp,
			Eulerx,columns_in,columns_out,AppliedDirichlet)

	MainData.NRConvergence = ResidualNorm

	return TotalDisp
