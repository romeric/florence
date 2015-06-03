import numpy as np
from Core.FiniteElements.Solvers.NewtonRaphsonDynamic import *
from Core.FiniteElements.Assembly import *

def DynamicSolver(LoadIncrement,MainData,K,F,M,NodalForces,Residual,ResidualNorm,mesh,nmesh,TotalSol,TotalDisp,
	Eulerx,SolutionComponent,columns_in,columns_out,AppliedDirichlet,Domain,Boundary,Quadrature,MaterialArgs,BoundaryData):

	# LoadFactor = 1./LoadIncrement
	LoadIncrement = BoundaryData.nstep
	LoadFactor = 1./LoadIncrement
	

	# Alpha method parameter
	class AlphaMehtod(object):
		alpha = -0.1
		gamma = 0.25*(1-alpha)**2
		delta = 0.5-alpha

	dt = BoundaryData.dt
	alpha = AlphaMehtod.alpha
	gamma = AlphaMehtod.gamma
	delta = AlphaMehtod.delta

	V = np.zeros((F.shape[0],1)); A = np.zeros((F.shape[0],1))

	# # First step of alpha method - compute initial accelerations
	K_b, F_b, F_dum, M_b = ApplyIncrementalDirichletBoundaryConditions(K,Residual,columns_in,columns_out,AppliedDirichlet,MainData.Minimal,nmesh,M,MainData.Analysis)
	# Perform static condensation to take out electric DoFs out of the system
	K_b, F_b, stiffness_uu, stiffness_up, stiffness_pu, stiffness_pp, inv_stiffness_pp, F_p, F_u, uu, t = StaticCondensationGlobal(K,
		F_dum,MainData.Minimal.nvar,nmesh,columns_in,columns_out)
	# Call again for mass matrix
	M_b = StaticCondensationGlobal(M,F_dum,MainData.Minimal.nvar,nmesh,columns_in,columns_out)[0]
	# A = 
	MainData.DynDisp = np.zeros((F_b.shape[0],LoadIncrement)) 
	MainData.DynVel = np.zeros((F_b.shape[0],LoadIncrement)) 
	MainData.DynAcc = np.zeros((F_b.shape[0],LoadIncrement))

	MainData.DynForce = np.zeros((K.shape[0],LoadIncrement))
	MainData.DynForce_b = np.zeros((F_b.shape[0],LoadIncrement))

	# Reinitiate nodal forces and residual
	NodalForces2 = np.zeros((M_b.shape[0],1))
	Residual2 = np.zeros((M_b.shape[0],1))

	for Increment in range(1,LoadIncrement):

		# DeltaF = []
		# if Increment >0:
		DeltaF = AssemblyForces_Cheap(MainData,mesh,nmesh,Quadrature,Domain,MaterialArgs,BoundaryData,Boundary,Increment)
		# Save
		MainData.DynForce[:,Increment] = DeltaF[:,0] 

		# Apply Dirichlet incremental Dirichlet boundary conditions (on the coupled problem - note that mass matrix only has mechanical DOFS)
		K_b, F_b, F_dum, M_b = ApplyIncrementalDirichletBoundaryConditions(K,Residual,columns_in,columns_out,AppliedDirichlet,MainData.Minimal,nmesh,M,MainData.Analysis)

		# Perform static condensation to take out electric DoFs out of the system
		K_b, F_b, stiffness_uu, stiffness_up, stiffness_pu, stiffness_pp, inv_stiffness_pp, F_p, F_u, uu, t = StaticCondensationGlobal(K,
			F_dum,MainData.Minimal.nvar,nmesh,columns_in,columns_out)
		# Call again for mass matrix
		M_b = StaticCondensationGlobal(M,F_dum,MainData.Minimal.nvar,nmesh,columns_in,columns_out)[0]

		MainData.DynForce_b[:,Increment] = F_b

		# Get the equivalent stiffness matrix for Alpha mehtod
		K_dyn = (1./dt**2/gamma)*M_b + (1+alpha)*K_b
		F_dyn = ((1./dt**2/gamma)*M_b + alpha*K_b).dot(MainData.DynDisp[:,Increment-1])  + (1+alpha)*MainData.DynForce_b[:,Increment] -\
		alpha*MainData.DynForce_b[:,Increment-1] + 1.0/dt/gamma*M_b.dot(MainData.DynVel[:,Increment-1]) + (1.0/2.0/gamma-1.0)*M_b*MainData.DynAcc[:,Increment]

		# # DeltaF = LoadFactor*F
		NodalForces2 += F_dyn.reshape(F_dyn.shape[0],1)
		# Residual -= DeltaF
		Residual2 -= F_dyn.reshape(F_dyn.shape[0],1)
		AppliedDirichlet *= LoadFactor

		
		# Call the iterative routine
		TotalDisp, MainData.DynVel, MainData.DynAcc, TotalSol = NewtonRaphson(Increment,MainData,K,F,M,NodalForces2,Residual2,ResidualNorm,
			mesh,nmesh,TotalSol,TotalDisp,Eulerx,SolutionComponent,columns_in,columns_out,AppliedDirichlet,
			Domain,Boundary,Quadrature,MaterialArgs,BoundaryData,AlphaMehtod,MainData.DynAcc,MainData.DynVel)

		MainData.MainDict['ElectricPotential'][:,:,Increment] = MainData.IncrementalPotential 

		print '\nFinished Load increment', Increment
		print 'Norm of Residual is', np.abs(la.norm(Residual)/la.norm(NodalForces)), '\n'
		
		# sys.exit("STOPPED")

	return TotalDisp, TotalSol 