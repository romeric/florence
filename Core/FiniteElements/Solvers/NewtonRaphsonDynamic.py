from time import time
import numpy as np
import numpy.linalg as la
from scipy.sparse.linalg import spsolve  
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *
from Core.FiniteElements.StaticCondensationGlobal import *
from Core.FiniteElements.PostProcess import *
from Core.FiniteElements.Assembly import *


def NewtonRaphson(Increment,MainData,K,F,M,NodalForces,Residual,ResidualNorm,mesh,nmesh,TotalSol,TotalDisp,Eulerx,SolutionComponent,
	columns_in,columns_out,AppliedDirichlet,Domain,Boundary,Quadrature,MaterialArgs,BoundaryData,AlphaMehtod,DynAcc,DynVel):

	Tolerance = MainData.AssemblyParameters.NRTolerance
	LoadIncrement = MainData.AssemblyParameters.LoadIncrements
	# alpha = AlphaMehtod.alpha
	# gamma = AlphaMehtod.gamma
	# delta = AlphaMehtod.delta

	

	Iter = 0
	print np.abs(la.norm(Residual))
	while np.abs(la.norm(Residual)/la.norm(NodalForces)) > Tolerance:

		print 'Iteration number ', Iter, 'for load increment', Increment,'and residual is \t\t', np.abs(la.norm(Residual)/la.norm(NodalForces)) 

		# Apply Dirichlet incremental Dirichlet boundary conditions (on the coupled problem - note that mass matrix only has mechanical DOFS)
		K_b, F_b, F_dum, M_b = ApplyIncrementalDirichletBoundaryConditions(K,Residual,columns_in,columns_out,AppliedDirichlet,MainData.Minimal,nmesh,M,MainData.Analysis)

		# Perform static condensation to take out electric DoFs out of the system
		K_b, F_b, stiffness_uu, stiffness_up, stiffness_pu, stiffness_pp, inv_stiffness_pp, F_p, F_u, uu, t = StaticCondensationGlobal(K,
			F_dum,MainData.Minimal.nvar,nmesh,columns_in,columns_out)
		# Call again for mass matrix
		M_b = StaticCondensationGlobal(M,F_dum,MainData.Minimal.nvar,nmesh,columns_in,columns_out)[0]

		# Get the equivalent stiffness matrix for Alpha mehtod
		K_dyn = (1./dt**2/gamma)*M_b + (1+alpha)*K_b
		F_dyn = ((1./dt**2/gamma)*M_b + alpha*K_b).dot(MainData.DynDisp[:,Increment-1])  + (1+alpha)*MainData.DynForce_b[:,Increment] -\
		alpha*MainData.DynForce_b[:,Increment-1] + 1.0/dt/gamma*M_b.dot(MainData.DynVel[:,Increment-1]) + (1.0/2.0/gamma-1.0)*M_b*MainData.DynAcc[:,Increment]

		# Solve for only mechanical field
		# sol_mech = spsolve(K_b,-F_b)
		sol_mech = spsolve(K_dyn,-F_dyn)
		# print sol

		sol = StaticCondensationGlobalPost(sol_mech,uu,t,F.shape[0], stiffness_up, stiffness_pu, stiffness_pp, inv_stiffness_pp, F_p, F_u)

		# Get Total solution
		TotalSol[:,:,Increment], SolutionComponent, dU = PostProcess().TotalComponentSol(sol,columns_in,columns_out,AppliedDirichlet,MainData.nvar,
			MainData.MaterialModelName,F.shape[0],MainData.ndim,SolutionComponent,MainData.Analysis) 

		# Update the fields
		TotalDisp[:,:,Increment] +=  dU
		MainData.IncrementalPotential = MainData.IncrementalPotential + SolutionComponent.ElectricPotential 
		# Update Geometry
		Eulerx = nmesh.points + TotalDisp[:,:,Increment]			
		# 
		MainData.AssemblyParameters.IterationNumber +=1
		# Re-assemble - compute internal traction froces
		# print 'Assembling and computing traction forces...'
		t1=time()
		K, TractionForces, _, _ = Assembly(MainData,mesh,nmesh,Eulerx,Quadrature,Domain,MaterialArgs,BoundaryData,SolutionComponent,Boundary)
		# print 'Finished assembly and computation of traction forces. Time taken', time()-t1, 'sec' 

		# Find Residual
		# print np.concatenate((TractionForces[columns_in],NodalForces[columns_in]),axis=1)
		Residual[columns_in] = TractionForces[columns_in] - NodalForces[columns_in]
		# Residual[columns_in1] = TractionForces[columns_in1] - NodalForces[columns_in1]
		# print TractionForces[columns_in]
		# print NodalForces[columns_in]
		# print Residual[columns_in]

		# Save the norm 
		ResidualNorm['Increment_'+str(Increment)] = np.append(ResidualNorm['Increment_'+str(Increment)],np.abs(la.norm(Residual[columns_in])/la.norm(NodalForces[columns_in])))

		Iter +=1 
		# sys.exit("STOPPED")
		if Iter==MainData.AssemblyParameters.MaxIter:
			sys.exit("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

		# if Iter==1:
			# sys.exit("STOPPED")

	
	return TotalDisp, DynVel, DynAcc, TotalSol