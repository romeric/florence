from __future__ import print_function
import gc, os, sys
import numpy as np 
import scipy as sp
import numpy.linalg as la
import scipy.linalg as sla 
from time import time
from copy import deepcopy
from warnings import warn
from time import time


class StructuralDynamicIntegrators(object):
    """docstring for StructuralDynamicIntegrators"""

    def __init__(self):
        super(StructuralDynamicIntegrators, self).__init__()
        self.Alpha_alpha = -0.1     
        self.Alpha_gamma = 0.5
        self.Alpha_delta = 0.
            
    def Alpha(self,K,M,F1,freedof,nstep,dt,napp,alpha,delta,gam):

        # Input/Output Data
        # M - Mass Matrix of the System
        # K - Stiffness Matrix of the System
        # F1 - Vector of Dynamic Nodal Forces
        # freedof - Free Degrees of Freedom
        # napp - Degrees of Freedom at Which Excitation is Applied
        # nstep - No of Time Steps
        # dt - Time Step Size
        # delta, gam and alpha - 3 Integration parameters
        #                     With -1/3=<alpha<=0;  delta = 0.5-gam; alpha =
        #                     0.25*(1-gam)**2

        # Reference:
        # Hilber, H.M, Hughes,T.J.R and Talor, R.L. 
        # "Improved Numerical Dissipation for Time Integration Algorithms in
        # Structural Dynamics" Earthquake Engineering and Structural Dynamics,
        # 5:282-292, 1977.


        # U(n,nstep) - Matrix storing nodal displacements at each step.
        # V(n,nstep) - Matrix storing nodal Velocities at each step.
        # A(n,nstep) - Matrix storing nodal Accelerations at each step.
        # n - is No of DOF's.
        
        # Allocate Space for Vectors and Matrices
        u = np.zeros(K.shape[0]); u = u[freedof]
        u0=u; v = u

        F = np.zeros(K.shape[0]);    F = F[freedof];  F2=F
        M = M[:,freedof][freedof,:]; K = K[:,freedof][freedof,:]
        U = np.zeros((u.shape[0],nstep)); V=U; A=U

        # Initial Calculations
        U[:,0] = u.reshape(u.shape[0])
        A[:,0] =  sla.solve(M,(F - np.dot(K,u0)))
        V[:,0] = v


        # Initialize The Algorithm (Compute Displacements, Vel's and Accel's)
        for istep in range(0,nstep-1):
            if istep==nstep-1:
                break

            # print nstep, istep

            # F[napp] = F1[istep]
            # F2[napp] = F1[istep+1]
            # U[:,istep+1] = sla.solve((1/dt**2/gam*M+(1.0+alpha)*K),((1+alpha)*F2-alpha*F +\
         #        np.dot((1.0/dt**2/gam*M+alpha*K),U[:,istep])+ 1.0/dt/gam*np.dot(M,V[:,istep])+(1/2/gam-1)*np.dot(M,A[:,istep])))
            # A[:,istep+1] = 1.0/dt**2/gam*(U[:,istep+1]-U[:,istep]) - 1.0/dt/gam*V[:,istep]+(1-1.0/2.0/gam)*A[:,istep]
            # V[:,istep+1] = V[:,istep] + dt*(delta*A[:,istep+1]+(1-delta)*A[:,istep])

            F[napp] = F1[istep]
            F2[napp] = F1[istep+1]
            U[:,istep+1] = sla.solve((1/dt**2/gam*M+(1.0+alpha)*K),((1+alpha)*F2-alpha*F +\
                np.dot((1.0/dt**2/gam*M+alpha*K),U[:,istep])+ 1.0/dt/gam*np.dot(M,V[:,istep])+(1/2/gam-1)*np.dot(M,A[:,istep])))
            A[:,istep+1] = 1.0/dt**2/gam*(U[:,istep+1]-U[:,istep]) - 1.0/dt/gam*V[:,istep]+(1-1.0/2.0/gam)*A[:,istep]
            V[:,istep+1] = V[:,istep] + dt*(delta*A[:,istep+1]+(1-delta)*A[:,istep])


        return U, V, A


    def Solver(self, function_spaces, formulation, solver, 
        K, M, NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):
    
        self.NRConvergence = fem_solver.NRConvergence
        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = 1./LoadIncrement
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)
        
        for Increment in range(LoadIncrement):

            t_increment = time()

            # APPLY NEUMANN BOUNDARY CONDITIONS
            DeltaF = LoadFactor*NeumannForces
            NodalForces += DeltaF
            # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
            Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,NodalForces,
                boundary_condition.applied_dirichlet,LoadFactor=LoadFactor, mass=M)[2]
            # GET THE INCREMENTAL DISPLACEMENT
            AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet


            self.NormForces = np.linalg.norm(Residual)

            Eulerx = self.NewtonRaphson(function_spaces, formulation, solver, 
                Increment,K,M,NodalForces,Residual,mesh,Eulerx,Eulerp,
                material,boundary_condition,AppliedDirichletInc, fem_solver)

            # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
            TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
            if formulation.fields == "electro_mechanics":
                TotalDisp[:,-1,Increment] = Eulerp


            print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')

            try:
                print('Norm of Residual is', 
                    np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces), '\n')
            except RuntimeWarning:
                print("Invalid value encountered in norm of Newton-Raphson residual")

            if fem_solver.newton_raphson_failed_to_converge:
                break

        return TotalDisp


    def NewtonRaphson(self, function_spaces, formulation, solver, 
        Increment, K, M, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
        boundary_condition, AppliedDirichletInc, fem_solver):

        Tolerance = fem_solver.newton_raphson_tolerance
        LoadIncrement = fem_solver.number_of_load_increments
        Iter = 0

        # AVOID DIVISION BY ZERO
        if self.NormForces < 1e-14:
            self.NormForces = 1e-14

        # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
        IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
            K.shape[0],formulation.nvar)
        # UPDATE EULERIAN COORDINATE
        Eulerx += IncDirichlet[:,:formulation.ndim]
        Eulerp += IncDirichlet[:,-1]

        while np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces) > Tolerance:
            # GET THE REDUCED SYSTEM OF EQUATIONS
            K_b, F_b, M_b = boundary_condition.GetReducedMatrices(K,Residual,M)

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b,-F_b)

            # GET ITERATIVE SOLUTION
            dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar) 

            # UPDATE THE GEOMETRY
            Eulerx += dU[:,:formulation.ndim]
            Eulerp += dU[:,-1]

            # GET ITERATIVE ELECTRIC POTENTIAL
            # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
            K, TractionForces, _, M = fem_solver.Assemble(function_spaces[0], formulation, mesh, material, solver,
                Eulerx,Eulerp)

            # FIND THE RESIDUAL
            Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] \
            - NodalForces[boundary_condition.columns_in]

            # SAVE THE NORM 
            # NormForces = self.NormForces
            self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
                np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces))
            
            print('Iteration number', Iter, 'for load increment', Increment, 'with a residual of \t\t', \
                np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces)) 

            # UPDATE ITERATION NUMBER
            Iter +=1

            if Iter==fem_solver.maximum_iteration_for_newton_raphson and formulation.fields == "electro_mechanics":
                raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

            if Iter==fem_solver.maximum_iteration_for_newton_raphson:
                fem_solver.newton_raphson_failed_to_converge = True
                break
            if np.isnan(np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces)):
                fem_solver.newton_raphson_failed_to_converge = True
                break


        return Eulerx





# import matplotlib.pyplot as plt

# dyn = StructuralDynamicIntegrators()
# n = 500
# stiffness = np.random.rand(n,n)
# mass = np.random.rand(n,n)
# # mass = np.eye(n,n)
# alpha = 0.2
# delta=0.5
# gamma = 0.4
# freedof = np.arange(0,10)
# nstep = 2*n
# F = np.random.rand(nstep,1)
# napp=8
# dt = 1.0/nstep
# U, A, V = dyn.Alpha(stiffness,mass,F,freedof,nstep,dt,napp,alpha,delta,gamma)

# plt.plot(U[2,:])
# plt.show()