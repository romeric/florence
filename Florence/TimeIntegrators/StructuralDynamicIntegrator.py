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

from Florence.FiniteElements.Assembly import Assemble
from Florence import Mesh

__all__ = ["StructuralDynamicIntegrators"]


class StructuralDynamicIntegrators(object):
    """docstring for StructuralDynamicIntegrators"""

    def __init__(self):
        super(StructuralDynamicIntegrators, self).__init__()
        self.Alpha_alpha = -0.1     
        self.Alpha_gamma = 0.5
        self.Alpha_delta = 0.


        self.alpha_f = 1.0
        self.alpha_m = 1.0
        self.gamma   = 0.5
        self.beta    = 0.25


    def GetBoundaryInfo(self, mesh, formulation, boundary_condition):


        all_dofs = np.arange(mesh.points.shape[0]*formulation.nvar)
        self.electric_dofs = all_dofs[formulation.nvar-1::formulation.nvar]
        self.mechanical_dofs = np.array([],dtype=np.int64)
        self.mechanical_dofs = np.setdiff1d(all_dofs,self.electric_dofs)

        # # GET BOUNDARY CONDITON FOR THE REDUCED MECHANICAL SYSTEM
        # self.columns_in_mech = np.intersect1d(boundary_condition.columns_in,self.mechanical_dofs)
        # self.columns_in_mech_idx = np.in1d(self.mechanical_dofs,boundary_condition.columns_in)

        # # GET BOUNDARY CONDITON FOR THE REDUCED ELECTROSTATIC SYSTEM
        # self.columns_in_electric = np.intersect1d(boundary_condition.columns_in,self.electric_dofs)
        # self.columns_in_electric_idx = np.in1d(self.electric_dofs,boundary_condition.columns_in)


        # # GET FREE MECHANICAL DOFs
        # self.columns_out_mech = np.intersect1d(boundary_condition.columns_out,self.mechanical_dofs)
        # self.columns_out_mech_idx = np.in1d(self.mechanical_dofs,boundary_condition.columns_out)

        # # GET FREE ELECTROSTATIC DOFs
        # self.columns_out_electric = np.intersect1d(boundary_condition.columns_out,self.electric_dofs)
        # self.columns_out_electric_idx = np.in1d(self.electric_dofs,boundary_condition.columns_out)


        # self.applied_dirichlet_mech = boundary_condition.applied_dirichlet[np.in1d(boundary_condition.columns_out,self.columns_out_mech)]
        # self.applied_dirichlet_electric = boundary_condition.applied_dirichlet[np.in1d(boundary_condition.columns_out,self.columns_out_electric)]
        
        # # MAPPED QUANTITIES
        # out_idx = np.in1d(all_dofs,boundary_condition.columns_out)
        # idx_electric = all_dofs[formulation.nvar-1::formulation.nvar]
        # idx_mech = np.setdiff1d(all_dofs,idx_electric)

        # self.all_electric_dofs = np.arange(mesh.points.shape[0])
        # self.electric_out = self.all_electric_dofs[out_idx[idx_electric]]
        # self.electric_in = np.setdiff1d(self.all_electric_dofs,self.electric_out)

        # self.all_mech_dofs = np.arange(mesh.points.shape[0]*formulation.ndim)
        # self.mech_out = self.all_mech_dofs[out_idx[idx_mech]]
        # self.mech_in = np.setdiff1d(self.all_mech_dofs,self.mech_out)


    def Solver(self, function_spaces, formulation, solver, 
        K, M, NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):

        # GET BOUNDARY CONDITIONS INFROMATION
        if formulation.fields == "electro_mechanics":
            self.GetBoundaryInfo(mesh, formulation,boundary_condition)
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]

        # accelerations  = np.zeros_like(TotalDisp)
        velocities     = np.zeros((mesh.points.shape[0],formulation.ndim,fem_solver.number_of_load_increments))
        accelerations  = np.zeros((mesh.points.shape[0],formulation.ndim,fem_solver.number_of_load_increments))

        Res = Residual - NeumannForces[:,0][:,None]
        if formulation.fields == "electro_mechanics":
            accelerations[:,:,0] = solver.Solve(M_mech, -Res[self.mechanical_dofs].ravel() ).reshape(mesh.points.shape[0],formulation.ndim)
        else:
            accelerations[:,:,0] = solver.Solve(M, -Res.ravel() ).reshape(mesh.points.shape[0],formulation.ndim)
        # print(np.linalg.norm(accelerations[:,:,0]))
        # print(np.linalg.norm(NeumannForces[:,0]))
        # exit()
    
        self.NRConvergence = fem_solver.NRConvergence
        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = 1./LoadIncrement
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)
        # print(np.isnan(AppliedDirichletInc).any())
        # exit()
        # print(NeumannForces.shape)
        # exit()
        if NeumannForces.ndim == 2 and NeumannForces.shape[1]==1:
            tmp = np.zeros((NeumannForces.shape[0],LoadIncrement))
            tmp[:,0] = NeumannForces[:,0]
            NeumannForces = tmp
        
        for Increment in range(LoadIncrement):

            t_increment = time()

            # APPLY NEUMANN BOUNDARY CONDITIONS
            # DeltaF = LoadFactor*NeumannForces
            DeltaF = NeumannForces[:,Increment][:,None]
            # NodalForces += DeltaF
            NodalForces = DeltaF
            # print(NodalForces.shape)

            # print(boundary_condition.applied_dirichlet)
            # exit()
            # K += (1./self.beta/LoadFactor**2)*M

            # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
            # Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
            #     boundary_condition.applied_dirichlet[:,Increment],LoadFactor=LoadFactor,mass=M,only_residual=True)
            Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                boundary_condition.applied_dirichlet[:,Increment],LoadFactor=1.0,mass=M,only_residual=True)
            Residual -= DeltaF
            # Residual = -DeltaF
            # print(Residual)
            # GET THE INCREMENTAL DISPLACEMENT
            # AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet[:,Increment]
            AppliedDirichletInc = boundary_condition.applied_dirichlet[:,Increment]


            # COMPUTE INITIAL ACCELERATION
            # print(solver.Solve(M,Residual.ravel() - K.dot(TotalDisp[:,:,Increment].ravel())).shape)
            # print(K.dot(TotalDisp[:,:,Increment].ravel()).shape)
            # print(Residual.shape)
            # exit()
            # np.arange(0,100)
            # M_mech = 
            # accelerations[:,:,Increment] = solver.Solve(M, Residual.ravel() - \
            #     K.dot(TotalDisp[:,:,Increment].ravel())).reshape(mesh.points.shape[0],formulation.nvar)
            
            # exit()


            # LET NORM OF THE FIRST RESIDUAL BE THE NORM WITH RESPECT TO WHICH WE
            # HAVE TO CHECK THE CONVERGENCE OF NEWTON RAPHSON. TYPICALLY THIS IS 
            # NORM OF NODAL FORCES
            if Increment==0:
                self.NormForces = np.linalg.norm(Residual)
                # AVOID DIVISION BY ZERO
                if np.isclose(self.NormForces,0.0):
                    self.NormForces = 1e-14

            self.norm_residual = np.linalg.norm(Residual)/self.NormForces

            Eulerx, Eulerp, K, Residual, velocities, accelerations = self.NewtonRaphson(function_spaces, formulation, solver, 
                Increment, K, M, NodalForces, Residual, mesh, Eulerx, Eulerp,
                material,boundary_condition,AppliedDirichletInc, fem_solver, velocities, accelerations)

            # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
            TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
            if formulation.fields == "electro_mechanics":
                TotalDisp[:,-1,Increment] = Eulerp

            if fem_solver.compute_energy_dissipation:
                energy_dissipation = self.ComputeEnergyDissipation(function_spaces[0],mesh,material,formulation,fem_solver, 
                    Eulerx, TotalDisp, NeumannForces, M, velocities, Increment)
                fem_solver.energy_dissipation.append(energy_dissipation)

            # PRINT LOG IF ASKED FOR
            if fem_solver.print_incremental_log:
                dmesh = Mesh()
                dmesh.points = TotalDisp[:,:formulation.ndim,Increment]
                dmesh_bounds = dmesh.Bounds
                if formulation.fields == "electro_mechanics":
                    _bounds = np.zeros((2,formulation.nvar))
                    _bounds[:,:formulation.ndim] = dmesh_bounds
                    _bounds[:,-1] = [TotalDisp[:,-1,Increment].min(),TotalDisp[:,-1,Increment].max()]
                    print("\nMinimum and maximum incremental solution values at increment {} are \n".format(Increment),_bounds)
                else:
                    print("\nMinimum and maximum incremental solution values at increment {} are \n".format(Increment),dmesh_bounds)


            print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')

            try:
                print('Norm of Residual is', 
                    np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces), '\n')
            except RuntimeWarning:
                print("Invalid value encountered in norm of Newton-Raphson residual")

            # STORE THE INFORMATION IF NEWTON-RAPHSON FAILS
            if fem_solver.newton_raphson_failed_to_converge:
                solver.condA = np.NAN
                TotalDisp = TotalDisp[:,:,:Increment-1]
                fem_solver.number_of_load_increments = Increment - 1
                break

        # TotalDisp = np.concatenate((np.zeros_like(mesh.points),TotalDisp),axis=2)
        # for i in range(TotalDisp.shape[2]-1,0,-1):
            # TotalDisp[:,:,i] = np.sum(TotalDisp[:,:,:i+1],axis=2)
        return TotalDisp


    def NewtonRaphson(self, function_spaces, formulation, solver, 
        Increment, K, M, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
        boundary_condition, AppliedDirichletInc, fem_solver, velocities, accelerations):

        Tolerance = fem_solver.newton_raphson_tolerance
        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = 1./fem_solver.number_of_load_increments
        Iter = 0

        # if Increment >= 179 and Increment < 188:
        #     M *=1.2
            # exit()
        # print(AppliedDirichletInc.min(),AppliedDirichletInc.max())
        # print(Eulerp.min(),Eulerp.max())

        EulerV = np.copy(velocities[:,:,Increment])
        EulerA = np.copy(accelerations[:,:,Increment])
        EulerxPrev = np.copy(Eulerx)
        # PREDICTOR STEP
        dumV = (1. - self.gamma/self.beta)*velocities[:,:,Increment] + (1. - self.gamma/2./self.beta)*LoadFactor*accelerations[:,:,Increment]
        dumA = (-1./self.beta/LoadFactor)*velocities[:,:,Increment] - (1./2./self.beta)*(1.- 2.*self.beta)*accelerations[:,:,Increment]
        velocities[:,:,Increment]    = dumV
        accelerations[:,:,Increment] = dumA

        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            # # print(M_mech.shape,accelerations[:,:,Increment].ravel().shape)
            InertiaResidual = np.zeros((Residual.shape[0],1))
            InertiaResidual[self.mechanical_dofs,0] = M_mech.dot(accelerations[:,:,Increment].ravel())
            # InertiaResidual[self.mechanical_dofs,0] = M_mech.dot(accelerations_prev.ravel())

            # InertiaResidual[self.mechanical_dofs,0] += 0.1*M_mech.dot(velocities[:,:,Increment].ravel())
        else:
            InertiaResidual = np.zeros((Residual.shape[0],1))
            InertiaResidual[:,0] = M.dot(accelerations[:,:,Increment].ravel())
        Residual[boundary_condition.columns_in] += InertiaResidual[boundary_condition.columns_in]




        # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
        IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
            K.shape[0],formulation.nvar)
        # UPDATE EULERIAN COORDINATE
        Eulerx += IncDirichlet[:,:formulation.ndim]
        Eulerp += IncDirichlet[:,-1]
        # print(Eulerp.min(),Eulerp.max())
        # EulerGeom = np.copy(Eulerx)

        # UPDATE VELOCITY AND ACCELERATION
        # velocities[:,:,Increment]    += self.gamma/self.beta/LoadFactor*IncDirichlet[:,:]
        # accelerations[:,:,Increment] += 1./self.beta/LoadFactor**2*IncDirichlet[:,:]
        # accelerations[:,formulation.nvar-1,Increment] = 0.
        # velocities[:,:,Increment]    += self.gamma/self.beta/LoadFactor*IncDirichlet[:,:formulation.ndim]
        # accelerations[:,:,Increment] += 1./self.beta/LoadFactor**2*IncDirichlet[:,:formulation.ndim]


        np.set_printoptions(linewidth=1000,threshold=1000)

        # K_0 = K.copy()
        
        # D = 0.1*M


        while np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces) > Tolerance:
        # for ii in range(7):

            # # PREDICTOR STEP
            # dumV = (1. - self.gamma/self.beta)*velocities[:,:,Increment] + (1. - self.gamma/2./self.beta)*LoadFactor*accelerations[:,:,Increment]
            # dumA = (-1./self.beta/LoadFactor)*velocities[:,:,Increment] - (1./2./self.beta)*(1.- 2.*self.beta)*accelerations[:,:,Increment]
            # velocities[:,:,Increment]    = dumV
            # accelerations[:,:,Increment] = dumA

            # GET THE REDUCED SYSTEM OF EQUATIONS
            # K_b, F_b, M_b = boundary_condition.GetReducedMatrices(K,Residual,M)
            # print(M.todense())
            # print(M[self.electric_dofs,:][:,self.electric_dofs].todense())
            # print(self.electric_dofs)
            # print(mesh.points.shape)
            # exit()
            # print(M.todense())
            # print(K.todense())

            K += (1./self.beta/LoadFactor**2)*M
            # K += self.gamma/self.beta/LoadFactor*D + (1./self.beta/LoadFactor**2)*M
            K_b, F_b, _ = boundary_condition.GetReducedMatrices(K,Residual)
            # K1 = K + (1./self.beta/LoadFactor**2)*M
            # K1 = K_0 + (1./self.beta/LoadFactor**2)*M
            # K_b, F_b, _ = boundary_condition.GetReducedMatrices(K1,Residual,M)

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b,-F_b)
            # print(sol)

            # GET ITERATIVE SOLUTION
            dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar) 
            # print()
            # print(np.linalg.norm(dU))
            # print(dU)
            # exit()
            # print(np.linalg.norm(M.todense()))

            # UPDATE THE GEOMETRY
            Eulerx += dU[:,:formulation.ndim]
            Eulerp += dU[:,-1]
            # print(Eulerp.min(),Eulerp.max())
            # print(Eulerx.min(),Eulerx.max())
            # print(dU)
            # print(1./self.beta/LoadFactor**2*(Eulerx - TotalDisp[:,:formulation.ndim,Increment]))


            # UPDATE VELOCITY AND ACCELERATION
            # dumA = 1./self.beta/LoadFactor**2*(Eulerx - TotalDisp[:,:formulation.ndim,Increment]) -\
            #     1./self.beta/LoadFactor*velocities[:,:,Increment] -\
            #     1./2./self.beta*(1. - 2.*self.beta)*accelerations[:,:,Increment]
            # dumV = (1. - self.gamma/self.beta)*velocities[:,:,Increment] +\
            #     (1. - self.gamma/2./self.beta)*LoadFactor*accelerations[:,:,Increment] +\
            #     self.gamma/self.beta/LoadFactor*(Eulerx - TotalDisp[:,:formulation.ndim,Increment])

            # dumA = 1./self.beta/LoadFactor**2*(Eulerx - EulerxPrev) -\
            #     1./self.beta/LoadFactor*EulerV -\
            #     1./2./self.beta*(1. - 2.*self.beta)*EulerA
            # dumV = (1. - self.gamma/self.beta)*EulerV +\
            #     (1. - self.gamma/2./self.beta)*LoadFactor*EulerA +\
            #     self.gamma/self.beta/LoadFactor*(Eulerx - EulerxPrev)

            # accelerations_prev = np.copy(accelerations[:,:,Increment])

            # dumA = 1./self.beta/LoadFactor**2*(Eulerx - EulerxPrev) -\
            #     1./self.beta/LoadFactor*(EulerV) -\
            #     1./2./self.beta*(1. - 2.*self.beta)*(EulerA)
            # dumV = (1. - self.gamma/self.beta)*(EulerV) +\
            #     (1. - self.gamma/2./self.beta)*LoadFactor*(EulerA) +\
            #     self.gamma/self.beta/LoadFactor*(Eulerx - EulerxPrev)

            # velocities[:,:,Increment]    = dumV
            # accelerations[:,:,Increment] = dumA


            # velocities[:,:,Increment]    += dU[:,:formulation.ndim]
            # accelerations[:,:,Increment] = dumA


            velocities[:,:,Increment]    += self.gamma/self.beta/LoadFactor*dU[:,:formulation.ndim]
            accelerations[:,:,Increment] += 1./self.beta/LoadFactor**2*dU[:,:formulation.ndim]

            # velocities[:,:,Increment]    += self.gamma/self.beta/LoadFactor*dU[:,:formulation.ndim]
            # accelerations[:,:,Increment] += 1./self.beta/LoadFactor**2*dU[:,:formulation.ndim]
            # accelerations[:,formulation.nvar-1,Increment] = 0.
            # print(accelerations[:,2,Increment])
            # exit()
            # big_acceleration = np.zeros((mesh.points.shape[0]*formulation.nvar))
            # accelerations

            # GET ITERATIVE ELECTRIC POTENTIAL
            # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
            K, TractionForces, _, _ = Assemble(fem_solver,function_spaces[0], formulation, mesh, material, solver,
                Eulerx, Eulerp)

            # FIND THE RESIDUAL
            # Residual[boundary_condition.columns_in,0] = TractionForces[boundary_condition.columns_in,0] \
            # - NodalForces[boundary_condition.columns_in,0] + M.dot(accelerations[:,:,Increment].ravel())[boundary_condition.columns_in]


            # mech_aranger = np.arange(mesh.points.shape[0])
            # for ii in range(1,formulation.ndim):
            #     mech_aranger = np.concatenate((formulation.nvar*mech_aranger,formulation.nvar*mech_aranger+ii))
            # mech_aranger.sort()
            # coupled_aranger = np.arange(mesh.points.shape[0]*formulation.nvar)
            # electro_aranger = np.delete(coupled_aranger,mech_aranger)
            # print(coupled_aranger)
            # print(mech_aranger)
            # print(electro_aranger)

            # print(mech_aranger)
            # print(self.mechanical_dofs)
            # exit()
            # M_mech = M[mech_aranger,:][:,mech_aranger]

            if formulation.fields == "electro_mechanics":
                M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
                # # print(M_mech.shape,accelerations[:,:,Increment].ravel().shape)
                InertiaResidual = np.zeros((TractionForces.shape[0],1))
                InertiaResidual[self.mechanical_dofs,0] = M_mech.dot(accelerations[:,:,Increment].ravel())
                # InertiaResidual[self.mechanical_dofs,0] = M_mech.dot(accelerations_prev.ravel())

                # InertiaResidual[self.mechanical_dofs,0] += 0.1*M_mech.dot(velocities[:,:,Increment].ravel())
            else:
                InertiaResidual = np.zeros((TractionForces.shape[0],1))
                InertiaResidual[:,0] = M.dot(accelerations[:,:,Increment].ravel())


            # print(InertiaResidual.shape,TractionForces.shape)
            # print(boundary_condition.columns_in)
            # # hard_coded_aranger = [7,9,11,13,15,17]
            # print(np.linalg.norm(InertiaResidual[boundary_condition.columns_in]))
            # print(TractionForces[boundary_condition.columns_in].shape)
            # print(InertiaResidual[boundary_condition.columns_in].shape)


            # exit()
            # # [boundary_condition.columns_in]
            Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] \
            - NodalForces[boundary_condition.columns_in] + InertiaResidual[boundary_condition.columns_in]
            # exit()

            # SAVE THE NORM
            self.rel_norm_residual = la.norm(Residual[boundary_condition.columns_in])
            if Iter==0:
                self.NormForces = la.norm(Residual[boundary_condition.columns_in])
            self.norm_residual = np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces) 

            # SAVE THE NORM 
            self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
                self.norm_residual)
            
            print("Iteration {} for increment {}.".format(Iter, Increment) +\
                " Residual (abs) {0:>16.7g}".format(self.rel_norm_residual), 
                "\t Residual (rel) {0:>16.7g}".format(self.norm_residual))

            if np.abs(self.rel_norm_residual) < Tolerance:
                break

            # UPDATE ITERATION NUMBER
            Iter +=1

            if Iter==fem_solver.maximum_iteration_for_newton_raphson and formulation.fields == "electro_mechanics":
                raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

            if Iter==fem_solver.maximum_iteration_for_newton_raphson:
                fem_solver.newton_raphson_failed_to_converge = True
                break
            if np.isnan(self.norm_residual) or self.norm_residual>1e16:
                fem_solver.newton_raphson_failed_to_converge = True
                break


        return Eulerx, Eulerp, K, Residual, velocities, accelerations




    def ComputeEnergyDissipation(self,function_space,mesh,material,formulation,fem_solver, Eulerx, TotalDisp, NeumannForces, M, velocities, Increment):

        internal_energy = 0.
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
            
            internal_energy += formulation.GetEnergy(function_space, material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem)

        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            kinetic_energy = 0.5*np.dot(velocities[:,:,Increment].ravel(),M_mech.dot(velocities[:,:,Increment].ravel()))

        else:
            kinetic_energy = 0.5*np.dot(velocities[:,:,Increment].ravel(),M.dot(velocities[:,:,Increment].ravel()))

        external_energy = np.dot(TotalDisp[:,:,Increment].ravel(),NeumannForces[:,Increment])

        return internal_energy + kinetic_energy - external_energy
        # return internal_energy
        # return kinetic_energy
        # return external_energy




##########################
    # def Alpha(self,K,M,F1,freedof,nstep,dt,napp,alpha,delta,gam):

    #     # Input/Output Data
    #     # M - Mass Matrix of the System
    #     # K - Stiffness Matrix of the System
    #     # F1 - Vector of Dynamic Nodal Forces
    #     # freedof - Free Degrees of Freedom
    #     # napp - Degrees of Freedom at Which Excitation is Applied
    #     # nstep - No of Time Steps
    #     # dt - Time Step Size
    #     # delta, gam and alpha - 3 Integration parameters
    #     #                     With -1/3=<alpha<=0;  delta = 0.5-gam; alpha =
    #     #                     0.25*(1-gam)**2

    #     # Reference:
    #     # Hilber, H.M, Hughes,T.J.R and Talor, R.L. 
    #     # "Improved Numerical Dissipation for Time Integration Algorithms in
    #     # Structural Dynamics" Earthquake Engineering and Structural Dynamics,
    #     # 5:282-292, 1977.


    #     # U(n,nstep) - Matrix storing nodal displacements at each step.
    #     # V(n,nstep) - Matrix storing nodal Velocities at each step.
    #     # A(n,nstep) - Matrix storing nodal Accelerations at each step.
    #     # n - is No of DOF's.
        
    #     # Allocate Space for Vectors and Matrices
    #     u = np.zeros(K.shape[0]); u = u[freedof]
    #     u0=u; v = u

    #     F = np.zeros(K.shape[0]);    F = F[freedof];  F2=F
    #     M = M[:,freedof][freedof,:]; K = K[:,freedof][freedof,:]
    #     U = np.zeros((u.shape[0],nstep)); V=U; A=U

    #     # Initial Calculations
    #     U[:,0] = u.reshape(u.shape[0])
    #     A[:,0] =  sla.solve(M,(F - np.dot(K,u0)))
    #     V[:,0] = v


    #     # Initialize The Algorithm (Compute Displacements, Vel's and Accel's)
    #     for istep in range(0,nstep-1):
    #         if istep==nstep-1:
    #             break

    #         # print nstep, istep

    #         # F[napp] = F1[istep]
    #         # F2[napp] = F1[istep+1]
    #         # U[:,istep+1] = sla.solve((1/dt**2/gam*M+(1.0+alpha)*K),((1+alpha)*F2-alpha*F +\
    #      #        np.dot((1.0/dt**2/gam*M+alpha*K),U[:,istep])+ 1.0/dt/gam*np.dot(M,V[:,istep])+(1/2/gam-1)*np.dot(M,A[:,istep])))
    #         # A[:,istep+1] = 1.0/dt**2/gam*(U[:,istep+1]-U[:,istep]) - 1.0/dt/gam*V[:,istep]+(1-1.0/2.0/gam)*A[:,istep]
    #         # V[:,istep+1] = V[:,istep] + dt*(delta*A[:,istep+1]+(1-delta)*A[:,istep])

    #         F[napp] = F1[istep]
    #         F2[napp] = F1[istep+1]
    #         U[:,istep+1] = sla.solve((1/dt**2/gam*M+(1.0+alpha)*K),((1+alpha)*F2-alpha*F +\
    #             np.dot((1.0/dt**2/gam*M+alpha*K),U[:,istep])+ 1.0/dt/gam*np.dot(M,V[:,istep])+(1/2/gam-1)*np.dot(M,A[:,istep])))
    #         A[:,istep+1] = 1.0/dt**2/gam*(U[:,istep+1]-U[:,istep]) - 1.0/dt/gam*V[:,istep]+(1-1.0/2.0/gam)*A[:,istep]
    #         V[:,istep+1] = V[:,istep] + dt*(delta*A[:,istep+1]+(1-delta)*A[:,istep])


    #     return U, V, A

# import matplotlib.pyplot as plt

# dyn = StructuralDynamicIntegrators()
# n = 500
# stiffness = np.random.rand(n,n)
# stiffness += stiffness.T
# # mass = np.random.rand(n,n)
# mass = np.eye(n,n)
# # mass = np.eye(n,n)
# alpha = 0.2
# delta=0.5
# gamma = 0.4
# freedof = np.arange(0,10)
# nstep = 2*n
# # F = np.random.rand(nstep,1)
# F = np.sin(np.linspace(0,2*np.pi,nstep))
# napp=8
# dt = 1.0/nstep
# U, A, V = dyn.Alpha(stiffness,mass,F,freedof,nstep,dt,napp,alpha,delta,gamma)

# plt.plot(U[2,:])
# plt.show()