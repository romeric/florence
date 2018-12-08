from __future__ import print_function
import gc, os, sys
import numpy as np
import scipy as sp
import numpy.linalg as la
import scipy.linalg as sla
from numpy.linalg import norm
from time import time
from copy import deepcopy
from warnings import warn
from time import time

from Florence.FiniteElements.Assembly import Assemble
from Florence import Mesh
from Florence.PostProcessing import PostProcess
from .StructuralDynamicIntegrator import StructuralDynamicIntegrator

__all__ = ["NonlinearImplicitStructuralDynamicIntegrator", "LinearImplicitStructuralDynamicIntegrator"]


class NonlinearImplicitStructuralDynamicIntegrator(StructuralDynamicIntegrator):
    """Implicit dynamic solver for nonlinear problems based on Newmark's beta
    """

    def __init__(self):
        super(NonlinearImplicitStructuralDynamicIntegrator, self).__init__()
        self.gamma   = 0.5
        self.beta    = 0.25


    def Solver(self, function_spaces, formulation, solver,
        K, M, NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):


        # COMPUTE DAMPING MATRIX BASED ON MASS
        D = 0.0
        if fem_solver.include_physical_damping:
            D = fem_solver.damping_factor*M

        # GET BOUNDARY CONDITIONS INFROMATION
        self.GetBoundaryInfo(mesh, formulation, boundary_condition)
        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            if fem_solver.include_physical_damping:
                D_mech = D[self.mechanical_dofs,:][:,self.mechanical_dofs]

        # INITIALISE VELOCITY AND ACCELERATION
        velocities     = np.zeros((mesh.points.shape[0],formulation.ndim))
        accelerations  = np.zeros((mesh.points.shape[0],formulation.ndim))

        # COMPUTE INITIAL ACCELERATION FOR TIME STEP 0
        if NeumannForces.ndim == 2 and NeumannForces.shape[1]>1:
            InitResidual = Residual - NeumannForces[:,0][:,None]
        else:
            InitResidual = Residual

        if formulation.fields == "electro_mechanics":
            accelerations[:,:] = solver.Solve(M_mech, -InitResidual[self.mechanical_dofs].ravel()
                ).reshape(mesh.points.shape[0],formulation.ndim)
        else:
            accelerations[:,:] = solver.Solve(M, -InitResidual.ravel() ).reshape(mesh.points.shape[0],formulation.ndim)

        self.NRConvergence = fem_solver.NRConvergence
        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = fem_solver.total_time/LoadIncrement
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)

        save_counter = 1
        nincr_last = float(LoadIncrement-1) if LoadIncrement !=1 else 1
        if boundary_condition.compound_dirichlet_bcs:
            ChangedTotalDisp = np.zeros((mesh.nnode, formulation.nvar))
        # TIME LOOP
        for Increment in range(1,LoadIncrement):

            t_increment = time()

            # GET INCREMENTAL DIRICHLET BC
            if not boundary_condition.has_step_wise_dirichlet_loading:
                if boundary_condition.applied_dirichlet.ndim == 2:
                    AppliedDirichletInc = boundary_condition.applied_dirichlet[:,Increment]
                else:
                    if boundary_condition.make_loading == "ramp":
                        AppliedDirichletInc = boundary_condition.applied_dirichlet*(1.*Increment/LoadIncrement)
                    else:
                        AppliedDirichletInc = boundary_condition.applied_dirichlet/nincr_last
            else:
                boundary_condition.ApplyStepWiseDirichletFunc(formulation, mesh, increment=Increment)
                self.GetBoundaryInfo(mesh, formulation, boundary_condition, increment=Increment)
                AppliedDirichletInc = boundary_condition.applied_dirichlet
                if self.bc_changed_at_this_step and boundary_condition.compound_dirichlet_bcs:
                    ChangedTotalDisp += np.copy(U)

            # GET INCREMENTAL NEUMANN DIRICHLET BC
            if not boundary_condition.has_step_wise_neumann_loading:
                if NeumannForces.ndim == 2 and NeumannForces.shape[1]>1:
                    NodalForces = NeumannForces[:,Increment][:,None]
                else:
                    if boundary_condition.make_loading == "ramp":
                        NodalForces = NeumannForces*(1.*Increment/LoadIncrement)
                    else:
                        NodalForces = NeumannForces/nincr_last
            else:
                NodalForces = boundary_condition.ApplyStepWiseNeumannFunc(formulation, mesh,
                    material, increment=Increment)
            NodalForces = NodalForces.ravel()[:,None]

            # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
            # OLD WAY - RESIDUAL WAS GETTING CARRIED OVER FROM PREV NR STEP BUT AT THIS
            # POINT IT WAS TINY (AS NR HAD CONVERGED) THAT IT DIDN'T MATTER AND WORKED AS EXPECTED
            # Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
            #     AppliedDirichletInc,LoadFactor=1.0,mass=M,only_residual=True)
            # ACTUAL WAY
            Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,np.zeros_like(Residual),
                AppliedDirichletInc,LoadFactor=1.0,mass=M,only_residual=True)
            Residual -= NodalForces

            # COMPUTE INITIAL ACCELERATION - ONLY NEEDED IN CASES OF PRESTRETCHED CONFIGURATIONS
            # accelerations[:,:] = solver.Solve(M, Residual.ravel() - \
            #     K.dot(TotalDisp[:,:,Increment].ravel())).reshape(mesh.points.shape[0],formulation.nvar)

            # LET NORM OF THE FIRST RESIDUAL BE THE NORM WITH RESPECT TO WHICH WE
            # HAVE TO CHECK THE CONVERGENCE OF NEWTON RAPHSON. TYPICALLY THIS IS
            # NORM OF NODAL FORCES
            if Increment==1:
                self.NormForces = np.linalg.norm(Residual)
                # AVOID DIVISION BY ZERO
                if np.isclose(self.NormForces,0.0):
                    self.NormForces = 1e-14
            self.norm_residual = np.linalg.norm(Residual)/self.NormForces


            Eulerx, Eulerp, K, Residual, velocities, accelerations = self.NewtonRaphson(function_spaces, formulation, solver,
                Increment, K, D, M, NodalForces, Residual, mesh, Eulerx, Eulerp,
                material,boundary_condition,AppliedDirichletInc, fem_solver, velocities, accelerations)

            # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
            U = np.zeros((mesh.points.shape[0], formulation.nvar))
            U[:,:formulation.ndim] = Eulerx - mesh.points
            if formulation.fields == "electro_mechanics":
                U[:,-1] = Eulerp

            # SAVE RESULTS
            if Increment % fem_solver.save_frequency == 0 or\
                (Increment == LoadIncrement - 1 and save_counter<TotalDisp.shape[2]):
                TotalDisp[:,:,save_counter] = U
                if boundary_condition.compound_dirichlet_bcs:
                    TotalDisp[:,:,save_counter] += ChangedTotalDisp
                save_counter += 1


            # COMPUTE DISSIPATION OF ENERGY THROUGH TIME
            if fem_solver.compute_energy_dissipation:
                energy_info = self.ComputeEnergyDissipation(function_spaces[0], mesh, material, formulation, fem_solver,
                    Eulerx, U, NodalForces, M, velocities)
                formulation.energy_dissipation.append(energy_info[0])
                formulation.internal_energy.append(energy_info[1])
                formulation.kinetic_energy.append(energy_info[2])
                formulation.external_energy.append(energy_info[3])
            # COMPUTE DISSIPATION OF LINEAR MOMENTUM THROUGH TIME
            if fem_solver.compute_linear_momentum_dissipation:
                power_info = self.ComputePowerDissipation(function_spaces[0], mesh, material, formulation, fem_solver,
                    Eulerx, U, NodalForces, M, velocities, accelerations)
                formulation.power_dissipation.append(power_info[0])
                formulation.internal_power.append(power_info[1])
                formulation.kinetic_power.append(power_info[2])
                formulation.external_power.append(power_info[3])


            # LOG IF ASKED FOR
            self.LogSave(fem_solver, formulation, U[:,:formulation.ndim], Eulerp, Increment)

            print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')

            try:
                print('Norm of Residual is',
                    np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces), '\n')
            except RuntimeWarning:
                print("Invalid value encountered in norm of Newton-Raphson residual")

            # STORE THE INFORMATION IF NEWTON-RAPHSON FAILS
            if fem_solver.newton_raphson_failed_to_converge:
                solver.condA = np.NAN
                TotalDisp = TotalDisp[:,:,:save_counter-1]
                fem_solver.number_of_load_increments = save_counter - 1
                break

            # BREAK AT A SPECIFICED LOAD INCREMENT IF ASKED FOR
            if fem_solver.break_at_increment != -1 and fem_solver.break_at_increment is not None:
                if fem_solver.break_at_increment == Increment:
                    if fem_solver.break_at_increment < LoadIncrement - 1:
                        print("\nStopping at increment {} as specified\n\n".format(Increment))
                        TotalDisp = TotalDisp[:,:,:save_counter]
                        fem_solver.number_of_load_increments = save_counter
                    break


        if fem_solver.save_frequency != 1:
            if TotalDisp.shape[2] > save_counter:
                # IN CASE SOLVER BLEW UP
                TotalDisp = TotalDisp[:,:,:save_counter]
                fem_solver.number_of_load_increments = TotalDisp.shape[2]
            else:
                fem_solver.number_of_load_increments = save_counter

        return TotalDisp


    def NewtonRaphson(self, function_spaces, formulation, solver,
        Increment, K, D, M, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
        boundary_condition, AppliedDirichletInc, fem_solver, velocities, accelerations):

        Tolerance = fem_solver.newton_raphson_tolerance
        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = fem_solver.total_time/fem_solver.number_of_load_increments
        Iter = 0
        self.iterative_norm_history = []

        # EulerxPrev = np.copy(Eulerx)
        # EulerVPrev = np.copy(velocities[:,:,Increment-1])
        # EulerAPrev = np.copy(accelerations[:,:,Increment-1])

        # PREDICTOR STEP
        tmpV = (1. - self.gamma/self.beta)*velocities + (1. - self.gamma/2./self.beta)*LoadFactor*accelerations
        tmpA = (-1./self.beta/LoadFactor)*velocities - (1./2./self.beta)*(1.- 2.*self.beta)*accelerations
        velocities    = tmpV
        accelerations = tmpA

        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            InertiaResidual = np.zeros((Residual.shape[0],1))
            InertiaResidual[self.mechanical_dofs,0] = M_mech.dot(accelerations.ravel())
            if fem_solver.include_physical_damping:
                D_mech = D[self.mechanical_dofs,:][:,self.mechanical_dofs]
                InertiaResidual[self.mechanical_dofs,0] += D_mech.dot(velocities.ravel())
        else:
            InertiaResidual = np.zeros((Residual.shape[0],1))
            InertiaResidual[:,0] = M.dot(accelerations.ravel())
            if fem_solver.include_physical_damping:
                InertiaResidual[:,0] += D.dot(velocities.ravel())
        Residual[boundary_condition.columns_in] += InertiaResidual[boundary_condition.columns_in]


        # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
        IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
            K.shape[0],formulation.nvar)
        # UPDATE EULERIAN COORDINATE
        # Eulerx += IncDirichlet[:,:formulation.ndim]
        Eulerx[:,:] = mesh.points + IncDirichlet[:,:formulation.ndim]
        Eulerp[:] = IncDirichlet[:,-1] # ENSURES Eulerp IS CONTIGUOUS - NECESSARY FOR LOW-LEVEL DISPATCHER


        while np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces) > Tolerance or Iter==0:

            # GET EFFECTIVE STIFFNESS
            # K += (1./self.beta/LoadFactor**2)*M
            K += (self.gamma/self.beta/LoadFactor)*D + (1./self.beta/LoadFactor**2)*M
            # GET THE REDUCED SYSTEM OF EQUATIONS
            K_b, F_b, _ = boundary_condition.GetReducedMatrices(K,Residual)

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b,-F_b)

            # GET ITERATIVE SOLUTION
            dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar)

            # UPDATE THE EULERIAN COMPONENTS
            # UPDATE THE GEOMETRY
            Eulerx += dU[:,:formulation.ndim]
            # GET ITERATIVE ELECTRIC POTENTIAL
            Eulerp += dU[:,-1]

            # UPDATE VELOCITY AND ACCELERATION
            velocities    += self.gamma/self.beta/LoadFactor*dU[:,:formulation.ndim]
            accelerations += 1./self.beta/LoadFactor**2*dU[:,:formulation.ndim]

            # OR ALTERNATIVELY
            # dumA = 1./self.beta/LoadFactor**2*(Eulerx - EulerxPrev) -\
            #     1./self.beta/LoadFactor*(EulerVPrev) -\
            #     1./2./self.beta*(1. - 2.*self.beta)*(EulerAPrev)
            # dumV = (1. - self.gamma/self.beta)*(EulerVPrev) +\
            #     (1. - self.gamma/2./self.beta)*LoadFactor*(EulerAPrev) +\
            #     self.gamma/self.beta/LoadFactor*(Eulerx - EulerxPrev)
            # velocities    = dumV
            # accelerations = dumA

            # RE-ASSEMBLE - COMPUTE STIFFNESS AND INTERNAL TRACTION FORCES
            K, TractionForces, _, _ = Assemble(fem_solver,function_spaces[0], formulation, mesh, material,
                Eulerx, Eulerp)

            # FIND INITIAL RESIDUAL
            if formulation.fields == "electro_mechanics":
                InertiaResidual = np.zeros((TractionForces.shape[0],1))
                InertiaResidual[self.mechanical_dofs,0] = M_mech.dot(accelerations.ravel())
                if fem_solver.include_physical_damping:
                    InertiaResidual[self.mechanical_dofs,0] += D_mech.dot(velocities.ravel())
            else:
                InertiaResidual = np.zeros((TractionForces.shape[0],1))
                InertiaResidual[:,0] = M.dot(accelerations.ravel())
                if fem_solver.include_physical_damping:
                    InertiaResidual[:,0] += D.dot(velocities.ravel())


            # UPDATE RESIDUAL
            Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] \
            - NodalForces[boundary_condition.columns_in] + InertiaResidual[boundary_condition.columns_in]


            # SAVE THE NORM
            self.abs_norm_residual = la.norm(Residual[boundary_condition.columns_in])
            if Iter==0:
                self.NormForces = la.norm(Residual[boundary_condition.columns_in])
            self.norm_residual = np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces)

            # SAVE THE NORM
            self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
                self.norm_residual)

            print("Iteration {} for increment {}.".format(Iter, Increment) +\
                " Residual (abs) {0:>16.7g}".format(self.abs_norm_residual),
                "\t Residual (rel) {0:>16.7g}".format(self.norm_residual))

            # BREAK BASED ON RELATIVE NORM
            if np.abs(self.abs_norm_residual) < Tolerance:
                break

            # BREAK BASED ON INCREMENTAL SOLUTION - KEEP IT AFTER UPDATE
            if norm(dU) <=  fem_solver.newton_raphson_solution_tolerance:
                print("Incremental solution within tolerance i.e. norm(dU): {}".format(norm(dU)))
                break

            # UPDATE ITERATION NUMBER
            Iter +=1

            if Iter==fem_solver.maximum_iteration_for_newton_raphson and formulation.fields == "electro_mechanics":
                raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

            if Iter==fem_solver.maximum_iteration_for_newton_raphson:
                fem_solver.newton_raphson_failed_to_converge = True
                break

            if np.isnan(self.norm_residual) or self.norm_residual>1e06:
                fem_solver.newton_raphson_failed_to_converge = True
                break

            # IF BREAK WHEN NEWTON RAPHSON STAGNATES IS ACTIVATED
            if fem_solver.break_at_stagnation:
                self.iterative_norm_history.append(self.norm_residual)
                if Iter >= 5 and self.abs_norm_residual<1e06:
                    if np.mean(self.iterative_norm_history) < 1.:
                        break

            # USER DEFINED CRITERIA TO BREAK OUT OF NEWTON-RAPHSON
            if fem_solver.user_defined_break_func != None:
                if fem_solver.user_defined_break_func(Increment,Iter,self.norm_residual,self.abs_norm_residual, Tolerance):
                    break

            # USER DEFINED CRITERIA TO STOP NEWTON-RAPHSON AND THE WHOLE ANALYSIS
            if fem_solver.user_defined_stop_func != None:
                if fem_solver.user_defined_stop_func(Increment,Iter,self.norm_residual,self.abs_norm_residual, Tolerance):
                    fem_solver.newton_raphson_failed_to_converge = True
                    break

        return Eulerx, Eulerp, K, Residual, velocities, accelerations












#------------------------------------------ LINEAR IMPLICIT SOLVER ----------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#


class LinearImplicitStructuralDynamicIntegrator(StructuralDynamicIntegrator):
    """Implicit dynamic solver for linear problems based on Newmark's beta
    """

    def __init__(self,**kwargs):
        super(LinearImplicitStructuralDynamicIntegrator, self).__init__()
        self.lump_rhs = False
        self.gamma   = 0.5
        self.beta    = 0.25


    def Solver(self, function_spaces, formulation, solver,
        K, M, NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):

        # CHECK FORMULATION
        if formulation.fields != "mechanics" and formulation.fields != "electro_mechanics":
            raise NotImplementedError("Linear implicit solver for {} is not available".format(formulation.fields))
        if formulation.fields == "electro_mechanics":
            warn("Linear implicit solver for electromechanics formulation is not thoroughly checked and may return incorrect results. "
                "Please use nonlinear explicit dynamic solver instead")

        # GET BOUNDARY CONDITIONS INFROMATION
        self.GetBoundaryInfo(mesh, formulation, boundary_condition)

        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = fem_solver.total_time/LoadIncrement

        post_process = PostProcess(formulation.ndim,formulation.nvar)
        post_process.SetAnalysis(analysis_type=fem_solver.analysis_type, analysis_nature=fem_solver.analysis_nature)

        if NeumannForces.ndim == 2 and NeumannForces.shape[1]==1:
            tmp = np.zeros((NeumannForces.shape[0],LoadIncrement))
            tmp[:,0] = NeumannForces[:,0]
            NeumannForces = tmp

        dU = boundary_condition.UpdateFixDoFs(boundary_condition.applied_dirichlet[:,0],
            mesh.points.shape[0]*formulation.nvar, formulation.nvar)
        TotalDisp[:,:formulation.nvar,0] = dU
        # INITIALISE VELOCITY AND ACCELERATION
        velocities     = np.zeros((mesh.points.shape[0]*formulation.ndim))
        accelerations  = np.zeros((mesh.points.shape[0]*formulation.ndim))
        # COMPUTE DAMPING MATRIX BASED ON MASS
        D = 0.0
        if fem_solver.include_physical_damping:
            D = fem_solver.damping_factor*M

        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            if fem_solver.include_physical_damping:
                D_mech = D[self.mechanical_dofs,:][:,self.mechanical_dofs]
        else:
            M_mech = M
            D_mech = D

        # COMPUTE INITIAL ACCELERATION FOR TIME STEP 0
        Residual = np.zeros_like(Residual)
        InitResidual = Residual + NeumannForces[:,0][:,None]
        if formulation.fields == "electro_mechanics":
            accelerations[:] = solver.Solve(M_mech, -InitResidual[self.mechanical_dofs].ravel())
        else:
            accelerations[:] = solver.Solve(M, InitResidual.ravel() )

        # COMPUTE AUGMENTED K (INCLUDES INERTIA EFFECT)
        K          += (self.gamma/self.beta/LoadFactor)*D + (1./self.beta/LoadFactor**2)*M
        # GET REDUCED VARIABLES
        K_b, F_b, _ = boundary_condition.GetReducedMatrices(K,Residual)

        if self.lump_rhs:
            M_mech = M_mech.sum(axis=1).A.ravel() # FOR CSR
            # M_mech = M_mech.sum(axis=0).ravel() # FOR CSC
            if self.include_physical_damping:
                D_mech = D_mech.sum(axis=1).A.ravel()

        reuse_factorisation = False if formulation.fields == "electro_mechanics" else True

        for Increment in range(1,LoadIncrement):
            t_increment=time()

            # FIXED INCREMENTAL DIRICHLET
            AppliedDirichletInc = boundary_condition.applied_dirichlet[:,Increment-1]

            # APPLY NEUMANN BOUNDARY CONDITIONS
            DeltaF = NeumannForces[:,Increment][:,None]
            NodalForces = DeltaF

            # ACCUMULATED FORCE
            if fem_solver.include_physical_damping:
                if self.lump_rhs:
                    Residual[self.mechanical_dofs,0] = (1./self.beta/LoadFactor**2)*M_mech*TotalDisp[:,:formulation.ndim,Increment-1].ravel() +\
                        (1./self.beta/LoadFactor)*M_mech*velocities + (0.5/self.beta - 1.)*M_mech*accelerations +\
                        (self.gamma/self.beta/LoadFactor)*D_mech*TotalDisp[:,:formulation.ndim,Increment-1].ravel() +\
                        (self.gamma/self.beta - 1.)*D_mech*velocities -\
                        LoadFactor*((1-self.gamma)-self.gamma*(0.5/self.beta - 1.))*D_mech*accelerations
                else:
                    Residual[self.mechanical_dofs,0] = (1./self.beta/LoadFactor**2)*M_mech.dot(TotalDisp[:,:formulation.ndim,Increment-1].ravel()) +\
                        (1./self.beta/LoadFactor)*M_mech.dot(velocities) + (0.5/self.beta - 1.)*M_mech.dot(accelerations) +\
                        (self.gamma/self.beta/LoadFactor)*D_mech.dot(TotalDisp[:,:formulation.ndim,Increment-1].ravel()) +\
                        (self.gamma/self.beta - 1.)*D_mech.dot(velocities) -\
                        LoadFactor*((1-self.gamma)-self.gamma*(0.5/self.beta - 1.))*D_mech.dot(accelerations)
            else:
                if self.lump_rhs:
                    Residual[self.mechanical_dofs,0] = (1./self.beta/LoadFactor**2)*M_mech*TotalDisp[:,:formulation.ndim,Increment-1].ravel() +\
                        (1./self.beta/LoadFactor)*M_mech*velocities + (0.5/self.beta - 1.)*M_mech*accelerations
                else:
                    Residual[self.mechanical_dofs,0] = (1./self.beta/LoadFactor**2)*M_mech.dot(TotalDisp[:,:formulation.ndim,Increment-1].ravel()) +\
                        (1./self.beta/LoadFactor)*M_mech.dot(velocities) + (0.5/self.beta - 1.)*M_mech.dot(accelerations)
            Residual += DeltaF

            if formulation.fields == "electro_mechanics":
                K           = Assemble(fem_solver,function_spaces[0], formulation, mesh, material, Eulerx, Eulerp)[0]
                K          += (self.gamma/self.beta/LoadFactor)*D + (1./self.beta/LoadFactor**2)*M

            # CHECK CONTACT AND ASSEMBLE IF DETECTED
            if fem_solver.has_contact:
                Eulerx = mesh.points + TotalDisp[:,:formulation.ndim,Increment-1]
                TractionForcesContact = np.zeros_like(Residual)
                TractionForcesContact = fem_solver.contact_formulation.AssembleTractions(mesh,material,Eulerx).ravel()*LoadFactor

                if formulation.fields == "electro_mechanics" or formulation.fields == "flexoelectric":
                    Residual[self.mechanical_dofs,0] -= TractionForcesContact
                elif formulation.fields == "mechanics" or formulation.fields == "couple_stress":
                    Residual[:,0] -= TractionForcesContact
                else:
                    raise NotImplementedError("Contact algorithm for {} is not available".format(formulation.fields))

            # REDUCED ACCUMULATED FORCE
            if formulation.fields == "mechanics":
                F_b = boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                    boundary_condition.applied_dirichlet[:,Increment],LoadFactor=1.0,
                    mass=M,only_residual=True)[boundary_condition.columns_in,0]
            else:
                K_b, F_b = boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                    boundary_condition.applied_dirichlet[:,Increment],LoadFactor=1.0,
                    mass=M)[:2]

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b, F_b, reuse_factorisation=reuse_factorisation)

            dU = post_process.TotalComponentSol(sol, boundary_condition.columns_in,
                boundary_condition.columns_out, AppliedDirichletInc,0,K.shape[0])

            # STORE TOTAL SOLUTION DATA
            TotalDisp[:,:,Increment] += dU

            # UPDATE VELOCITY AND ACCELERATION
            accelerations_old = np.copy(accelerations)
            accelerations = (1./self.beta/LoadFactor**2)*(TotalDisp[:,:formulation.ndim,Increment] -\
                TotalDisp[:,:formulation.ndim,Increment-1]).ravel() -\
                1./self.beta/LoadFactor*velocities + (1.-0.5/self.beta)*accelerations_old
            velocities += LoadFactor*(self.gamma*accelerations + (1-self.gamma)*accelerations_old)

            # UPDATE
            Eulerx += dU[:,:formulation.ndim]
            Eulerp += dU[:,-1]

            # LOG REQUESTS
            fem_solver.LogSave(formulation, TotalDisp, Increment)

            # BREAK AT A SPECIFICED LOAD INCREMENT IF ASKED FOR
            if fem_solver.break_at_increment != -1 and fem_solver.break_at_increment is not None:
                if fem_solver.break_at_increment == Increment:
                    if fem_solver.break_at_increment < LoadIncrement - 1:
                        print("\nStopping at increment {} as specified\n\n".format(Increment))
                        TotalDisp = TotalDisp[:,:,:Increment]
                        fem_solver.number_of_load_increments = Increment
                    break

            # STORE THE INFORMATION IF THE SOLVER BLOWS UP
            if Increment > 0:
                U0 = TotalDisp[:,:,Increment-1].ravel()
                U = TotalDisp[:,:,Increment].ravel()
                tol = 1e200 if Increment < 5 else 10.
                if np.isnan(norm(U)) or np.abs(U.max()/(U0.max()+1e-14)) > tol:
                    print("Solver blew up! Norm of incremental solution is too large")
                    TotalDisp = TotalDisp[:,:,:Increment]
                    fem_solver.number_of_load_increments = Increment
                    break

            print('Finished Load increment', Increment, 'in', time()-t_increment, 'seconds\n')

        solver.CleanUp()

        return TotalDisp

