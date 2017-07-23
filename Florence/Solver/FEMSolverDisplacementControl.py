from __future__ import print_function
import gc, os, sys
import multiprocessing
from copy import deepcopy
from warnings import warn
from time import time
import numpy as np
from numpy.linalg import norm
import scipy as sp
from Florence.Utils import insensitive

from Florence.FiniteElements.Assembly import Assemble
from Florence.PostProcessing import *
from Florence.Solver import LinearSolver
from Florence.TimeIntegrators import StructuralDynamicIntegrators
from Florence import Mesh, FEMSolver




def StaticSolverDisplacementControl(self, function_spaces, formulation, solver, K,
        NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition):

    LoadIncrement = self.number_of_load_increments
    self.accumulated_load_factor = 0.0
    AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)

    # GET TOTAL FORCE
    TotalForce = np.copy(NeumannForces)
    # TotalForce = boundary_condition.ApplyDirichletGetReducedMatrices(K,TotalForce,
            # boundary_condition.applied_dirichlet,LoadFactor=1.0,only_residual=True)

    # self.max_prescribed_displacement = np.max(np.abs(boundary_condition.applied_dirichlet))
    self.max_prescribed_displacement = 20.
    self.incremental_displacement = self.max_prescribed_displacement/self.number_of_load_increments
    # print(self.incremental_displacement)
    # exit()
    
    for Increment in range(LoadIncrement):

        # CHECK ADAPTIVE LOAD FACTOR
        # if self.load_factor is not None:
        #     self.local_load_factor = self.load_factor[Increment]
        # else:
            # if Increment <= 1:
            #     self.local_load_factor = 1./LoadIncrement
            # else:

        #         # GET THE REDUCED SYSTEM OF EQUATIONS
        #         K_b, F_bb = boundary_condition.GetReducedMatrices(K,TotalForce)[:2]
        #         # SOLVE THE SYSTEM
        #         sol_b = solver.Solve(K_b,F_bb)
        #         # GET ITERATIVE SOLUTION
        #         dU_b = boundary_condition.UpdateFreeDoFs(sol_b,K.shape[0],formulation.nvar)

        #         # max_occured_displacement = np.max(np.abs((TotalDisp[:,:,Increment-1] - TotalDisp[:,:,Increment-2])))
        #         max_occured_displacement = np.max(np.abs(dU_b))
        #         self.local_load_factor = max_occured_displacement/self.max_prescribed_displacement
        #         # print(self.local_load_factor)
        #         # exit()
        # self.accumulated_load_factor += self.local_load_factor
        # # print(self.accumulated_load_factor)



        # GET THE REDUCED SYSTEM OF EQUATIONS
        K_b, F_bb = boundary_condition.GetReducedMatrices(K,TotalForce)[:2]
        # SOLVE THE SYSTEM
        sol_b = solver.Solve(K_b,F_bb)
        # GET ITERATIVE SOLUTION
        dU_b = boundary_condition.UpdateFreeDoFs(sol_b,K.shape[0],formulation.nvar)

        max_occured_displacement = np.max(np.abs(dU_b))
        self.local_load_factor = self.incremental_displacement/max_occured_displacement
        if self.local_load_factor > 1.0:
            raise ValueError("Raise max displacements")
        # print(self.local_load_factor,max_occured_displacement)
                # exit()

        self.accumulated_load_factor += self.local_load_factor
        # print(self.accumulated_load_factor)


        # APPLY NEUMANN BOUNDARY CONDITIONS
        DeltaF = self.local_load_factor*NeumannForces
        NodalForces += DeltaF
        # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
        Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
            boundary_condition.applied_dirichlet,LoadFactor=self.local_load_factor,only_residual=True)
        Residual -= DeltaF
        # GET THE INCREMENTAL DISPLACEMENT
        AppliedDirichletInc = self.local_load_factor*boundary_condition.applied_dirichlet

        t_increment = time()

        # LET NORM OF THE FIRST RESIDUAL BE THE NORM WITH RESPECT TO WHICH WE
        # HAVE TO CHECK THE CONVERGENCE OF NEWTON RAPHSON. TYPICALLY THIS IS 
        # NORM OF NODAL FORCES
        if Increment==0:
            self.NormForces = np.linalg.norm(Residual)
            # AVOID DIVISION BY ZERO
            if np.isclose(self.NormForces,0.0):
                self.NormForces = 1e-14

        self.norm_residual = np.linalg.norm(Residual)/self.NormForces

        Eulerx, Eulerp, K, Residual = NewtonRaphsonDisplacementControl(self, function_spaces, formulation, solver, 
            Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp,
            material, boundary_condition, AppliedDirichletInc, NeumannForces, TotalForce, TotalDisp)

        # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
        TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
        if formulation.fields == "electro_mechanics":
            TotalDisp[:,-1,Increment] = Eulerp

        # PRINT LOG IF ASKED FOR
        if self.print_incremental_log:
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

        # SAVE INCREMENTAL SOLUTION IF ASKED FOR
        if self.save_incremental_solution:
            from scipy.io import savemat
            if self.incremental_solution_filename is not None:
                savemat(self.incremental_solution_filename+"_"+str(Increment),{'solution':TotalDisp[:,:,Increment]},do_compression=True)
            else:
                raise ValueError("No file name provided to save incremental solution")


        print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')
        try:
            print('Norm of Residual is', 
                np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces), '\n')
        except RuntimeWarning:
            print("Invalid value encountered in norm of Newton-Raphson residual")

        # STORE THE INFORMATION IF NEWTON-RAPHSON FAILS
        if self.newton_raphson_failed_to_converge:
            solver.condA = np.NAN
            TotalDisp = TotalDisp[:,:,:Increment]
            self.number_of_load_increments = Increment
            break

        # BREAK AT A SPECIFICED LOAD INCREMENT IF ASKED FOR
        if self.break_at_increment != -1 and self.break_at_increment is not None:
            if self.break_at_increment == Increment:
                if self.break_at_increment < LoadIncrement - 1:
                    print("\nStopping at increment {} as specified\n\n".format(Increment))
                    TotalDisp = TotalDisp[:,:,:Increment]
                break


    return TotalDisp


def NewtonRaphsonDisplacementControl(self, function_spaces, formulation, solver, 
    Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
    boundary_condition, AppliedDirichletInc, NeumannForces, TotalForce, TotalDisp):

    Tolerance = self.newton_raphson_tolerance
    LoadIncrement = self.number_of_load_increments
    Iter = 0
    iterative_load_factor = 0.0


    # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
    IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
        K.shape[0],formulation.nvar)

    # UPDATE EULERIAN COORDINATE
    Eulerx += IncDirichlet[:,:formulation.ndim]
    Eulerp += IncDirichlet[:,-1]

    while self.norm_residual > Tolerance or Iter==0:
        # GET THE REDUCED SYSTEM OF EQUATIONS
        K_b, F_b = boundary_condition.GetReducedMatrices(K,Residual)[:2]
        # SOLVE THE SYSTEM
        sol = solver.Solve(K_b,-F_b)
        # GET ITERATIVE SOLUTION
        dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar)

        # GET THE REDUCED SYSTEM OF EQUATIONS
        F_bb = boundary_condition.GetReducedMatrices(K,TotalForce)[1]
        # SOLVE THE SYSTEM
        sol_b = solver.Solve(K_b,F_bb)
        # GET ITERATIVE SOLUTION
        dU_b = boundary_condition.UpdateFreeDoFs(sol_b,K.shape[0],formulation.nvar)

        # ratio = np.max(np.abs(dU))/np.max(np.abs(dU_b))
        # ratio = np.max(np.abs(dU))/self.max_prescribed_displacement
        max_occured_displacement = np.max(np.abs(dU_b))
        ratio = np.max(np.abs(dU))/max_occured_displacement
        iterative_load_factor += ratio
        # self.local_load_factor += iterative_load_factor
        print(ratio)
        # print(iterative_load_factor)
        # print(self.local_load_factor)
        # print(self.accumulated_load_factor)


        # # GET THE REDUCED SYSTEM OF EQUATIONS
        # K_b, F_b = boundary_condition.GetReducedMatrices(K, Residual - ratio*TotalForce)[:2]
        # # SOLVE THE SYSTEM
        # sol = solver.Solve(K_b,-F_b)
        # # GET ITERATIVE SOLUTION
        # dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar)


        # # UPDATE THE EULERIAN COMPONENTS
        # Eulerx += dU[:,:formulation.ndim]
        # Eulerp += dU[:,-1]

        # UPDATE THE EULERIAN COMPONENTS
        Eulerx += dU[:,:formulation.ndim] + ratio*dU_b[:,:formulation.ndim]
        Eulerp += dU[:,-1] + ratio*dU_b[:,-1]

        # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
        K, TractionForces = Assemble(self, function_spaces[0], formulation, mesh, material,
            Eulerx,Eulerp)[:2]

        # FIND THE RESIDUAL
        Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] -\
            NodalForces[boundary_condition.columns_in] - ratio*TotalForce[boundary_condition.columns_in]

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

        if Iter==self.maximum_iteration_for_newton_raphson and formulation.fields == "electro_mechanics":
            # raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")
            warn("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")
            self.newton_raphson_failed_to_converge = True
            break

        if Iter==self.maximum_iteration_for_newton_raphson:
            self.newton_raphson_failed_to_converge = True
            break
        if np.isnan(self.norm_residual) or self.norm_residual>1e06:
            self.newton_raphson_failed_to_converge = True
            break

        # USER DEFINED CRITERIA TO BREAK OUT OF NEWTON-RAPHSON
        if self.user_defined_break_func != None:
            if self.user_defined_break_func(Increment,Iter,self.norm_residual,self.rel_norm_residual, Tolerance):
                break

        # USER DEFINED CRITERIA TO STOP NEWTON-RAPHSON AND THE WHOLE ANALYSIS
        if self.user_defined_stop_func != None:
            if self.user_defined_stop_func(Increment,Iter,self.norm_residual,self.rel_norm_residual, Tolerance):
                self.newton_raphson_failed_to_converge = True
                break

        if self.accumulated_load_factor >= 1.0:
            print("Load factor: 1.0, Breaking")
            self.newton_raphson_failed_to_converge = True
            break

    self.local_load_factor += iterative_load_factor

    return Eulerx, Eulerp, K, Residual