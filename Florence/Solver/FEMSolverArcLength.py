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


# class FEMSolverArcLength(FEMSolver):

#     def __init__(self):
#         pass


def StaticSolverArcLength(self, function_spaces, formulation, solver, K,
        NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition):

    LoadIncrement = self.number_of_load_increments
    LoadFactor = 1./LoadIncrement
    AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)
    
    for Increment in range(LoadIncrement):

        # CHECK ADAPTIVE LOAD FACTOR
        if self.load_factor is not None:
            LoadFactor = self.load_factor[Increment]

        # APPLY NEUMANN BOUNDARY CONDITIONS
        DeltaF = LoadFactor*NeumannForces
        NodalForces += DeltaF
        # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
        Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
            boundary_condition.applied_dirichlet,LoadFactor=LoadFactor,only_residual=True)
        Residual -= DeltaF
        # GET THE INCREMENTAL DISPLACEMENT
        AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet

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

        Eulerx, Eulerp, K, Residual = NewtonRaphsonArchLength(self, function_spaces, formulation, solver, 
            Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp,
            material, boundary_condition, AppliedDirichletInc, DeltaF, TotalDisp)

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


def NewtonRaphsonArchLength(self, function_spaces, formulation, solver, 
    Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
    boundary_condition, AppliedDirichletInc, DeltaF, TotalDisp):

    Tolerance = self.newton_raphson_tolerance
    LoadIncrement = self.number_of_load_increments
    LoadFactor = 1./LoadIncrement
    accumulated_load_factor = Increment/LoadIncrement
    Iter = 0
    dL = 1.
    psi = 1.
    # NodalForces = DeltaF

    Dlam = 0.
    dU = np.zeros((mesh.points.shape[0],formulation.nvar))
    dU_b = np.zeros((mesh.points.shape[0],formulation.nvar))

    # SOLVE WITH INCREMENTAL LOAD
    K_b, DF_b = boundary_condition.GetReducedMatrices(K,NodalForces)[:2]
    dU_t = solver.Solve(K_b,DF_b)
    dU_t = boundary_condition.UpdateFreeDoFs(dU_t,K.shape[0],formulation.nvar)
    # print(NodalForces)

    # dU = IncDirichlet
    # GET TOTAL ITERATIVE SOLUTION
    # dU = dU_actual + LoadFactor*dU_current

    # GET ARC LENGTH QUADRATIC EQUATIONS COEFFICIENTS
    # c1 = np.dot(dU.ravel(),dU.ravel()) + psi**2 * np.dot(DeltaF.ravel(),DeltaF.ravel())
    # c2 = 2.*np.dot(DU.ravel()+dU_actual.ravel(),dU_current.ravel()) + 2.*psi**2 * LoadFactor * np.dot(DeltaF.ravel(),DeltaF.ravel())
    # c3 = np.dot((DU+dU_actual).ravel(),(DU+dU_actual).ravel()) + psi**2 * LoadFactor**2 * np.dot(DeltaF.ravel(),DeltaF.ravel()) - dL**2
    # coeffs = [c1,c2,c3]

    # c1 = np.dot(dU_t.ravel(),dU_t.ravel()) + psi**2 * np.dot(NodalForces.ravel(),NodalForces.ravel())
    # c2 = 2.*np.dot(dU.ravel()+dU_b.ravel(),dU_t.ravel()) + 2.*psi**2 * Dlam * np.dot(NodalForces.ravel(),NodalForces.ravel())
    # c3 = np.dot((dU+dU_b).ravel(),(dU+dU_b).ravel()) + psi**2 * Dlam**2 * np.dot(NodalForces.ravel(),NodalForces.ravel()) - dL**2
    # coeffs = [c1,c2,c3]

    # # FIND THE NEW LOAD FACTOR
    # dlams = np.roots(coeffs)
    # dlam = np.real(dlams.max())
    # # print(c1,c2,c3,dlams, dlam)

    # # CORRECTOR
    # dU_iter = dU_b + dlam*dU_t
    # # print (dU_iter)
    # # exit()


    # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
    IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
        K.shape[0],formulation.nvar)

    # UPDATE EULERIAN COORDINATE
    Eulerx += IncDirichlet[:,:formulation.ndim]
    Eulerp += IncDirichlet[:,-1]
    # Eulerx += IncDirichlet[:,:formulation.ndim] + dU_iter[:,:formulation.ndim]
    # Eulerp += IncDirichlet[:,-1] + dU_iter[:,-1]

    # accumulated_load_factor += dlam
    # if Increment>0:
    #     DU = TotalDisp[:,:,Increment] - TotalDisp[:,:,Increment-1]
    # else:
    #     DU = np.zeros((mesh.points.shape[0],formulation.nvar))

    # DU = np.zeros((mesh.points.shape[0],formulation.nvar))


    while self.norm_residual > Tolerance or Iter==0:
        # GET THE REDUCED SYSTEM OF EQUATIONS
        K_b, F_b = boundary_condition.GetReducedMatrices(K,Residual)[:2]
        # SOLVE THE SYSTEM
        sol = solver.Solve(K_b,-F_b)
        # GET ITERATIVE SOLUTION
        # dU_b = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar) 
        dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar)


        # print(dlams)
        # exit()
        # LoadFactor += np.real(np.max(dlams))
        # print(LoadFactor)

        c1 = np.dot(dU_t.ravel(),dU_t.ravel()) + psi**2 * np.dot(NodalForces.ravel(),NodalForces.ravel())
        c2 = 2.*np.dot(dU.ravel()+dU_b.ravel(),dU_t.ravel()) + 2.*psi**2 * Dlam * np.dot(NodalForces.ravel(),NodalForces.ravel())
        c3 = np.dot((dU+dU_b).ravel(),(dU+dU_b).ravel()) + psi**2 * Dlam**2 * np.dot(NodalForces.ravel(),NodalForces.ravel()) - dL**2
        coeffs = [c1,c2,c3]

        # FIND THE NEW LOAD FACTOR
        dlams = np.roots(coeffs)
        dlam = np.real(dlams.max())
        print(dlam)

        # CORRECTOR
        dU_iter = dU_b + dlam*dU_t
        accumulated_load_factor += dlam


        # UPDATE THE EULERIAN COMPONENTS
        Eulerx += dU[:,:formulation.ndim]
        Eulerp += dU[:,-1]
        # Eulerx += dU_iter[:,:formulation.ndim]
        # Eulerp += dU_iter[:,-1]

        # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
        K, TractionForces = Assemble(self, function_spaces[0], formulation, mesh, material, solver,
            Eulerx,Eulerp)[:2]

        # FIND THE RESIDUAL
        Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] -\
            NodalForces[boundary_condition.columns_in]

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
        if np.isnan(self.norm_residual) or self.norm_residual>1e10:
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


    return Eulerx, Eulerp, K, Residual