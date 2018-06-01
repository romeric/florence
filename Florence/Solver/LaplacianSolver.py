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

from Florence.FiniteElements.Assembly import Assemble, AssembleExplicit
from Florence.PostProcessing import *
from Florence.Solver import LinearSolver
from Florence import Mesh


__all__ = ["LaplacianSolver"]


class LaplacianSolver(object):
    """Solver for linear and non-linear finite elements for Laplacian type operators.
    """

    def __init__(self, fem_solver=None):
        # super(LaplacianSolver, self).__init__(*args,**kwargs)
        if fem_solver is not None:
            self.__dict__.update(fem_solver.__dict__)
        # self.fem_solver = fem_solver
        self.__makeoutput__ = fem_solver.__makeoutput__


    def Solve(self, formulation=None, mesh=None,
        material=None, boundary_condition=None,
        function_spaces=None, solver=None, Eulerx=None, Eulerp=None):
        """Main solution routine for LaplacianSolver """

        # CHECK FOR ATTRIBUTE FOR LOWLEVEL ASSEMBLY
        if material.nature == "linear" and material.has_low_level_dispatcher and self.has_low_level_dispatcher:
            if hasattr(material,'e'):
                if material.e is None or isinstance(material.e, float):
                    if material.mtype == "IdealDielectric":
                        material.e = material.eps_1*np.eye(formulation.ndim, formulation.ndim)
                    else:
                        raise ValueError("For optimise=True, you need to provide the material permittivity tensor (ndimxndim)")
            else:
                raise ValueError("For optimise=True, you need to provide the material permittivity tensor (ndimxndim)")


        # INITIATE DATA FOR THE ANALYSIS
        NodalForces, Residual = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64), \
            np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64)
        # SET NON-LINEAR PARAMETERS
        self.NRConvergence = { 'Increment_'+str(Increment) : [] for Increment in range(self.number_of_load_increments) }

        # ALLOCATE FOR SOLUTION FIELDS
        if self.save_frequency == 1:
            # TotalDisp = np.zeros((mesh.points.shape[0],self.number_of_load_increments),dtype=np.float32)
            TotalDisp = np.zeros((mesh.points.shape[0],self.number_of_load_increments),dtype=np.float64)
        else:
            TotalDisp = np.zeros((mesh.points.shape[0],
                int(self.number_of_load_increments/self.save_frequency)),dtype=np.float64)

        # PRE-ASSEMBLY
        print('Assembling the system and acquiring neccessary information for the analysis...')
        tAssembly=time()

        # APPLY DIRICHELT BOUNDARY CONDITIONS AND GET DIRICHLET RELATED FORCES
        boundary_condition.GetDirichletBoundaryConditions(formulation, mesh, material, solver, self)

        # ALLOCATE FOR GEOMETRY - GetDirichletBoundaryConditions CHANGES THE MESH
        # SO EULERX SHOULD BE ALLOCATED AFTERWARDS
        if Eulerx is None:
            Eulerx = np.copy(mesh.points)
        if Eulerp is None:
            Eulerp = np.zeros((mesh.points.shape[0]))

        # FIND PURE NEUMANN (EXTERNAL) NODAL FORCE VECTOR
        NeumannForces = boundary_condition.ComputeNeumannForces(mesh, material, function_spaces,
            compute_traction_forces=True, compute_body_forces=self.add_self_weight)

        # ASSEMBLE STIFFNESS MATRIX AND TRACTION FORCES FOR THE FIRST TIME
        if self.analysis_type == "static":
            K, TractionForces, _, _ = Assemble(self, function_spaces[0], formulation, mesh, material,
                Eulerx, Eulerp)
        else:
            pass

        if self.analysis_nature == 'nonlinear':
            print('Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'seconds')
        else:
            print('Finished the assembly stage. Time elapsed was', time()-tAssembly, 'seconds')


        if self.analysis_type != 'static':
            pass
        else:
            if self.iterative_technique == "newton_raphson" or self.iterative_technique == "modified_newton_raphson":
                TotalDisp = self.StaticSolver(function_spaces, formulation, solver,
                    K,NeumannForces,NodalForces,Residual,
                    mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition)


        return self.__makeoutput__(mesh, TotalDisp, formulation, function_spaces, material)



    def StaticSolver(self, function_spaces, formulation, solver, K,
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

            if self.iterative_technique == "newton_raphson":
                Eulerx, Eulerp, K, Residual = self.NewtonRaphson(function_spaces, formulation, solver,
                    Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp,
                    material, boundary_condition, AppliedDirichletInc)
            elif self.iterative_technique == "modified_newton_raphson":
                Eulerx, Eulerp, K, Residual = self.ModifiedNewtonRaphson(function_spaces, formulation, solver,
                    Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp,
                    material, boundary_condition, AppliedDirichletInc)

            # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
            TotalDisp[:,Increment] = Eulerp

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
                if Increment==0:
                    TotalDisp = TotalDisp.ravel()
                else:
                    TotalDisp = TotalDisp[:,:Increment]
                self.number_of_load_increments = Increment
                break

            # BREAK AT A SPECIFICED LOAD INCREMENT IF ASKED FOR
            if self.break_at_increment != -1 and self.break_at_increment is not None:
                if self.break_at_increment == Increment:
                    if self.break_at_increment < LoadIncrement - 1:
                        print("\nStopping at increment {} as specified\n\n".format(Increment))
                        TotalDisp = TotalDisp[:,:,:Increment]
                        self.number_of_load_increments = Increment
                    break

        # print(TotalDisp.shape)
        return TotalDisp


    def NewtonRaphson(self, function_spaces, formulation, solver,
        Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
        boundary_condition, AppliedDirichletInc):

        Tolerance = self.newton_raphson_tolerance
        LoadIncrement = self.number_of_load_increments
        Iter = 0


        # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
        IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
            K.shape[0],formulation.nvar)
        # UPDATE EULERIAN COORDINATE
        # Eulerx += IncDirichlet[:,:formulation.ndim]
        # Eulerp += IncDirichlet[:,-1]
        Eulerp += IncDirichlet[:,0]

        while self.norm_residual > Tolerance or Iter==0:
            # GET THE REDUCED SYSTEM OF EQUATIONS
            K_b, F_b = boundary_condition.GetReducedMatrices(K,Residual)[:2]

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b,-F_b)

            # GET ITERATIVE SOLUTION
            dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar)

            # UPDATE THE EULERIAN COMPONENTS
            # # UPDATE THE GEOMETRY
            # Eulerx += dU[:,:formulation.ndim]
            # # GET ITERATIVE ELECTRIC POTENTIAL
            # Eulerp += dU[:,-1]
            Eulerp += dU[:,0]

            # BREAK FROM HERE IF ANALYSIS IS LINEAR
            if material.nature == 'linear':
                break

            # RE-ASSEMBLE - COMPUTE STIFFNESS AND INTERNAL TRACTION FORCES
            K, TractionForces = Assemble(self, function_spaces[0], formulation, mesh, material,
                Eulerx,Eulerp)[:2]

            # FIND THE RESIDUAL
            Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] -\
                NodalForces[boundary_condition.columns_in]

            # SAVE THE NORM
            self.rel_norm_residual = la.norm(Residual[boundary_condition.columns_in])
            if Iter==0:
                self.NormForces = la.norm(Residual[boundary_condition.columns_in])
                # AVOID DIVISION BY ZERO
                if np.isclose(self.NormForces,0.0):
                    self.NormForces = 1e-14
            self.norm_residual = np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces)

            # SAVE THE NORM
            self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
                self.norm_residual)

            print("Iteration {} for increment {}.".format(Iter, Increment) +\
                " Residual (abs) {0:>16.7g}".format(self.rel_norm_residual),
                "\t Residual (rel) {0:>16.7g}".format(self.norm_residual))

            # BREAK BASED ON RELATIVE NORM
            if np.abs(self.rel_norm_residual) < Tolerance:
                break

            # BREAK BASED ON INCREMENTAL SOLUTION - KEEP IT AFTER UPDATE
            if norm(dU) <=  self.newton_raphson_solution_tolerance:
                print("Incremental solution within tolerance i.e. norm(dU): {}".format(norm(dU)))
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


        return Eulerx, Eulerp, K, Residual





    def ModifiedNewtonRaphson(self, function_spaces, formulation, solver,
        Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
        boundary_condition, AppliedDirichletInc):

        from Florence.FiniteElements.Assembly import AssembleInternalTractionForces

        Tolerance = self.newton_raphson_tolerance
        LoadIncrement = self.number_of_load_increments
        Iter = 0


        # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
        IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
            K.shape[0],formulation.nvar)
        # UPDATE EULERIAN COORDINATE
        Eulerx += IncDirichlet[:,:formulation.ndim]
        Eulerp += IncDirichlet[:,-1]

        # ASSEMBLE STIFFNESS PER TIME STEP
        K, TractionForces = Assemble(self, function_spaces[0], formulation, mesh, material,
            Eulerx,Eulerp)[:2]

        while self.norm_residual > Tolerance or Iter==0:
            # GET THE REDUCED SYSTEM OF EQUATIONS
            K_b, F_b = boundary_condition.GetReducedMatrices(K,Residual)[:2]

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b,-F_b)

            # GET ITERATIVE SOLUTION
            dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar)

            # UPDATE THE EULERIAN COMPONENTS
            # UPDATE THE GEOMETRY
            Eulerx += dU[:,:formulation.ndim]
            # GET ITERATIVE ELECTRIC POTENTIAL
            Eulerp += dU[:,-1]

            # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
            TractionForces = AssembleInternalTractionForces(self, function_spaces[0], formulation, mesh, material,
                Eulerx,Eulerp)

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

            # BREAK BASED ON RELATIVE NORM
            if np.abs(self.rel_norm_residual) < Tolerance:
                break

            # BREAK BASED ON INCREMENTAL SOLUTION - KEEP IT AFTER UPDATE
            if norm(dU) <=  self.newton_raphson_solution_tolerance:
                print("Incremental solution within tolerance i.e. norm(dU): {}".format(norm(dU)))
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

        return Eulerx, Eulerp, K, Residual

