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
    """Generic structural time integerator based on Newmark's Beta"""

    def __init__(self):
        super(StructuralDynamicIntegrators, self).__init__()
        self.alpha_f = 1.0
        self.alpha_m = 1.0
        self.gamma   = 0.5
        self.beta    = 0.25


    def GetBoundaryInfo(self, mesh, formulation, boundary_condition):

        all_dofs = np.arange(mesh.points.shape[0]*formulation.nvar)
        self.electric_dofs = all_dofs[formulation.nvar-1::formulation.nvar]
        self.mechanical_dofs = np.array([],dtype=np.int64)
        self.mechanical_dofs = np.setdiff1d(all_dofs,self.electric_dofs)


    def Solver(self, function_spaces, formulation, solver, 
        K, M, NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):


        # COMPUTE DAMPING MATRIX BASED ON MASS
        D = 0.0
        if fem_solver.include_physical_damping:
            D = fem_solver.damping_factor*M

        # GET BOUNDARY CONDITIONS INFROMATION
        if formulation.fields == "electro_mechanics":
            self.GetBoundaryInfo(mesh, formulation,boundary_condition)
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            if fem_solver.include_physical_damping:
                D_mech = D[self.mechanical_dofs,:][:,self.mechanical_dofs]

        # INITIALISE VELOCITY AND ACCELERATION
        velocities     = np.zeros((mesh.points.shape[0],formulation.ndim))
        accelerations  = np.zeros((mesh.points.shape[0],formulation.ndim))

        # COMPUTE INITIAL ACCELERATION FOR TIME STEP 0
        InitResidual = Residual - NeumannForces[:,0][:,None]
        if formulation.fields == "electro_mechanics":
            accelerations[:,:] = solver.Solve(M_mech, -InitResidual[self.mechanical_dofs].ravel() 
                ).reshape(mesh.points.shape[0],formulation.ndim)
        else:
            accelerations[:,:] = solver.Solve(M, -InitResidual.ravel() ).reshape(mesh.points.shape[0],formulation.ndim)

        self.NRConvergence = fem_solver.NRConvergence
        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = fem_solver.total_time/LoadIncrement
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)

        if NeumannForces.ndim == 2 and NeumannForces.shape[1]==1:
            tmp = np.zeros((NeumannForces.shape[0],LoadIncrement))
            tmp[:,0] = NeumannForces[:,0]
            NeumannForces = tmp


        # TIME LOOP
        for Increment in range(1,LoadIncrement):

            # # print(np.linalg.norm(K.todense()))
            # print(np.linalg.norm(M.todense()))
            # # solver.SparsityPattern(M)
            # # exit()

            t_increment = time()

            # APPLY NEUMANN BOUNDARY CONDITIONS
            DeltaF = NeumannForces[:,Increment][:,None]
            NodalForces = DeltaF

            # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
            Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                boundary_condition.applied_dirichlet[:,Increment],LoadFactor=1.0,mass=M,only_residual=True)
            Residual -= DeltaF
            # Residual = -DeltaF
            # GET THE INCREMENTAL DIRICHLET
            AppliedDirichletInc = boundary_condition.applied_dirichlet[:,Increment]

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
            TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
            if formulation.fields == "electro_mechanics":
                TotalDisp[:,-1,Increment] = Eulerp

            # COMPUTE DISSIPATION OF ENERGY THROUGH TIME
            if fem_solver.compute_energy_dissipation:
                energy_info = self.ComputeEnergyDissipation(function_spaces[0],mesh,material,formulation,fem_solver, 
                    Eulerx, TotalDisp, NeumannForces, M, velocities, Increment)
                formulation.energy_dissipation.append(energy_info[0])
                formulation.internal_energy.append(energy_info[1])
                formulation.kinetic_energy.append(energy_info[2])
                formulation.external_energy.append(energy_info[3])
            # COMPUTE DISSIPATION OF LINEAR MOMENTUM THROUGH TIME
            if fem_solver.compute_linear_momentum_dissipation:
                power_info = self.ComputePowerDissipation(function_spaces[0],mesh,material,formulation,fem_solver, 
                    Eulerx, TotalDisp, NeumannForces, M, velocities, accelerations, Increment)
                formulation.power_dissipation.append(power_info[0])
                formulation.power_energy.append(power_info[1])
                formulation.power_energy.append(power_info[2])
                formulation.power_energy.append(power_info[3])


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

            # SAVE INCREMENTAL SOLUTION IF ASKED FOR
            if fem_solver.save_incremental_solution:
                from scipy.io import savemat
                if fem_solver.incremental_solution_filename is not None:
                    savemat(fem_solver.incremental_solution_filename+"_"+str(Increment),{'solution':TotalDisp[:,:,Increment]},do_compression=True)
                else:
                    raise ValueError("No file name provided to save incremental solution")


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

        return TotalDisp


    def NewtonRaphson(self, function_spaces, formulation, solver, 
        Increment, K, D, M, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
        boundary_condition, AppliedDirichletInc, fem_solver, velocities, accelerations):

        Tolerance = fem_solver.newton_raphson_tolerance
        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = fem_solver.total_time/fem_solver.number_of_load_increments
        Iter = 0

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
        Eulerx += IncDirichlet[:,:formulation.ndim]
        Eulerp = IncDirichlet[:,-1]


        while np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces) > Tolerance or Iter==0:
        # for ii in range(7):

            # GET EFFECTIVE STIFFNESS
            # K += (1./self.beta/LoadFactor**2)*M
            K += (self.gamma/self.beta/LoadFactor)*D + (1./self.beta/LoadFactor**2)*M
            # GET THE REDUCED SYSTEM OF EQUATIONS
            K_b, F_b, _ = boundary_condition.GetReducedMatrices(K,Residual)

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b,-F_b)

            # GET ITERATIVE SOLUTION
            dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar) 

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

            # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
            K, TractionForces, _, _ = Assemble(fem_solver,function_spaces[0], formulation, mesh, material, solver,
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
            if np.isnan(self.norm_residual) or self.norm_residual>1e12:
                fem_solver.newton_raphson_failed_to_converge = True
                break

        return Eulerx, Eulerp, K, Residual, velocities, accelerations





    def ComputeEnergyDissipation(self,function_space,mesh,material,formulation,fem_solver, 
        Eulerx, TotalDisp, NeumannForces, M, velocities, Increment):

        internal_energy = 0.
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
            
            internal_energy += formulation.GetEnergy(function_space, material, 
                LagrangeElemCoords, EulerElemCoords, fem_solver, elem)

        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            kinetic_energy = 0.5*np.dot(velocities.ravel(),M_mech.dot(velocities.ravel()))
        else:
            kinetic_energy = 0.5*np.dot(velocities.ravel(),M.dot(velocities.ravel()))

        external_energy = np.dot(TotalDisp[:,:,Increment].ravel(),NeumannForces[:,Increment])

        total_energy = internal_energy + kinetic_energy - external_energy
        return total_energy, internal_energy, kinetic_energy, external_energy



    def ComputePowerDissipation(self,function_space,mesh,material,formulation,fem_solver, Eulerx, TotalDisp, 
        NeumannForces, M, velocities, accelerations, Increment):

        internal_energy = 0.
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords    = Eulerx[mesh.elements[elem,:],:]
            VelocityElem       = velocities[mesh.elements[elem,:],:]
            
            internal_energy += formulation.GetPower(function_space, material, 
                LagrangeElemCoords, EulerElemCoords, VelocityElem, fem_solver, elem)

        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            kinetic_energy = np.dot(velocities.ravel(),M_mech.dot(accelerations.ravel()))
        else:
            kinetic_energy = np.dot(velocities.ravel(),M.dot(accelerations.ravel()))

        external_energy = np.dot(velocities.ravel(),NeumannForces[:,Increment])

        total_energy = internal_energy + kinetic_energy - external_energy
        return total_energy, internal_energy, kinetic_energy, external_energy


























# from __future__ import print_function
# import gc, os, sys
# import numpy as np 
# import scipy as sp
# import numpy.linalg as la
# import scipy.linalg as sla 
# from time import time
# from copy import deepcopy
# from warnings import warn
# from time import time

# from Florence.FiniteElements.Assembly import Assemble
# from Florence import Mesh

# __all__ = ["StructuralDynamicIntegrators"]


# class StructuralDynamicIntegrators(object):
#     """Generic structural time integerator based on Newmark's Beta"""

#     def __init__(self):
#         super(StructuralDynamicIntegrators, self).__init__()
#         self.alpha_f = 1.0
#         self.alpha_m = 1.0
#         self.gamma   = 0.5
#         self.beta    = 0.25


#     def GetBoundaryInfo(self, mesh, formulation, boundary_condition):

#         all_dofs = np.arange(mesh.points.shape[0]*formulation.nvar)
#         self.electric_dofs = all_dofs[formulation.nvar-1::formulation.nvar]
#         self.mechanical_dofs = np.array([],dtype=np.int64)
#         self.mechanical_dofs = np.setdiff1d(all_dofs,self.electric_dofs)


#     def Solver(self, function_spaces, formulation, solver, 
#         K, M, NeumannForces, NodalForces, Residual,
#         mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):

#         # COMPUTE DAMPING MATRIX BASED ON MASS
#         D = 0.0
#         if fem_solver.include_physical_damping:
#             D = fem_solver.damping_factor*M

#         # GET BOUNDARY CONDITIONS INFROMATION
#         if formulation.fields == "electro_mechanics":
#             self.GetBoundaryInfo(mesh, formulation,boundary_condition)
#             M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
#             if fem_solver.include_physical_damping:
#                 D_mech = D[self.mechanical_dofs,:][:,self.mechanical_dofs]

#         # INITIALISE VELOCITY AND ACCELERATION
#         velocities     = np.zeros((mesh.points.shape[0],formulation.ndim,fem_solver.number_of_load_increments))
#         accelerations  = np.zeros((mesh.points.shape[0],formulation.ndim,fem_solver.number_of_load_increments))

#         # COMPUTE INITIAL ACCELERATION FOR TIME STEP 0
#         Res = Residual - NeumannForces[:,0][:,None]
#         if formulation.fields == "electro_mechanics":
#             accelerations[:,:,0] = solver.Solve(M_mech, -Res[self.mechanical_dofs].ravel() ).reshape(mesh.points.shape[0],formulation.ndim)
#         else:
#             accelerations[:,:,0] = solver.Solve(M, -Res.ravel() ).reshape(mesh.points.shape[0],formulation.ndim)

#         self.NRConvergence = fem_solver.NRConvergence
#         LoadIncrement = fem_solver.number_of_load_increments
#         LoadFactor = fem_solver.total_time/LoadIncrement
#         AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)

#         if NeumannForces.ndim == 2 and NeumannForces.shape[1]==1:
#             tmp = np.zeros((NeumannForces.shape[0],LoadIncrement))
#             tmp[:,0] = NeumannForces[:,0]
#             NeumannForces = tmp


#         # TIME LOOP
#         for Increment in range(1,LoadIncrement):

#             t_increment = time()

#             # APPLY NEUMANN BOUNDARY CONDITIONS
#             DeltaF = NeumannForces[:,Increment][:,None]
#             NodalForces = DeltaF

#             # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
#             Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
#                 boundary_condition.applied_dirichlet[:,Increment],LoadFactor=1.0,mass=M,only_residual=True)
#             Residual -= DeltaF
#             # Residual = -DeltaF
#             # GET THE INCREMENTAL DIRICHLET
#             AppliedDirichletInc = boundary_condition.applied_dirichlet[:,Increment]

#             # COMPUTE INITIAL ACCELERATION - ONLY NEEDED IN CASES OF PRESTRETCHED CONFIGURATIONS
#             # accelerations[:,:,Increment] = solver.Solve(M, Residual.ravel() - \
#             #     K.dot(TotalDisp[:,:,Increment].ravel())).reshape(mesh.points.shape[0],formulation.nvar)

#             # LET NORM OF THE FIRST RESIDUAL BE THE NORM WITH RESPECT TO WHICH WE
#             # HAVE TO CHECK THE CONVERGENCE OF NEWTON RAPHSON. TYPICALLY THIS IS 
#             # NORM OF NODAL FORCES
#             if Increment==1:
#                 self.NormForces = np.linalg.norm(Residual)
#                 # AVOID DIVISION BY ZERO
#                 if np.isclose(self.NormForces,0.0):
#                     self.NormForces = 1e-14
#             self.norm_residual = np.linalg.norm(Residual)/self.NormForces


#             Eulerx, Eulerp, K, Residual, velocities, accelerations = self.NewtonRaphson(function_spaces, formulation, solver, 
#                 Increment, K, D, M, NodalForces, Residual, mesh, Eulerx, Eulerp,
#                 material,boundary_condition,AppliedDirichletInc, fem_solver, velocities, accelerations)

#             # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
#             TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
#             if formulation.fields == "electro_mechanics":
#                 TotalDisp[:,-1,Increment] = Eulerp

#             # COMPUTE DISSIPATION OF ENERGY THROUGH TIME
#             if fem_solver.compute_energy_dissipation:
#                 energy_info = self.ComputeEnergyDissipation(function_spaces[0],mesh,material,formulation,fem_solver, 
#                     Eulerx, TotalDisp, NeumannForces, M, velocities, Increment)
#                 formulation.energy_dissipation.append(energy_info[0])
#                 formulation.internal_energy.append(energy_info[1])
#                 formulation.kinetic_energy.append(energy_info[2])
#                 formulation.external_energy.append(energy_info[3])
#             # COMPUTE DISSIPATION OF LINEAR MOMENTUM THROUGH TIME
#             if fem_solver.compute_linear_momentum_dissipation:
#                 power_info = self.ComputePowerDissipation(function_spaces[0],mesh,material,formulation,fem_solver, 
#                     Eulerx, TotalDisp, NeumannForces, M, velocities, accelerations, Increment)
#                 formulation.power_dissipation.append(power_info[0])
#                 formulation.power_energy.append(power_info[1])
#                 formulation.power_energy.append(power_info[2])
#                 formulation.power_energy.append(power_info[3])


#             # PRINT LOG IF ASKED FOR
#             if fem_solver.print_incremental_log:
#                 dmesh = Mesh()
#                 dmesh.points = TotalDisp[:,:formulation.ndim,Increment]
#                 dmesh_bounds = dmesh.Bounds
#                 if formulation.fields == "electro_mechanics":
#                     _bounds = np.zeros((2,formulation.nvar))
#                     _bounds[:,:formulation.ndim] = dmesh_bounds
#                     _bounds[:,-1] = [TotalDisp[:,-1,Increment].min(),TotalDisp[:,-1,Increment].max()]
#                     print("\nMinimum and maximum incremental solution values at increment {} are \n".format(Increment),_bounds)
#                 else:
#                     print("\nMinimum and maximum incremental solution values at increment {} are \n".format(Increment),dmesh_bounds)


#             print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')

#             try:
#                 print('Norm of Residual is', 
#                     np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces), '\n')
#             except RuntimeWarning:
#                 print("Invalid value encountered in norm of Newton-Raphson residual")

#             # STORE THE INFORMATION IF NEWTON-RAPHSON FAILS
#             if fem_solver.newton_raphson_failed_to_converge:
#                 solver.condA = np.NAN
#                 TotalDisp = TotalDisp[:,:,:Increment-1]
#                 fem_solver.number_of_load_increments = Increment - 1
#                 break

#         return TotalDisp


#     def NewtonRaphson(self, function_spaces, formulation, solver, 
#         Increment, K, D, M, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
#         boundary_condition, AppliedDirichletInc, fem_solver, velocities, accelerations):

#         Tolerance = fem_solver.newton_raphson_tolerance
#         LoadIncrement = fem_solver.number_of_load_increments
#         LoadFactor = fem_solver.total_time/fem_solver.number_of_load_increments
#         Iter = 0

#         # EulerxPrev = np.copy(Eulerx)
#         # EulerVPrev = np.copy(velocities[:,:,Increment-1])
#         # EulerAPrev = np.copy(accelerations[:,:,Increment-1])

#         # PREDICTOR STEP
#         dumV = (1. - self.gamma/self.beta)*velocities[:,:,Increment-1] + (1. - self.gamma/2./self.beta)*LoadFactor*accelerations[:,:,Increment-1]
#         dumA = (-1./self.beta/LoadFactor)*velocities[:,:,Increment-1] - (1./2./self.beta)*(1.- 2.*self.beta)*accelerations[:,:,Increment-1]
#         velocities[:,:,Increment]    = dumV
#         accelerations[:,:,Increment] = dumA

#         if formulation.fields == "electro_mechanics":
#             M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
#             D_mech = D[self.mechanical_dofs,:][:,self.mechanical_dofs]
#             InertiaResidual = np.zeros((Residual.shape[0],1))
#             InertiaResidual[self.mechanical_dofs,0] = M_mech.dot(accelerations[:,:,Increment].ravel())
#             if fem_solver.include_physical_damping:
#                 InertiaResidual[self.mechanical_dofs,0] += D_mech.dot(velocities[:,:,Increment].ravel())
#         else:
#             InertiaResidual = np.zeros((Residual.shape[0],1))
#             InertiaResidual[:,0] = M.dot(accelerations[:,:,Increment].ravel())
#             if fem_solver.include_physical_damping:
#                 InertiaResidual[:,0] += D.dot(velocities[:,:,Increment].ravel())
#         Residual[boundary_condition.columns_in] += InertiaResidual[boundary_condition.columns_in]
        

#         # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
#         IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
#             K.shape[0],formulation.nvar)
#         # UPDATE EULERIAN COORDINATE
#         Eulerx += IncDirichlet[:,:formulation.ndim]
#         Eulerp = IncDirichlet[:,-1]


#         while np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces) > Tolerance or Iter==0:
#         # for ii in range(7):

#             # GET EFFECTIVE STIFFNESS
#             # K += (1./self.beta/LoadFactor**2)*M
#             K += (self.gamma/self.beta/LoadFactor)*D + (1./self.beta/LoadFactor**2)*M
#             # GET THE REDUCED SYSTEM OF EQUATIONS
#             K_b, F_b, _ = boundary_condition.GetReducedMatrices(K,Residual)

#             # SOLVE THE SYSTEM
#             sol = solver.Solve(K_b,-F_b)

#             # GET ITERATIVE SOLUTION
#             dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar) 

#             # UPDATE THE GEOMETRY
#             Eulerx += dU[:,:formulation.ndim]
#             # GET ITERATIVE ELECTRIC POTENTIAL
#             Eulerp += dU[:,-1]

#             # UPDATE VELOCITY AND ACCELERATION
#             velocities[:,:,Increment]    += self.gamma/self.beta/LoadFactor*dU[:,:formulation.ndim]
#             accelerations[:,:,Increment] += 1./self.beta/LoadFactor**2*dU[:,:formulation.ndim]

#             # OR ALTERNATIVELY
#             # dumA = 1./self.beta/LoadFactor**2*(Eulerx - EulerxPrev) -\
#             #     1./self.beta/LoadFactor*(EulerV) -\
#             #     1./2./self.beta*(1. - 2.*self.beta)*(EulerA)
#             # dumV = (1. - self.gamma/self.beta)*(EulerV) +\
#             #     (1. - self.gamma/2./self.beta)*LoadFactor*(EulerA) +\
#             #     self.gamma/self.beta/LoadFactor*(Eulerx - EulerxPrev)
#             # velocities[:,:,Increment]    = dumV
#             # accelerations[:,:,Increment] = dumA

#             # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
#             K, TractionForces, _, _ = Assemble(fem_solver,function_spaces[0], formulation, mesh, material, solver,
#                 Eulerx, Eulerp)

#             # FIND INITIAL RESIDUAL
#             if formulation.fields == "electro_mechanics":
#                 # M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
#                 InertiaResidual = np.zeros((TractionForces.shape[0],1))
#                 InertiaResidual[self.mechanical_dofs,0] = M_mech.dot(accelerations[:,:,Increment].ravel())
#                 if fem_solver.include_physical_damping:
#                     InertiaResidual[self.mechanical_dofs,0] += D_mech.dot(velocities[:,:,Increment].ravel())

#             else:
#                 InertiaResidual = np.zeros((TractionForces.shape[0],1))
#                 InertiaResidual[:,0] = M.dot(accelerations[:,:,Increment].ravel())
#                 if fem_solver.include_physical_damping:
#                     InertiaResidual[:,0] += D.dot(velocities[:,:,Increment].ravel())


#             # UPDATE RESIDUAL
#             Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] \
#             - NodalForces[boundary_condition.columns_in] + InertiaResidual[boundary_condition.columns_in]


#             # SAVE THE NORM
#             self.rel_norm_residual = la.norm(Residual[boundary_condition.columns_in])
#             if Iter==0:
#                 self.NormForces = la.norm(Residual[boundary_condition.columns_in])
#             self.norm_residual = np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces) 

#             # SAVE THE NORM 
#             self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
#                 self.norm_residual)
            
#             print("Iteration {} for increment {}.".format(Iter, Increment) +\
#                 " Residual (abs) {0:>16.7g}".format(self.rel_norm_residual), 
#                 "\t Residual (rel) {0:>16.7g}".format(self.norm_residual))

#             if np.abs(self.rel_norm_residual) < Tolerance:
#                 break

#             # UPDATE ITERATION NUMBER
#             Iter +=1

#             if Iter==fem_solver.maximum_iteration_for_newton_raphson and formulation.fields == "electro_mechanics":
#                 raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

#             if Iter==fem_solver.maximum_iteration_for_newton_raphson:
#                 fem_solver.newton_raphson_failed_to_converge = True
#                 break
#             if np.isnan(self.norm_residual) or self.norm_residual>1e16:
#                 fem_solver.newton_raphson_failed_to_converge = True
#                 break

#         return Eulerx, Eulerp, K, Residual, velocities, accelerations





#     def ComputeEnergyDissipation(self,function_space,mesh,material,formulation,fem_solver, 
#         Eulerx, TotalDisp, NeumannForces, M, velocities, Increment):

#         internal_energy = 0.
#         for elem in range(mesh.nelem):
#             LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
#             EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
            
#             internal_energy += formulation.GetEnergy(function_space, material, 
#                 LagrangeElemCoords, EulerElemCoords, fem_solver, elem)

#         if formulation.fields == "electro_mechanics":
#             M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
#             kinetic_energy = 0.5*np.dot(velocities[:,:,Increment].ravel(),M_mech.dot(velocities[:,:,Increment].ravel()))

#         else:
#             kinetic_energy = 0.5*np.dot(velocities[:,:,Increment].ravel(),M.dot(velocities[:,:,Increment].ravel()))

#         external_energy = np.dot(TotalDisp[:,:,Increment].ravel(),NeumannForces[:,Increment])

#         total_energy = internal_energy + kinetic_energy - external_energy
#         return total_energy, internal_energy, kinetic_energy, external_energy



#     def ComputePowerDissipation(self,function_space,mesh,material,formulation,fem_solver, Eulerx, TotalDisp, 
#         NeumannForces, M, velocities, accelerations, Increment):

#         internal_energy = 0.
#         for elem in range(mesh.nelem):
#             LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
#             EulerElemCoords    = Eulerx[mesh.elements[elem,:],:]
#             VelocityElem       = velocities[mesh.elements[elem,:],:,Increment]
            
#             internal_energy += formulation.GetPower(function_space, material, 
#                 LagrangeElemCoords, EulerElemCoords, VelocityElem, fem_solver, elem)

#         if formulation.fields == "electro_mechanics":
#             M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
#             kinetic_energy = np.dot(velocities[:,:,Increment].ravel(),M_mech.dot(accelerations[:,:,Increment].ravel()))

#         else:
#             kinetic_energy = np.dot(velocities[:,:,Increment].ravel(),M.dot(accelerations[:,:,Increment].ravel()))

#         external_energy = np.dot(velocities[:,:,Increment].ravel(),NeumannForces[:,Increment])

#         total_energy = internal_energy + kinetic_energy - external_energy
#         return total_energy, internal_energy, kinetic_energy, external_energy

