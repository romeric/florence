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

from Florence.FiniteElements.Assembly import Assemble, AssembleExplicit
from Florence import Mesh

__all__ = ["ExplicitStructuralDynamicIntegrators"]


class ExplicitStructuralDynamicIntegrators(object):
    """Generic explicit structural time integerator based on central difference"""

    def __init__(self):
        super(ExplicitStructuralDynamicIntegrators, self).__init__()


    def GetBoundaryInfo(self, mesh, formulation, boundary_condition):

        all_dofs = np.arange(mesh.points.shape[0]*formulation.nvar)
        self.electric_dofs = all_dofs[formulation.nvar-1::formulation.nvar]
        self.mechanical_dofs = np.array([],dtype=np.int64)
        self.mechanical_dofs = np.setdiff1d(all_dofs,self.electric_dofs)


    def Solver(self, function_spaces, formulation, solver,
        TractionForces, M, NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):

        # CHECK FORMULATION
        if formulation.fields != "mechanics":
            raise NotImplementedError("Explicit solver for {} is not implemented yet".format(formulation.fields))

        # COMPUTE DAMPING MATRIX BASED ON MASS
        if fem_solver.include_physical_damping:
            raise NotImplementedError("Damping is not included in the explicit solver")

        TractionForces = TractionForces.ravel()
        Residual = Residual.ravel()
        if fem_solver.mass_type == "lumped":
            M = M.ravel()
            # COMPUTE INVERSE OF LUMPED MASS MATRIX
            invM = np.reciprocal(M)

        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = fem_solver.total_time/LoadIncrement
        dt = LoadFactor
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)

        if NeumannForces.ndim == 1:
            NeumannForces = NeumannForces[:,None]

        # INITIALISE VELOCITY AND ACCELERATION
        U0      = np.zeros((mesh.points.shape[0]*formulation.ndim))
        V0      = np.zeros((mesh.points.shape[0]*formulation.ndim))
        A0      = np.zeros((mesh.points.shape[0]*formulation.ndim))

        # COMPUTE INITIAL ACCELERATION FOR TIME STEP 0
        if fem_solver.mass_type == "lumped":
            A0[:] = (NeumannForces[:,0] - TractionForces).ravel()*invM
        else:
            # CHECK
            # A0[:] = solver.Solve(M,(NeumannForces[:,0] - TractionForces).ravel())
            M_b, F_b = boundary_condition.GetReducedMatrices(M,NeumannForces - TractionForces[:,None])[:2]
            sola = solver.Solve(M_b,F_b)
            A0[:] = boundary_condition.UpdateFreeDoFs(sola,TractionForces.shape[0],formulation.nvar).ravel()

        U00  = U0 - dt*V0 + (dt**2/2.)*A0
        U0   = boundary_condition.UpdateFreeDoFs(U0[boundary_condition.columns_in],TractionForces.shape[0],formulation.nvar)
        U00  = boundary_condition.UpdateFreeDoFs(U00[boundary_condition.columns_in],TractionForces.shape[0],formulation.nvar)

        TotalDisp[:,:,0] = U00
        TotalDisp[:,:,1] = U0

        save_counter = 1
        # TIME LOOP
        for Increment in range(2,LoadIncrement):

            t_increment = time()

            if boundary_condition.applied_dirichlet.ndim == 2:
                AppliedDirichletInc = boundary_condition.applied_dirichlet[:,Increment-1]
            else:
                # RAMP TYPE LOAD
                AppliedDirichletInc = boundary_condition.applied_dirichlet*(1.*Increment/LoadIncrement)

            # APPLY NEUMANN BOUNDARY CONDITIONS
            if NeumannForces.ndim == 2 and NeumannForces.shape[1]>1:
                NodalForces = NeumannForces[:,Increment-1]
            else:
                # RAMP TYPE LOAD
                NodalForces = NeumannForces.ravel()*(1.*Increment/LoadIncrement)

            Residual[:] = NodalForces - TractionForces

            if fem_solver.mass_type == "lumped":
                # Residual   += (2./dt**2)*M*TotalDisp[:,:,Increment-1].ravel() - (1./dt**2)*M*TotalDisp[:,:,Increment-2].ravel()
                Residual   += (2./dt**2)*M*U0.ravel() - (1./dt**2)*M*U00.ravel()
                U = dt**2*invM*Residual
                U = boundary_condition.UpdateFreeDoFs(U[boundary_condition.columns_in],TractionForces.shape[0],formulation.nvar)
                IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,TractionForces.shape[0],formulation.nvar)

            elif fem_solver.mass_type == "consistent":
                Residual   += (2./dt**2)*M.dot(U0.ravel()) - (1./dt**2)*M.dot(U00.ravel())
                # M_b, F_b = boundary_condition.GetReducedMatrices(M,Residual[:,None])[:2]
                F_b = boundary_condition.GetReducedVectors(Residual[:,None],only_residual=True)[0]
                U = solver.Solve(M_b,F_b*dt**2)
                U = boundary_condition.UpdateFreeDoFs(U,TractionForces.shape[0],formulation.nvar)
                IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,TractionForces.shape[0],formulation.nvar)

            # COMPUTE VELOCITY AND ACCELERATION
            V0[:] = (1./2./dt)*(U+U0).ravel()
            A0[:] = (1./dt**2)*(U-2.*U0+U00).ravel()

            # UPDATE GEOMETRY
            Eulerx[:,:] = mesh.points + U[:,:formulation.ndim]
            Eulerx[:,:] += IncDirichlet[:,:formulation.ndim]

            # UPDATE RESULTS FOR NEXT STEP
            U00[:,:formulation.ndim] = U0
            U0[:,:formulation.ndim]  = U

            # SAVE RESULTS
            if Increment % fem_solver.save_frequency == 0 or\
                (Increment == LoadIncrement - 1 and save_counter<TotalDisp.shape[2]):
                TotalDisp[:,:,save_counter] = Eulerx - mesh.points
                save_counter += 1

            # STORE THE INFORMATION IF EXPLICIT BLOWS UP
            if np.isnan(norm(U)) or norm(U[:,:formulation.ndim] - U0)/(norm(U0)+1e-12)>1e04:
                print("Explicit solver blew up! Norm of incremental solution is too large")
                TotalDisp = TotalDisp[:,:,:Increment]
                self.number_of_load_increments = Increment
                break

            # ASSEMBLE INTERNAL TRACTION FORCES
            t_assembly = time()
            TractionForces = AssembleExplicit(fem_solver,function_spaces[0], formulation, mesh, material,
                Eulerx, Eulerp)[0].ravel()
            print("Explicit assembly time is {} seconds".format(time()-t_assembly))


            # COMPUTE DISSIPATION OF ENERGY THROUGH TIME
            if fem_solver.compute_energy_dissipation:
                energy_info = self.ComputeEnergyDissipation(function_spaces[0],mesh,material,formulation,fem_solver,
                    Eulerx, U, NeumannForces, M, V0, Increment)
                formulation.energy_dissipation.append(energy_info[0])
                formulation.internal_energy.append(energy_info[1])
                formulation.kinetic_energy.append(energy_info[2])
                formulation.external_energy.append(energy_info[3])
            # COMPUTE DISSIPATION OF LINEAR MOMENTUM THROUGH TIME
            if fem_solver.compute_linear_momentum_dissipation:
                power_info = self.ComputePowerDissipation(function_spaces[0],mesh,material,formulation,fem_solver,
                    Eulerx, U, NeumannForces, M, V0, A0, Increment)
                formulation.power_dissipation.append(power_info[0])
                formulation.internal_power.append(power_info[1])
                formulation.kinetic_power.append(power_info[2])
                formulation.external_power.append(power_info[3])


            # PRINT LOG IF ASKED FOR
            if fem_solver.print_incremental_log:
                dmesh = Mesh()
                dmesh.points = U
                dmesh_bounds = dmesh.Bounds
                if formulation.fields == "electro_mechanics":
                    _bounds = np.zeros((2,formulation.nvar))
                    _bounds[:,:formulation.ndim] = dmesh_bounds
                    _bounds[:,-1] = [U[:,-1].min(),U[:,-1].max()]
                    print("\nMinimum and maximum incremental solution values at increment {} are \n".format(Increment),_bounds)
                else:
                    print("\nMinimum and maximum incremental solution values at increment {} are \n".format(Increment),dmesh_bounds)

            # SAVE INCREMENTAL SOLUTION IF ASKED FOR
            if fem_solver.save_incremental_solution:
                from scipy.io import savemat
                if fem_solver.incremental_solution_filename is not None:
                    savemat(fem_solver.incremental_solution_filename+"_"+str(Increment),{'solution':U},do_compression=True)
                else:
                    raise ValueError("No file name provided to save incremental solution")

            print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')

            # BREAK AT A SPECIFICED LOAD INCREMENT IF ASKED FOR
            if fem_solver.break_at_increment != -1 and fem_solver.break_at_increment is not None:
                if fem_solver.break_at_increment == Increment:
                    if fem_solver.break_at_increment < LoadIncrement - 1:
                        print("\nStopping at increment {} as specified\n\n".format(Increment))
                        TotalDisp = TotalDisp[:,:,:Increment]
                        fem_solver.number_of_load_increments = Increment
                    break

        if fem_solver.save_frequency != 1:
            if TotalDisp.shape[2] > save_counter:
                # IN CASE EXPLICIT SOLVER BLEW UP
                fem_solver.number_of_load_increments = TotalDisp.shape[2]
            else:
                fem_solver.number_of_load_increments = save_counter

        return TotalDisp



    def ComputeEnergyDissipation(self,function_space,mesh,material,formulation,fem_solver,
        Eulerx, TotalDisp, NeumannForces, M, velocities, Increment):

        ndim = material.ndim
        velocities = velocities.reshape(mesh.nnode,ndim)
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

        external_energy = np.dot(U.ravel(),NeumannForces.ravel())

        total_energy = internal_energy + kinetic_energy - external_energy
        return total_energy, internal_energy, kinetic_energy, external_energy



    def ComputePowerDissipation(self,function_space,mesh,material,formulation,fem_solver, Eulerx, TotalDisp,
        NeumannForces, M, velocities, accelerations, Increment):

        ndim = material.ndim
        velocities = velocities.reshape(mesh.nnode,ndim)
        accelerations = accelerations.reshape(mesh.nnode,ndim)
        internal_energy = 0.
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords    = Eulerx[mesh.elements[elem,:],:]
            VelocityElem       = velocities[mesh.elements[elem,:],:]

            internal_energy += formulation.GetLinearMomentum(function_space, material,
                LagrangeElemCoords, EulerElemCoords, VelocityElem, fem_solver, elem)

        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            kinetic_energy = np.dot(velocities.ravel(),M_mech.dot(accelerations.ravel()))
        else:
            kinetic_energy = np.dot(velocities.ravel(),M.dot(accelerations.ravel()))

        external_energy = np.dot(velocities.ravel(),NeumannForces.ravel())

        total_energy = internal_energy + kinetic_energy - external_energy
        return total_energy, internal_energy, kinetic_energy, external_energy
