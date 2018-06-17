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
from Florence.PostProcessing import PostProcess
from .StructuralDynamicIntegrator import StructuralDynamicIntegrator

__all__ = ["ExplicitStructuralDynamicIntegrator"]


class ExplicitStructuralDynamicIntegrator(StructuralDynamicIntegrator):
    """Generic explicit structural time integerator based on central difference"""

    def __init__(self):
        super(ExplicitStructuralDynamicIntegrator, self).__init__()


    def Solver(self, function_spaces, formulation, solver,
        TractionForces, M, NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):

        # CHECK FORMULATION
        if formulation.fields != "mechanics" and formulation.fields != "electro_mechanics":
            raise NotImplementedError("Explicit solver for {} is not available".format(formulation.fields))

        # GET BOUNDARY CONDITIONS INFROMATION
        self.GetBoundaryInfo(mesh, formulation, boundary_condition)

        # COMPUTE INVERSE OF LUMPED MASS MATRIX
        if formulation.fields == "electro_mechanics":
            if fem_solver.mass_type == "lumped":
                M = M.ravel()
                invM = np.zeros_like(M)
                invM[self.mechanical_dofs] = np.reciprocal(M[self.mechanical_dofs])
                # M_mech = M[self.mechanical_dofs]
                M_mech = M[self.mechanical_dofs]
            else:
                M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
        else:
            if fem_solver.mass_type == "lumped":
                M = M.ravel()
                M_mech = M
                invM = np.reciprocal(M)
            else:
                M_mech = M

        # COMPUTE DAMPING MATRIX BASED ON MASS
        if fem_solver.include_physical_damping:
            raise NotImplementedError("Damping is not included in the explicit solver")

        TractionForces = TractionForces.ravel()
        Residual = Residual.ravel()

        LoadIncrement = fem_solver.number_of_load_increments
        LoadFactor = fem_solver.total_time/LoadIncrement
        dt = LoadFactor
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)
        nnode = mesh.points.shape[0]

        if NeumannForces.ndim == 1:
            NeumannForces = NeumannForces[:,None]

        # INITIALISE VELOCITY AND ACCELERATION
        U0      = np.zeros((mesh.points.shape[0]*formulation.ndim))
        V0      = np.zeros((mesh.points.shape[0]*formulation.ndim))
        A0      = np.zeros((mesh.points.shape[0]*formulation.ndim))

        # COMPUTE INITIAL ACCELERATION FOR TIME STEP 0
        InitResidual = (NeumannForces[:,0] - TractionForces).ravel()
        if fem_solver.mass_type == "lumped":
            # A0[:] = (NeumannForces[:,0] - TractionForces).ravel()*invM
            A0[:] = InitResidual[self.mechanical_dofs]*invM[self.mechanical_dofs]
        else:
            #
            if formulation.fields == "electro_mechanics":
                M_b, F_b = M_mech[self.mech_in,:][:,self.mech_in], InitResidual[:,None][self.mech_in,0]
                sola = solver.Solve(M_b,F_b)
                A0[self.mech_in] = sola
            else:
                M_b, F_b = boundary_condition.GetReducedMatrices(M,InitResidual[:,None])[:2]
                sola = solver.Solve(M_b,F_b)
                A0[:] = boundary_condition.UpdateFreeDoFs(sola,TractionForces.shape[0],formulation.ndim).ravel()

        U00  = U0 - dt*V0 + (dt**2/2.)*A0
        U0   = self.UpdateFreeMechanicalDoFs(U0[self.mech_in],formulation.ndim*nnode,formulation.ndim)
        U00  = self.UpdateFreeMechanicalDoFs(U00[self.mech_in],formulation.ndim*nnode,formulation.ndim)

        TotalDisp[:,:formulation.ndim,0] = U00
        TotalDisp[:,:formulation.ndim,1] = U0

        # SET UP THE ELECTROSTATICS SOLVER PARAMETERS ONCE
        if formulation.fields == "electro_mechanics":
            self.SetupElectrostaticsImplicit(mesh, formulation, boundary_condition, material, fem_solver, solver, Eulerx, 0)

        save_counter = 2 if fem_solver.save_frequency == 1 else 1
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

            # Residual[:] = NodalForces - TractionForces
            Residual = NodalForces - TractionForces
            Residual = Residual[self.mechanical_dofs]

            if fem_solver.mass_type == "lumped":
                # Residual   += (2./dt**2)*M*TotalDisp[:,:,Increment-1].ravel() - (1./dt**2)*M*TotalDisp[:,:,Increment-2].ravel()
                Residual   += (2./dt**2)*M_mech*U0.ravel() - (1./dt**2)*M_mech*U00.ravel()
                U = dt**2*invM[self.mechanical_dofs]*Residual
                U = self.UpdateFreeMechanicalDoFs(U[self.mech_in],formulation.ndim*nnode,formulation.ndim)
                IncDirichlet = self.UpdateFixMechanicalDoFs(AppliedDirichletInc[self.columns_out_mech_reverse_idx],
                    formulation.ndim*nnode,formulation.ndim)

            elif fem_solver.mass_type == "consistent":
                # Residual   += (2./dt**2)*M.dot(U0.ravel()) - (1./dt**2)*M.dot(U00.ravel())
                # F_b = boundary_condition.GetReducedVectors(Residual[:,None],only_residual=True)[0]
                Residual   += (2./dt**2)*M_mech.dot(U0.ravel()) - (1./dt**2)*M_mech.dot(U00.ravel())
                F_b = Residual[:,None][self.mech_in,0]
                U = solver.Solve(M_b,F_b*dt**2)
                U = self.UpdateFreeMechanicalDoFs(U,formulation.ndim*nnode,formulation.ndim)
                IncDirichlet = self.UpdateFixMechanicalDoFs(AppliedDirichletInc[self.columns_out_mech_reverse_idx],
                    formulation.ndim*nnode,formulation.ndim)

            # COMPUTE VELOCITY AND ACCELERATION
            V0[:] = (1./2./dt)*(U+U0).ravel()
            A0[:] = (1./dt**2)*(U-2.*U0+U00).ravel()

            # UPDATE GEOMETRY
            Eulerx[:,:] = mesh.points + U[:,:formulation.ndim]
            Eulerx[:,:] += IncDirichlet[:,:formulation.ndim]

            # SOLVE ELECTROSTATICS PROBLEM
            if formulation.fields == "electro_mechanics":
                Eulerp[:] = self.SolveElectrostaticsImplicit(mesh, formulation, boundary_condition, material, fem_solver, solver, Eulerx, Increment)

            # SAVE RESULTS
            if Increment % fem_solver.save_frequency == 0 or\
                (Increment == LoadIncrement - 1 and save_counter<TotalDisp.shape[2]):
                TotalDisp[:,:formulation.ndim,save_counter] = Eulerx - mesh.points
                if formulation.fields == "electro_mechanics":
                    TotalDisp[:,-1,save_counter] = Eulerp.ravel()
                save_counter += 1

            # STORE THE INFORMATION IF EXPLICIT BLOWS UP
            tol = 1e200 if Increment < 5 else 10.
            if np.isnan(norm(U)) or np.abs(U.max()/(U0.max()+1e-14)) > tol:
                print("Explicit solver blew up! Norm of incremental solution is too large")
                TotalDisp = TotalDisp[:,:,:Increment]
                fem_solver.number_of_load_increments = Increment
                break

            # UPDATE RESULTS FOR NEXT STEP
            U00[:,:formulation.ndim] = U0
            U0[:,:formulation.ndim]  = U

            # ASSEMBLE INTERNAL TRACTION FORCES
            t_assembly = time()
            TractionForces = AssembleExplicit(fem_solver,function_spaces[0], formulation, mesh, material,
                Eulerx, Eulerp)[0].ravel()
            # CHECK CONTACT AND ASSEMBLE IF DETECTED
            if fem_solver.has_contact:
                TractionForcesContact = np.zeros_like(TractionForces)
                TractionForcesContact = fem_solver.contact_formulation.AssembleTractions(mesh,material,Eulerx).ravel()
                if formulation.fields == "electro_mechanics":
                    TractionForces[self.mechanical_dofs] += TractionForcesContact
                elif formulation.fields == "mechanics":
                    TractionForces += TractionForcesContact
                else:
                    raise NotImplementedError("Contact algorithm for {} is not available".format(formulation.fields))

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


            # LOG IF ASKED FOR
            self.LogSave(fem_solver, formulation, U, Eulerp, Increment)

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
                TotalDisp = TotalDisp[:,:,:save_counter]
                fem_solver.number_of_load_increments = TotalDisp.shape[2]
            else:
                fem_solver.number_of_load_increments = save_counter

        return TotalDisp


    def SetupElectrostaticsImplicit(self, mesh, formulation, boundary_condition, material, fem_solver, solver, Eulerx, Increment):
        """setup implicit electrostatic problem
        """

        from Florence import BoundaryCondition, FEMSolver, LaplacianSolver, IdealDielectric, AnisotropicIdealDielectric
        from Florence.VariationalPrinciple import LaplacianFormulation, ExplicitPenaltyContactFormulation

        # EMULATE ELECTROSTATICS MODEL
        emesh = deepcopy(mesh)

        if fem_solver.activate_explicit_multigrid:
            # WE GET THE MAX EPS - NOT ELEGANT BUT SEEMINGLY WORKS WELL
            eps_s = []
            for key, value in list(material.__dict__.items()):
                if "eps" in key:
                    eps_s.append(value)
            max_eps = max(eps_s)
            ematerial = IdealDielectric(emesh.InferSpatialDimension(),eps_1=max_eps)

            eanalysis_nature = "linear"
            eoptimise = True
        else:
            ematerial = deepcopy(material)
            ematerial.Hessian = ematerial.Permittivity
            ematerial.KineticMeasures = ematerial.ElectrostaticMeasures
            ematerial.H_VoigtSize = formulation.ndim
            ematerial.nvar = 1
            ematerial.fields = "electrostatics"

            eanalysis_nature = "nonlinear"
            eoptimise = False

        # SET UP BOUNDARY CONDITION DURING SOLUTION
        eboundary_condition = BoundaryCondition()

        eformulation = LaplacianFormulation(mesh)
        efem_solver = FEMSolver(number_of_load_increments=1,analysis_nature=eanalysis_nature,
            newton_raphson_tolerance=fem_solver.newton_raphson_tolerance,optimise=eoptimise)

        self.emesh = emesh
        self.ematerial = ematerial
        self.eformulation = eformulation
        self.eboundary_condition = eboundary_condition
        self.efem_solver = efem_solver


    def SolveElectrostaticsImplicit(self, mesh, formulation, boundary_condition, material, fem_solver, solver, Eulerx, Increment):
        """Solve implicit electrostatic problem
        """

        LoadIncrement = fem_solver.number_of_load_increments
        # IF ALL ELECTRIC DoFs ARE FIXED
        if mesh.points.shape[0] == self.electric_out.shape[0]:
            if self.applied_dirichlet_electric.ndim == 2:
                return self.applied_dirichlet_electric[:,Increment]
            else:
                # RAMP TYPE
                return self.applied_dirichlet_electric*(1.*Increment/LoadIncrement)

        # GET BOUNDARY CONDITIONS
        if boundary_condition.dirichlet_flags.ndim==3:
            self.eboundary_condition.dirichlet_flags = boundary_condition.dirichlet_flags[:,-1,Increment]
        else:
            # RAMP TYPE
            self.eboundary_condition.dirichlet_flags = boundary_condition.dirichlet_flags[:,-1]*(1.*Increment/LoadIncrement)
        if boundary_condition.neumann_flags is not None:
            if boundary_condition.neumann_data_applied_at == "node":
                if boundary_condition.neumann_flags.ndim==3:
                    self.eboundary_condition.neumann_flags = boundary_condition.neumann_flags[:,-1, Increment]
                else:
                    # RAMP TYPE
                    self.eboundary_condition.neumann_flags = boundary_condition.neumann_flags[:,-1]*(1.*Increment/LoadIncrement)
            if boundary_condition.neumann_data_applied_at == "face":
                if boundary_condition.neumann_flags.ndim==2:
                    self.eboundary_condition.neumann_flags = boundary_condition.neumann_flags[:,Increment]
                    self.eboundary_condition.applied_neumann = boundary_condition.applied_neumann[:,-1, Increment]
                    self.eboundary_condition.applied_neumann = self.eboundary_condition.applied_neumann[:,None]
                else:
                    # RAMP TYPE
                    self.eboundary_condition.neumann_flags = boundary_condition.neumann_flags[:]
                    self.eboundary_condition.applied_neumann = boundary_condition.applied_neumann[:,-1]*(1.*Increment/LoadIncrement)
                    self.eboundary_condition.applied_neumann = self.eboundary_condition.applied_neumann[:,None]

        # FILTER OUT CASES WHERE BOUNDARY CONDITION IS APPLIED BUT IS ZERO - RELEASE LOAD CYCLE IN DYNAMICS
        if np.allclose(self.eboundary_condition.dirichlet_flags[~np.isnan(self.eboundary_condition.dirichlet_flags)],0.):
            if self.eboundary_condition.neumann_flags is not None:
                if np.allclose(self.eboundary_condition.neumann_flags[~np.isnan(self.eboundary_condition.neumann_flags)],0.):
                    return 0.

        print("\nSolving the electrostatics problem iteratively")
        esolution = self.efem_solver.Solve(formulation=self.eformulation, mesh=self.emesh,
            material=self.ematerial, boundary_condition=self.eboundary_condition, solver=solver, Eulerx=Eulerx)
        print("Finished solving the electrostatics problem\n")

        return esolution.sol.ravel()





