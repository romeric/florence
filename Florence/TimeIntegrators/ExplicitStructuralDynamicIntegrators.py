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
        if formulation.fields == "electro_mechanics":
            self.electric_dofs = all_dofs[formulation.nvar-1::formulation.nvar]
            self.mechanical_dofs = np.array([],dtype=np.int64)
            self.mechanical_dofs = np.setdiff1d(all_dofs,self.electric_dofs)

            # GET BOUNDARY CONDITON FOR THE REDUCED MECHANICAL SYSTEM
            self.columns_in_mech = np.intersect1d(boundary_condition.columns_in,self.mechanical_dofs)
            self.columns_in_mech_idx = np.in1d(self.mechanical_dofs,boundary_condition.columns_in)

            # GET BOUNDARY CONDITON FOR THE REDUCED ELECTROSTATIC SYSTEM
            self.columns_in_electric = np.intersect1d(boundary_condition.columns_in,self.electric_dofs)
            self.columns_in_electric_idx = np.in1d(self.electric_dofs,boundary_condition.columns_in)


            # GET FREE MECHANICAL DOFs
            self.columns_out_mech = np.intersect1d(boundary_condition.columns_out,self.mechanical_dofs)
            self.columns_out_mech_idx = np.in1d(self.mechanical_dofs,boundary_condition.columns_out)

            # GET FREE ELECTROSTATIC DOFs
            self.columns_out_electric = np.intersect1d(boundary_condition.columns_out,self.electric_dofs)
            self.columns_out_electric_idx = np.in1d(self.electric_dofs,boundary_condition.columns_out)

            self.applied_dirichlet_mech = boundary_condition.applied_dirichlet[np.in1d(boundary_condition.columns_out,self.columns_out_mech)]
            self.applied_dirichlet_electric = boundary_condition.applied_dirichlet[np.in1d(boundary_condition.columns_out,self.columns_out_electric)]

            # MAPPED QUANTITIES
            # all_dofs = np.arange(0,K.shape[0])
            out_idx = np.in1d(all_dofs,boundary_condition.columns_out)
            idx_electric = all_dofs[formulation.nvar-1::formulation.nvar]
            idx_mech = np.setdiff1d(all_dofs,idx_electric)

            # self.all_electric_dofs = np.arange(M.shape[0]/formulation.nvar)
            self.all_electric_dofs = np.arange(mesh.points.shape[0])
            self.electric_out = self.all_electric_dofs[out_idx[idx_electric]]
            self.electric_in = np.setdiff1d(self.all_electric_dofs,self.electric_out)

            self.all_mech_dofs = np.arange(mesh.points.shape[0]*formulation.ndim)
            self.mech_out = self.all_mech_dofs[out_idx[idx_mech]]
            self.mech_in = np.setdiff1d(self.all_mech_dofs,self.mech_out)

        elif formulation.fields == "mechanics":
            self.electric_dofs = []
            self.mechanical_dofs = all_dofs
            self.columns_out_mech = boundary_condition.columns_out

            self.mech_in = boundary_condition.columns_in
            self.mech_out = boundary_condition.columns_out

            self.applied_dirichlet_mech = boundary_condition.applied_dirichlet


    def Solver(self, function_spaces, formulation, solver,
        TractionForces, M, NeumannForces, NodalForces, Residual,
        mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, fem_solver):

        # CHECK FORMULATION
        # if formulation.fields != "mechanics":
            # raise NotImplementedError("Explicit solver for {} is not implemented yet".format(formulation.fields))

        # GET BOUNDARY CONDITIONS INFROMATION
        self.GetBoundaryInfo(mesh, formulation,boundary_condition)

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
            self.SetupElectrostaticsImplicit(mesh, formulation, boundary_condition, material, solver, Eulerx)

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

            # Residual[:] = NodalForces - TractionForces
            Residual = NodalForces - TractionForces
            Residual = Residual[self.mechanical_dofs]

            if fem_solver.mass_type == "lumped":
                # Residual   += (2./dt**2)*M*TotalDisp[:,:,Increment-1].ravel() - (1./dt**2)*M*TotalDisp[:,:,Increment-2].ravel()
                Residual   += (2./dt**2)*M_mech*U0.ravel() - (1./dt**2)*M_mech*U00.ravel()
                U = dt**2*invM[self.mechanical_dofs]*Residual
                U = self.UpdateFreeMechanicalDoFs(U[self.mech_in],formulation.ndim*nnode,formulation.ndim)
                IncDirichlet = self.UpdateFixMechanicalDoFs(AppliedDirichletInc[np.in1d(boundary_condition.columns_out,self.columns_out_mech)],
                    formulation.ndim*nnode,formulation.ndim)

            elif fem_solver.mass_type == "consistent":
                # Residual   += (2./dt**2)*M.dot(U0.ravel()) - (1./dt**2)*M.dot(U00.ravel())
                # F_b = boundary_condition.GetReducedVectors(Residual[:,None],only_residual=True)[0]
                Residual   += (2./dt**2)*M_mech.dot(U0.ravel()) - (1./dt**2)*M_mech.dot(U00.ravel())
                F_b = Residual[:,None][self.mech_in,0]
                U = solver.Solve(M_b,F_b*dt**2)
                U = self.UpdateFreeMechanicalDoFs(U,formulation.ndim*nnode,formulation.ndim)
                IncDirichlet = self.UpdateFixMechanicalDoFs(AppliedDirichletInc[np.in1d(boundary_condition.columns_out,self.columns_out_mech)],
                    formulation.ndim*nnode,formulation.ndim)

            # COMPUTE VELOCITY AND ACCELERATION
            V0[:] = (1./2./dt)*(U+U0).ravel()
            A0[:] = (1./dt**2)*(U-2.*U0+U00).ravel()

            # UPDATE GEOMETRY
            Eulerx[:,:] = mesh.points + U[:,:formulation.ndim]
            Eulerx[:,:] += IncDirichlet[:,:formulation.ndim]

            # SOLVE ELECTROSTATICS PROBLEM
            if formulation.fields == "electro_mechanics":
                Eulerp[:] = self.SolveElectrostaticsImplicit(mesh, formulation, boundary_condition, material, solver, Eulerx)

            # SAVE RESULTS
            if Increment % fem_solver.save_frequency == 0 or\
                (Increment == LoadIncrement - 1 and save_counter<TotalDisp.shape[2]):
                TotalDisp[:,:formulation.ndim,save_counter] = Eulerx - mesh.points
                if formulation.fields == "electro_mechanics":
                    TotalDisp[:,-1,save_counter] = Eulerp.ravel()
                save_counter += 1

            # STORE THE INFORMATION IF EXPLICIT BLOWS UP
            tol = 1e200 if Increment < 5 else 10.
            # if np.isnan(norm(U)) or norm(U - U0)/(norm(U0)+1e-14)> tol:
            if np.isnan(norm(U)) or np.abs(U.max()/(U0.max()+1e-14)) > tol:
                print("Explicit solver blew up! Norm of incremental solution is too large")
                TotalDisp = TotalDisp[:,:,:Increment]
                self.number_of_load_increments = Increment
                break

            # UPDATE RESULTS FOR NEXT STEP
            U00[:,:formulation.ndim] = U0
            U0[:,:formulation.ndim]  = U

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
                    _bounds[:,-1] = [Eulerp.min(),Eulerp.max()]
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
                TotalDisp = TotalDisp[:,:,:save_counter]
            else:
                fem_solver.number_of_load_increments = save_counter

        return TotalDisp


    def SetupElectrostaticsImplicit(self, mesh, formulation, boundary_condition, material, solver, Eulerx):
        """setup implicit electrostatic problem
        """

        from Florence import BoundaryCondition, FEMSolver, IdealDielectric
        from Florence.VariationalPrinciple import LaplacianFormulation
        emesh = deepcopy(mesh)
        emesh.points = np.copy(Eulerx)
        # ematerial = IdealDielectric(emesh.InferSpatialDimension(),eps_1=material.eps_2)
        ematerial = deepcopy(material)
        # ematerial.has_low_level_dispatcher = False
        ematerial.Hessian = ematerial.Permittivity
        ematerial.KineticMeasures = ematerial.ElectrostaticMeasures
        ematerial.H_VoigtSize = formulation.ndim
        ematerial.nvar = 1

        eboundary_condition = BoundaryCondition()
        # ONLY CONSTANT LOAD AT THE MOMENT
        eboundary_condition.dirichlet_flags = boundary_condition.dirichlet_flags[:,-1]
        if boundary_condition.neumann_flags is not None:
            eboundary_condition.neumann_flags = boundary_condition.neumann_flags[:,-1]
        eformulation = LaplacianFormulation(mesh)
        efem_solver = FEMSolver(number_of_load_increments=1,analysis_nature="nonlinear")

        self.emesh = emesh
        self.ematerial = ematerial
        self.eformulation = eformulation
        self.eboundary_condition = eboundary_condition
        self.efem_solver = efem_solver


    def SolveElectrostaticsImplicit(self, mesh, formulation, boundary_condition, material, solver, Eulerx):
        """Solve implicit electrostatic problem
        """

        # IF ALL ELECTRIC DoFs ARE FIXED
        if mesh.points.shape[0] == self.electric_out.shape[0]:
            return self.applied_dirichlet_electric

        print("\nSolving the electrostatics problem iteratively")
        esolution = self.efem_solver.Solve(formulation=self.eformulation, mesh=self.emesh,
            material=self.ematerial, boundary_condition=self.eboundary_condition, solver=solver, Eulerx=Eulerx)
        print("Finished solving the electrostatics problem\n")

        return esolution.sol.ravel()



    def UpdateFixMechanicalDoFs(self, AppliedDirichletInc, fsize, nvar):
        """Updates the geometry (DoFs) with incremental Dirichlet boundary conditions
            for fixed/constrained degrees of freedom only. Needs to be applied per time steps"""

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[self.mech_out,0] = AppliedDirichletInc

        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(int(TotalSol.shape[0]/nvar),nvar)

        return dU

    def UpdateFreeMechanicalDoFs(self, sol, fsize, nvar):
        """Updates the geometry with iterative solutions of Newton-Raphson
            for free degrees of freedom only. Needs to be applied per time NR iteration"""

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[self.mech_in,0] = sol

        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(int(TotalSol.shape[0]/nvar),nvar)

        return dU





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
