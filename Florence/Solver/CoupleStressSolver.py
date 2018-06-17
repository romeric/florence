from __future__ import print_function
import gc, os, sys, inspect
from copy import deepcopy
from warnings import warn
from time import time
import numpy as np
import scipy as sp
from numpy.linalg import norm
from Florence.Utils import insensitive
from Florence.Tensor import trace, Voigt

from Florence.PostProcessing import *
from Florence.Solver import LinearSolver, FEMSolver

__all__ = ["CoupleStressSolver"]

class CoupleStressSolver(FEMSolver):

    def __init__(self,**kwargs):
        if 'static_condensation' in kwargs.keys():
            self.static_condensation = kwargs['static_condensation']
        else:
            self.static_condensation = True

        if 'assembly_print_counter' in kwargs.keys():
            self.assembly_print_counter = kwargs['assembly_print_counter']
            del kwargs['assembly_print_counter']
        else:
            self.assembly_print_counter = 500

        if 'lump_rhs' in kwargs.keys():
            self.lump_rhs = kwargs['lump_rhs']
            del kwargs['lump_rhs']
        else:
            self.lump_rhs = False


        super(CoupleStressSolver, self).__init__(**kwargs)
        self.gamma   = 0.5
        self.beta    = 0.25

    def GetBoundaryInfo(self, mesh, formulation, boundary_condition):

        all_dofs = np.arange(mesh.points.shape[0]*formulation.nvar)
        if formulation.fields == "electro_mechanics" or formulation.fields == "flexoelectric":
            self.electric_dofs = all_dofs[formulation.nvar-1::formulation.nvar]
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
            out_idx = np.in1d(all_dofs,boundary_condition.columns_out)
            idx_electric = all_dofs[formulation.nvar-1::formulation.nvar]
            idx_mech = np.setdiff1d(all_dofs,idx_electric)

            self.all_electric_dofs = np.arange(mesh.points.shape[0])
            self.electric_out = self.all_electric_dofs[out_idx[idx_electric]]
            self.electric_in = np.setdiff1d(self.all_electric_dofs,self.electric_out)

            self.all_mech_dofs = np.arange(mesh.points.shape[0]*formulation.ndim)
            self.mech_out = self.all_mech_dofs[out_idx[idx_mech]]
            self.mech_in = np.setdiff1d(self.all_mech_dofs,self.mech_out)

            # LOCAL NUMBERING OF FIELDS
            self.all_local_dofs = np.arange(mesh.elements.shape[1]*formulation.nvar)
            self.all_local_electric_dofs = self.all_local_dofs[formulation.nvar-1::formulation.nvar]
            self.all_local_mech_dofs = np.setdiff1d(self.all_local_dofs,self.all_local_electric_dofs)

            # INTERMIX RAVELLED LOCAL INDICES
            matrix_shape = (mesh.elements.shape[1]*formulation.nvar,mesh.elements.shape[1]*formulation.nvar)
            self.idx_uu = np.ravel_multi_index(np.meshgrid(self.all_local_mech_dofs,self.all_local_mech_dofs), matrix_shape)
            self.idx_up = np.ravel_multi_index(np.meshgrid(self.all_local_mech_dofs,self.all_local_electric_dofs), matrix_shape)
            self.idx_pu = np.ravel_multi_index(np.meshgrid(self.all_local_electric_dofs,self.all_local_mech_dofs), matrix_shape)
            self.idx_pp = np.ravel_multi_index(np.meshgrid(self.all_local_electric_dofs,self.all_local_electric_dofs), matrix_shape)


        elif formulation.fields == "mechanics" or formulation.fields == "couple_stress":
            self.electric_dofs = []
            self.mechanical_dofs = all_dofs
            self.columns_out_mech = boundary_condition.columns_out

            self.mech_in = boundary_condition.columns_in
            self.mech_out = boundary_condition.columns_out

            self.applied_dirichlet_mech = boundary_condition.applied_dirichlet



    def __checkdata__(self, material, boundary_condition,
        formulation, mesh, function_spaces, solver, contact_formulation=None):

        ## IMPORTANT
        self.requires_geometry_update = False
        self.analysis_nature = "linear"

        function_spaces, solver = super(CoupleStressSolver,self).__checkdata__(material, boundary_condition,
            formulation, mesh, function_spaces, solver, contact_formulation=contact_formulation)

        # BUILD THE TANGENT OPERATORS BEFORE HAND
        if material.mtype == "CoupleStressModel":
            I = np.eye(material.ndim,material.ndim)

            material.elasticity_tensor = Voigt(material.lamb*np.einsum('ij,kl',I,I)+material.mu*(np.einsum('ik,jl',I,I)+np.einsum('il,jk',I,I)),1)
            material.gradient_elasticity_tensor = 2.*material.eta*I
            material.coupling_tensor0 = np.zeros((material.elasticity_tensor.shape[0],
                material.gradient_elasticity_tensor.shape[0]))

            # print material.elasticity_tensor
            ngauss = function_spaces[0].AllGauss.shape[0]
            d0 = material.elasticity_tensor.shape[0]
            d1 = material.elasticity_tensor.shape[1]
            d2 = material.gradient_elasticity_tensor.shape[0]
            d3 = material.gradient_elasticity_tensor.shape[1]
            material.elasticity_tensors = np.tile(material.elasticity_tensor.ravel(),ngauss).reshape(ngauss,d0,d1)
            material.gradient_elasticity_tensors = np.tile(material.gradient_elasticity_tensor.ravel(),ngauss).reshape(ngauss,d2,d3)

        elif material.mtype == "IsotropicLinearFlexoelectricModel":
            I = np.eye(material.ndim,material.ndim)

            material.elasticity_tensor = Voigt(material.lamb*np.einsum('ij,kl',I,I)+material.mu*(np.einsum('ik,jl',I,I)+np.einsum('il,jk',I,I)),1)
            material.gradient_elasticity_tensor = 2.*material.eta*I
            material.coupling_tensor0 = np.zeros((material.elasticity_tensor.shape[0],
                material.gradient_elasticity_tensor.shape[0]))
            material.piezoelectric_tensor = material.P
            material.flexoelectric_tensor = material.f
            material.dielectric_tensor = -material.eps*np.eye(material.ndim,material.ndim)

            # print material.elasticity_tensor
            ngauss = function_spaces[0].AllGauss.shape[0]
            d0 = material.elasticity_tensor.shape[0]
            d1 = material.elasticity_tensor.shape[1]
            d2 = material.gradient_elasticity_tensor.shape[0]
            d3 = material.gradient_elasticity_tensor.shape[1]
            material.elasticity_tensors = np.tile(material.elasticity_tensor.ravel(),ngauss).reshape(ngauss,d0,d1)
            material.gradient_elasticity_tensors = np.tile(material.gradient_elasticity_tensor.ravel(),ngauss).reshape(ngauss,d2,d3)
            material.piezoelectric_tensors = np.tile(material.P.ravel(),ngauss).reshape(ngauss,material.P.shape[0],material.P.shape[1])
            material.flexoelectric_tensors = np.tile(material.f.ravel(),ngauss).reshape(ngauss,material.f.shape[0],material.f.shape[1])
            material.dielectric_tensors = np.tile(material.dielectric_tensor.ravel(),ngauss).reshape(ngauss,material.ndim,material.ndim)

            factor = -1.
            material.H_Voigt = np.zeros((ngauss,material.H_VoigtSize,material.H_VoigtSize))
            for i in range(ngauss):
                H1 = np.concatenate((material.elasticity_tensor,factor*material.piezoelectric_tensor),axis=1)
                H2 = np.concatenate((factor*material.piezoelectric_tensor.T,material.dielectric_tensor),axis=1)
                H_Voigt = np.concatenate((H1,H2),axis=0)
                material.H_Voigt[i,:,:] = H_Voigt


        return function_spaces, solver



    def Solve(self, formulation=None, mesh=None,
        material=None, boundary_condition=None,
        function_spaces=None, solver=None,
        contact_formulation=None):
        """Main solution routine for FEMSolver """

        # CHECK DATA CONSISTENCY
        mesh = formulation.meshes[0]
        #---------------------------------------------------------------------------#
        function_spaces, solver = self.__checkdata__(material, boundary_condition,
            formulation, mesh, function_spaces, solver, contact_formulation=contact_formulation)
        #---------------------------------------------------------------------------#
        caller = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
        if caller != "Solve":
            self.PrintPreAnalysisInfo(mesh, formulation)
        #---------------------------------------------------------------------------#

        # INITIATE DATA FOR NON-LINEAR ANALYSIS
        NodalForces, Residual = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64), \
            np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64)
        # SET NON-LINEAR PARAMETERS
        self.NRConvergence = { 'Increment_'+str(Increment) : [] for Increment in range(self.number_of_load_increments) }

        # ALLOCATE FOR SOLUTION FIELDS
        TotalDisp = np.zeros((mesh.points.shape[0],formulation.nvar,self.number_of_load_increments),dtype=np.float64)
        TotalW = np.zeros((formulation.meshes[1].points.shape[0],formulation.ndim,self.number_of_load_increments),dtype=np.float64)
        TotalS = np.zeros((formulation.meshes[2].points.shape[0],formulation.ndim,self.number_of_load_increments),dtype=np.float64)
        # TotalDisp = np.zeros((mesh.points.shape[0],int(formulation.ndim**2),self.number_of_load_increments),dtype=np.float64)

        # PRE-ASSEMBLY
        if caller != "Solve":
            print('Assembling the system and acquiring neccessary information for the analysis...')
        tAssembly=time()

        # APPLY DIRICHELT BOUNDARY CONDITIONS AND GET DIRICHLET RELATED FORCES
        boundary_condition.GetDirichletBoundaryConditions(formulation, mesh, material, solver, self)

        # GET BOUNDARY INFO
        self.GetBoundaryInfo(mesh,formulation,boundary_condition)

        # ALLOCATE FOR GEOMETRY - GetDirichletBoundaryConditions CHANGES THE MESH
        # SO EULERX SHOULD BE ALLOCATED AFTERWARDS
        Eulerx = np.copy(formulation.meshes[0].points)
        Eulerw = np.zeros_like(formulation.meshes[1].points)
        Eulers = np.zeros_like(formulation.meshes[1].points)
        Eulerp = np.zeros((formulation.meshes[0].points.shape[0]))

        # FIND PURE NEUMANN (EXTERNAL) NODAL FORCE VECTOR
        NeumannForces = boundary_condition.ComputeNeumannForces(mesh, material, function_spaces,
            compute_traction_forces=True, compute_body_forces=self.add_self_weight)
        # NeumannForces = np.zeros((mesh.points.shape[0]*formulation.ndim))

        # ASSEMBLE STIFFNESS MATRIX AND TRACTION FORCES
        if self.analysis_type != 'static':
            if formulation.fields == "couple_stress" or formulation.fields == "flexoelectric":
                K, TractionForces, _, M = formulation.Assemble(self, material, Eulerx, Eulerw, Eulers, Eulerp)
            elif formulation.fields == "mechanics" or formulation.fields == "electro_mechanics":
                # STANDARD MECHANINCS ELECTROMECHANICS DYNAMIC ANALYSIS ARE DISPATCHED HERE
                from Florence.FiniteElements.Assembly import Assemble
                fspace = function_spaces[0] if (mesh.element_type=="hex" or mesh.element_type=="quad") else function_spaces[1]
                K, TractionForces, _, M = Assemble(self, fspace, formulation, mesh, material,
                    Eulerx, Eulerp)
        else:
            K, TractionForces = formulation.Assemble(self, material, Eulerx, Eulerw, Eulers, Eulerp)[:2]

        print('Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'seconds')

        if self.analysis_type != 'static':
            boundary_condition.ConvertStaticsToDynamics(mesh, self.number_of_load_increments)
            TotalDisp, TotalW, TotalS = self.DynamicSolver(formulation, solver,
                K, M, NeumannForces, NodalForces, Residual,
                mesh, TotalDisp, TotalW, TotalS, Eulerx, Eulerw, Eulers, Eulerp, material, boundary_condition)
        else:
            TotalDisp, TotalW, TotalS = self.StaticSolver(formulation, solver,
                K, NeumannForces, NodalForces, Residual,
                mesh, TotalDisp, TotalW, TotalS, Eulerx, Eulerw, Eulers, Eulerp, material, boundary_condition)


        solution = self.__makeoutput__(mesh, TotalDisp, formulation, function_spaces, material)
        if formulation.fields == "couple_stress" or formulation.fields == "flexoelectric":
            solution.solW = TotalW
            solution.solS = TotalS
        return solution



    def StaticSolver(self, formulation, solver, K,
            NeumannForces, NodalForces, Residual,
            mesh, TotalDisp, TotalW, TotalS, Eulerx, Eulerw, Eulers, Eulerp, material, boundary_condition):


        LoadIncrement = self.number_of_load_increments
        LoadFactor = 1./LoadIncrement
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)
        Increment = 0

        post_process = PostProcess(formulation.ndim,formulation.nvar)
        post_process.SetAnalysis(analysis_type=self.analysis_type, analysis_nature=self.analysis_nature)

        # APPLY NEUMANN BOUNDARY CONDITIONS
        DeltaF = LoadFactor*NeumannForces
        Residual[:,:] += DeltaF

        K_b, F_b, _ = boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                boundary_condition.applied_dirichlet,LoadFactor=LoadFactor)


        for Increment in range(LoadIncrement):
            t_increment=time()
            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b, F_b, reuse_factorisation=True)

            AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet
            dU = post_process.TotalComponentSol(sol, boundary_condition.columns_in,
                boundary_condition.columns_out, AppliedDirichletInc,0,K.shape[0])

            # STORE TOTAL SOLUTION DATA
            TotalDisp[:,:,Increment] += dU

            # REDUCED ACCUMULATED FORCE
            Residual[:,:] += DeltaF
            F_b = boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                AppliedDirichletInc,LoadFactor=1.0,
                only_residual=True)[boundary_condition.columns_in,0]
            # F_b = boundary_condition.GetReducedVectors(Residual,only_residual=True)[0]

            # UPDATE
            Eulerx += dU[:,:formulation.ndim]
            Eulerp += dU[:,-1]
            TotalW[:,:,Increment], TotalS[:,:,Increment] = formulation.GetAugmentedSolution(self,
                material, TotalDisp, Eulerx, Eulerw, Eulers, Eulerp)

            # LOG REQUESTS
            self.LogSave(formulation, TotalDisp, Increment)

            # BREAK AT A SPECIFICED LOAD INCREMENT IF ASKED FOR
            if self.break_at_increment != -1 and self.break_at_increment is not None:
                if self.break_at_increment == Increment:
                    if self.break_at_increment < LoadIncrement - 1:
                        print("\nStopping at increment {} as specified\n\n".format(Increment))
                        TotalDisp = TotalDisp[:,:,:Increment]
                        self.number_of_load_increments = Increment
                    break

            print('Finished Load increment', Increment, 'in', time()-t_increment, 'seconds\n')

        # ADD EACH INCREMENTAL CONTRIBUTION TO MAKE IT CONSISTENT WITH THE NONLINEAR ANALYSIS
        for i in range(TotalDisp.shape[2]-1,0,-1):
            TotalDisp[:,:,i] = np.sum(TotalDisp[:,:,:i+1],axis=2)

        # COMPUTE DISSIPATION OF ENERGY THROUGH TIME
        if self.compute_energy:
            energy_info = self.ComputeEnergy(formulation.function_spaces[0],mesh,material,formulation,Eulerx,Eulerp)
            formulation.strain_energy = energy_info[0]
            formulation.electrical_energy = energy_info[1]

        solver.CleanUp()

        return TotalDisp, TotalW, TotalS



    def DynamicSolver(self, formulation, solver, K, M,
            NeumannForces, NodalForces, Residual,
            mesh, TotalDisp, TotalW, TotalS, Eulerx, Eulerw, Eulers, Eulerp, material, boundary_condition):

        LoadIncrement = self.number_of_load_increments
        LoadFactor = self.total_time/LoadIncrement

        post_process = PostProcess(formulation.ndim,formulation.nvar)
        post_process.SetAnalysis(analysis_type=self.analysis_type, analysis_nature=self.analysis_nature)

        if NeumannForces.ndim == 2 and NeumannForces.shape[1]==1:
            tmp = np.zeros((NeumannForces.shape[0],LoadIncrement))
            tmp[:,0] = NeumannForces[:,0]
            NeumannForces = tmp

        dumU = np.zeros((mesh.points.shape[0]*formulation.nvar))
        dumU[boundary_condition.columns_out] = boundary_condition.applied_dirichlet[:,0]
        TotalDisp[:,:formulation.nvar,0] = dumU.reshape(mesh.points.shape[0],formulation.nvar)
        # INITIALISE VELOCITY AND ACCELERATION
        velocities     = np.zeros((mesh.points.shape[0]*formulation.ndim))
        accelerations  = np.zeros((mesh.points.shape[0]*formulation.ndim))
        # COMPUTE DAMPING MATRIX BASED ON MASS
        D = 0.0
        if self.include_physical_damping:
            D = self.damping_factor*M

        if formulation.fields == "electro_mechanics" or formulation.fields == "flexoelectric":
            # self.GetBoundaryInfo(mesh, formulation,boundary_condition)
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            if self.include_physical_damping:
                D_mech = D[self.mechanical_dofs,:][:,self.mechanical_dofs]
        else:
            M_mech = M
            D_mech = D

        # COMPUTE INITIAL ACCELERATION FOR TIME STEP 0
        Residual = np.zeros_like(Residual)
        InitResidual = Residual + NeumannForces[:,0][:,None]
        if formulation.fields == "electro_mechanics" or formulation.fields == "flexoelectric":
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


        for Increment in range(1,LoadIncrement):
            t_increment=time()

            # FIXED INCREMENTAL DIRICHLET
            AppliedDirichletInc = boundary_condition.applied_dirichlet[:,Increment-1]

            # APPLY NEUMANN BOUNDARY CONDITIONS
            DeltaF = NeumannForces[:,Increment][:,None]
            NodalForces = DeltaF

            # ACCUMULATED FORCE
            if self.include_physical_damping:
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


            # CHECK CONTACT AND ASSEMBLE IF DETECTED
            if self.has_contact:
                Eulerx = mesh.points + TotalDisp[:,:formulation.ndim,Increment-1]
                TractionForcesContact = np.zeros_like(Residual)
                TractionForcesContact = self.contact_formulation.AssembleTractions(mesh,material,Eulerx).ravel()*LoadFactor

                if formulation.fields == "electro_mechanics" or formulation.fields == "flexoelectric":
                    Residual[self.mechanical_dofs,0] -= TractionForcesContact
                elif formulation.fields == "mechanics" or formulation.fields == "couple_stress":
                    Residual[:,0] -= TractionForcesContact
                else:
                    raise NotImplementedError("Contact algorithm for {} is not available".format(formulation.fields))

            # REDUCED ACCUMULATED FORCE
            F_b = boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                boundary_condition.applied_dirichlet[:,Increment],LoadFactor=1.0,
                mass=M,only_residual=True)[boundary_condition.columns_in,0]

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b, F_b, reuse_factorisation=True)

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
            if formulation.fields == "couple_stress" or formulation.fields == "flexoelectric":
                TotalW[:,:,Increment], TotalW[:,:,Increment] = formulation.GetAugmentedSolution(self, material,
                    TotalDisp, Eulerx, Eulerw, Eulers, Eulerp)

            # LOG REQUESTS
            self.LogSave(formulation, TotalDisp, Increment)

            # BREAK AT A SPECIFICED LOAD INCREMENT IF ASKED FOR
            if self.break_at_increment != -1 and self.break_at_increment is not None:
                if self.break_at_increment == Increment:
                    if self.break_at_increment < LoadIncrement - 1:
                        print("\nStopping at increment {} as specified\n\n".format(Increment))
                        TotalDisp = TotalDisp[:,:,:Increment]
                        self.number_of_load_increments = Increment
                    break

            # STORE THE INFORMATION IF THE SOLVER BLOWS UP
            if Increment > 0:
                U0 = TotalDisp[:,:,Increment-1].ravel()
                U = TotalDisp[:,:,Increment].ravel()
                tol = 1e200 if Increment < 5 else 10.
                if np.isnan(norm(U)) or np.abs(U.max()/(U0.max()+1e-14)) > tol:
                    print("Solver blew up! Norm of incremental solution is too large")
                    TotalDisp = TotalDisp[:,:,:Increment]
                    self.number_of_load_increments = Increment
                    break

            print('Finished Load increment', Increment, 'in', time()-t_increment, 'seconds\n')

        # COMPUTE DISSIPATION OF ENERGY THROUGH TIME
        if self.compute_energy:
            energy_info = self.ComputeEnergy(formulation.function_spaces[0],mesh,material,formulation,Eulerx,Eulerp)
            formulation.strain_energy = energy_info[0]
            formulation.electrical_energy = energy_info[1]

        solver.CleanUp()

        return TotalDisp, TotalW, TotalS





    def ComputeEnergy(self,function_space,mesh,material,formulation,Eulerx,Eulerp):

        strain_energy = 0.
        electrical_energy = 0.
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
            ElectricPotentialElem = Eulerp[mesh.elements[elem,:]]

            energy = formulation.GetEnergy(function_space, material,
                LagrangeElemCoords, EulerElemCoords, ElectricPotentialElem, self, elem)
            strain_energy += energy[0]
            electrical_energy += energy[1]

        return strain_energy, electrical_energy



    def ComputeEnergyDissipation(self,function_space,mesh,material,formulation,fem_solver,
        Eulerx, TotalDisp, NeumannForces, M, velocities, Increment):

        internal_energy = 0.
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords = Eulerx[mesh.elements[elem,:],:]

            internal_energy += formulation.GetEnergy(function_space, material,
                LagrangeElemCoords, EulerElemCoords, fem_solver, elem)

        if formulation.fields == "flexoelectric":
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

            internal_energy += formulation.GetLinearMomentum(function_space, material,
                LagrangeElemCoords, EulerElemCoords, VelocityElem, fem_solver, elem)

        if formulation.fields == "electro_mechanics":
            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            kinetic_energy = np.dot(velocities.ravel(),M_mech.dot(accelerations.ravel()))
        else:
            kinetic_energy = np.dot(velocities.ravel(),M.dot(accelerations.ravel()))

        external_energy = np.dot(velocities.ravel(),NeumannForces[:,Increment])

        total_energy = internal_energy + kinetic_energy - external_energy
        return total_energy, internal_energy, kinetic_energy, external_energy
