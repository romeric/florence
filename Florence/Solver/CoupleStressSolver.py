from __future__ import print_function
import gc, os, sys
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
        super(CoupleStressSolver, self).__init__(**kwargs)
        if 'static_condensation' in kwargs.keys():
            self.static_condensation = kwargs['static_condensation']
        else:
            self.static_condensation = True

        self.gamma   = 0.5
        self.beta    = 0.25
        self.assembly_print_counter = 500

    def GetBoundaryInfo(self,K,boundary_condition,formulation):
        pass


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

        print('Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation info etc...')
        print('Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*formulation.nvar)
        if formulation.ndim==2:
            print('Number of elements is', mesh.elements.shape[0], \
                 'and number of boundary nodes is', np.unique(mesh.edges).shape[0])
        elif formulation.ndim==3:
            print('Number of elements is', mesh.elements.shape[0], \
                 'and number of boundary nodes is', np.unique(mesh.faces).shape[0])
        #---------------------------------------------------------------------------#

        ## IMPORTANT
        self.requires_geometry_update = False
        self.analysis_nature = "linear"

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



        # INITIATE DATA FOR NON-LINEAR ANALYSIS
        NodalForces, Residual = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64), \
            np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64)
        # SET NON-LINEAR PARAMETERS
        self.NRConvergence = { 'Increment_'+str(Increment) : [] for Increment in range(self.number_of_load_increments) }

        # ALLOCATE FOR SOLUTION FIELDS
        # TotalDisp = np.zeros((mesh.points.shape[0],formulation.nvar,self.number_of_load_increments),dtype=np.float32)
        TotalDisp = np.zeros((mesh.points.shape[0],formulation.nvar,self.number_of_load_increments),dtype=np.float64)

        # PRE-ASSEMBLY
        print('Assembling the system and acquiring neccessary information for the analysis...')
        tAssembly=time()

        # APPLY DIRICHELT BOUNDARY CONDITIONS AND GET DIRICHLET RELATED FORCES
        boundary_condition.GetDirichletBoundaryConditions(formulation, mesh, material, solver, self)

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
            # M = AssembleMass(fem_solver, function_space, formulation, mesh, material, Eulerx)
            K, TractionForces, _, M = formulation.Assemble(self, material, Eulerx, Eulerw, Eulers, Eulerp)
        else:
            K, TractionForces = formulation.Assemble(self, material, Eulerx, Eulerw, Eulers, Eulerp)[:2]

        print('Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'seconds')

        if self.analysis_type != 'static':

            TotalDisp = self.DynamicSolver(formulation, solver,
                K, M, NeumannForces, NodalForces, Residual,
                mesh, TotalDisp, Eulerx, Eulerw, Eulers, Eulerp, material, boundary_condition)
        else:
            TotalDisp = self.StaticSolver(formulation, solver,
                K, NeumannForces, NodalForces, Residual,
                mesh, TotalDisp, Eulerx, Eulerw, Eulers, Eulerp, material, boundary_condition)


        return self.__makeoutput__(mesh, TotalDisp, formulation, function_spaces, material)



    def StaticSolver(self, formulation, solver, K,
            NeumannForces, NodalForces, Residual,
            mesh, TotalDisp, Eulerx, Eulerw, Eulers, Eulerp, material, boundary_condition):


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
            sol = solver.Solve(K_b,F_b)

            AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet
            dU = post_process.TotalComponentSol(sol, boundary_condition.columns_in,
                boundary_condition.columns_out, AppliedDirichletInc,0,K.shape[0])

            # STORE TOTAL SOLUTION DATA
            TotalDisp[:,:,Increment] += dU

            # REDUCED ACCUMULATED FORCE
            Residual[:,:] = DeltaF
            F_b = boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                AppliedDirichletInc,LoadFactor=1.0,
                only_residual=True)[boundary_condition.columns_in,0]
            # F_b = boundary_condition.GetReducedVectors(Residual,only_residual=True)[0]


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

        return TotalDisp



    def DynamicSolver(self, formulation, solver, K, M,
            NeumannForces, NodalForces, Residual,
            mesh, TotalDisp, Eulerx, Eulerw, Eulers, Eulerp, material, boundary_condition):

        LoadIncrement = self.number_of_load_increments
        LoadFactor = self.total_time/LoadIncrement

        post_process = PostProcess(formulation.ndim,formulation.nvar)
        post_process.SetAnalysis(analysis_type=self.analysis_type, analysis_nature=self.analysis_nature)

        if NeumannForces.ndim == 2 and NeumannForces.shape[1]==1:
            tmp = np.zeros((NeumannForces.shape[0],LoadIncrement))
            tmp[:,0] = NeumannForces[:,0]
            NeumannForces = tmp

        dumU = np.zeros((mesh.points.shape[0]*formulation.ndim))
        dumU[boundary_condition.columns_out] = boundary_condition.applied_dirichlet[:,0]
        TotalDisp[:,:,0] = dumU.reshape(mesh.points.shape[0],formulation.ndim)
        # INITIALISE VELOCITY AND ACCELERATION
        velocities     = np.zeros((mesh.points.shape[0]*formulation.ndim))
        accelerations  = np.zeros((mesh.points.shape[0]*formulation.ndim))

        # COMPUTE INITIAL ACCELERATION FOR TIME STEP 0
        Residual = np.zeros_like(Residual)
        InitResidual = Residual + NeumannForces[:,0][:,None]
        if formulation.fields == "electro_mechanics":
            accelerations[:] = solver.Solve(M_mech, -InitResidual[self.mechanical_dofs].ravel())
        else:
            accelerations[:] = solver.Solve(M, InitResidual.ravel() )

        # COMPUTE DAMPING MATRIX BASED ON MASS
        D = 0.0
        if self.include_physical_damping:
            D = self.damping_factor*M
        # COMPUTE AUGMENTED K (INCLUDES INERTIA EFFECT)
        K          += (self.gamma/self.beta/LoadFactor)*D + (1./self.beta/LoadFactor**2)*M
        # GET REDUCED VARIABLES
        K_b, F_b, _ = boundary_condition.GetReducedMatrices(K,Residual)


        for Increment in range(1,LoadIncrement):
            t_increment=time()

            # FIXED INCREMENTAL DIRICHLET
            AppliedDirichletInc = boundary_condition.applied_dirichlet[:,Increment-1]

            # APPLY NEUMANN BOUNDARY CONDITIONS
            DeltaF = NeumannForces[:,Increment][:,None]
            NodalForces = DeltaF

            # ACCUMULATED FORCE
            Residual[:,0] = (1./self.beta/LoadFactor**2)*M.dot(TotalDisp[:,:,Increment-1].ravel()) +\
                (1./self.beta/LoadFactor)*M.dot(velocities) + (0.5/self.beta - 1.)*M.dot(accelerations)
            Residual += DeltaF

            # CHECK CONTACT AND ASSEMBLE IF DETECTED
            if self.has_contact:
                Eulerx = mesh.points + TotalDisp[:,:,Increment-1]
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
            sol = solver.Solve(K_b,F_b)

            dU = post_process.TotalComponentSol(sol, boundary_condition.columns_in,
                boundary_condition.columns_out, AppliedDirichletInc,0,K.shape[0])

            # STORE TOTAL SOLUTION DATA
            TotalDisp[:,:,Increment] += dU

            # UPDATE VELOCITY AND ACCELERATION
            accelerations_old = np.copy(accelerations)
            accelerations = (1./self.beta/LoadFactor**2)*(TotalDisp[:,:,Increment] - TotalDisp[:,:,Increment-1]).ravel() -\
                1./self.beta/LoadFactor*velocities + (1.-0.5/self.beta)*accelerations
            velocities += LoadFactor*(self.gamma*accelerations + (1-self.gamma)*accelerations_old)

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

        return TotalDisp