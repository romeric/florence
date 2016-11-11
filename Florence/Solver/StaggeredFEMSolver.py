from __future__ import print_function
import gc, os, sys
from copy import deepcopy
from warnings import warn
from time import time
import numpy as np
import scipy as sp
from Florence.Utils import insensitive

from Florence.PostProcessing import *
from Florence.Solver import LinearSolver, FEMSolver

class StaggeredFEMSolver(FEMSolver):

    def __init__(self,**kwargs):
        super(StaggeredFEMSolver, self).__init__(**kwargs)

        self.electric_dofs = None
        self.mechanical_dofs = None
        self.columns_in_mech = None
        self.columns_in_mech_idx = None
        self.columns_in_electric = None
        self.columns_in_electric_idx = None
        self.columns_out_mech = None
        self.columns_out_mech_idx = None
        self.columns_out_electric = None
        self.columns_out_electric_idx = None

    def GetBoundaryInfo(self,K,boundary_condition,formulation):

        all_dofs = np.arange(K.shape[0])
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

        self.all_electric_dofs = np.arange(K.shape[0]/formulation.nvar)
        self.electric_out = self.all_electric_dofs[out_idx[idx_electric]]
        self.electric_in = np.setdiff1d(self.all_electric_dofs,self.electric_out)

        self.all_mech_dofs = np.arange(K.shape[0]/formulation.nvar*formulation.ndim)
        self.mech_out = self.all_mech_dofs[out_idx[idx_mech]]
        self.mech_in = np.setdiff1d(self.all_mech_dofs,self.mech_out)




    def Solve(self, formulation=None, mesh=None, 
        material=None, boundary_condition=None, 
        function_spaces=None, solver=None):
        """Main solution routine for FEMSolver """


        # CHECK DATA CONSISTENCY
        #---------------------------------------------------------------------------#
        if mesh is None:
            raise ValueError("No mesh detected for the analysis")
        if boundary_condition is None:
            raise ValueError("No boundary conditions detected for the analysis")
        if material is None:
            raise ValueError("No material model chosen for the analysis")
        if formulation is None:
            raise ValueError("No variational form specified")

        # GET FUNCTION SPACES FROM THE FORMULATION 
        if function_spaces is None:
            if formulation.function_spaces is None:
                raise ValueError("No interpolation functions specified")
            else:
                function_spaces = formulation.function_spaces

        # CHECK IF A SOLVER IS SPECIFIED
        if solver is None:
            solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")

        self.__checkdata__(material, boundary_condition, formulation, mesh)
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
        Eulerx = np.copy(mesh.points)
        Eulerp = np.zeros((mesh.points.shape[0]))

        # GET EXTERNAL NODAL FORCES
        # boundary_condition.GetExternalForces(mesh,material)

        # FIND PURE NEUMANN (EXTERNAL) NODAL FORCE VECTOR
        NeumannForces = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64)
        # FORCES RESULTING FROM DIRICHLET BOUNDARY CONDITIONS

        # ASSEMBLE STIFFNESS MATRIX AND TRACTION FORCES
        K, TractionForces = self.Assemble(function_spaces[0], formulation, mesh, material, solver, 
            Eulerx, Eulerp)[:2]

        if self.analysis_nature == 'nonlinear':
            print('Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'seconds')
        else:
            print('Finished the assembly stage. Time elapsed was', time()-tAssembly, 'seconds')


        self.StaggeredSolver(function_spaces, formulation, solver, 
                K,NeumannForces,NodalForces,Residual,
                mesh,TotalDisp,Eulerx,Eulerp,material, boundary_condition)


        return self.__makeoutput__(mesh, TotalDisp, formulation, function_spaces, material)





    def StaggeredSolver(self, function_spaces, formulation, solver, K,
            NeumannForces,NodalForces,Residual,
            mesh,TotalDisp,Eulerx,Eulerp,material, boundary_condition):
    
        Tolerance = self.newton_raphson_tolerance
        LoadIncrement = self.number_of_load_increments
        LoadFactor = 1./LoadIncrement
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)

        # GET BOUNDARY CONDITIONS INFO FOR EACH INDIVIDUAL PROBLEM
        self.GetBoundaryInfo(K,boundary_condition,formulation)

        # SOLVE THE FIRST MECHANICAL PROBLEM
        dUm = self.SolveMechanics(K, LoadFactor, mesh, formulation, solver, initial_solution=True)
        self.force_up = np.zeros(formulation.ndim*mesh.points.shape[0])
        self.force_pu = np.zeros(mesh.points.shape[0])
  
        for Increment in range(LoadIncrement):
            
            t_increment = time()
            
            # UPDATE FIXED DOFs FOR ELECTROSTATICS
            dirichlet_forces_electric = np.zeros((mesh.points.shape[0],1),dtype=np.float64)
            dirichlet_forces_electric = self.ApplyDirichlet(K[self.electric_dofs,:][:,self.electric_dofs],
                dirichlet_forces_electric, self.electric_out, self.electric_in,self.applied_dirichlet_electric)
            residual_electric = -LoadFactor*dirichlet_forces_electric
            
            # COMPUTE FORCE TO BE TRANSMITTED TO ELECTROSTATIC
            K_pu = K[self.electric_dofs,:][:,self.mechanical_dofs]
            self.force_pu = K_pu.dot(dUm.flatten())

            # STORE PREVIOUS ELECTRIC POTENTIAL AT THIS INCREMENT
            Eulerp_n = np.copy(Eulerp)
            # APPLY INCREMENTAL POTENTIAL FOR FIXED POTENTIAL DoFs
            applied_dirichlet_electric_inc = LoadFactor*self.applied_dirichlet_electric
            Eulerp[self.electric_out] += applied_dirichlet_electric_inc

            # GET ONLY NORM OF FIXED DOFs (THAT IS WHERE RESIDUAL FORCES GET GENERATED)
            if Increment==0:
                self.NormForces = la.norm(residual_electric[self.electric_in])
            # AVOID DIVISION BY ZERO
            if np.abs(la.norm(residual_electric[self.electric_in])) < 1e-14:
                self.NormForces = 1e-14

            Ke = deepcopy(K)

            Iter = 0
            # ENTER NEWTON-RAPHSON FOR ELECTROSTATICS
            # xx= np.zeros_like(Eulerp)
            while np.abs(la.norm(residual_electric[self.electric_in])/self.NormForces) > Tolerance:

                # SOLVE ELECTROSTATIC PROBLEM ITERATIVELY
                dUe = self.SolveElectrostatics(Ke, residual_electric, formulation, solver, Iter)

                # UPDATE EULERIAN POTENTIAL - GET ITERATIVE ELECTRIC POTENTIAL
                Eulerp[self.electric_in] += dUe

                # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES FOR ELECTROSTATICS
                Ke, TractionForces = self.Assemble(function_spaces[0], formulation, mesh, material, solver,
                    Eulerx,Eulerp)[:2]

                # FIND THE ITERATIVE RESIDUAL
                residual_electric[self.electric_in] = TractionForces[self.columns_in_electric]

                self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
                    np.abs(la.norm(residual_electric[self.electric_in])/self.NormForces))
                
                print('Iteration number', Iter, 'for load increment', Increment, 'with a residual of \t\t', \
                    np.abs(la.norm(residual_electric[self.electric_in])/self.NormForces)) 

                # UPDATE ITERATION NUMBER
                Iter +=1

                if Iter==self.maximum_iteration_for_newton_raphson or self.NRConvergence['Increment_'+str(Increment)][-1] > 500:
                    self.newton_raphson_failed_to_converge = True
                    break
                if np.isnan(np.abs(la.norm(residual_electric[self.electric_in])/self.NormForces)):
                    self.newton_raphson_failed_to_converge = True
                    break

            if self.newton_raphson_failed_to_converge:
                break


            # COMPUTE FORCE TO BE TRANSMITTED TO MECHANICS
            K_up = Ke[self.mechanical_dofs,:][:,self.electric_dofs]
            dUe = Eulerp - Eulerp_n
            self.force_up = K_up.dot(dUe)
            
            # SOLVE MECHANICS PROBLEM WITH OLD GEOMETRY (K), AND THE FORCE self.force_up AS A RESIDUAL  
            dUm = self.SolveMechanics(K, LoadFactor, mesh, formulation, solver, initial_solution=False)

            # UPDATE GEOMETRY INCREMENTALLY
            Eulerx += dUm

            # UPDATE SOLUTION FOR THE CURRENT LOAD INCREMENT
            TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
            TotalDisp[:,-1,Increment] = Eulerp

            K = self.Assemble(function_spaces[0], formulation, mesh, material, solver,
                    Eulerx,Eulerp)[0]

            print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')


        return TotalDisp


    def ApplyDirichlet(self, stiffness, F, columns_out, columns_in, AppliedDirichlet, mass=None):
        """AppliedDirichlet is a non-member because it can be external incremental Dirichlet,
            which is currently not implemented as member of BoundaryCondition. F also does not 
            correspond to Dirichlet forces, as it can be residual in incrementally linearised
            framework.
        """

        # APPLY DIRICHLET BOUNDARY CONDITIONS
        for i in range(columns_out.shape[0]):
            F = F - AppliedDirichlet[i]*stiffness.getcol(columns_out[i])

        return F


    def SolveMechanics(self, K, LoadFactor, mesh, formulation, solver, initial_solution=True):
        """ Solves for mechanical variables. This solves the upper row 
            of the following system

                [K_uu K_up][U_u] = [F_u]
                [K_pu K_pp][U_p] = [F_p]

            i.e. 

                K_uu*U_u = F_u - K_up*U_p


            input:

                K:              [scipy.sparse.csc/csr_matrix] Total electromechanical stiffness
                                Matrix, no boundary conditions applied

        """
        dirichlet_forces_mech = np.zeros((mesh.points.shape[0]*formulation.ndim,1),dtype=np.float64)
        dirichlet_forces_mech = self.ApplyDirichlet(K[self.mechanical_dofs,:][:,self.mechanical_dofs],
            dirichlet_forces_mech, self.mech_out, self.mech_in,self.applied_dirichlet_mech)
        residual_mech = -LoadFactor*dirichlet_forces_mech
        K_uu_b = K[self.columns_in_mech,:][:,self.columns_in_mech]

        if initial_solution:
            F_b = residual_mech[self.mech_in,0]
        else:
            rhs_mech = residual_mech + self.force_up[:,None]
            F_b = rhs_mech[self.mech_in]
 
        sol = solver.Solve(K_uu_b,-F_b)
        # REARRANGE
        dUm = np.zeros(self.all_mech_dofs.shape[0],dtype=np.float64)
        dUm[self.mech_in] = sol
        dUm[self.mech_out] = LoadFactor*self.applied_dirichlet_mech
        dUm = dUm.reshape(dUm.shape[0]/formulation.ndim,formulation.ndim)

        return dUm



    def SolveElectrostatics(self, K, residual_electric,formulation,solver,iteration):
        """ Solves for mechanical variables. This solves the lower row 
            of the following system

                [K_uu K_up][U_u] = [F_u]
                [K_pu K_pp][U_p] = [F_p]

            i.e. 

                K_uu*U_u = F_u - K_up*U_p


            input:

                K:              [scipy.sparse.csc/csr_matrix] Total electromechanical stiffness
                                Matrix, no boundary conditions applied

        """
        K_pp_b = K[self.columns_in_electric,:][:,self.columns_in_electric]
        if iteration == 0:
            # rhs_electric = residual_electric - self.force_pu[:,None]
            rhs_electric = residual_electric + self.force_pu[:,None]
            # print(self.force_pu[self.electric_in])
        else:
            rhs_electric = residual_electric           

        F_b = rhs_electric[self.electric_in]
        sol = solver.Solve(K_pp_b,-F_b)
        # REARRANGE
        # dUe = np.zeros_like(self.all_electric_dofs)
        # dUe[self.electric_in] = sol
        # for i in range(dUe)
        # print(dUe.shape, sol.shape, dUe[self.electric_in])
        # print(dUe[self.electric_in], self.electric_in, dUe.shape, F_b.shape)
        # dUe = dUe.reshape(dUe.shape[0]/formulation.ndim,formulation.ndim)

        return sol