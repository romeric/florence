from __future__ import print_function
import gc, os, sys
from copy import deepcopy
from warnings import warn
from time import time
import numpy as np
import scipy as sp
# from scipy.sparse import coo_matrix, csc_matrix, csr_matrix 
from Florence.Utils import insensitive

from Florence.FiniteElements.PostProcess import *
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
        NodalForces, Residual = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float32), \
            np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float32)
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
        dUm = self.SolveMechanics(K, LoadFactor, mesh, formulation, solver, Increment=0)
        # print(dUm)
  
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

            
            Eulerp_n = np.copy(Eulerp)
            Eulerp[self.electric_out] += self.applied_dirichlet_electric

            # GET ONLY NORM OF FIXED DOFs (THAT IS WHERE RESIDUAL FORCES GET GENERATED)
            if Increment==0:
                self.NormForces = la.norm(residual_electric[self.electric_in])
            # AVOID DIVISION BY ZERO
            if np.abs(la.norm(residual_electric[self.electric_in])) < 1e-14:
                self.NormForces = 1e-14

            Ke = K

            Iter = 0
            # ENTER NEWTON-RAPHSON FOR ELECTROSTATICS
            while np.abs(la.norm(residual_electric[self.electric_in])/self.NormForces) > Tolerance:

                dUe = self.SolveElectrostatics(Ke, residual_electric,formulation,solver)
                # print(dUe)
                # print(Eulerp[self.electric_in])
                # UPDATE EULERIAN POTENTIAL - GET ITERATIVE ELECTRIC POTENTIAL
                Eulerp[self.electric_in] += dUe
                # print(Eulerp[self.electric_in])

                # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
                # print(Eulerx - mesh.points)
                Ke, TractionForces = self.Assemble(function_spaces[0], formulation, mesh, material, solver,
                    Eulerx,Eulerp)[:2]
                
                # FIND THE RESIDUAL
                residual_electric[self.electric_in] = TractionForces[self.columns_in_electric]

                # SAVE THE NORM 
                self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
                    np.abs(la.norm(Residual[self.columns_in_electric])/self.NormForces))
                
                print('Iteration number', Iter, 'for load increment', Increment, 'with a residual of \t\t', \
                    np.abs(la.norm(residual_electric[self.electric_in])/self.NormForces)) 

                # UPDATE ITERATION NUMBER
                Iter +=1

                # if Iter==self.maximum_iteration_for_newton_raphson:
                    # raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

                if Iter==self.maximum_iteration_for_newton_raphson or self.NRConvergence['Increment_'+str(Increment)][-1] > 500:
                    self.newton_raphson_failed_to_converge = True
                    break
                if np.isnan(np.abs(la.norm(residual_electric[self.electric_in])/self.NormForces)):
                    self.newton_raphson_failed_to_converge = True
                    break
            ##

            if self.newton_raphson_failed_to_converge:
                break


            # COMPUTE FORCE TO BE TRANSMITTED TO ELECTROSTATIC
            K_up = K[self.mechanical_dofs,:][:,self.electric_dofs]
            dUe = Eulerp - Eulerp_n
            self.force_up = K_up.dot(dUe.flatten())

            # SOLVE MECHANICS
            dUm = self.SolveMechanics(K, LoadFactor, mesh, formulation, solver, Increment=0)
            # UPDATE THE GEOMETRY NOW
            Eulerx += dUm


            # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
            TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
            TotalDisp[:,-1,Increment] = Eulerp - Eulerp_n

            print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')

        return TotalDisp


    def ApplyDirichlet(self, stiffness, F, columns_out, columns_in,AppliedDirichlet, mass=None):
        """AppliedDirichlet is a non-member because it can be external incremental Dirichlet,
            which is currently not implemented as member of BoundaryCondition. F also does not 
            correspond to Dirichlet forces, as it can be residual in incrementally linearised
            framework.
        """

        # APPLY DIRICHLET BOUNDARY CONDITIONS
        for i in range(0,columns_out.shape[0]):
            F = F - AppliedDirichlet[i]*stiffness.getcol(columns_out[i])

        return F


    def SolveMechanics(self, K, LoadFactor, mesh, formulation, solver, Increment=0):
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

        if Increment == 0:
            F_b = residual_mech[self.mech_in,0]
        else:
            rhs_mech = residual_mech - self.force_up[:,None]
            F_b = rhs_mech[self.mech_in]

        sol = solver.Solve(K_uu_b,-F_b)
        # REARRANGE
        dUm = np.zeros(self.all_mech_dofs.shape[0],dtype=np.float64)
        dUm[self.mech_in] = sol
        dUm[self.mech_out] = LoadFactor*self.applied_dirichlet_mech
        dUm = dUm.reshape(dUm.shape[0]/formulation.ndim,formulation.ndim)

        return dUm
        # return sol



    def SolveElectrostatics(self, K, residual_electric,formulation,solver):
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
        rhs_electric = residual_electric - self.force_pu[:,None]
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














    # def GetIndividualBoundaryInfo():

    #     mech_dofs = np.arange(0,K.shape[0])
    #     mech_dofs[formulation.nvar-1::formulation.nvar] = -1
    #     accum = 0
    #     for i in range(mech_dofs.shape[0]):
    #         if mech_dofs[i] == -1:
    #             accum -= 1
    #         else:
    #             mech_dofs[i] += accum
    #     mech_dofs = mech_dofs[np.where(mech_dofs!=-1)[0]]
































    # def StaggeredSolver(self, function_spaces, formulation, solver, K,
    #         NeumannForces,NodalForces,Residual,
    #         mesh,TotalDisp,Eulerx,Eulerp,material, boundary_condition):
    
    #     Tolerance = self.newton_raphson_tolerance
    #     LoadIncrement = self.number_of_load_increments
    #     LoadFactor = 1./LoadIncrement
    #     AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)

    #     # GET BOUNDARY CONDITIONS INFO FOR EACH INDIVIDUAL PROBLEM
    #     self.GetBoundaryInfo(K,boundary_condition,formulation)
        
    #     for Increment in range(LoadIncrement):

    #         # APPLY NEUMANN BOUNDARY CONDITIONS
    #         DeltaF = LoadFactor*NeumannForces
    #         NodalForces += DeltaF
    #         # RESIDUAL FORCES CONTAIN CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
    #         DirichletForces = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64)
    #         DirichletForces = boundary_condition.ApplyDirichletGetReducedMatrices(K,DirichletForces,
    #             boundary_condition.applied_dirichlet)[2]
    #         # OBRTAIN THE INCREMENTAL RESIDUAL
    #         Residual -= LoadFactor*DirichletForces
    #         # GET THE INCREMENTAL DISPLACEMENT
    #         AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet


    #         t_increment = time()
    #         # GET ONLY NORM OF FIXED DOFs (THAT IS WHERE RESIDUAL FORCES GET GENERATED)
    #         if Increment==0:
    #             self.NormForces = np.linalg.norm(Residual[self.columns_out_electric])
    #         # AVOID DIVISION BY ZERO
    #         if np.linalg.norm(Residual[self.columns_in_electric]) < 1e-14:
    #             self.NormForces = 1e-14

    #         # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
    #         IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
    #             K.shape[0],formulation.nvar)
    #         # UPDATE EULERIAN COORDINATE FOR FIXED DOFs
    #         Eulerx += IncDirichlet[:,:formulation.ndim]
    #         # GET EULERIAN POTENTIAL
    #         Eulerp += IncDirichlet[:,-1]

    #         ##
    #         # SOLVE MECHANICAL PROBLEM
    #         dUm = self.SolveMechanics(K,Residual,IncDirichlet[:,-1],formulation,solver)
    #         # dU = self.SolveMechanics(K,Residual,Eulerp,formulation,solver)
    #         # # UPDATE EULERIAN COORDINATE FOR FREE DOFs
    #         # Eulerx += dUm[:,:formulation.ndim]

    #         # ENTER NEWTON-RAPHSON FOR ELECTROSTATICS
    #         Iter = 0
    #         while np.abs(la.norm(Residual[self.columns_in_electric])/self.NormForces) > Tolerance:
    #             # SOLVE ELECTROSTATICS
    #             dUe = self.SolveElectrostatics(K,Residual,IncDirichlet[:,:formulation.ndim],formulation,solver)

    #             # UPDATE
    #             Eulerp += dUe[:,-1]

    #             # RE-ASSEMBLE THE WHOLE SYSTEM
    #             K, TractionForces = self.Assemble(function_spaces[0], formulation, mesh, material, solver,
    #                 Eulerx,Eulerp)[:2]
    #             # FIND THE RESIDUAL
    #             Residual[self.columns_in_electric] = TractionForces[self.columns_in_electric] \
    #             - NodalForces[self.columns_in_electric]

    #             # SAVE THE NORM 
    #             self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
    #                 np.abs(la.norm(Residual[self.columns_in_electric])/self.NormForces))
                
    #             print('Iteration number', Iter, 'for load increment', Increment, 'with a residual of \t\t', \
    #                 np.abs(la.norm(Residual[self.columns_in_electric])/self.NormForces)) 

    #             # UPDATE ITERATION NUMBER
    #             Iter +=1

    #             # if Iter==self.maximum_iteration_for_newton_raphson:
    #                 # raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

    #             if Iter==self.maximum_iteration_for_newton_raphson or self.NRConvergence['Increment_'+str(Increment)][-1] > 500:
    #                 self.newton_raphson_failed_to_converge = True
    #                 break
    #             if np.isnan(np.abs(la.norm(Residual[self.columns_in_electric])/self.NormForces)):
    #                 self.newton_raphson_failed_to_converge = True
    #                 break
    #         ##

    #         if self.newton_raphson_failed_to_converge:
    #             break

    #         # UPDATE EULERIAN COORDINATE FOR FREE DOFs
    #         Eulerx += dUm[:,:formulation.ndim]

    #         # Eulerx = self.NewtonRaphson(function_spaces, formulation, solver, 
    #         #     Increment,K,NodalForces,Residual,mesh,Eulerx,Eulerp,
    #         #     material,boundary_condition,AppliedDirichletInc)

    #         # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
    #         TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
    #         TotalDisp[:,-1,Increment] = Eulerp

    #         print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')

    #     return TotalDisp


    # def SolveMechanics(self, K, Residual,dUe,formulation,solver):
    #     """ Solves for mechanical variables. This solves the upper row 
    #         of the following system

    #             [K_uu K_up][U_u] = [F_u]
    #             [K_pu K_pp][U_p] = [F_p]

    #         i.e. 

    #             K_uu*U_u = F_u - K_up*U_p


    #         input:

    #             K:              [scipy.sparse.csc/csr_matrix] Total electromechanical stiffness
    #                             Matrix, no boundary conditions applied

    #     """


    #     # COMPUTE THE RHS CONTRIBUTION OF ELECTROSTATIC
    #     K_up = K[self.mechanical_dofs,:][:,self.electric_dofs]
    #     ResidualU = Residual[self.mechanical_dofs] - K_up.dot(dUe)[:,None]

    #     # SOLVE MECHANICAL SYSTEM
    #     K_uu_b = K[self.columns_in_mech,:][:,self.columns_in_mech]
    #     F_b = ResidualU[self.columns_in_mech_idx,0]
    #     # print(ResidualU[columns_in_mech_idx,0])
    #     sol = solver.Solve(K_uu_b,-F_b)
    #     # REARRANGE
    #     dU = np.zeros((K.shape[0],1))
    #     dU[self.columns_in_mech,0] = sol
    #     dU = dU.reshape(dU.shape[0]/formulation.nvar,formulation.nvar)

    #     return dU



    # def SolveElectrostatics(self, K, Residual,dUm,formulation,solver):
    #     """ Solves for mechanical variables. This solves the lower row 
    #         of the following system

    #             [K_uu K_up][U_u] = [F_u]
    #             [K_pu K_pp][U_p] = [F_p]

    #         i.e. 

    #             K_uu*U_u = F_u - K_up*U_p


    #         input:

    #             K:              [scipy.sparse.csc/csr_matrix] Total electromechanical stiffness
    #                             Matrix, no boundary conditions applied

    #     """
    #     dUm = dUm.ravel()
    #     # COMPUTE THE RHS CONTRIBUTION OF ELECTROSTATIC
    #     K_pu = K[self.electric_dofs,:][:,self.mechanical_dofs]
    #     ResidualP = Residual[self.electric_dofs] - K_pu.dot(dUm)[:,None]

    #     # SOLVE MECHANICAL SYSTEM
    #     K_pp_b = K[self.columns_in_electric,:][:,self.columns_in_electric]
    #     F_b = ResidualP[self.columns_in_electric_idx,0]
    #     # print(ResidualU[columns_in_mech_idx,0])
    #     sol = solver.Solve(K_pp_b,-F_b)
    #     # REARRANGE
    #     dU = np.zeros((K.shape[0],1))
    #     dU[self.columns_in_electric,0] = sol
    #     dU = dU.reshape(dU.shape[0]/formulation.nvar,formulation.nvar)

    #     return dU



    #     # for dim in range(formulation.ndim):
    #     #     mechanical_dofs = np.append(mechanical_dofs,all_dofs[dim::formulation.nvar])
    #     # mechanical_dofs = np.sort(mechanical_dofs)