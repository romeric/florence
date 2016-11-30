from __future__ import print_function
import gc, os, sys
from copy import deepcopy
from warnings import warn
from time import time
import numpy as np
import scipy as sp
from scipy.io import loadmat, savemat
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix 
from Florence.Utils import insensitive

from Florence.FiniteElements.SparseAssembly import SparseAssembly_Step_2
from Florence.FiniteElements.SparseAssemblySmall import SparseAssemblySmall
from Florence.PostProcessing import *
from Florence.Solver import LinearSolver
from Florence.TimeIntegrators import StructuralDynamicIntegrators

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from Florence.FiniteElements.SparseAssemblyNative import SparseAssemblyNative

# PARALLEL PROCESSING ROUTINES
import multiprocessing
import Florence.ParallelProcessing.parmap as parmap


class FEMSolver(object):
    """Solver for linear and non-linear finite elements.
        This is different from the LinearSolver, as linear solver
        specifically deals with solution of matrices, whereas FEM
        solver is essentially responsible for linear, linearised
        and nonlinear finite element formulations
    """

    def __init__(self, analysis_type="static", analysis_nature="nonlinear",
        is_geometrically_linearised=False, requires_geometry_update=True,
        requires_line_search=False, requires_arc_length=False, has_moving_boundary=False,
        has_prestress=True, number_of_load_increments=1, 
        newton_raphson_tolerance=1.0e-6, maximum_iteration_for_newton_raphson=50,
        compute_mesh_qualities=True,  
        parallelise=False, memory_model="shared", platform="cpu", backend="opencl"):

        self.analysis_type = analysis_type
        self.analysis_nature = analysis_nature

        self.is_geometrically_linearised = is_geometrically_linearised
        self.requires_geometry_update = requires_geometry_update
        self.requires_line_search = requires_line_search
        self.requires_arc_length = requires_arc_length
        self.has_moving_boundary = has_moving_boundary
        self.has_prestress = has_prestress

        self.number_of_load_increments = number_of_load_increments
        self.newton_raphson_tolerance = newton_raphson_tolerance
        self.maximum_iteration_for_newton_raphson = maximum_iteration_for_newton_raphson
        self.newton_raphson_failed_to_converge = False
        self.NRConvergence = None

        self.compute_mesh_qualities = compute_mesh_qualities
        self.isScaledJacobianComputed = False

        self.vectorise = True
        self.parallel = parallelise
        self.no_of_cpu_cores = multiprocessing.cpu_count()
        self.memory_model = memory_model
        self.platform = platform
        self.backend = backend
        self.debug = False


    def __checkdata__(self, material, boundary_condition, formulation, mesh):
        """Checks the state of data for FEMSolver"""

        if material.mtype == "LinearModel" and self.number_of_load_increments > 1:
            warn("Can not solve a linear elastic model in multiple step. "
                "The number of load increments is going to be set to 1")
            self.number_of_load_increments = 1

        self.has_prestress = False
        if "nonlinear" not in insensitive(self.analysis_nature) and formulation.fields == "mechanics":
            # RUN THE SIMULATION WITHIN A NONLINEAR ROUTINE
            if material.mtype != "IncrementalLinearElastic" and \
                material.mtype != "LinearModel" and material.mtype != "TranservselyIsotropicLinearElastic":
                self.has_prestress = True
            else:
                self.has_prestress = False

        # GEOMETRY UPDATE FLAGS
        ###########################################################################
        # DO NOT UPDATE THE GEOMETRY IF THE MATERIAL MODEL NAME CONTAINS 
        # INCREMENT (CASE INSENSITIVE). VALID FOR ELECTROMECHANICS FORMULATION. 
        self.requires_geometry_update = False
        if formulation.fields == "electro_mechanics":
            if "Increment" in insensitive(material.mtype):
                # RUN THE SIMULATION WITHIN A NONLINEAR ROUTINE WITHOUT UPDATING THE GEOMETRY
                self.requires_geometry_update = False
            else:
                self.requires_geometry_update = True
        elif formulation.fields == "mechanics":
            if self.analysis_nature == "nonlinear" or self.has_prestress:
                self.requires_geometry_update = True

        # CHECK IF MATERIAL MODEL AND ANALYSIS TYPE ARE COMPATIBLE
        #############################################################################
        if "nonlinear" in insensitive(self.analysis_nature):
            if "linear" in  insensitive(material.mtype) or \
                "increment" in insensitive(material.mtype):
                warn("Incompatible material model and analysis type. I'm going to change analysis type")
                self.analysis_nature = "linear"
                formulation.analysis_nature = "linear"

        if material.is_transversely_isotropic or material.is_anisotropic:
            if material.anisotropic_orientations is None:
                material.GetFibresOrientation(mesh)
        ##############################################################################

        ##############################################################################
        if boundary_condition.boundary_type == "straight":
            self.compute_mesh_qualities = False
        ##############################################################################

        # CHANGE MESH DATA TYPE
        mesh.ChangeType()
        # ASSIGN ANALYSIS PARAMTER TO BOUNDARY CONDITION
        boundary_condition.analysis_type = self.analysis_type
        boundary_condition.analysis_nature = self.analysis_nature



    def __makeoutput__(self, mesh, TotalDisp, formulation=None, function_spaces=None, material=None):
        post_process = PostProcess(formulation.ndim,formulation.nvar)
        post_process.SetBases(postdomain=function_spaces[1], domain=function_spaces[0], boundary=None)
        post_process.SetAnalysis(analysis_type=self.analysis_type, 
            analysis_nature=self.analysis_nature)
        post_process.SetMesh(mesh)
        post_process.SetSolution(TotalDisp)
        post_process.SetFormulation(formulation)
        post_process.SetMaterial(material)
        post_process.SetFEMSolver(self)

        if self.analysis_nature == "nonlinear" and self.compute_mesh_qualities:
            # COMPUTE QUALITY MEASURES
            # self.ScaledJacobian=post_process.MeshQualityMeasures(mesh,TotalDisp,False,False)[3]
            post_process.ScaledJacobian=post_process.MeshQualityMeasures(mesh,TotalDisp,False,False)[3]
        elif self.isScaledJacobianComputed:
            post_process.ScaledJacobian=self.ScaledJacobian

        if self.analysis_nature == "nonlinear":
            post_process.newton_raphson_convergence = self.NRConvergence
            
        return post_process


    @property
    def WhichFEMSolver():
        pass

    @property
    def WhichFEMSolvers():
        print(["IncrementalLinearElasticitySolver","NewtonRaphson","NewtonRaphsonArchLength"])


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

        # INITIATE DATA FOR THE ANALYSIS
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

        # FIND PURE NEUMANN (EXTERNAL) NODAL FORCE VECTOR
        NeumannForces = boundary_condition.ComputeNeumannForces(mesh, material)

        # ADOPT A DIFFERENT PATH FOR INCREMENTAL LINEAR ELASTICITY
        if formulation.fields == "mechanics" and self.analysis_nature != "nonlinear":     
            # MAKE A COPY OF MESH, AS MESH POINTS WILL BE OVERWRITTEN
            vmesh = deepcopy(mesh)
            TotalDisp = self.IncrementalLinearElasticitySolver(function_spaces, formulation, vmesh, material,
                boundary_condition, solver, TotalDisp, Eulerx, NeumannForces)
            del vmesh

            # ADD EACH INCREMENTAL CONTRIBUTION TO MAKE IT CONSISTENT WITH THE NONLINEAR ANALYSYS
            for i in range(TotalDisp.shape[2]-1,0,-1):
                TotalDisp[:,:,i] = np.sum(TotalDisp[:,:,:i+1],axis=2)

            return self.__makeoutput__(mesh, TotalDisp, formulation, function_spaces, material)

        # ASSEMBLE STIFFNESS MATRIX AND TRACTION FORCES
        K, TractionForces, _, M = self.Assemble(function_spaces[0], formulation, mesh, material, solver, 
            Eulerx, Eulerp)

        if self.analysis_nature == 'nonlinear':
            print('Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'seconds')
        else:
            print('Finished the assembly stage. Time elapsed was', time()-tAssembly, 'seconds')


        if self.analysis_type != 'static':
            structural_integrator = StructuralDynamicIntegrators()
            TotalDisp = structural_integrator.Solver(function_spaces, formulation, solver, 
                K, M, NeumannForces, NodalForces, Residual,
                mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, self)
        else:
            TotalDisp = self.StaticSolver(function_spaces, formulation, solver, 
                K,NeumannForces,NodalForces,Residual,
                mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition)

        return self.__makeoutput__(mesh, TotalDisp, formulation, function_spaces, material)


    def IncrementalLinearElasticitySolver(self, function_spaces, formulation, mesh, material,
                boundary_condition, solver, TotalDisp, Eulerx, NeumannForces):
        """An icremental linear elasticity solver, in which the geometry is updated 
            and the remaining quantities such as stresses and Hessians are based on Prestress flag. 
            In this approach instead of solving the problem inside a non-linear routine,
            a somewhat explicit and more efficient way is adopted to avoid pre-assembly of the system
            of equations needed for non-linear analysis
        """

        # CREATE POST-PROCESS OBJECT ONCE
        post_process = PostProcess(formulation.ndim,formulation.nvar)
        post_process.SetAnalysis(self.analysis_type, self.analysis_nature)

        LoadIncrement = self.number_of_load_increments

        LoadFactor = 1./LoadIncrement
        for Increment in range(LoadIncrement):
            # COMPUTE INCREMENTAL FORCES
            NodalForces = LoadFactor*NeumannForces
            # NodalForces = LoadFactor*boundary_condition.neumann_forces
            AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet
            # DIRICHLET FORCES IS SET TO ZERO EVERY TIME
            DirichletForces = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64)
            Residual = DirichletForces + NodalForces

            t_assembly = time()
            # IF STRESSES ARE TO BE CALCULATED 
            if self.has_prestress:
                # GET THE MESH COORDINATES FOR LAST INCREMENT 
                mesh.points -= TotalDisp[:,:formulation.ndim,Increment-1]
                # ASSEMBLE
                K, TractionForces = self.Assemble(function_spaces[0], formulation, mesh, material, 
                    solver, Eulerx, np.zeros_like(mesh.points))[:2]
                # UPDATE MESH AGAIN
                mesh.points += TotalDisp[:,:formulation.ndim,Increment-1]
                # FIND THE RESIDUAL
                Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] \
                - NodalForces[boundary_condition.columns_in]
            else:
                # ASSEMBLE
                K = self.Assemble(function_spaces[0], formulation, mesh, material, solver,
                    Eulerx, np.zeros_like(mesh.points))[0]
            print('Finished assembling the system of equations. Time elapsed is', time() - t_assembly, 'seconds')

            # APPLY DIRICHLET BOUNDARY CONDITIONS & GET REDUCED MATRICES 
            K_b, F_b = boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,AppliedDirichletInc)[:2]
            # SOLVE THE SYSTEM
            t_solver=time()
            sol = solver.Solve(K_b,F_b)
            t_solver = time()-t_solver

            dU = post_process.TotalComponentSol(sol, boundary_condition.columns_in,
                boundary_condition.columns_out, AppliedDirichletInc,0,K.shape[0]) 

            # STORE TOTAL SOLUTION DATA
            TotalDisp[:,:,Increment] += dU

            # UPDATE MESH GEOMETRY
            mesh.points += TotalDisp[:,:formulation.ndim,Increment]    
            Eulerx = np.copy(mesh.points)

            if LoadIncrement > 1:
                print("Finished load increment "+str(Increment)+" for incrementally linearised problem. Solver time is", t_solver)
            else:
                print("Finished load increment "+str(Increment)+" for linear problem. Solver time is", t_solver)
            gc.collect()


            # COMPUTE SCALED JACBIAN FOR THE MESH
            if Increment == LoadIncrement - 1:
                smesh = deepcopy(mesh)
                smesh.points -= TotalDisp[:,:,-1] 

                if material.is_transversely_isotropic:
                    post_process.is_material_anisotropic = True
                    post_process.SetAnisotropicOrientations(material.anisotropic_orientations)

                post_process.SetBases(postdomain=function_spaces[1])    
                qualities = post_process.MeshQualityMeasures(smesh,TotalDisp,plot=False,show_plot=False)
                self.isScaledJacobianComputed = qualities[0]
                self.ScaledJacobian = qualities[3]

                del smesh, post_process
                gc.collect()


        return TotalDisp



    def StaticSolver(self, function_spaces, formulation, solver, K,
            NeumannForces,NodalForces,Residual,
            mesh,TotalDisp,Eulerx,Eulerp,material, boundary_condition):
    
        LoadIncrement = self.number_of_load_increments
        LoadFactor = 1./LoadIncrement
        AppliedDirichletInc = np.zeros(boundary_condition.applied_dirichlet.shape[0],dtype=np.float64)
        
        for Increment in range(LoadIncrement):

            # APPLY NEUMANN BOUNDARY CONDITIONS
            DeltaF = LoadFactor*NeumannForces
            NodalForces += DeltaF
            # OBRTAIN INCREMENTAL RESIDUAL - CONTRIBUTION FROM BOTH NEUMANN AND DIRICHLET
            Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,NodalForces,
                boundary_condition.applied_dirichlet,LoadFactor=LoadFactor)[2]
            # GET THE INCREMENTAL DISPLACEMENT
            AppliedDirichletInc = LoadFactor*boundary_condition.applied_dirichlet

            t_increment = time()

            # LET NORM OF THE FIRST RESIDUAL BE THE NORM WITH RESPECT TO WHICH WE
            # HAVE TO CHECK THE CONVERGENCE OF NEWTON RAPHSON. TYPICALLY THIS IS 
            # NORM OF NODAL FORCES
            if Increment==0:
                # self.NormForces = np.linalg.norm(Residual[boundary_condition.columns_out])
                # self.NormForces = np.linalg.norm(Residual[boundary_condition.columns_in])
                self.NormForces = np.linalg.norm(Residual)
                # AVOID DIVISION BY ZERO
                # if np.abs(self.NormForces) < 1e-14:
                if np.isclose(self.NormForces,0.0):
                    self.NormForces = 1e-14

            if np.isclose(self.NormForces,0.0):
                self.norm_residual = np.abs(la.norm(Residual[boundary_condition.columns_in]))
            else:
                self.norm_residual = np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces)

            Eulerx, Eulerp = self.NewtonRaphson(function_spaces, formulation, solver, 
                Increment,K,NodalForces,Residual,mesh,Eulerx,Eulerp,
                material,boundary_condition,AppliedDirichletInc)

            # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
            TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
            if formulation.fields == "electro_mechanics":
                TotalDisp[:,-1,Increment] = Eulerp


            print('\nFinished Load increment', Increment, 'in', time()-t_increment, 'seconds')
            try:
                print('Norm of Residual is', 
                    np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces), '\n')
            except RuntimeWarning:
                print("Invalid value encountered in norm of Newton-Raphson residual")

            # STORE THE INFORMATION IF NEWTON-RAPHSON FAILS
            if self.newton_raphson_failed_to_converge:
                solver.condA = np.NAN
                solver.scaledA = np.NAN
                solver.scaledAFF = np.NAN
                solver.scaledAHH = np.NAN
                break

        return TotalDisp


    def NewtonRaphson(self, function_spaces, formulation, solver, 
        Increment,K,NodalForces,Residual,mesh,Eulerx,Eulerp,material,
        boundary_condition,AppliedDirichletInc):

        Tolerance = self.newton_raphson_tolerance
        LoadIncrement = self.number_of_load_increments
        Iter = 0


        # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
        IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
            K.shape[0],formulation.nvar)
        # UPDATE EULERIAN COORDINATE
        Eulerx += IncDirichlet[:,:formulation.ndim]
        Eulerp += IncDirichlet[:,-1]


        while self.norm_residual > Tolerance:
        # while np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces) > Tolerance:
            # GET THE REDUCED SYSTEM OF EQUATIONS
            K_b, F_b = boundary_condition.GetReducedMatrices(K,Residual)[:2]

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b,-F_b)

            # GET ITERATIVE SOLUTION
            dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar) 

            # UPDATE THE GEOMETRY
            Eulerx += dU[:,:formulation.ndim]
            Eulerp += dU[:,-1]

            # GET ITERATIVE ELECTRIC POTENTIAL
            # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
            K, TractionForces = self.Assemble(function_spaces[0], formulation, mesh, material, solver,
                Eulerx,Eulerp)[:2]

            # FIND THE RESIDUAL
            Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] \
            - NodalForces[boundary_condition.columns_in]

            # SAVE THE NORM 
            if np.isclose(self.NormForces,0.0):
                self.norm_residual = np.abs(la.norm(Residual[boundary_condition.columns_in]))
            else:
                self.norm_residual = np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces)

            self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
                self.norm_residual)

            
            print('Iteration number', Iter, 'for load increment', 
                Increment, 'with a residual of \t\t', self.norm_residual) 

            # # SAVE THE NORM 
            # self.NRConvergence['Increment_'+str(Increment)] = np.append(self.NRConvergence['Increment_'+str(Increment)],\
            #     np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces))
            
            # print('Iteration number', Iter, 'for load increment', Increment, 'with a residual of \t\t', \
            #     np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces)) 

            # UPDATE ITERATION NUMBER
            Iter +=1

            if Iter==self.maximum_iteration_for_newton_raphson and formulation.fields == "electro_mechanics":
                raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

            if Iter==self.maximum_iteration_for_newton_raphson:
                self.newton_raphson_failed_to_converge = True
                break
            if np.isnan(self.norm_residual):
                self.newton_raphson_failed_to_converge = True
                break


            # if Iter==self.maximum_iteration_for_newton_raphson and formulation.fields == "electro_mechanics":
            #     raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

            # if Iter==self.maximum_iteration_for_newton_raphson:
            #     self.newton_raphson_failed_to_converge = True
            #     break
            # if np.isnan(np.abs(la.norm(Residual[boundary_condition.columns_in])/self.NormForces)):
            #     self.newton_raphson_failed_to_converge = True
            #     break


        return Eulerx, Eulerp



    def Assemble(self, function_space, formulation, mesh, material, solver, Eulerx, Eulerp):

        if self.memory_model == "shared" or self.memory_model is None:
            if mesh.nelem <= 500000:
                return self.AssemblySmall(function_space, formulation, mesh, material, Eulerx, Eulerp)
            elif mesh.nelem > 500000:
                print("Larger than memory system. Dask on disk parallel assembly is turned on")
                return self.OutofCoreAssembly(function_space,mesh,material,formulation,Eulerx,Eulerp)

        elif self.memory_model == "distributed":
            # RUN THIS PROGRAM FROM SHELL WITH python RunSession.py INSTEAD
            if not __PARALLEL__:
                warn("parallelisation is going to be turned on")

            import subprocess, os, shutil
            from time import time
            from Florence.Utils import par_unpickle

            tmp_dir = par_unpickle(function_space,mesh,material,Eulerx,Eulerp)
            pwd = os.path.dirname(os.path.realpath(__file__))
            distributed_caller = os.path.join(pwd,"DistributedAssembly.py")

            t_dassembly = time()
            p = subprocess.Popen("time mpirun -np "+str(MP.cpu_count())+" Florence/FiniteElements/DistributedAssembly.py"+" /home/roman/tmp/", 
                cwd="/home/roman/Dropbox/florence/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            # p = subprocess.Popen("./mpi_runner.sh", cwd="/home/roman/Dropbox/florence/", shell=True)
            p.wait()
            print('MPI took', time() - t_dassembly, 'seconds for distributed assembly')
            Dict = loadmat(os.path.join(tmp_dir,"results.mat"))

            try:
                shutil.rmtree(tmp_dir)
            except IOError:
                raise IOError("Could not delete the directory")

            return Dict['stiffness'], Dict['T'], Dict['F'], []


    def AssemblySmall(self, function_space, formulation, mesh, material, Eulerx, Eulerp):

        # GET MESH DETAILS
        C = mesh.InferPolynomialDegree() - 1
        nvar = formulation.nvar
        ndim = formulation.ndim
        nelem = mesh.nelem
        nodeperelem = mesh.elements.shape[1]

        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
        I_stiffness=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        J_stiffness=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        # V_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.float32)
        V_stiffness=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)

        I_mass=[]; J_mass=[]; V_mass=[]
        if self.analysis_type !='static':
            # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
            I_mass=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int32)
            J_mass=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int32)
            V_mass=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.float64)

        T = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)
        # T = np.zeros((mesh.points.shape[0]*nvar,1),np.float32)  

        mass, F = [], []
        if self.has_moving_boundary:
            F = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)


        if self.parallel:
            # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
            # ParallelTuple = parmap.map(formulation.GetElementalMatrices,np.arange(0,nelem,dtype=np.int32),
                # function_space, mesh, material, self, Eulerx, Eulerp)

            ParallelTuple = parmap.map(formulation,np.arange(0,nelem,dtype=np.int32),
                function_space, mesh, material, self, Eulerx, Eulerp)

        for elem in range(nelem):

            if self.parallel:
                # UNPACK PARALLEL TUPLE VALUES
                I_stiff_elem = ParallelTuple[elem][0]; J_stiff_elem = ParallelTuple[elem][1]; V_stiff_elem = ParallelTuple[elem][2]
                t = ParallelTuple[elem][3]; f = ParallelTuple[elem][4]
                I_mass_elem = ParallelTuple[elem][5]; J_mass_elem = ParallelTuple[elem][6]; V_mass_elem = ParallelTuple[elem][6]

            else:
                # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
                I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, \
                I_mass_elem, J_mass_elem, V_mass_elem = formulation.GetElementalMatrices(elem, 
                    function_space, mesh, material, self, Eulerx, Eulerp)
            # SPARSE ASSEMBLY - STIFFNESS MATRIX
            SparseAssemblyNative(I_stiff_elem,J_stiff_elem,V_stiff_elem,I_stiffness,J_stiffness,V_stiffness,
                elem,nvar,nodeperelem,mesh.elements)

            # SparseAssemblySmall(I_stiff_elem,J_stiff_elem,V_stiff_elem,
            #   I_stiffness,J_stiffness,V_stiffness,elem,nvar,nodeperelem,mesh.elements)

            if self.analysis_type != 'static':
                # SPARSE ASSEMBLY - MASS MATRIX
                # I_mass, J_mass, V_mass = SparseAssemblySmall(I_mass_elem,J_mass_elem,V_mass_elem,
                #     I_mass,J_mass,V_mass,elem,ndim,nodeperelem,mesh.elements)
                SparseAssemblyNative(I_mass_elem,J_mass_elem,V_mass_elem,I_mass,J_mass,V_mass,
                    elem,ndim,nodeperelem,mesh.elements)

            if self.has_moving_boundary:
                # RHS ASSEMBLY
                for iterator in range(0,nvar):
                    F[mesh.elements[elem,:]*nvar+iterator,0]+=f[iterator::nvar,0]
            # INTERNAL TRACTION FORCE ASSEMBLY
            for iterator in range(0,nvar):
                T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]

        if self.parallel:
            del ParallelTuple
        gc.collect()

        # REALLY DANGEROUS FOR MULTIPHYSICS PROBLEMS
        # V_stiffness[np.isclose(V_stiffness,0.)] = 0.

        stiffness = coo_matrix((V_stiffness,(I_stiffness,J_stiffness)),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64).tocsc()
        # stiffness = csc_matrix((V_stiffness,(I_stiffness,J_stiffness)),
            # shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float32)
        # stiffness = csc_matrix((V_stiffness,(I_stiffness,J_stiffness)),
            # shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64)
        
        # GET STORAGE/MEMORY DETAILS
        self.spmat = stiffness.data.nbytes/1024./1024.
        self.ijv = (I_stiffness.nbytes + J_stiffness.nbytes + V_stiffness.nbytes)/1024./1024.

        del I_stiffness, J_stiffness, V_stiffness
        gc.collect()

        if self.analysis_type != 'static':
            mass = csc_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],
                nvar*mesh.points.shape[0])),dtype=np.float32)

        return stiffness, T, F, mass



    #-------------- ASSEMBLY ROUTINE FOR RELATIVELY LARGER MATRICES ( 1e06 < NELEM < 1e07 3D)------------------------#
    #----------------------------------------------------------------------------------------------------------------#

    def AssemblyLarge(MainData,mesh,material,Eulerx,TotalPot):

        # GET MESH DETAILS
        C = MainData.C
        nvar = MainData.nvar
        ndim = MainData.ndim

        # nelem = mesh.elements.shape[0]
        nelem = mesh.nelem
        nodeperelem = mesh.elements.shape[1]

        from tempfile import mkdtemp

        # WRITE TRIPLETS ON DESK
        pwd = os.path.dirname(os.path.realpath(__file__))
        tmp_dir = mkdtemp()
        I_filename = os.path.join(tmp_dir, 'I_stiffness.dat')
        J_filename = os.path.join(tmp_dir, 'J_stiffness.dat')
        V_filename = os.path.join(tmp_dir, 'V_stiffness.dat')

        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX
        I_stiffness = np.memmap(I_filename,dtype=np.int32,mode='w+',shape=((nvar*nodeperelem)**2*nelem,))
        J_stiffness = np.memmap(J_filename,dtype=np.int32,mode='w+',shape=((nvar*nodeperelem)**2*nelem,))
        V_stiffness = np.memmap(V_filename,dtype=np.float32,mode='w+',shape=((nvar*nodeperelem)**2*nelem,))

        # I_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int64)
        # J_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int64)
        # V_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.float64)

        # THE I & J VECTORS OF LOCAL STIFFNESS MATRIX DO NOT CHANGE, HENCE COMPUTE THEM ONCE
        I_stiff_elem = np.repeat(np.arange(0,nvar*nodeperelem),nvar*nodeperelem,axis=0)
        J_stiff_elem = np.tile(np.arange(0,nvar*nodeperelem),nvar*nodeperelem)

        I_mass=[];J_mass=[];V_mass=[]; I_mass_elem = []; J_mass_elem = []
        if MainData.Analysis !='Static':
            # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX
            I_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.int64)
            J_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.int64)
            V_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.float64)

            # THE I & J VECTORS OF LOCAL MASS MATRIX DO NOT CHANGE, HENCE COMPUTE THEM ONCE
            I_mass_elem = np.repeat(np.arange(0,nvar*nodeperelem),nvar*nodeperelem,axis=0)
            J_mass_elem = np.tile(np.arange(0,nvar*nodeperelem),nvar*nodeperelem)


        # ALLOCATE RHS VECTORS
        F = np.zeros((mesh.points.shape[0]*nvar,1)) 
        T =  np.zeros((mesh.points.shape[0]*nvar,1)) 
        # ASSIGN OTHER NECESSARY MATRICES
        full_current_row_stiff = []; full_current_column_stiff = []; coeff_stiff = [] 
        full_current_row_mass = []; full_current_column_mass = []; coeff_mass = []
        mass = []

        if MainData.Parallel:
            # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
            ParallelTuple = parmap.map(GetElementalMatrices,np.arange(0,nelem),MainData,mesh.elements,mesh.points,nodeperelem,
                Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem,pool=MP.Pool(processes=MainData.nCPU))

        for elem in range(nelem):

            if MainData.Parallel:
                # UNPACK PARALLEL TUPLE VALUES
                full_current_row_stiff = ParallelTuple[elem][0]; full_current_column_stiff = ParallelTuple[elem][1]
                coeff_stiff = ParallelTuple[elem][2]; t = ParallelTuple[elem][3]; f = ParallelTuple[elem][4]
                full_current_row_mass = ParallelTuple[elem][5]; full_current_column_mass = ParallelTuple[elem][6]; coeff_mass = ParallelTuple[elem][6]

            else:
                # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
                full_current_row_stiff, full_current_column_stiff, coeff_stiff, t, f, \
                full_current_row_mass, full_current_column_mass, coeff_mass = GetElementalMatrices(elem,
                    MainData,mesh.elements,mesh.points,nodeperelem,Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem)

            # SPARSE ASSEMBLY - STIFFNESS MATRIX
            # I_stiffness, J_stiffness, V_stiffness = SparseAssembly_Step_2(I_stiffness,J_stiffness,V_stiffness,
                # full_current_row_stiff,full_current_column_stiff,coeff_stiff,
            #   nvar,nodeperelem,elem)

            I_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row_stiff
            J_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column_stiff
            V_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff_stiff

            if MainData.Analysis != 'Static':
                # SPARSE ASSEMBLY - MASS MATRIX
                I_mass, J_mass, V_mass = SparseAssembly_Step_2(I_mass,J_mass,V_mass,full_current_row_mass,full_current_column_mass,coeff_mass,
                    nvar,nodeperelem,elem)

            if MainData.AssemblyParameters.ExternalLoadNature == 'Nonlinear':
                # RHS ASSEMBLY
                for iterator in range(0,nvar):
                    F[mesh.elements[elem,:]*nvar+iterator,0]+=f[iterator::nvar]
            # INTERNAL TRACTION FORCE ASSEMBLY
            for iterator in range(0,nvar):
                    T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]

        # CALL BUILT-IN SPARSE ASSEMBLER 
        stiffness = coo_matrix((V_stiffness,(I_stiffness,J_stiffness)),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64).tocsc()

        # GET STORAGE/MEMORY DETAILS
        MainData.spmat = stiffness.data.nbytes/1024./1024.
        MainData.ijv = (I_stiffness.nbytes + J_stiffness.nbytes + V_stiffness.nbytes)/1024./1024.
        del I_stiffness, J_stiffness, V_stiffness
        gc.collect()

        if MainData.Analysis != 'Static':
            # CALL BUILT-IN SPARSE ASSEMBLER
            mass = coo_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0]))).tocsc()


        return stiffness, T, F, mass



    def OutofCoreAssembly(MainData, mesh, material, Eulerx, TotalPot, calculate_rhs=True, filename=None, chunk_size=None):
        """Assembly routine for larger than memory system of equations. 
            Usage of h5py and dask allow us to store the triplets and build a sparse matrix out of 
            them on disk.

            Note: The sparse matrix itself is created on the memory.
        """

        import sys, os
        from warnings import warn
        from time import time
        import psutil
        # from Core.Supplementary.dsparse.sparse import dok_matrix


        if MainData.Parallel is True or MainData.__PARALLEL__ is True:
            warn("Parallel assembly cannot performed on large arrays. \n" 
                "Out of core 'i.e. Dask' parallelisation is turned on instead. "
                "This is an innocuous warning")

        try:
            import h5py
        except ImportError:
            has_h5py = False
            raise ImportError('h5py is not installed. Please install it first by running "pip install h5py"')

        try:
            import dask.array as da
        except ImportError:
            has_dask = False
            raise ImportError('dask is not installed. Please install it first by running "pip install toolz && pip install dask"')

        if filename is None:
            warn("filename not given. I am going to write the output in the current directory")
            pwd = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(pwd,"output.hdf5")


        # GET MESH DETAILS
        C = MainData.C
        nvar = MainData.nvar
        ndim = MainData.ndim

        nelem = mesh.nelem
        nodeperelem = mesh.elements.shape[1]

        # GET MEMORY INFO
        memory = psutil.virtual_memory()
        size_of_triplets_gbytes = (mesh.points.shape[0]*nvar)**2*nelem*(4)*(3)//1024**3
        if memory.available//1024**3 > 2*size_of_triplets_gbytes:
            warn("Out of core assembly is only efficient for larger than memory "
                "system of equations. Using it on smaller matrices can be very inefficient")

        # hdf_file = h5py.File(filename,'w')
        # IJV_triplets = hdf_file.create_dataset("IJV_triplets",((nvar*nodeperelem)**2*nelem,3),dtype=np.float32)


        # THE I & J VECTORS OF LOCAL STIFFNESS MATRIX DO NOT CHANGE, HENCE COMPUTE THEM ONCE
        I_stiff_elem = np.repeat(np.arange(0,nvar*nodeperelem),nvar*nodeperelem,axis=0)
        J_stiff_elem = np.tile(np.arange(0,nvar*nodeperelem),nvar*nodeperelem)

        I_mass=[];J_mass=[];V_mass=[]; I_mass_elem = []; J_mass_elem = []

        if calculate_rhs is False:
            F = []
            T = []
        else:
            F = np.zeros((mesh.points.shape[0]*nvar,1),np.float32)
            T = np.zeros((mesh.points.shape[0]*nvar,1),np.float32)  

        # ASSIGN OTHER NECESSARY MATRICES
        full_current_row_stiff = []; full_current_column_stiff = []; coeff_stiff = [] 
        full_current_row_mass = []; full_current_column_mass = []; coeff_mass = []
        mass = []

        gc.collect()

        print('Writing the triplets to disk')
        t_hdf5 = time()
        for elem in range(nelem):

            full_current_row_stiff, full_current_column_stiff, coeff_stiff, t, f, \
                full_current_row_mass, full_current_column_mass, coeff_mass = GetElementalMatrices(elem,
                    MainData,mesh.elements,mesh.points,nodeperelem,Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem)

            IJV_triplets[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1),0] = full_current_row_stiff.flatten()
            IJV_triplets[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1),1] = full_current_column_stiff.flatten()
            IJV_triplets[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1),2] = coeff_stiff.flatten()

            if calculate_rhs is True:

                if MainData.Analysis != 'Static':
                    # SPARSE ASSEMBLY - MASS MATRIX
                    I_mass, J_mass, V_mass = SparseAssembly_Step_2(I_mass,J_mass,V_mass,full_current_row_mass,full_current_column_mass,coeff_mass,
                        nvar,nodeperelem,elem)

                if MainData.AssemblyParameters.ExternalLoadNature == 'Nonlinear':
                    # RHS ASSEMBLY
                    for iterator in range(0,nvar):
                        F[mesh.elements[elem,:]*nvar+iterator,0]+=f[iterator::nvar]
                # INTERNAL TRACTION FORCE ASSEMBLY
                for iterator in range(0,nvar):
                        T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]

            if elem % 10000 == 0:
                print("Processed ", elem, " elements")

        hdf_file.close()
        print('Finished writing the triplets to disk. Time taken', time() - t_hdf5, 'seconds')


        print('Reading the triplets back from disk')
        hdf_file = h5py.File(filename,'r')
        if chunk_size is None:
            chunk_size = mesh.points.shape[0]*nvar // 300

        print('Creating dask array from triplets')
        IJV_triplets = da.from_array(hdf_file['IJV_triplets'],chunks=(chunk_size,3))


        print('Creating the sparse matrix')
        t_sparse = time()

        stiffness = csr_matrix((IJV_triplets[:,2].astype(np.float32),
            (IJV_triplets[:,0].astype(np.int32),IJV_triplets[:,1].astype(np.int32))),
            shape=((mesh.points.shape[0]*nvar,mesh.points.shape[0]*nvar)),dtype=np.float32)

        print('Done creating the sparse matrix, time taken', time() - t_sparse)
        
        hdf_file.close()

        return stiffness, T, F, mass



    def AssemblyForces(self,mesh,Quadrature,Domain,BoundaryData,Boundary):

        C = MainData.C
        nvar = MainData.nvar
        ndim = MainData.ndim

        mesh.points = np.array(mesh.points)
        mesh.elements = np.array(mesh.elements)
        nelem = mesh.elements.shape[0]
        nodeperelem = mesh.elements.shape[1]


        F = np.zeros((mesh.points.shape[0]*nvar,1)) 
        f = []

        for elem in range(0,nelem):
            LagrangeElemCoords = np.zeros((nodeperelem,ndim))
            for i in range(0,nodeperelem):
                LagrangeElemCoords[i,:] = mesh.points[mesh.elements[elem,i],:]

            
            if ndim==2:
                # Compute Force vector
                f = np.zeros(k.shape[0])
            elif ndim==3:
                # Compute Force vector
                f = ApplyNeumannBoundaryConditions3D(MainData, mesh, 
                    BoundaryData, Domain, Boundary, Quadrature.weights, elem, LagrangeElemCoords)


            # Static Condensation
            # if C>0:
            #   k,f = St.StaticCondensation(k,f,C,nvar)

            # RHS Assembly
            for iter in range(0,nvar):
                F[mesh.elements[elem]*nvar+iter,0]+=f[iter:f.shape[0]:nvar]


        return F
