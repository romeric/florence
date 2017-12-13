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
from Florence.TimeIntegrators import StructuralDynamicIntegrators
from Florence.TimeIntegrators import ExplicitStructuralDynamicIntegrators
from .LaplacianSolver import LaplacianSolver
from Florence import Mesh


__all__ = ["FEMSolver"]


class FEMSolver(object):
    """Solver for linear and non-linear finite elements.
        This class is fundamentally different from the LinearSolver, as linear solver
        specifically deals with the solution of linear system of equations, whereas FEM
        solver is essentially responsible for building the discretised linear, linearised
        and nonlinear systems arising from variational formulations
    """

    def __init__(self, has_low_level_dispatcher=False,
        analysis_type="static", analysis_nature="nonlinear", analysis_subtype="implicit",
        is_geometrically_linearised=False, requires_geometry_update=True,
        requires_line_search=False, requires_arc_length=False, has_moving_boundary=False,
        has_prestress=True, number_of_load_increments=1, load_factor=None,
        newton_raphson_tolerance=1.0e-6, newton_raphson_solution_tolerance=None,
        maximum_iteration_for_newton_raphson=50, iterative_technique="newton_raphson",
        add_self_weight=False, mass_type=None,
        compute_mesh_qualities=True,
        parallelise=False, memory_model="shared", platform="cpu", backend="opencl",
        print_incremental_log=False, save_incremental_solution=False, incremental_solution_filename=None,
        break_at_increment=-1,
        include_physical_damping=False, damping_factor=0.1,
        compute_energy_dissipation=False, compute_linear_momentum_dissipation=False, total_time=1.,
        user_defined_break_func=None, user_defined_stop_func=None,
        save_results=True, save_frequency=1):

        self.has_low_level_dispatcher = has_low_level_dispatcher

        self.analysis_nature = analysis_nature
        self.analysis_type = analysis_type
        self.analysis_subtype = analysis_subtype

        self.is_geometrically_linearised = is_geometrically_linearised
        self.requires_geometry_update = requires_geometry_update
        self.requires_line_search = requires_line_search
        self.requires_arc_length = requires_arc_length
        self.has_moving_boundary = has_moving_boundary
        self.has_prestress = has_prestress
        self.is_mass_computed = False
        self.mass_type = mass_type # "consistent" or "lumped"
        self.total_time = float(total_time)
        self.save_results = save_results
        # SAVE AT EVERY N TIME STEP WHERE N=save_frequency
        self.save_frequency = int(save_frequency)

        self.number_of_load_increments = number_of_load_increments
        self.load_factor = load_factor
        self.newton_raphson_tolerance = newton_raphson_tolerance
        self.newton_raphson_solution_tolerance = newton_raphson_solution_tolerance
        self.maximum_iteration_for_newton_raphson = maximum_iteration_for_newton_raphson
        self.newton_raphson_failed_to_converge = False
        self.NRConvergence = None
        self.iterative_technique = iterative_technique
        self.include_physical_damping = include_physical_damping
        self.damping_factor = damping_factor
        self.add_self_weight = add_self_weight

        self.compute_energy_dissipation = compute_energy_dissipation
        self.compute_linear_momentum_dissipation = compute_linear_momentum_dissipation

        self.compute_mesh_qualities = compute_mesh_qualities
        self.isScaledJacobianComputed = False

        self.vectorise = True
        self.parallel = parallelise
        self.no_of_cpu_cores = multiprocessing.cpu_count()
        self.memory_model = memory_model
        self.platform = platform
        self.backend = backend
        self.debug = False

        self.print_incremental_log = print_incremental_log
        self.save_incremental_solution = save_incremental_solution
        self.incremental_solution_filename = incremental_solution_filename
        self.break_at_increment = break_at_increment
        self.user_defined_break_func = user_defined_break_func
        self.user_defined_stop_func = user_defined_stop_func

        self.fem_timer = 0.

        if self.newton_raphson_solution_tolerance is None:
            self.newton_raphson_solution_tolerance = 10.*self.newton_raphson_tolerance


    def __checkdata__(self, material, boundary_condition, formulation, mesh, function_spaces, solver):
        """Checks the state of data for FEMSolver"""

        # INITIAL CHECKS
        ###########################################################################
        if mesh is None:
            raise ValueError("No mesh detected for the analysis")
        elif not isinstance(mesh,Mesh):
            raise ValueError("mesh has to be an instance of Florence.Mesh")
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
            solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack", geometric_discretisation=mesh.element_type)

        if material.ndim != mesh.InferSpatialDimension():
            # THIS HAS TO BE AN ERROR BECAUSE OF THE DYNAMIC NATURE OF MATERIAL
            raise ValueError("Material model and mesh are incompatible. Change the dimensionality of the material")
        ###########################################################################


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

        if "Explicit" in material.mtype:
            if self.analysis_subtype == "implicit":
                raise ValueError("Incorrect material model ({}) used for implicit analysis".format(material.mtype))
        if self.analysis_subtype == "explicit":
            if self.mass_type is None:
                self.mass_type = "lumped"
        if self.analysis_type == "static":
            if self.save_frequency != 1:
                warn("save_frequency must be one")
                self.save_frequency = 1
        if self.analysis_type == "dynamics" and self.analysis_subtype=="implicit":
            if self.save_frequency != 1:
                warn("save_frequency must be one")
                self.save_frequency = 1



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

        ##############################################################################
        if self.load_factor is not None:
            self.load_factor = np.array(self.load_factor).ravel()
            if self.load_factor.shape[0] != self.number_of_load_increments:
                raise ValueError("Supplied load factor should have the same length as the number of load increments")
            if not np.isclose(self.load_factor.sum(),1.0):
                raise ValueError("Load factor should sum up to one")
        ##############################################################################

        ##############################################################################
        if self.include_physical_damping and self.compute_energy_dissipation:
            warn("Energy is not going to be preserved due to physical damping")
        ##############################################################################

        # CHANGE MESH DATA TYPE
        mesh.ChangeType()
        # ASSIGN ANALYSIS PARAMTER TO BOUNDARY CONDITION
        boundary_condition.analysis_type = self.analysis_type
        boundary_condition.analysis_nature = self.analysis_nature

        return function_spaces, solver



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

        # self.compute_mesh_qualities = True
        if self.analysis_nature == "nonlinear" and self.compute_mesh_qualities:
            # COMPUTE QUALITY MEASURES
            post_process.ScaledJacobian = post_process.MeshQualityMeasures(mesh,TotalDisp,False,False)[3]
        elif self.isScaledJacobianComputed:
            post_process.ScaledJacobian=self.ScaledJacobian
        # self.isScaledJacobianComputed = False
        # post_process.is_scaledjacobian_computed = False
        # post_process.ScaledJacobian=post_process.MeshQualityMeasures(mesh,TotalDisp,False,False)[3]

        if self.analysis_nature == "nonlinear":
            post_process.newton_raphson_convergence = self.NRConvergence

        if self.analysis_type == "dynamic":
            if self.compute_energy_dissipation:
                post_process.energy_dissipation = formulation.energy_dissipation
                post_process.internal_energy = formulation.internal_energy
                post_process.kinetic_energy = formulation.kinetic_energy
                post_process.external_energy = formulation.external_energy
            if self.compute_linear_momentum_dissipation:
                post_process.power_dissipation = formulation.power_dissipation
                post_process.internal_power = formulation.internal_power
                post_process.kinetic_power = formulation.kinetic_power
                post_process.external_power = formulation.external_power

        return post_process


    @property
    def WhichFEMSolver():
        solver = None
        if self.analysis_type == "dynamic":
            solver = "StructuralDynamicIntegrator"
        else:
            if self.analysis_nature == "linear":
                if self.number_of_load_increments > 1:
                    solver = "IncrementalLinearElasticitySolver"
                else:
                    solver = "LinearElasticity"
            else:
                solver = self.iterative_technique
        print(solver)
        return solver

    @property
    def WhichFEMSolvers():
        solvers = ["LinearElasticity","IncrementalLinearElasticitySolver","NewtonRaphson","ModifiedNewtonRaphson",
            "NewtonRaphsonLineSearch","NewtonRaphsonArchLength","StructuralDynamicIntegrator"]
        print(solvers)
        return solvers


    def Solve(self, formulation=None, mesh=None,
        material=None, boundary_condition=None,
        function_spaces=None, solver=None,
        Eulerx=None, Eulerp=None):
        """Main solution routine for FEMSolver """


        # CHECK DATA CONSISTENCY
        #---------------------------------------------------------------------------#
        function_spaces, solver = self.__checkdata__(material, boundary_condition, formulation, mesh, function_spaces, solver)
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

        # QUICK REDIRECT TO LAPLACIAN SOLVER
        if formulation.fields == "electrostatics":
            laplacian_solver = LaplacianSolver(self)
            return laplacian_solver.Solve(formulation=formulation, mesh=mesh,
                material=material, boundary_condition=boundary_condition,
                function_spaces=function_spaces, solver=solver, Eulerx=Eulerx, Eulerp=Eulerp)


        # INITIATE DATA FOR THE ANALYSIS
        NodalForces, Residual = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64), \
            np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float64)
        # SET NON-LINEAR PARAMETERS
        self.NRConvergence = { 'Increment_'+str(Increment) : [] for Increment in range(self.number_of_load_increments) }

        # ALLOCATE FOR SOLUTION FIELDS
        if self.save_frequency == 1:
            # TotalDisp = np.zeros((mesh.points.shape[0],formulation.nvar,self.number_of_load_increments),dtype=np.float32)
            TotalDisp = np.zeros((mesh.points.shape[0],formulation.nvar,self.number_of_load_increments),dtype=np.float64)
        else:
            TotalDisp = np.zeros((mesh.points.shape[0],formulation.nvar,
                int(self.number_of_load_increments/self.save_frequency)),dtype=np.float64)

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
        NeumannForces = boundary_condition.ComputeNeumannForces(mesh, material, function_spaces,
            compute_traction_forces=True, compute_body_forces=self.add_self_weight)

        # ADOPT A DIFFERENT PATH FOR INCREMENTAL LINEAR ELASTICITY
        if formulation.fields == "mechanics" and self.analysis_nature != "nonlinear":
            # MAKE A COPY OF MESH, AS MESH POINTS WILL BE OVERWRITTEN
            vmesh = deepcopy(mesh)
            TotalDisp = self.IncrementalLinearElasticitySolver(function_spaces, formulation, vmesh, material,
                boundary_condition, solver, TotalDisp, Eulerx, NeumannForces)
            del vmesh

            # ADD EACH INCREMENTAL CONTRIBUTION TO MAKE IT CONSISTENT WITH THE NONLINEAR ANALYSIS
            for i in range(TotalDisp.shape[2]-1,0,-1):
                TotalDisp[:,:,i] = np.sum(TotalDisp[:,:,:i+1],axis=2)

            return self.__makeoutput__(mesh, TotalDisp, formulation, function_spaces, material)

        # ASSEMBLE STIFFNESS MATRIX AND TRACTION FORCES FOR THE FIRST TIME
        if self.analysis_type == "static":
            K, TractionForces, _, _ = Assemble(self, function_spaces[0], formulation, mesh, material,
                Eulerx, Eulerp)
        else:
            fspace = function_spaces[0] if (mesh.element_type=="hex" or mesh.element_type=="quad") else function_spaces[1]
            # fspace = function_spaces[1]
            # COMPUTE CONSTANT PART OF MASS MATRIX
            formulation.GetConstantMassIntegrand(fspace,material)

            if self.analysis_subtype != "explicit":
                # COMPUTE BOTH STIFFNESS AND MASS USING HIGHER QUADRATURE RULE
                K, TractionForces, _, M = Assemble(self, fspace, formulation, mesh, material,
                    Eulerx, Eulerp)
            else:
                # lmesh = mesh.ConvertToLinearMesh()
                # COMPUTE BOTH STIFFNESS AND MASS USING HIGHER QUADRATURE RULE
                TractionForces, _, M = AssembleExplicit(self, fspace, formulation, mesh, material,
                    Eulerx, Eulerp)

        if self.analysis_nature == 'nonlinear':
            print('Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'seconds')
        else:
            print('Finished the assembly stage. Time elapsed was', time()-tAssembly, 'seconds')


        if self.analysis_type != 'static':
            if self.analysis_subtype != "explicit":
                structural_integrator = StructuralDynamicIntegrators()
                TotalDisp = structural_integrator.Solver(function_spaces, formulation, solver,
                    K, M, NeumannForces, NodalForces, Residual,
                    mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, self)
            else:
                structural_integrator = ExplicitStructuralDynamicIntegrators()
                TotalDisp = structural_integrator.Solver(function_spaces, formulation, solver,
                    TractionForces, M, NeumannForces, NodalForces, Residual,
                    mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition, self)

        else:
            if self.iterative_technique == "newton_raphson" or self.iterative_technique == "modified_newton_raphson":
                TotalDisp = self.StaticSolver(function_spaces, formulation, solver,
                    K,NeumannForces,NodalForces,Residual,
                    mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition)
            elif self.iterative_technique == "arc_length":
                from FEMSolverArcLength import StaticSolverArcLength
                TotalDisp = StaticSolverArcLength(self,function_spaces, formulation, solver,
                    K,NeumannForces,NodalForces,Residual,
                    mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition)

            # from FEMSolverDisplacementControl import StaticSolverDisplacementControl
            # TotalDisp = StaticSolverDisplacementControl(self,function_spaces, formulation, solver,
            #     K,NeumannForces,NodalForces,Residual,
            #     mesh, TotalDisp, Eulerx, Eulerp, material, boundary_condition)

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
                K, TractionForces = Assemble(self, function_spaces[0], formulation, mesh, material,
                    Eulerx, np.zeros(mesh.points.shape[0]))[:2]
                # UPDATE MESH AGAIN
                mesh.points += TotalDisp[:,:formulation.ndim,Increment-1]
                # FIND THE RESIDUAL
                Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] \
                - NodalForces[boundary_condition.columns_in]
            else:
                # ASSEMBLE
                K = Assemble(self, function_spaces[0], formulation, mesh, material,
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
            # Residual = -boundary_condition.ApplyDirichletGetReducedMatrices(K,Residual,
                # boundary_condition.applied_dirichlet,LoadFactor=LoadFactor)[2]
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

            # Eulerx, Eulerp, K, Residual = self.NewtonRaphsonLineSearch(function_spaces, formulation, solver,
            #     Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp,
            #     material, boundary_condition, AppliedDirichletInc)

            # UPDATE DISPLACEMENTS FOR THE CURRENT LOAD INCREMENT
            TotalDisp[:,:formulation.ndim,Increment] = Eulerx - mesh.points
            if formulation.fields == "electro_mechanics":
                TotalDisp[:,-1,Increment] = Eulerp

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
                TotalDisp = TotalDisp[:,:,:Increment]
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
        Eulerx += IncDirichlet[:,:formulation.ndim]
        Eulerp += IncDirichlet[:,-1]

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





    def NewtonRaphsonLineSearch(self, function_spaces, formulation, solver,
        Increment, K, NodalForces, Residual, mesh, Eulerx, Eulerp, material,
        boundary_condition, AppliedDirichletInc):

        Tolerance = self.newton_raphson_tolerance
        LoadIncrement = self.number_of_load_increments
        Iter = 0


        # APPLY INCREMENTAL DIRICHLET PER LOAD STEP (THIS IS INCREMENTAL NOT ACCUMULATIVE)
        IncDirichlet = boundary_condition.UpdateFixDoFs(AppliedDirichletInc,
            K.shape[0],formulation.nvar)
        # UPDATE EULERIAN COORDINATE
        Eulerx += IncDirichlet[:,:formulation.ndim]
        Eulerp += IncDirichlet[:,-1]

        eta = 1.

        while self.norm_residual > Tolerance or Iter==0:
            # GET THE REDUCED SYSTEM OF EQUATIONS
            K_b, F_b = boundary_condition.GetReducedMatrices(K,Residual)[:2]

            # SOLVE THE SYSTEM
            sol = solver.Solve(K_b,-F_b)

            # GET ITERATIVE SOLUTION
            dU = boundary_condition.UpdateFreeDoFs(sol,K.shape[0],formulation.nvar)

            # UPDATE THE EULERIAN COMPONENTS
            Eulerx += eta*dU[:,:formulation.ndim]
            Eulerp += eta*dU[:,-1]

            # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES
            K, TractionForces = Assemble(self, function_spaces[0], formulation, mesh, material,
                Eulerx,Eulerp)[:2]

            # print(Residual.shape,dU.shape)
            R0 = np.dot(Residual.ravel(),dU.ravel())

            # FIND THE RESIDUAL
            Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] -\
                NodalForces[boundary_condition.columns_in]

            # print("eta" ,eta)
            R1 = np.dot(Residual.ravel(),dU.ravel())
            alpha = R0/R1
            rho = 0.5

            from scipy.optimize import newton
            if alpha < 0.:
                eta = alpha/2. + np.sqrt((alpha/2.)**2. - alpha)
            else:
                eta = alpha/2.

            def func(x):
                return (1-x)*R0 + R1*x**2
            # def ffunc(x):
            #     return -R0 + 2.*R1*x
            # etaa = newton(func,eta,fprime=ffunc)
            # print(etaa )
            # eta = etaa
            # print(func(eta),rho*func(0))
            # if Increment == 0:
                # eta = 1.0
            if np.abs(func(eta)) < np.abs(rho*func(0)):
                eta = func(eta)
            else:
                eta = 1.


            # print(norm(R1-R0))

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