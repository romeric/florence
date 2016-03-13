import os, sys
from time import time
import numpy as np
from copy import deepcopy

from Florence.FiniteElements.Assembly import *
# from Florence.FiniteElements.ApplyDirichletBoundaryConditions import *
from Florence.FiniteElements.PostProcess import *
from Florence.FiniteElements.InitiateNonlinearAnalysisData import *
from Florence.FiniteElements.Solvers.StaticSolver import *
from Florence.FiniteElements.Solvers.IncrementalLinearElasticitySolver import *
# from Core.FiniteElements.Solvers.DynamicSolver import *
# from Florence.FiniteElements.StaticCondensationGlobal import *

# def MainSolver(MainData, mesh, material, boundary_condition):
def MainSolver(function_spaces, formulation, mesh, material, boundary_condition, solver, fem_solver):


    # INITIATE DATA FOR NON-LINEAR ANALYSIS
    # NodalForces, Residual = InitiateNonlinearAnalysisData(MainData,mesh,material)
    NodalForces, Residual = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float32), \
        np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float32)
    # SET NON-LINEAR PARAMETERS
    # Tolerance = MainData.AssemblyParameters.NRTolerance
    # LoadIncrement = MainData.AssemblyParameters.LoadIncrements
    LoadIncrement = fem_solver.number_of_load_increments
    ResidualNorm = { 'Increment_'+str(Increment) : [] for Increment in range(0,LoadIncrement) }
    
    # ALLOCATE FOR SOLUTION FIELDS
    # TotalDisp = np.zeros((mesh.points.shape[0],MainData.nvar,LoadIncrement),dtype=np.float64)
    TotalDisp = np.zeros((mesh.points.shape[0],formulation.nvar,LoadIncrement),dtype=np.float32)

    # PRE-ASSEMBLY
    print 'Assembling the system and acquiring neccessary information for the analysis...'
    tAssembly=time()

    # APPLY DIRICHELT BOUNDARY CONDITIONS AND GET DIRICHLET RELATED FORCES
    # ColumnsIn, ColumnsOut, AppliedDirichlet = MainData.boundary_condition.GetDirichletBoundaryConditions(MainData,mesh,material)
    # boundary_condition.GetDirichletBoundaryConditions(MainData,mesh,material)
    boundary_condition.GetDirichletBoundaryConditions(formulation, mesh, material, solver, fem_solver)
    # ALLOCATE FOR GEOMETRY - GetDirichletBoundaryConditions CHANGES THE MESH 
    # SO EULERX SHOULD BE ALLOCATED AFTERWARDS 
    Eulerx = np.copy(mesh.points)
    # GET EXTERNAL NODAL FORCES
    # boundary_condition.GetExternalForces(mesh,material)

    # FIND PURE NEUMANN (EXTERNAL) NODAL FORCE VECTOR
    # NeumannForces = AssemblyForces(MainData,mesh)
    # NeumannForces = AssemblyForces_Cheap(MainData,mesh)
    # NeumannForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
    NeumannForces = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float32)
    # FORCES RESULTING FROM DIRICHLET BOUNDARY CONDITIONS
    # DirichletForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
    DirichletForces = np.zeros((mesh.points.shape[0]*formulation.nvar,1),dtype=np.float32)


    # ADOPT A DIFFERENT PATH FOR INCREMENTAL LINEAR ELASTICITY
    if formulation.fields == "mechanics" and fem_solver.analysis_nature != "nonlinear":     
        # MAKE A COPY OF MESH, AS MESH POINTS WILL BE OVERWRITTEN
        vmesh = deepcopy(mesh)
        # TotalDisp = IncrementalLinearElasticitySolver(MainData,vmesh,material,TotalDisp,
            # Eulerx,LoadIncrement,NeumannForces,ColumnsIn,ColumnsOut,AppliedDirichlet)
        TotalDisp = IncrementalLinearElasticitySolver(function_spaces, formulation, vmesh, material,
            boundary_condition, solver, fem_solver, TotalDisp, Eulerx, LoadIncrement, NeumannForces)
        del vmesh

        # ADD EACH INCREMENTAL CONTRIBUTION TO MAKE IT CONSISTENT WITH THE NONLINEAR ANALYSYS
        for i in range(TotalDisp.shape[2]-1,0,-1):
            TotalDisp[:,:,i] = np.sum(TotalDisp[:,:,:i+1],axis=2)

        return TotalDisp

    # ASSEMBLE STIFFNESS MATRIX AND TRACTION FORCES
    K, TractionForces = Assembly(function_spaces[0], formulation, mesh, material, fem_solver, Eulerx, 
        np.zeros((mesh.points.shape[0],1),dtype=np.float64))[:2]

    
    # GET DIRICHLET FORCES
    DirichletForces = boundary_condition.ApplyDirichletGetReducedMatrices(K,DirichletForces,
        boundary_condition.applied_dirichlet)[2]
    # boundary_condition.ApplyDirichletGetReducedMatrices(K,boundary_condition.dirichlet_forces,
        # boundary_condition.applied_dirichlet)

    if fem_solver.analysis_nature == 'nonlinear':
        print 'Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'sec'
    else:
        print 'Finished the assembly stage. Time elapsed was', time()-tAssembly, 'sec'


    if fem_solver.analysis_type != 'static':
        TotalDisp = DynamicSolver(function_spaces, formulation, solver, fem_solver, LoadIncrement,K,M,
            DirichletForces,NeumannForces,NodalForces,Residual,
            ResidualNorm,mesh,TotalDisp,Eulerx,material, boundary_condition)
    else:
        TotalDisp = StaticSolver(function_spaces, formulation, solver, fem_solver, LoadIncrement,K,
            DirichletForces,NeumannForces,NodalForces,Residual,
            ResidualNorm,mesh,TotalDisp,Eulerx,material, boundary_condition)


    fem_solver.NRConvergence = ResidualNorm

    return TotalDisp






