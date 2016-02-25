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

def MainSolver(MainData, mesh, material, boundary_condition):

    # INITIATE DATA FOR NON-LINEAR ANALYSIS
    NodalForces, Residual = InitiateNonlinearAnalysisData(MainData,mesh,material)
    # SET NON-LINEAR PARAMETERS
    Tolerance = MainData.AssemblyParameters.NRTolerance
    LoadIncrement = MainData.AssemblyParameters.LoadIncrements
    ResidualNorm = { 'Increment_'+str(Increment) : [] for Increment in range(0,LoadIncrement) }
    
    # ALLOCATE FOR SOLUTION FIELDS
    # TotalDisp = np.zeros((mesh.points.shape[0],MainData.nvar,LoadIncrement),dtype=np.float64)
    TotalDisp = np.zeros((mesh.points.shape[0],MainData.nvar,LoadIncrement),dtype=np.float32)

    # PRE-ASSEMBLY
    print 'Assembling the system and acquiring neccessary information for the analysis...'
    tAssembly=time()

    # APPLY DIRICHELT BOUNDARY CONDITIONS AND GET DIRICHLET RELATED FORCES
    # ColumnsIn, ColumnsOut, AppliedDirichlet = MainData.boundary_condition.GetDirichletBoundaryConditions(MainData,mesh,material)
    boundary_condition.GetDirichletBoundaryConditions(MainData,mesh,material)
    # ALLOCATE FOR GEOMETRY - GetDirichletBoundaryConditions CHANGES THE MESH 
    # SO EULERX SHOULD BE ALLOCATED AFTERWARDS 
    Eulerx = np.copy(mesh.points)

    # GET EXTERNAL NODAL FORCES
    # boundary_condition.GetExternalForces(mesh,material)

    # FIND PURE NEUMANN (EXTERNAL) NODAL FORCE VECTOR
    # NeumannForces = AssemblyForces(MainData,mesh)
    # NeumannForces = AssemblyForces_Cheap(MainData,mesh)
    # NeumannForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
    NeumannForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float32)
    # FORCES RESULTING FROM DIRICHLET BOUNDARY CONDITIONS
    # DirichletForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
    DirichletForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float32)


    # ADOPT A DIFFERENT PATH FOR INCREMENTAL LINEAR ELASTICITY
    if MainData.Fields == "Mechanics" and MainData.AnalysisType != "Nonlinear":     
        # MAKE A COPY OF MESH, AS MESH POINTS WILL BE OVERWRITTEN
        vmesh = deepcopy(mesh)
        # TotalDisp = IncrementalLinearElasticitySolver(MainData,vmesh,material,TotalDisp,
            # Eulerx,LoadIncrement,NeumannForces,ColumnsIn,ColumnsOut,AppliedDirichlet)
        TotalDisp = IncrementalLinearElasticitySolver(MainData,vmesh,material,boundary_condition,
            TotalDisp,Eulerx,LoadIncrement,NeumannForces)
        del vmesh

        # ADD EACH INCREMENTAL CONTRIBUTION TO MAKE IT CONSISTENT WITH THE NONLINEAR ANALYSYS
        for i in range(TotalDisp.shape[2]-1,0,-1):
            TotalDisp[:,:,i] = np.sum(TotalDisp[:,:,:i+1],axis=2)

        return TotalDisp

    # ASSEMBLE STIFFNESS MATRIX AND TRACTION FORCES
    K,TractionForces = Assembly(MainData,mesh,material,Eulerx,np.zeros((mesh.points.shape[0],1),dtype=np.float32))[:2]
    
    # GET DIRICHLET FORCES
    DirichletForces = boundary_condition.ApplyDirichletGetReducedMatrices(K,DirichletForces,
        boundary_condition.applied_dirichlet)[2]
    # boundary_condition.ApplyDirichletGetReducedMatrices(K,boundary_condition.dirichlet_forces,
        # boundary_condition.applied_dirichlet)

    if MainData.AnalysisType=='Nonlinear':
        print 'Finished all pre-processing stage. Time elapsed was', time()-tAssembly, 'sec'
    else:
        print 'Finished the assembly stage. Time elapsed was', time()-tAssembly, 'sec'


    if MainData.Analysis != 'Static':
        TotalDisp = DynamicSolver(LoadIncrement,MainData,K,M,NodalForces,Residual,
            ResidualNorm,mesh,TotalDisp,Eulerx,material,boundary_condition)
    else:
        TotalDisp = StaticSolver(MainData,LoadIncrement,K,DirichletForces,NeumannForces,NodalForces,Residual,
            ResidualNorm,mesh,TotalDisp,Eulerx,material, boundary_condition)
        # TotalDisp = StaticSolver(MainData,LoadIncrement,K,NodalForces,Residual,
            # ResidualNorm,mesh,TotalDisp,Eulerx,material,boundary_condition)


    MainData.NRConvergence = ResidualNorm

    return TotalDisp






