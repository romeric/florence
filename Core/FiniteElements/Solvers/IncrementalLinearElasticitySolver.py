import numpy as np
from scipy.sparse.linalg import onenormest 
from SparseSolver import SparseSolver

from Core.FiniteElements.Assembly import *
from Core.FiniteElements.ApplyDirichletBoundaryConditions import *
from Core.FiniteElements.PostProcess import *
from copy import deepcopy

def IncrementalLinearElasticitySolver(MainData,mesh,TotalDisp,Eulerx,LoadIncrement,NeumannForces,
        ColumnsIn,ColumnsOut,AppliedDirichlet):
    """An icremental linear elasticity solver, in which the geometry is updated 
        and the remaining quantities such as stresses and Hessians are based on Prestress flag. 
        In this approach instead of solving the problem inside a non-linear routine,
        a somewhat explicit and more efficient way is adopted to avoid pre-assembly of the system
        of equations needed for non-linear analysis
    """

    jacobian_postprocess = PostProcess()

    LoadFactor = 1./LoadIncrement
    for Increment in range(LoadIncrement):
        # COMPUTE INCREMENTAL FORCES
        NodalForces = LoadFactor*NeumannForces
        AppliedDirichletInc = LoadFactor*AppliedDirichlet
        # DIRICHLET FORCES IS SET TO ZERO EVERY TIME
        DirichletForces = np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float64)
        Residual = DirichletForces + NodalForces

        t_assembly = time()
        # IF STRESSES ARE TO BE CALCULATED 
        if MainData.Prestress:
            # GET THE MESH COORDINATES FOR LAST INCREMENT 
            mesh.points -= TotalDisp[:,:MainData.ndim,Increment-1]
            # ASSEMBLE
            K, TractionForces = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[:2]
            # PUT THE MESH POINTS BACK TO CURRENT COORDINATE
            mesh.points += TotalDisp[:,:MainData.ndim,Increment-1]
            # FIND THE RESIDUAL
            Residual[ColumnsIn] = TractionForces[ColumnsIn] - NodalForces[ColumnsIn]
        else:
            # ASSEMBLE
            K = Assembly(MainData,mesh,Eulerx,np.zeros_like(mesh.points))[0]
        print 'Finished assembling the system of equations. Time elapsed is', time() - t_assembly, 'seconds'
        # APPLY DIRICHLET BOUNDARY CONDITIONS & GET REDUCED MATRICES 
        K_b, F_b = ApplyDirichletGetReducedMatrices(K,Residual,ColumnsIn,ColumnsOut,AppliedDirichletInc,MainData.Analysis,[])[:2]

        # SOLVE THE SYSTEM
        t_solver=time()

        sol = SparseSolver(K_b,F_b,MainData.solve.type,sub_type=MainData.solve.sub_type,tol=MainData.solve.tol)

        if Increment==MainData.AssemblyParameters.LoadIncrements-1:
            # MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
            MainData.solve.condA = onenormest(K_b) # REMOVE THIS
        t_solver = time()-t_solver

        dU = PostProcess().TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,0,K.shape[0]) 
        # STORE TOTAL SOLUTION DATA
        TotalDisp[:,:,Increment] += dU

        # UPDATE MESH GEOMETRY
        mesh.points += TotalDisp[:,:MainData.ndim,Increment]    
        Eulerx = np.copy(mesh.points)

        if LoadIncrement > 1:
            print "Finished load increment "+str(Increment)+" for incrementally linearised problem. Solver time is", t_solver
        else:
            print "Finished load increment "+str(Increment)+" for linear problem. Solver time is", t_solver

        # COMPUTE SCALED JACBIAN FOR THE MESH
        if Increment == LoadIncrement - 1:
            smesh = deepcopy(mesh)
            smesh.points -= TotalDisp[:,:,-1] 
            # jacobian_postprocess.MeshQualityMeasures(MainData,mesh,np.zeros_like(TotalDisp[:,:,:Increment+1]),show_plot=False)
            jacobian_postprocess.MeshQualityMeasures(MainData,smesh,TotalDisp,show_plot=False)
            # jacobian_postprocess.MeshQualityMeasures(MainData,smesh,TotalDisp[:,:,:-1],show_plot=False)

            # CHECK IF THE WING2D MESH P2 IS DISPLACED CORRECTLY
            # print mesh.points[906,:], TotalDisp[906,:,-1]
            # print mesh.points[887,:], TotalDisp[887,:,-1]
            # print mesh.points[3715,:], TotalDisp[3715,:,-1]

            # PostProcess.HighOrderPatchPlot(MainData,mesh,np.zeros_like(TotalDisp))
            # import matplotlib.pyplot as plt
            # plt.show()

    jacobian_postprocess.is_scaledjacobian_computed
    MainData.isScaledJacobianComputed = True


    return TotalDisp