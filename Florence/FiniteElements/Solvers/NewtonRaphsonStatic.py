from time import time
import numpy as np
import numpy.linalg as la 
# from SparseSolver import SparseSolver

from Florence.FiniteElements.StaticCondensationGlobal import *
from Florence.FiniteElements.PostProcess import *
from Florence.FiniteElements.Assembly import *


def NewtonRaphson(MainData,Increment,K,NodalForces,Residual,
        ResidualNorm,mesh,TotalDisp,Eulerx,material,
        boundary_condition,AppliedDirichletInc):

    Tolerance = MainData.AssemblyParameters.NRTolerance
    LoadIncrement = MainData.AssemblyParameters.LoadIncrements
    Iter = 0

    # NormForces = la.norm(NodalForces[ColumnsIn])
    NormForces = MainData.NormForces

    # AVOID DIVISION BY ZERO
    if la.norm(Residual[boundary_condition.columns_in]) < 1e-14:
        NormForces = 1e-14

    # CREATE POST-PROCESS OBJECT ONCE
    post_process = PostProcess(MainData.ndim,MainData.nvar)
    post_process.SetAnalysis(MainData.AnalysisType,MainData.Analysis)


    while np.abs(la.norm(Residual[boundary_condition.columns_in])/NormForces) > Tolerance:
        # APPLY INCREMENTAL DIRICHLET BOUNDARY CONDITIONS
        K_b, F_b = boundary_condition.GetReducedMatrices(K,Residual)[:2]

        # SOLVE THE SYSTEM
        # # CHECK FOR THE CONDITION NUMBER OF THE SYSTEM
        # if Increment==MainData.AssemblyParameters.LoadIncrements-1 and Iter>1:
        #     # MainData.solve.condA = np.linalg.cond(K_b.todense()) # REMOVE THIS
        #     MainData.solver.condA = onenormest(K_b) # REMOVE THIS
        sol = MainData.solver.Solve(K_b,-F_b)

        # GET THE TOTAL SOLUTION AND ITS COMPONENTS SUCH AS UX, UY, UZ, PHI ETC
        dU = post_process.TotalComponentSol(sol,boundary_condition.columns_in,
            boundary_condition.columns_out,AppliedDirichletInc,Iter,K.shape[0]) 

        # UPDATE THE FIELDS
        TotalDisp[:,:,Increment] += dU
        # UPDATE THE GEOMETRY
        Eulerx = mesh.points + TotalDisp[:,:MainData.ndim,Increment]            
        # UPDATE & SAVE ITERATION NUMBER
        MainData.AssemblyParameters.IterationNumber +=1
        # RE-ASSEMBLE - COMPUTE INTERNAL TRACTION FORCES (BE CAREFUL ABOUT THE -1 INDEX IN HERE)
        K, TractionForces = Assembly(MainData,mesh,material,Eulerx,TotalDisp[:,MainData.nvar-1,Increment,None])[:2]
        # FIND THE RESIDUAL
        Residual[boundary_condition.columns_in] = TractionForces[boundary_condition.columns_in] \
        - NodalForces[boundary_condition.columns_in]
        # SAVE THE NORM 
        NormForces = MainData.NormForces
        ResidualNorm['Increment_'+str(Increment)] = np.append(ResidualNorm['Increment_'+str(Increment)],\
            np.abs(la.norm(Residual[boundary_condition.columns_in])/NormForces))
        
        print 'Iteration number', Iter, 'for load increment', Increment, 'with a residual of \t\t', \
            np.abs(la.norm(Residual[boundary_condition.columns_in])/NormForces)          

        # UPDATE ITERATION NUMBER
        Iter +=1

        # if Iter==MainData.AssemblyParameters.MaxIter:
            # raise StopIteration("\n\nNewton Raphson did not converge! Maximum number of iterations reached.")

        if Iter==MainData.AssemblyParameters.MaxIter or ResidualNorm['Increment_'+str(Increment)][-1] > 500:
            MainData.AssemblyParameters.FailedToConverge = True
            break
        if np.isnan(np.abs(la.norm(Residual[boundary_condition.columns_in])/NormForces)):
            MainData.AssemblyParameters.FailedToConverge = True
            break


    return TotalDisp





def NewtonRaphsonWithArcLength(Increment,MainData,K,F,M,NodalForces,Residual,
    ResidualNorm,mesh,TotalDisp,Eulerx,AppliedDirichletInc):

    raise NotImplementedError('Arc length not implemented yet')